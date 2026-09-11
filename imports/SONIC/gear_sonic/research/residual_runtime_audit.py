"""Zero-update integration audit, NOT a trainer or a stochastic PPO rollout.

The released actor alone supplies physics actions. This records learner states
and values before auto-reset, checks non-reset next-state parity, resets only
the corresponding physical histories and tests time-limit bootstrapping. No
optimizer, exploration, added physics or observation-manager calls are used.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .learner_state import LearnerStateSampler
from .observation_shadow import state_hash
from .pre_reset_capture import PreResetCapture
from .residual_learning import ResidualActorCritic, generalized_advantage
from .runtime_observation import RuntimeObservation


class ResidualRuntimeAudit:
    def __init__(self, wrapper, cat_audit, policy, output):
        self.wrapper, self.env, self.policy = wrapper, wrapper.env, policy
        if policy.training or not 1 <= self.env.num_envs <= 4:
            raise ValueError("Runtime audit requires evaluation and 1..4 environments")
        self.output = Path(output)
        self.oracle = RuntimeObservation(wrapper, cat_audit, self.output.parent)
        self.state = LearnerStateSampler(wrapper)
        self.learner = ResidualActorCritic(len(cat_audit.probes), self.state.manifest()["state_dim"],
            latent_dim=policy.actor_module.token_total_dim).to(self.env.device).eval()
        self.clone = copy.deepcopy(policy).eval().requires_grad_(False)
        self.backbone_hash, self.learner_hash = state_hash(policy.actor_module), state_hash(self.learner)
        if state_hash(self.clone.actor_module) != self.backbone_hash:
            raise ValueError("Frozen runtime clone differs from restored actor")
        self.tap = PreResetCapture(self.env, self._pre_reset)
        self.rows, self.current = [], None
        self.completed = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        self.max_state_error = 0.
        self.report = dict(schema="grail-cat-residual-runtime-preflight-v1", simulation_only=True,
            num_envs=self.env.num_envs, optimizer_steps=0, exploration_enabled=False,
            action_source="untouched released policy.action_mean; learner and clone never supply env.step actions",
            training_data=False, data_note="Deterministic diagnostic transitions are NOT sampled PPO behavior data",
            learner=self.learner.manifest(), state=self.state.manifest(), oracle=self.oracle.contract,
            base_backbone_sha256=self.backbone_hash, learner_sha256=self.learner_hash,
            capture="reward_manager.compute return, before recorder/reset/command advancement",
            timeout_bootstrap="final physical history and +1 clamped reference query; never post-reset state",
            physics_steps_added=0, original_rewards_changed=hasattr(self.env, "research_avoidance"),
            original_terminations_changed=hasattr(self.env, "research_avoidance"), original_observations_changed=False)

    def __enter__(self):
        self.tap.__enter__()
        return self

    @torch.no_grad()
    def sample(self, buffered_obs, live_actions):
        if self.current is not None or len(self.rows) >= 500:
            raise ValueError("Each runtime sample must pair with exactly one bounded physics step")
        cpu_rng = torch.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(self.env.device).clone()
        packet, state = self.oracle.sample(), self.state.state()
        if not packet["valid"].all():
            raise ValueError("Invalid current oracle observation; no silent learning fallback")
        residual, value = self.learner.deterministic(packet, state)
        if torch.count_nonzero(residual):
            raise ValueError("Preflight only permits the untouched zero-initialized residual")
        inputs = {k: v.detach().clone() for k, v in buffered_obs.items()}
        candidate = self.clone.forward(inputs, latent_residual=residual,
                                       latent_residual_mode="post_quantization")[:, -1]
        if not torch.isfinite(live_actions).all() or not torch.equal(candidate, live_actions):
            raise ValueError("Runtime zero-residual clone changed released actor actions")
        if any(not torch.equal(v, buffered_obs[k]) for k, v in inputs.items()):
            raise ValueError("Runtime clone mutated the actor observation buffer")
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or not torch.equal(gpu_rng, torch.cuda.get_rng_state(self.env.device))):
            raise ValueError("Runtime observation/parity check consumed global RNG")
        self.current = dict(state=state.detach().clone(), value=value.detach().clone(),
                            reference_step=self.wrapper.motion_command.time_steps.clone())

    @torch.no_grad()
    def _pre_reset(self, reward):
        if self.current is None:
            raise ValueError("Pre-reset capture requires a corresponding pre-action state")
        cpu_rng, gpu_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(self.env.device).clone()
        state = self.state.state(phase_offset=1, next_physical=self.state.physical())
        packet = self.oracle.sample()
        # This integration control is expected to remain valid even at timeout.
        # Failure recovery/invalid-observation training semantics are separate.
        if not packet["valid"].all():
            raise ValueError("Invalid pre-reset oracle observation; cannot invent bootstrap values")
        _, value = self.learner.deterministic(packet, state)
        result = dict(next_state=state.detach().clone(), next_value=value.detach().clone(),
            reward=reward.detach().clone(), terminated=self.env.reset_terminated.clone(),
            truncated=self.env.reset_time_outs.clone())
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or not torch.equal(gpu_rng, torch.cuda.get_rng_state(self.env.device))):
            raise ValueError("Pre-reset capture consumed global RNG")
        return result

    @torch.no_grad()
    def outcome(self, dones, rewards):
        captured = self.tap.take()
        dones = dones.reshape(-1).bool()
        if not torch.equal(dones, captured["terminated"] | captured["truncated"]):
            raise ValueError("Wrapper reset/recovery differs from captured termination flags")
        if not torch.equal(rewards.reshape(-1), captured["reward"]):
            raise ValueError("Wrapper rewards differ from unchanged pre-reset rewards")
        physical = self.state.physical()
        post = self.state.state(next_physical=physical, reset=dones)
        packet = self.oracle.sample()
        if not packet["valid"].all():
            raise ValueError("Invalid post-step/reset oracle observation")
        _, post_value = self.learner.deterministic(packet, post)
        if (~dones).any():
            difference = float((post[~dones]-captured["next_state"][~dones]).abs().max())
            self.max_state_error = max(self.max_state_error, difference)
            if difference > 3e-5:
                raise ValueError(f"Pre-reset +1 reference state differs from actual non-reset next state: {difference}")
            if not torch.allclose(post_value[~dones], captured["next_value"][~dones], atol=3e-5, rtol=0):
                raise ValueError("Pre-reset value differs from actual non-reset next value")
        self.state.history.commit(physical, reset=dones)
        if dones.any() and not torch.equal(self.state.history.data[dones],
                physical[dones, None].expand_as(self.state.history.data[dones])):
            raise ValueError("Episode history was not cleared independently at reset")
        if not torch.equal(self.state.state(), post):
            raise ValueError("History commit differs from candidate post-step state")
        row = dict(**self.current, **captured, dones=dones, post_step_state=post, post_step_value=post_value)
        self.rows.append({k: v.detach().cpu().numpy().copy() for k, v in row.items()})
        self.current = None
        self.completed |= dones
        if len(self.rows) % 100 == 0 or dones.any():
            print(f"RESIDUAL_PREFLIGHT steps={len(self.rows)} completed={self.completed.tolist()} "
                  f"next_state_error={self.max_state_error:.3g}", flush=True)

    def __exit__(self, kind, error, traceback):
        self.tap.__exit__(kind, error, traceback)
        unchanged = state_hash(self.policy.actor_module) == self.backbone_hash == state_hash(self.clone.actor_module)
        learner_same = state_hash(self.learner) == self.learner_hash
        complete = bool(error is None and self.rows and self.current is None and self.completed.all()
                        and unchanged and learner_same and self.tap.calls == len(self.rows))
        if self.rows:
            arrays = {k: np.stack([row[k] for row in self.rows]) for k in self.rows[0]}
            tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
            advantages, returns = generalized_advantage(tensors["reward"], tensors["value"], tensors["next_value"],
                                                       tensors["terminated"], tensors["truncated"])
            arrays.update(advantages=advantages.numpy(), returns=returns.numpy())
            np.savez_compressed(self.output.with_suffix(".npz"), **arrays)
            timeout = arrays["truncated"] & ~arrays["terminated"]
            self.report.update(data_file=self.output.with_suffix(".npz").name,
                data_sha256=hashlib.sha256(self.output.with_suffix(".npz").read_bytes()).hexdigest(),
                failure_count=int(arrays["terminated"].sum()), timeout_count=int(timeout.sum()),
                timeout_state_reset_separation=float(np.max(np.abs(
                    arrays["next_state"][timeout]-arrays["post_step_state"][timeout]))) if timeout.any() else 0.)
            complete = complete and not arrays["terminated"].any() and timeout.any()
        self.report.update(complete=bool(complete), error=str(error) if error else None, steps=len(self.rows),
            completed_envs=self.completed.tolist(), capture_calls=self.tap.calls,
            base_backbone_unchanged=unchanged, learner_unchanged=learner_same,
            actor_action_parity_exact=bool(self.rows), max_nonreset_next_state_error=self.max_state_error)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")
        print(f"RESIDUAL_RUNTIME_PREFLIGHT {self.output} complete={complete}", flush=True)
        if error is None and not complete:
            raise ValueError("Incomplete zero-update residual runtime preflight")
        return False
