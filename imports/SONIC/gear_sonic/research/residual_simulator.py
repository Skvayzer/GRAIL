"""Isaac wrapper bridge for the future approved M2 pilot.

Importing/constructing this module does not start Isaac, access a robot, load
weights, or run a training update. Exploration must be explicitly opted into;
there is deliberately no user-facing training launcher until curriculum and
approval gates are finished. Existing frozen evaluation entry points are not
changed to use this bridge.
"""
import copy

import torch

from .observation_shadow import state_hash
from .pre_reset_capture import PreResetCapture
from .residual_learning import finite


class SimulatorCollector:
    def __init__(self, wrapper, frozen_policy, oracle, state_sampler, collector, *,
                 allow_simulation_exploration=False):
        if allow_simulation_exploration is not True:
            raise ValueError("Simulator exploration is opt-in; default is no changed actions")
        if frozen_policy.training or not 1 <= wrapper.env.num_envs <= 4:
            raise ValueError("Frozen evaluation decoder and bounded simulator batch required")
        self.wrapper, self.env = wrapper, wrapper.env
        self.original = frozen_policy
        self.actor = copy.deepcopy(frozen_policy).eval().requires_grad_(False)
        self.original_hash = state_hash(self.original)
        if state_hash(self.actor) != self.original_hash:
            raise ValueError("Frozen decoder clone changed checkpoint state")
        self.oracle, self.state, self.collector = oracle, state_sampler, collector
        self.task = getattr(self.env, "research_avoidance", None)
        if self.task is None:
            raise ValueError("Explicit avoidance task context required")
        self.tap = PreResetCapture(self.env, self._capture)
        self.current, self.active = None, False
        self.steps = self.terminations = self.timeouts = 0

    def __enter__(self):
        self.tap.__enter__()
        self.active = True
        return self

    @torch.no_grad()
    def _capture(self, reward):
        if self.current is None:
            raise ValueError("Pre-reset capture has no matching sampled action")
        before_packet, before_state = self.current
        term, timeout = self.env.reset_terminated.clone(), self.env.reset_time_outs.clone()
        # NaNs explicitly mark unqueried true-terminal observations. They never
        # enter a model: termination removes bootstrap. Select remaining physical
        # rows BEFORE quaternion/geometry operations to isolate a failed robot.
        final_state = torch.full_like(before_state, float("nan"))
        final_packet = {key: (torch.zeros_like(value) if value.dtype == torch.bool
                             else torch.full_like(value, float("nan")))
                        for key, value in before_packet.items()}
        needed = ~term
        if needed.any():
            packet = self.oracle.sample(env_mask=needed)
            state = self.state.state(phase_offset=1, next_physical=self.state.physical(env_mask=needed), env_mask=needed)
            final_state[needed] = state
            for key, value in packet.items():
                final_packet[key][needed] = value
        return dict(reward=reward.detach().clone(), terminated=term, truncated=timeout,
                    packet=final_packet, state=final_state)

    @torch.no_grad()
    def step(self, obs_dict):
        if not self.active or self.current is not None:
            raise ValueError("Active capture context and one paired step required")
        try:
            packet, state = self.oracle.sample(), self.state.state()
            residual = self.collector.sample(packet, state)
            self.current = packet, state
            # Match the pinned frozen evaluator's one-step decoder buffer. The
            # upstream observation-manager history and separate ten-step learner
            # history still exist; neither is replaced with a different window.
            self.actor.init_rollout()
            inputs = {key: value.detach().clone() for key, value in obs_dict.items()}
            joints = self.actor.act_inference(inputs, latent_residual=residual,
                                              latent_residual_mode="post_quantization")
            if joints.shape != (self.env.num_envs, 29):
                raise ValueError("Decoded actions differ from 29-joint simulator contract")
            finite(joints)
            if any(not torch.equal(value, obs_dict[key]) for key, value in inputs.items()):
                raise ValueError("Frozen decoder mutated its input observations")
            self.task.before_step()
            result = self.wrapper.step(dict(actions=joints, obs_dict=self.actor.obs_dict_buffer))
            captured = self.tap.take()
            obs, rewards, dones, infos = result
            dones = dones.reshape(-1).bool()
            if (not torch.equal(dones, captured["terminated"] | captured["truncated"])
                    or not torch.equal(rewards.reshape(-1), captured["reward"])):
                raise ValueError("Wrapper recovery changed pre-reset rewards/flags; discard rollout")
            self.collector.outcome(residual, captured["reward"], captured["terminated"], captured["truncated"],
                                   captured["packet"], captured["state"])
            self.task.outcome(dones)
            self.state.history.commit(self.state.physical(), reset=dones)
            self.current = None
            self.steps += 1
            self.terminations += int(captured["terminated"].sum())
            self.timeouts += int((captured["truncated"] & ~captured["terminated"]).sum())
            return obs, rewards, dones, infos
        except Exception:
            self.collector.failed = True
            raise

    def __exit__(self, kind, error, traceback):
        self.active = False
        self.tap.__exit__(kind, error, traceback)
        unchanged = state_hash(self.actor) == self.original_hash == state_hash(self.original)
        if error is not None or not unchanged or self.current is not None:
            self.collector.failed = True
        if error is None and (not unchanged or self.current is not None or self.tap.calls != self.steps):
            raise ValueError("Frozen decoder changed or simulator collection was incomplete")
        return False
