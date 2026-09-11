"""Bounded M2 collection/update engine, with gradient-only updates by default.

No Isaac startup, robot API, artifact downloads or automatic training here.
The separately enabled simulator launcher supplies PRE-RESET final states.
`optimize=True` is deliberately explicit and must be guarded by that launcher's
user-approval and scene-feasibility checks. Tests use synthetic data only and
never execute a real optimizer step.
"""
from dataclasses import asdict, dataclass, field
import math

import torch

from .learning_checkpoint import validate_optimizer, finite_tree
from .residual_learning import LossConfig, clipped_ppo_loss, finite, learner_precision
from .residual_rollout import ResidualRollout, minibatches


@dataclass(frozen=True)
class UpdateConfig:
    horizon: int = 32
    epochs: int = 4
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    max_gradient_norm: float = 1.
    target_kl: float = .02
    gamma: float = .99
    gae_lambda: float = .95
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        for name, lower, upper in (("horizon", 2, 512), ("epochs", 1, 8), ("minibatch_size", 2, 8192)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Bounded integer update configuration required")
        if (not all(math.isfinite(getattr(self, n)) for n in (
                "learning_rate", "max_gradient_norm", "target_kl", "gamma", "gae_lambda"))
                or not 0 < self.learning_rate <= .001 or not 0 < self.max_gradient_norm <= 10
                or not 0 < self.target_kl <= .1 or not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1
                or not isinstance(self.loss, LossConfig)):
            raise ValueError("Invalid learning/gradient/KL/discount configuration")

    def manifest(self):
        return dict(schema="grail-cat-bounded-ppo-v1", **asdict(self),
                    optimizer="Adam, fixed learning rate", entropy="pre-tanh", mixed_precision=False)


def packet_from(data):
    return {k: data["packet/"+k] for k in ("volume", "probes", "guidance", "valid")}


@torch.no_grad()
def bootstrap_value(model, packet, state, terminated):
    """Do not score true-terminal observations; timeout states MUST be valid.

    A physically failed terminal may lack geometry/guidance. Its value is zero
    by termination semantics, not an invented zero observation. Invalid ongoing
    or timeout states abort collection and must never reach an optimizer.
    """
    if terminated.shape != (len(state),) or terminated.dtype != torch.bool:
        raise ValueError("Explicit terminal flags required for bootstrap")
    value = torch.zeros(len(state), device=state.device, dtype=state.dtype)
    selected = ~terminated
    if selected.any():
        p = {key: item[selected] for key, item in packet.items()}
        if not p["valid"].all():
            raise ValueError("Invalid nonterminal/timeout final observation; discard rollout")
        _, current = model.deterministic(p, state[selected])
        value[selected] = current
    finite(value)
    return value


class PPOUpdater:
    def __init__(self, model, optimizer, generator, config=UpdateConfig()):
        validate_optimizer(model, optimizer)
        if (type(optimizer) is not torch.optim.Adam or not isinstance(generator, torch.Generator)
                or any(group["lr"] != config.learning_rate for group in optimizer.param_groups)):
            raise ValueError("Contracted Adam optimizer and explicit learner RNG required")
        self.model, self.optimizer, self.generator, self.config = model, optimizer, generator, config
        self.optimizer_steps = 0

    @torch.no_grad()
    def _validate_behavior(self, data):
        # Never train on a frozen-controller shadow trajectory or data from an
        # older residual version. Likelihood/value checks precede every update.
        for start in range(0, len(data["returns"]), self.config.minibatch_size):
            b = {key: value[start:start+self.config.minibatch_size] for key, value in data.items()}
            logp, _, value = self.model.evaluate_action(packet_from(b), b["state"], b["pre_tanh"])
            if (not torch.allclose(logp, b["old_log_prob"], atol=5e-4, rtol=0)
                    or not torch.allclose(value, b["old_values"], atol=5e-5, rtol=0)):
                raise ValueError("Stale or inconsistent sampled behavior likelihood/value: "
                    f"max logp error={float((logp-b['old_log_prob']).abs().max()):.8g}, "
                    f"value error={float((value-b['old_values']).abs().max()):.8g}")

    @learner_precision()
    def run(self, data, *, optimize=False):
        if type(optimize) is not bool:
            raise ValueError("Explicit boolean optimize choice required")
        validate_optimizer(self.model, self.optimizer)
        if (len(data["returns"]) < self.config.minibatch_size
                or any(v.requires_grad for v in data.values()) or not data["packet/valid"].all()):
            raise ValueError("Complete detached valid rollout required")
        finite(*data.values())
        self._validate_behavior(data)
        rng_before = self.generator.get_state().clone()
        was_training = self.model.training
        rows, steps, stopped = [], 0, False
        self.model.train()
        try:
            with torch.enable_grad():
                for epoch in range(self.config.epochs):
                    for batch in minibatches(data, self.config.minibatch_size, self.generator):
                        self.optimizer.zero_grad(set_to_none=True)
                        logp, entropy, value = self.model.evaluate_action(packet_from(batch), batch["state"], batch["pre_tanh"])
                        loss, metrics = clipped_ppo_loss(logp, batch["old_log_prob"], entropy, value,
                            batch["old_values"], batch["returns"], batch["advantages"], self.config.loss)
                        row = dict(epoch=epoch, samples=len(value), loss=float(loss.detach()), **metrics)
                        if metrics["approx_kl"] > self.config.target_kl:
                            row["skipped_kl"] = True
                            rows.append(row)
                            stopped = True
                            break  # No gradient or update for an over-KL batch.
                        loss.backward()
                        parameters = list(self.model.parameters())
                        if any(p.grad is None for p in parameters):
                            raise ValueError("A learner parameter is disconnected from the objective")
                        finite(*(p.grad for p in parameters))
                        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm,
                                                              error_if_nonfinite=True)
                        row.update(gradient_norm=float(norm), skipped_kl=False)
                        if optimize:
                            self.optimizer.step()
                            steps += 1
                            self.optimizer_steps += 1
                            finite_tree(self.model.state_dict())
                            finite_tree(self.optimizer.state_dict())
                        rows.append(row)
                    if stopped:
                        break
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train(was_training)
            if not optimize:
                self.generator.set_state(rng_before)  # Diagnostics don't change training shuffle state.
        return dict(schema="grail-cat-ppo-update-v1", mode="optimize" if optimize else "gradients_only",
                    optimizer_steps=steps, total_optimizer_steps=self.optimizer_steps,
                    kl_early_stop=stopped, samples=len(data["returns"]), batches=rows)


class ResidualCollector:
    """One live on-policy horizon, sample -> final capture -> outcome ordering.

    `sample`'s residual must be the value actually fed to the frozen decoder.
    The runtime must pass it back to `outcome`, which catches substituted zero
    or unrelated actions. Final packet/state refer to BEFORE any environment
    reset. After reset, the runtime independently clears histories and supplies
    the next new current state. This class never fabricates reset observations.
    """
    def __init__(self, model, generator, num_envs, config=UpdateConfig()):
        if not isinstance(generator, torch.Generator):
            raise ValueError("Dedicated learner RNG required")
        self.model, self.generator, self.config = model, generator, config
        self.buffer = ResidualRollout(config.horizon, num_envs)
        self.pending, self.failed = None, False
        self.parameter_versions = [p._version for p in model.parameters()]

    def _check(self):
        if self.failed:
            raise ValueError("Failed collection cannot be reused; discard its entire rollout")
        if self.parameter_versions != [p._version for p in self.model.parameters()]:
            self.failed = True
            raise ValueError("Learner weights changed during on-policy collection")

    @torch.no_grad()
    def sample(self, packet, state):
        self._check()
        if self.pending is not None or len(self.buffer.rows) >= self.buffer.steps:
            raise ValueError("Pair each sampled action with one outcome within the horizon")
        try:
            if packet["valid"].shape != (self.buffer.num_envs,):
                raise ValueError("Current observation batch differs from the simulator")
            action = self.model.sample(packet, state, self.generator)
        except Exception:
            self.failed = True
            raise
        self.pending = ({k: v.detach().clone() for k, v in packet.items()}, state.detach().clone(),
                        {k: v.detach().clone() for k, v in action.items()})
        return action["residual"].clone()

    @torch.no_grad()
    def outcome(self, executed_residual, reward, terminated, truncated, final_packet, final_state):
        self._check()
        try:
            if self.pending is None:
                raise ValueError("No corresponding sampled behavior action")
            packet, state, action = self.pending
            if not torch.equal(executed_residual, action["residual"]):
                raise ValueError("Executed latent differs from the sampled behavior action")
            value = bootstrap_value(self.model, final_packet, final_state, terminated)
            self.buffer.append(packet, state, action, reward, terminated, truncated, value)
            self.pending = None
        except Exception:
            self.failed = True
            raise

    def finish(self):
        self._check()
        if self.pending is not None:
            raise ValueError("A sampled action still lacks its final-state outcome")
        return self.buffer.finish(gamma=self.config.gamma, lam=self.config.gae_lambda)
