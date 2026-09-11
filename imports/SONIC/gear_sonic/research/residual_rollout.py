"""Bounded detached rollout storage for residual learning; no environment API."""
import torch

from .residual_learning import finite, generalized_advantage, normalized_advantages


class ResidualRollout:
    def __init__(self, steps, num_envs):
        if (type(steps) is not int or type(num_envs) is not int or not 2 <= steps <= 1024
                or not 1 <= num_envs <= 16 or steps*num_envs > 8192):
            raise ValueError("Bounded rollout capacity required")
        self.steps, self.num_envs = steps, num_envs
        self.rows = []
        self.closed = False

    def append(self, packet, state, action, reward, terminated, truncated, next_value):
        if self.closed or len(self.rows) >= self.steps:
            raise ValueError("Rollout is closed/full; create a fresh collection buffer")
        if (set(packet) != {"volume", "probes", "guidance", "valid"}
                or packet["valid"].shape != (self.num_envs,) or packet["valid"].dtype != torch.bool
                or not packet["valid"].all()):
            raise ValueError("Only explicitly valid action-state observations can be collected")
        scalar = dict(reward=reward, terminated=terminated, truncated=truncated, next_value=next_value,
                      old_log_prob=action["log_prob"], old_values=action["value"])
        if any(t.shape != (self.num_envs,) for t in scalar.values()):
            raise ValueError("Expected one reward/value/flag per environment")
        if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
            raise ValueError("Reset flags must be boolean and cannot be inferred from reward")
        row = dict(**{f"packet/{k}": v for k, v in packet.items()}, state=state,
                   pre_tanh=action["pre_tanh"], **scalar)
        if (state.ndim != 2 or action["pre_tanh"].ndim != 2
                or any(t.shape[0] != self.num_envs for t in row.values())
                or any(t.device != state.device for t in row.values())):
            raise ValueError("Inconsistent batch/device contract")
        finite(*row.values())
        if self.rows and any(t.shape != self.rows[0][k].shape or t.dtype != self.rows[0][k].dtype for k, t in row.items()):
            raise ValueError("Observation shape/dtype changed mid-rollout")
        self.rows.append({k: t.detach().clone() for k, t in row.items()})

    def finish(self, *, gamma=.99, lam=.95):
        if self.closed or len(self.rows) != self.steps:
            raise ValueError("Finish exactly one complete rollout")
        data = {k: torch.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        advantages, returns = generalized_advantage(data["reward"], data["old_values"], data["next_value"],
            data["terminated"], data["truncated"], gamma=gamma, lam=lam)
        data.update(returns=returns, advantages=normalized_advantages(advantages))
        self.closed = True
        # Time-major flattening is shared by observations, actions and targets.
        return {k: t.flatten(0, 1) for k, t in data.items()}


def minibatches(data, batch_size, generator):
    if not isinstance(generator, torch.Generator) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Explicit shuffle RNG and positive minibatch size required")
    n = len(data["returns"])
    if n < 2 or batch_size > n or any(len(t) != n for t in data.values()):
        raise ValueError("Invalid minibatch data")
    order = torch.randperm(n, generator=generator, device=data["returns"].device)
    for start in range(0, n, batch_size):
        selected = order[start:start+batch_size]
        yield {k: t[selected] for k, t in data.items()}
