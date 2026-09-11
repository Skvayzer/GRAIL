"""Simulator-independent latent actor/critic and PPO mathematics.

Preparation component, not a training launcher or hardware controller. Nothing
owns/loads a GRAIL checkpoint, starts an environment, or performs optimizer
updates here. The runtime must separately supply reviewed observations, correct
pre-reset bootstrap values and explicit training approval.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.distributions import Normal

from .obstacle_adapter import ObstacleAdapter


def finite(*values):
    if any(not torch.isfinite(value).all() for value in values):
        raise ValueError("Nonfinite learning tensor")


class ResidualActorCritic(nn.Module):
    """Shared obstacle/state trunk with Gaussian pre-tanh latent actor + value.

    State includes proprioception and reference conditioning (defined/versioned
    by the future runtime), not just obstacle geometry. The frozen GRAIL network
    is external. Sample z, store/log-probability z, decode bound*tanh(z). The
    transform's fixed Jacobian cancels in the PPO likelihood ratio; never score
    an already squashed action as if it were the underlying Gaussian sample.

    The deterministic mean starts at zero. Stochastic actions do NOT: enabling
    exploration changes the simulator actions even before the first update.
    Entropy reported here is PRE-TANH Gaussian entropy, not bounded-action entropy.
    """
    def __init__(self, probe_count, state_dim, latent_dim=64, bound=.1, initial_std=.1, seed=42):
        super().__init__()
        if (type(state_dim) is not int or not 1 <= state_dim <= 16384
                or not math.isfinite(initial_std) or not math.exp(-4) < initial_std < math.exp(-.5)):
            raise ValueError("Bounded state dimension and exploration std required")
        self.state_dim, self.latent_dim, self.bound = state_dim, latent_dim, bound
        self.obstacle = ObstacleAdapter(probe_count, latent_dim, bound=bound, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+1)
            self.state_encoder = nn.Sequential(nn.Linear(state_dim, 128), nn.SiLU())
            self.trunk = nn.Sequential(nn.Linear(256, 128), nn.SiLU())
            self.value_head = nn.Linear(128, 1)
            fraction = (math.log(initial_std)+4)/3.5
            self.std_parameter = nn.Parameter(torch.full((latent_dim,), math.log(fraction/(1-fraction))))

    def manifest(self):
        return dict(schema="grail-cat-residual-actor-critic-v1", state_dim=self.state_dim,
            probe_count=self.obstacle.probe_count, latent_dim=self.latent_dim, latent_bound=self.bound,
            log_std_bounds=[-4., -.5], action="pre-tanh Gaussian latent; bound*tanh(z) into frozen decoder",
            entropy="pre-tanh", value_encoder="shared obstacle and reference/proprioception trunk")

    def forward(self, packet, state):
        if state.shape != (len(packet["valid"]), self.state_dim):
            raise ValueError("State observation shape differs from policy contract")
        finite(state)
        features = self.trunk(torch.cat((self.obstacle.features(packet), self.state_encoder(state)), -1))
        mean = self.obstacle.head(features)
        log_std = -4.+3.5*torch.sigmoid(self.std_parameter)
        value = self.value_head(features).squeeze(-1)
        finite(mean, log_std, value)
        return mean, log_std.expand_as(mean), value

    def deterministic(self, packet, state):
        mean, _, value = self(packet, state)
        residual = torch.where(packet["valid"][:, None], self.bound*torch.tanh(mean), 0.)
        return residual, value

    def evaluate_action(self, packet, state, pre_tanh):
        if not packet["valid"].all():
            raise ValueError("Invalid obstacle packet cannot enter learning likelihoods")
        mean, log_std, value = self(packet, state)
        if pre_tanh.shape != mean.shape:
            raise ValueError("Store the full pre-tanh latent, not a joint or squashed action")
        finite(pre_tanh)
        distribution = Normal(mean, log_std.exp())
        return distribution.log_prob(pre_tanh).sum(-1), distribution.entropy().sum(-1), value

    @torch.no_grad()
    def sample(self, packet, state, generator):
        if not isinstance(generator, torch.Generator):
            raise ValueError("Exploration requires its own explicit RNG generator")
        if not packet["valid"].all():
            raise ValueError("Exploration requires valid observations; no stale/zero fallback")
        mean, log_std, value = self(packet, state)
        # A dedicated generator makes exploration reproducible without consuming
        # the frozen actor/scene RNG stream. Persist its state in runtime saves.
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        pre_tanh = mean+log_std.exp()*noise
        log_prob = Normal(mean, log_std.exp()).log_prob(pre_tanh).sum(-1)
        return dict(pre_tanh=pre_tanh, residual=self.bound*torch.tanh(pre_tanh),
                    log_prob=log_prob, value=value)


@torch.no_grad()
def generalized_advantage(rewards, values, next_values, terminated, truncated, *, gamma=.99, lam=.95):
    """GAE over T,N transitions; next_values refer to each PRE-RESET next state.

    True termination disables bootstrap. Timeout truncation bootstraps from its
    final observation but never recurses into a different episode. Termination
    wins when both flags occur. Horizon end still bootstraps from next_values.
    """
    if (rewards.ndim != 2 or min(rewards.shape) <= 0
            or any(t.shape != rewards.shape for t in (values, next_values, terminated, truncated))
            or terminated.dtype != torch.bool or truncated.dtype != torch.bool
            or not 0 <= gamma <= 1 or not 0 <= lam <= 1):
        raise ValueError("Expected T,N transitions and boolean reset flags")
    finite(rewards, values, next_values)
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        delta = rewards[step]+gamma*next_values[step]*(~terminated[step])-values[step]
        carry = delta+gamma*lam*(~(terminated[step] | truncated[step]))*carry
        advantages[step] = carry
    return advantages, advantages+values


@dataclass(frozen=True)
class LossConfig:
    clip_ratio: float = .2
    value_clip: float = .2
    value_weight: float = .5
    pre_tanh_entropy_weight: float = 0.

    def __post_init__(self):
        if (not all(math.isfinite(v) for v in vars(self).values())
                or not 0 < self.clip_ratio < 1 or self.value_clip <= 0
                or self.value_weight < 0 or self.pre_tanh_entropy_weight < 0):
            raise ValueError("Invalid PPO objective coefficients")


def normalized_advantages(advantages):
    """Population variance; normalize once over the rollout, not per minibatch."""
    finite(advantages)
    if advantages.numel() < 2:
        raise ValueError("At least two valid transitions required")
    return (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)


def clipped_ppo_loss(new_log_prob, old_log_prob, entropy, values, old_values,
                     returns, advantages, config=LossConfig()):
    """Pure clipped PPO objective; never applies a gradient or optimizer step."""
    tensors = (new_log_prob, old_log_prob, entropy, values, old_values, returns, advantages)
    if new_log_prob.ndim != 1 or len(new_log_prob) < 1 or any(t.shape != new_log_prob.shape for t in tensors):
        raise ValueError("Expected matching flat minibatch tensors")
    finite(*tensors)
    for frozen in (old_log_prob, old_values, returns, advantages):
        if frozen.requires_grad:
            raise ValueError("Rollout/target tensors must be detached from collection graphs")
    log_ratio = new_log_prob-old_log_prob
    ratio = log_ratio.exp()
    finite(ratio)
    surrogate = torch.minimum(ratio*advantages, ratio.clamp(1-config.clip_ratio, 1+config.clip_ratio)*advantages)
    clipped_value = old_values+(values-old_values).clamp(-config.value_clip, config.value_clip)
    value_loss = .5*torch.maximum((values-returns).square(), (clipped_value-returns).square()).mean()
    policy_loss = -surrogate.mean()
    total = policy_loss+config.value_weight*value_loss-config.pre_tanh_entropy_weight*entropy.mean()
    finite(total)
    with torch.no_grad():
        metrics = dict(policy_loss=float(policy_loss), value_loss=float(value_loss),
            pre_tanh_entropy=float(entropy.mean()), approx_kl=float(((ratio-1)-log_ratio).mean()),
            clip_fraction=float(((ratio-1).abs() > config.clip_ratio).float().mean()))
    return total, metrics
