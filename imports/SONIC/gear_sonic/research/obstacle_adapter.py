"""Zero-initialized bounded latent residual prototype, disconnected from actions.

This module has no actor ownership, checkpoint loader, optimizer or environment
reference. The bound is an experimental latent scale, NOT a robot safety limit.
"""
import math

import torch
from torch import nn


class ObstacleAdapter(nn.Module):
    def __init__(self, probe_count, token_dim, *, bound=.1, seed=42):
        super().__init__()
        if (type(probe_count) is not int or not 1 <= probe_count <= 512
                or type(token_dim) is not int or not 1 <= token_dim <= 256
                or not math.isfinite(bound) or not 0 < bound <= .1):
            raise ValueError("Bounded probe/token dimensions and residual scale required")
        self.probe_count, self.token_dim, self.bound = probe_count, token_dim, bound
        # CPU initialization must not consume the rollout's CPU/CUDA RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.volume_net = nn.Sequential(nn.Conv3d(4, 8, 3, padding=1), nn.SiLU(),
                nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(1))
            self.probe_net = nn.Sequential(nn.Flatten(1), nn.Linear(probe_count*8, 64), nn.SiLU())
            self.fusion = nn.Sequential(nn.Linear(16+64+9, 128), nn.SiLU())
            self.head = nn.Linear(128, token_dim)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def features(self, packet):
        """Validated obstacle features; also reused by the learning prototype.

        This extraction does not add parameters or change existing state keys.
        It preserves the shadow adapter's deterministic zero-residual forward.
        """
        v, p, g, valid = (packet[k] for k in ("volume", "probes", "guidance", "valid"))
        b = v.shape[0]
        if (v.ndim != 5 or v.shape[1] != 4 or p.shape != (b, self.probe_count, 8)
                or g.shape != (b, 9) or valid.shape != (b,) or valid.dtype != torch.bool
                or any(not torch.isfinite(t).all() for t in (v, p, g))):
            raise ValueError("Invalid obstacle packet dimensions/finite values")
        # Caller cannot claim valid while channels declare missing geometry.
        masks_ok = (v[:, 2:] == 1).flatten(1).all(1) & (p[..., 6:] == 1).flatten(1).all(1) & (g[:, 7] == 1)
        if (valid & ~masks_ok).any():
            raise ValueError("Packet validity disagrees with geometry/guidance channels")
        return self.fusion(torch.cat((self.volume_net(v), self.probe_net(p), g), -1))

    def forward(self, packet):
        residual = self.bound*torch.tanh(self.head(self.features(packet)))
        return torch.where(packet["valid"][:, None], residual, 0.)
