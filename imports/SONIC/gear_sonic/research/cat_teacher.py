"""Released CAT generalist actor for cross-framework distillation.

Input is the ORIGINAL CAT 162D observation, not a GRAIL/oracle observation.
Output is 12 normalized leg targets in CAT joint order, NOT 29 joint commands
or GRAIL latents. This module never writes robot/simulator actions. Keep frozen.
Architecture/weights are copied from the released Flax MLP, not learned anew.
"""
import numpy as np
import torch
from torch import nn

JOINTS = tuple(f"{side}_{joint}_joint" for side in ("left", "right") for joint in
               ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))
WIDTHS = (162, 512, 256, 128, 64, 24)


class CatTeacher(nn.Module):
    def __init__(self, weights):
        super().__init__()
        with np.load(weights, allow_pickle=False) as data, torch.random.fork_rng(devices=[]):
            self.layers = nn.ModuleList()
            for i, (n, m) in enumerate(zip(WIDTHS, WIDTHS[1:])):
                w, b = data[f"actor_{i}_kernel"], data[f"actor_{i}_bias"]
                if (w.shape != (n, m) or b.shape != (m,) or w.dtype != np.float32
                        or b.dtype != np.float32 or not np.isfinite(w).all() or not np.isfinite(b).all()):
                    raise ValueError("CAT checkpoint architecture/finite values mismatch")
                layer = nn.Linear(n, m)
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(w.copy()).T)
                    layer.bias.copy_(torch.from_numpy(b.copy()))
                self.layers.append(layer)
        self.requires_grad_(False).eval()

    def logits(self, obs):
        if obs.ndim != 2 or obs.shape[1] != 162 or not torch.isfinite(obs).all():
            raise ValueError("Expected finite Bx162 ORIGINAL CAT observations")
        x = obs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i+1 < len(self.layers):
                x = torch.nn.functional.silu(x)
        return x

    def forward(self, obs):
        # Brax NormalTanhDistribution.mode: tanh(mean), not tanh(log std).
        return torch.tanh(self.logits(obs)[..., :12])
