# Portions adapted from CAT env_cat.py, Copyright 2025 DeepMind Technologies Limited.
# Licensed under Apache-2.0: http://www.apache.org/licenses/LICENSE-2.0
# Distributed AS IS, without warranties or conditions of any kind.
# Modifications: batched PyTorch functions; caller-owned deterministic state.
"""CAT's field command and stop/gait transition, not a robot command publisher."""
import math
import torch


def field_command(gf, bf):
    # Native groups used by command projection: pelvis; head, feet, hands.
    v = gf[:, 1, :2]*.7
    selected = [0, 3, 4, 5, 6]
    cgf, cbf = gf[:, selected, :2], bf[:, selected, :2]
    bhat = cbf/(cbf.norm(dim=-1, keepdim=True)+1e-9)
    ls, bv = (bhat*cgf).sum(-1), (bhat*v[:, None]).sum(-1)
    delta = ((ls-bv)/(bhat.square().sum(-1)+1e-9))[..., None]*bhat
    corrected = v+torch.where((ls > bv)[..., None], delta, 0.).mean(1)
    command = torch.cat((torch.ones_like(v[:, :1]), corrected, torch.zeros_like(v[:, :1])), -1)*.75
    return torch.where((command[:, 1:].norm(dim=-1) < .2)[:, None], 0., command)


def gait_step(command, last_command, phase, stop_timestep, phase_dt):
    """Exact native _update_phase transition; not a redesigned restart controller."""
    before = stop_timestep > 50
    during = ~before & (stop_timestep > 0)
    after = ~before & ~during
    move2stop = (last_command[:, 0] == 1.) & (command[:, 0] == 0.) & before
    stop = torch.where(move2stop, 50, stop_timestep)
    stop = torch.where(during, stop-1, stop)
    command = torch.where(before[:, None], command, 0.).clone()
    command[:, 0] = (~after).to(command)
    phase = torch.fmod(phase+phase_dt+math.pi, 2*math.pi)-math.pi
    phase = torch.where(after[:, None], 0., phase)
    return command, phase, stop


def delayed_sites(sites, root, rotation, cached_root, cached_rotation):
    """Keep current articulation, but express it at CAT's cached root odometry."""
    relative = (sites-root[:, None]) @ rotation
    return relative @ cached_rotation.transpose(-1, -2)+cached_root[:, None]
