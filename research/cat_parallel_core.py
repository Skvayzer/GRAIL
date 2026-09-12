"""Batched CAT fields, GRAIL history and native CAT reward equations (torch).

CAT-derived equations: Apache-2.0, Copyright 2025 DeepMind Technologies Limited.
Reference: pinned cat_ppo/envs/g1/env_cat.py and env_loco.py. No hardware code.
"""
import json
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"imports/SONIC"))
from gear_sonic.research.cat_bridge import GROUP_SIZES, heading_matrix


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FieldBank:
    def __init__(self, directory, device="cpu"):
        directory = Path(directory)
        self.meta = json.loads((directory/"bank.json").read_text())
        if not self.meta["complete"] or self.meta["robot_actuation"]:
            raise ValueError("Incomplete or invalid generated bank")
        self.rows = self.meta["scenes"]
        grids = []
        for row in self.rows:
            path = directory/row["file"]
            if sha(path) != row["sha256"]:
                raise ValueError("Changed generated fields")
            with np.load(path, allow_pickle=False) as a:
                grids.append(np.concatenate((a["gf"], a["bf"], a["sdf"][..., None]), -1))
        self.grid = torch.as_tensor(np.stack(grids), device=device)
        self.origin = self.grid.new_tensor(self.rows[0]["origin"])
        self.dx = self.rows[0]["resolution"]
        if any(r["shape"] != self.rows[0]["shape"] or r["origin"] != self.rows[0]["origin"]
               or r["resolution"] != self.dx for r in self.rows):
            raise ValueError("Bank grids must share native geometry contract")
        self.upper = self.grid.new_tensor(self.grid.shape[1:4])-1
        self.offset = torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],
            [0,0,1],[1,0,1],[0,1,1],[1,1,1]], device=device)
        self.success_ema = torch.full((len(self.rows),), .5, device=device)
        self.visit_count = torch.zeros(len(self.rows), dtype=torch.long, device=device)
        self.hash = sha(directory/"bank.json")

    def candidates(self, family, split="train", max_difficulty=1.):
        ids = [i for i, r in enumerate(self.rows) if r["split"] == split
            and (family == "all" or r["family"] == family) and r["difficulty"] <= max_difficulty]
        if not ids:
            raise ValueError("No scenes in requested generated stratum")
        return torch.tensor(ids, device=self.grid.device)

    def choose(self, count, family, split="train", max_difficulty=1., generator=None):
        ids = self.candidates(family, split, max_difficulty)
        # CAT adaptive failure sampling, alpha=1; nonzero floor preserves coverage.
        weights = (.05+1-self.success_ema[ids]) if split == "train" else torch.ones_like(self.success_ema[ids])
        return ids[torch.multinomial(weights, count, replacement=True, generator=generator)]

    def sample(self, scene, points):
        idx = (points-self.origin)/self.dx
        valid = torch.isfinite(idx).all(-1) & ((idx >= 0) & (idx <= self.upper)).all(-1)
        clipped = torch.minimum(torch.nan_to_num(idx).clamp_min(0), self.upper-1)
        base = clipped.floor().long()
        frac = (clipped-base).flip(-1)  # Native teacher's legacy X/Z weight ordering.
        corners = base[..., None, :]+self.offset
        value = self.grid[scene[:, None, None], corners[...,0], corners[...,1], corners[...,2]]
        weights = torch.where(self.offset.bool(), frac[...,None,:], 1-frac[...,None,:]).prod(-1)
        value = (value*weights[...,None]).sum(-2)
        return dict(gf=value[...,:3], bf=value[...,3:6], sdf=value[...,6:], in_domain=valid)


class BatchHistory:
    def __init__(self, count, contract, names, device):
        self.ids = [names.index(n) for n in contract["joints"]]
        self.offset = torch.tensor(contract["offset"], device=device)
        self.scale = torch.tensor(contract["scale"], device=device)
        self.history = [torch.zeros(count, 10, width, device=device) for width in (3,29,29,29,3)]
        self.ready = torch.zeros(count, dtype=torch.bool, device=device)

    def sample(self, physical, root, targets):
        r = physical["pelvis_rotation"]
        values = (physical["gyro"], physical["joint_pos"][:,self.ids]-self.offset,
            physical["joint_vel"][:,self.ids], (targets[:,self.ids]-self.offset)/self.scale,
            physical["gravity"])
        for i, value in enumerate(values):
            self.history[i] = torch.cat((self.history[i][:,1:], value[:,None]), 1)
            self.history[i] = torch.where(self.ready[:,None,None], self.history[i], value[:,None])
        self.ready[:] = True
        count = len(root)
        identity = torch.eye(3, device=root.device)[:,:2].reshape(1,6).repeat(count,10)
        object_pos = -(r.transpose(-1,-2)@root[...,None]).squeeze(-1)
        return torch.cat([h.flatten(1) for h in self.history]+[
            torch.zeros(count,30,device=root.device), identity, object_pos,
            r.transpose(-1,-2)[...,:2].reshape(count,6)], -1)


def pack_cat(physical, obs_ids, defaults, last_action, targets, command, phase, fields, noise=None):
    """Same packing as the independently tested bridge; no per-term CUDA sync."""
    n = len(targets)
    nav = heading_matrix(physical["pelvis_rotation"])
    gf = fields["gf"]@nav
    bf = (fields["bf"]@nav)*(fields["sdf"] < .5)
    sdf = fields["sdf"].clamp(-1,.5)
    cmd = command.clone()
    cmd[:,1:] = (cmd[:,None,1:]@nav).squeeze(1)
    cmd[:,-1] = 0
    values = [physical["gyro"], physical["gravity"], physical["joint_pos"][:,obs_ids]-defaults,
        physical["joint_vel"][:,obs_ids]]
    if noise is not None:
        values = [v+e for v,e in zip(values,noise)]
    values += [last_action, targets, cmd, torch.full((n,1),.07,device=targets.device),phase.cos(),phase.sin()]
    start = 0
    for size in GROUP_SIZES:
        values.extend(v[:,start:start+size].reshape(n,-1) for v in (gf,bf,sdf))
        start += size
    return torch.cat(values,-1)


def rpy(rotation):
    return torch.stack((torch.atan2(rotation[:,2,1],rotation[:,2,2]),
        torch.asin((-rotation[:,2,0]).clamp(-1,1)),torch.atan2(rotation[:,1,0],rotation[:,0,0])), -1)


def native_rewards(s, scales):
    """Each term reduces only body/joint axes, never mixes different envs."""
    cmd, v, w = s["command"][:,1:], s["lin_vel"], s["torso_ang_nav"]
    sites, site_vel, sdf, gf = s["sites"], s["site_vel"], s["fields"]["sdf"].squeeze(-1), s["fields"]["gf"]
    move, gait = s["command"][:,0], s["gait"]
    p, t = s["pelvis_rpy"], s["torso_rpy"]
    bad_pitch = (-t[:,1]).clamp_min(0)
    ori_err = p[:,0].abs()+t[:,0].abs()+bad_pitch+(sites[:,0,2]>1.1)*t[:,1].abs()
    direction = cmd[:,:2]/cmd[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)
    orth = v[:,:2]-(v[:,:2]*direction).sum(-1,keepdim=True)*direction
    orth_cost = torch.where(cmd[:,:2].norm(dim=-1)>1e-8,orth.square().sum(-1),0.)
    leg_rot = s["legs_nav"]
    decay = ((.500001-cmd[:,2].abs())/.500001).clamp(0,1).square()
    rot_err = leg_rot[:,:,2,1].abs().mean(-1)+(decay[:,None]*leg_rot[:,:,0,1].abs()).mean(-1)
    stance, swing = (gait==1).float(), (gait==-1).float()
    contact = s["feet_contact"].float()
    contact_cost = (((contact-stance).abs()+((1-contact)-swing).abs())*(gait!=0)).sum(-1)*move
    feet = sites[:,3:5]
    center = (feet[:,:,:2]-s["com"][:,None,:2]).sum(1)
    balance = center.square().sum(-1)*(1+10*(.35-(feet[:,0]-feet[:,1]).norm(dim=-1)).clamp_min(0))
    action = s["action"]
    qvel = s["qvel"][:,s["obs_ids"]]
    qacc = (s["previous_qvel"][:,s["obs_ids"]]-qvel)/.02
    terms = dict(tracking_orientation=torch.exp(-.5*ori_err)-bad_pitch,
        tracking_root_field=torch.exp(-4*(cmd[:,:2]-v[:,:2]).square().sum(-1)),
        body_motion=1.2*orth_cost+.4*w[:,:2].abs().sum(-1),
        body_rotation=torch.exp(-5*rot_err), foot_contact=contact_cost,
        foot_clearance=(swing*(.07-feet[:,:,2]).clamp_min(0).square()).sum(-1)*move,
        foot_slip=(s["feet_sensor_vel"].square().sum(-1)*stance).sum(-1),
        foot_balance=balance, foot_far=(.35-(feet[:,0]-feet[:,1]).norm(dim=-1)).clamp_min(0),
        straight_knee=(.1-s["qpos"][:,s["knee_ids"]]).clamp_min(0).sum(-1),
        smoothness_joint=(.01*qvel.square()+qacc.square()).sum(-1),
        smoothness_action=(action.square()+(action-s["last_action"]).square()
            +(action-2*s["last_action"]+s["last_last_action"]).square()).sum(-1),
        joint_limits=((s["soft_lower"]-s["qpos"]).clamp_min(0)+(s["qpos"]-s["soft_upper"]).clamp_min(0)).sum(-1),
        joint_torque=s["torque"].square().sum(-1))
    for group, ids, tau in (("head",[0],.5),("feet",[3,4],.3),("hands",[5,6],.5)):
        g, vel, distance = gf[:,ids], site_vel[:,ids], sdf[:,ids]
        cosine = ((g/(g.norm(dim=-1,keepdim=True)+1e-6))*(vel/(vel.norm(dim=-1,keepdim=True)+1e-6))).sum(-1)
        crossed = (move[:,None]<.5)|(sites[:,ids,0]>1.5)
        if group == "feet":
            crossed |= gait==1
        terms[group+"gf"] = torch.where(crossed,4.,torch.sigmoid(40*(tau-distance))*5*cosine).mean(-1)
    for group, ids in (("head",[0]),("feet",[3,4]),("hands",[5,6]),("knees",[7,8]),("shlds",[9,10])):
        terms[group+"df"] = (-20*F.softplus((.05-sdf[:,ids])/.02)).mean(-1)
    if set(terms) != set(scales):
        raise ValueError("CAT reward configuration/source term mismatch")
    reward = (.02*sum(torch.nan_to_num(terms[k],nan=0.)*scales[k] for k in scales)).clamp(0,10000)
    return reward, terms
