# Portions adapted from Click-and-Traverse cat_ppo/envs/g1/env_cat.py:
# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modifications: batched PyTorch packing, named articulation mapping, leg-only
# supervision, explicit applied-target history. Source hashes in exported contract.
"""CAT checkpoint-compatible, noiseless teacher inputs and same-state leg labels.

No simulator/SDK imports or action writes. The caller supplies the ACTUALLY
applied action history, command, gait and (possibly delayed) HumanoidPF fields.
This is not a port of CAT's randomized MDP or a complete whole-body trainer.
"""
import torch

from .cat_teacher import JOINTS
from .geometry import sample_xyz_grid
from .learner_state import quaternion_matrix

GROUPS = ("head", "pelv", "tors", "feet", "hands", "knees", "shlds")
GROUP_SIZES = (1, 1, 1, 2, 2, 2, 2)
SITES = ("head", "imu_in_pelvis", "imu_in_torso", "left_foot", "right_foot",
         "left_palm", "right_palm", "left_knee", "right_knee", "left_shoulder", "right_shoulder")
OBS_JOINTS = JOINTS + ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint") + tuple(
    f"{side}_{joint}_joint" for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"))


def indices(names, wanted):
    names = tuple(names)
    if len(names) != len(set(names)) or not set(wanted).issubset(names):
        raise ValueError(f"Ambiguous/missing named joints or bodies: {set(wanted)-set(names)}")
    return [names.index(n) for n in wanted]


def finite_shape(value, shape):
    if tuple(value.shape) != tuple(shape) or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite {shape}, got {tuple(value.shape)}")


def heading_matrix(pelvis_rotation):
    """Native CAT navigation frame: heading only, not full tilted pelvis frame."""
    x = pelvis_rotation[..., :, 0].clone()
    x[..., 2] = 0
    length = x.norm(dim=-1, keepdim=True)
    if (length < 1e-6).any():
        raise ValueError("CAT heading undefined for vertical pelvis forward axis")
    x = x / length
    z = torch.zeros_like(x)
    z[..., 2] = 1
    y = torch.linalg.cross(z, x)
    return torch.stack((x, y, z), -1)


def sample_teacher_fields(fields, points, origin, resolution):
    """Reproduce the RELEASED teacher's X/Z weight ordering and edge clipping.

    Deliberately separate from corrected geometry/oracle queries. In-domain is
    returned for auditing: native edge clamping is not evidence of free space.
    Fields/positions must already be in the same scene coordinates.
    """
    values, masks = {}, []
    for key in ("gf", "bf", "sdf"):
        field = fields[key]
        if field.ndim == 3:
            field = field[..., None]
        values[key], valid = sample_xyz_grid(field, points, origin, resolution, legacy_cat=True)
        masks.append(valid)
    values["in_domain"] = torch.stack(masks).all(0)
    return values


def normalize_fields(gf, bf, move_flag):
    """Native step-time normalization, BEFORE field rotation/packing."""
    gf = gf * (move_flag[:, None, None] > .5) / (gf.norm(dim=-1, keepdim=True) + 1e-6)
    bf = bf / (bf.norm(dim=-1, keepdim=True) + 1e-6)
    return gf, bf


class CatFieldSampler:
    """Cached, vectorized equivalent of the legacy teacher's three field queries.

    One corner gather for all seven channels instead of separate Python loops
    per field/corner. Keep sample_teacher_fields as the independent slow check.
    """
    def __init__(self, fields, origin, resolution):
        sdf = fields["sdf"]
        if sdf.ndim == 3:
            sdf = sdf[..., None]
        self.grid = torch.cat((fields["gf"], fields["bf"], sdf), -1)
        if (self.grid.ndim != 4 or self.grid.shape[-1] != 7
                or min(self.grid.shape[:3]) < 2 or resolution <= 0 or not torch.isfinite(self.grid).all()):
            raise ValueError("Finite XYZx7 teacher field grid required")
        self.origin = self.grid.new_tensor(origin)
        self.resolution = resolution
        self.upper = self.grid.new_tensor(self.grid.shape[:3])-1
        # Ordering paired with weights; flipped fractions retain native X/Z bug.
        self.offsets = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                                     [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], device=self.grid.device)

    def sample(self, points):
        shape = points.shape[:-1]
        idx = ((points-self.origin)/self.resolution).reshape(-1, 3)
        valid = torch.isfinite(idx).all(-1) & ((idx >= 0) & (idx <= self.upper)).all(-1)
        clipped = torch.minimum(torch.nan_to_num(idx).clamp_min(0), self.upper-1)
        base = clipped.floor().long()
        frac = (clipped-base).flip(-1)
        corners = base[:, None]+self.offsets
        values = self.grid[corners[..., 0], corners[..., 1], corners[..., 2]]
        weights = torch.where(self.offsets.bool(), frac[:, None], 1-frac[:, None]).prod(-1)
        result = (values*weights[..., None]).sum(1).reshape(*shape, 7)
        return dict(gf=result[..., :3], bf=result[..., 3:6], sdf=result[..., 6:], in_domain=valid.reshape(shape))


class CatObservationBridge:
    def __init__(self, contract, joint_names, body_names=None):
        if (contract["schema"] != "cat-observation-action-contract-v1"
                or tuple(contract["action_joints"]) != JOINTS
                or tuple(contract["observation_joints"]) != OBS_JOINTS):
            raise ValueError("Unexpected CAT model contract")
        self.contract = contract
        self.joint_names = tuple(joint_names)
        self.obs_ids = indices(joint_names, OBS_JOINTS)
        self.action_ids = indices(joint_names, JOINTS)
        self.default_obs = [contract["default_joint_positions"][n] for n in OBS_JOINTS]
        self.body_names = tuple(body_names) if body_names is not None else None
        if body_names is not None:
            if tuple(s["site"] for s in contract["sites"]) != SITES:
                raise ValueError("CAT site order mismatch")
            self.site_ids = indices(body_names, [s["body"] for s in contract["sites"]])
            self.pelvis_id = indices(body_names, ("pelvis",))[0]

    def read_articulation(self, data, env_origins):
        """Read Isaac-style tensors; sites are LINK offsets, never COM offsets.

        Body angular velocity is world-frame; CAT gyro/gravity are pelvis-local.
        No tensors on the articulation are modified. Exported CAT sites must be
        attached to equivalent named G1 link frames in the destination model.
        """
        if self.body_names is None:
            raise ValueError("Named body mapping required")
        b = data.joint_pos.shape[0]
        finite_shape(env_origins, (b, 3))
        q = data.body_quat_w[:, self.site_ids]
        rotations = quaternion_matrix(q)
        offsets = q.new_tensor([s["position"] for s in self.contract["sites"]])
        sites = data.body_pos_w[:, self.site_ids] + (rotations @ offsets[..., None]).squeeze(-1)
        pelvis = quaternion_matrix(data.body_quat_w[:, self.pelvis_id])
        omega = data.body_ang_vel_w[:, self.pelvis_id]
        return dict(joint_pos=data.joint_pos, joint_vel=data.joint_vel,
                    pelvis_rotation=pelvis,
                    gyro=(pelvis.transpose(-1, -2) @ omega[..., None]).squeeze(-1),
                    gravity=-pelvis[:, 2, :], sites=sites-env_origins[:, None, :])

    def pack(self, *, joint_pos, joint_vel, pelvis_rotation, gyro, gravity,
             last_action, previous_targets, command_world, foot_height, phase, gf, bf, sdf):
        """Exactly 162D; command = [move flag, world vx, world vy, world vz].

        Last action and previous targets are 12D CAT-ordered histories (targets
        in radians). GF/BF are normalized world vectors supplied by the caller;
        command and fields may be delayed independently, as in CAT training.
        """
        b, j = joint_pos.shape
        for value, shape in ((joint_pos, (b, len(self.joint_names))), (joint_vel, (b, j)),
            (pelvis_rotation, (b, 3, 3)), (gyro, (b, 3)), (gravity, (b, 3)),
            (last_action, (b, 12)), (previous_targets, (b, 12)),
            (command_world, (b, 4)), (foot_height, (b, 1)), (phase, (b, 2)),
            (gf, (b, 11, 3)), (bf, (b, 11, 3)), (sdf, (b, 11, 1))):
            finite_shape(value, shape)
        nav = heading_matrix(pelvis_rotation)
        gf_nav, bf_nav = gf @ nav, bf @ nav
        bf_nav = bf_nav * (sdf < .5)
        distance = sdf.clamp(-1., .5)
        command = command_world.clone()
        command[:, 1:] = (command_world[:, None, 1:] @ nav).squeeze(1)
        command[:, -1] = 0
        packet = [gyro, gravity, joint_pos[:, self.obs_ids]-joint_pos.new_tensor(self.default_obs),
                  joint_vel[:, self.obs_ids], last_action, previous_targets,
                  command, foot_height, phase.cos(), phase.sin()]
        start = 0
        for size in GROUP_SIZES:
            # GROUP-major, then GF/BF/SDF. Not seven channels per point!
            packet.extend(v[:, start:start+size].reshape(b, -1) for v in (gf_nav, bf_nav, distance))
            start += size
        obs = torch.cat(packet, -1)
        finite_shape(obs, (b, 162))
        return obs

    def leg_targets(self, action, previous_targets):
        """Native integrator: clip(PREVIOUS target + .5 * action, soft limits)."""
        finite_shape(action, (previous_targets.shape[0], 12))
        finite_shape(previous_targets, action.shape)
        if (action.abs() > 1.00001).any():
            raise ValueError("Expected CAT's tanh-normalized leg increments")
        lo = action.new_tensor(self.contract["soft_lower"])
        hi = action.new_tensor(self.contract["soft_upper"])
        return torch.maximum(torch.minimum(previous_targets + self.contract["action_scale"]*action, hi), lo)


class AppliedTargetHistory:
    """DAgger histories follow the student/simulator, NEVER unexecuted teacher labels."""
    def __init__(self, targets, scale=.5):
        finite_shape(targets, (targets.shape[0], 12))
        if scale <= 0:
            raise ValueError("Positive action scale required")
        self.targets = targets.detach().clone()
        self.last_action = torch.zeros_like(targets)
        self.scale = scale

    def commit(self, applied_targets, reset=None):
        finite_shape(applied_targets, self.targets.shape)
        self.last_action = (applied_targets.detach()-self.targets)/self.scale
        self.targets = applied_targets.detach().clone()
        if reset is not None:
            if reset.dtype != torch.bool or reset.shape != (len(self.targets),):
                raise ValueError("Boolean per-environment reset mask required")
            self.last_action[reset] = 0


def teacher_leg_loss(student_joint_targets, student_joint_names, teacher_leg_targets):
    """Same-state imitation in radians. CAT provides NO arm/waist supervision.

    Supply decoded absolute GRAIL joint targets, not its normalized actions or
    64D latent. Upper-body/reference retention is a separate training loss.
    """
    ids = indices(student_joint_names, JOINTS)
    finite_shape(student_joint_targets, (teacher_leg_targets.shape[0], len(student_joint_names)))
    finite_shape(teacher_leg_targets, (student_joint_targets.shape[0], 12))
    return torch.nn.functional.mse_loss(student_joint_targets[:, ids], teacher_leg_targets.detach())
