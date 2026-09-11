"""Reviewed kinematic challenge admission and instance-local arm reset hook.

This replaces ONLY the reference-clear requirement for an explicit training
challenge. Physical mesh parity, protected-body clearance, role/terrain checks
and a separately checked collision-free posture remain required. Nothing here
starts physics or applies a policy. Kinematic admission is not balance proof.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import torch

from .capsule_screen import capsule_gaps
from .obstacle_roles import assess_roles


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_path(path, repository=None):
    """Resolve pinned evidence after copying the repo/artifact bundle to a new PC.

    Archived absolute paths remain provenance; only their research/runs suffix
    is rebound. No report or checksum is rewritten during relocation.
    """
    repository = Path(repository or Path(__file__).resolve().parents[4]).resolve()
    path = Path(path)
    parts = path.parts
    if path.is_absolute():
        candidates = [i for i in range(len(parts)-1) if parts[i:i+2] == ("research", "runs")]
        if len(candidates) != 1:
            raise ValueError("Evidence must be inside research/runs")
        path = Path(*parts[candidates[0]:])
    if path.parts[:2] != ("research", "runs") or ".." in path.parts:
        raise ValueError("Evidence must be inside research/runs")
    resolved = (repository/path).resolve()
    if not resolved.is_relative_to(repository/"research/runs"):
        raise ValueError("Evidence escaped the run directory")
    return resolved


class ChallengeAdmission:
    def __init__(self, path, expected_sha256, case=17):
        self.path = Path(path).resolve()
        if digest(self.path) != expected_sha256:
            raise ValueError("Reviewed challenge report checksum mismatch")
        self.report = json.loads(self.path.read_text())
        r = self.report
        if (r.get("schema") != "grail-cat-posture-witness-v1" or r.get("arm_mode") != "constant"
                or not r.get("initial_arm_pose_changed") or not r.get("role_retention", {}).get("all_roles_retained")
                or not r.get("geometric_route_found") or r.get("dynamic_feasibility_verified")):
            raise ValueError("Explicit constant-arm geometric witness required, not a dynamic success claim")
        item = next((item for item in r["witnesses"] if item["case"] == case), None)
        if item is None or not r["cases"][case]["kinematic_screen_passed"]:
            raise ValueError("Requested posture was not admitted")
        packet = (self.path.parent/item["file"]).resolve()
        if packet.parent != self.path.parent or digest(packet) != item["sha256"]:
            raise ValueError("Witness packet checksum/path mismatch")
        with np.load(packet, allow_pickle=False) as saved:
            self.centers = saved["centers"].copy()
            self.radii = saved["radii"].copy()
        self.targets = r["cases"][case]["targets"]
        expected = {f"{side}_{joint}_joint" for side in ("left", "right") for joint in
                    ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")}
        if set(self.targets) != expected or not np.isfinite(list(self.targets.values())).all():
            raise ValueError("Exactly 14 finite named arm targets required")
        self.contract = dict(schema="grail-cat-reviewed-posture-admission-v1", report_sha256=expected_sha256,
                             packet_sha256=item["sha256"], case=case, targets=self.targets,
                             reference_arms_changed=False, initial_arms_changed=True, dynamic_feasibility_verified=False)
        self.reference_checked = self.layout_checked = False

    def validate_reference(self, cat, wrapper):
        r = self.report
        if (digest(cat.fields.directory/"scene.json") != r["scene_sha256"]
                or asdict(cat.placement) != dict(r["placement"], translation=tuple(r["placement"]["translation"]))
                or json.loads(json.dumps([asdict(p) for p in cat.probes])) != r["reference"]["imported_probe_order"]):
            raise ValueError("Live CAT placement/probe geometry differs from reviewed example")
        with np.load(cat.output.with_name("reference_sweep.npz"), allow_pickle=False) as saved:
            original = saved["centers"].copy()
            radii = saved["radii"].copy()
        source = artifact_path(r["reference"]["reference_run"])/"reference_sweep.npz"
        if digest(source) != r["reference"]["reference_sweep_sha256"]:
            raise ValueError("Original reference sweep changed")
        with np.load(source, allow_pickle=False) as saved:
            if original.shape != saved["centers"].shape or not np.allclose(original, saved["centers"], atol=2e-5, rtol=0):
                raise ValueError("Live paired reference differs from reviewed motion")
        if (self.centers.shape != original.shape or not np.isfinite(self.centers).all()
                or not np.allclose(self.radii, radii, atol=1e-7, rtol=0)):
            raise ValueError("Posture geometry does not match the live embodiment")
        protected = [i for i, p in enumerate(cat.probes) if not any(x in p.link for x in ("shoulder", "elbow", "wrist"))]
        if not np.allclose(self.centers[:, protected], original[:, protected], atol=2e-5, rtol=0):
            raise ValueError("Arm-only witness changed protected reference geometry")
        centers = torch.as_tensor(self.centers, device=wrapper.env.device)
        gap = cat.gaps(centers, torch.as_tensor(self.radii, device=wrapper.env.device))
        if gap.min() < .03:
            raise ValueError("Live mesh rejects recorded witness clearance")
        inventory = cat.report["imported_colliders"]
        pairs = []
        # Reuse the exact pinned pair contract; never silently omit a named pair.
        for a, b in r["self_pairs"]:
            pairs.extend((i, j) for i, first in enumerate(inventory) for j, second in enumerate(inventory)
                         if first["body"] == a and second["body"] == b)
        self_gap = capsule_gaps(torch.as_tensor(self.centers), cat.probes, inventory, list(set(pairs)))
        if not pairs or self_gap.min() < .005:
            raise ValueError("Reviewed nonlocal self-clearance does not reproduce")
        self.contract.update(min_clutter_gap_m=float(gap.min()), min_self_gap_m=float(self_gap.min()))
        self.reference_checked = True
        return dict(self.contract)

    def validate_layout(self, surface, cat, layout):
        if not self.reference_checked:
            raise ValueError("Reference/body witness must be verified before scene admission")
        with np.load(cat.fields.directory/"role_trace.npz", allow_pickle=False) as saved:
            if saved["unresolved_added"].any():
                raise ValueError("Unresolved physical obstacle roles")
            masks = {key: saved[key].copy() for key in ("floor", "lateral", "overhead") if saved[key].any()}
        roles = assess_roles(surface, masks, cat.fields.meta["origin_corner"], cat.fields.meta["resolution"], cat.placement)
        arms = [i for i, p in enumerate(cat.probes) if any(x in p.link for x in ("shoulder", "elbow", "wrist"))]
        td, _, valid = surface.distance(torch.as_tensor(self.centers[:, arms], device=surface.device))
        gap = td-torch.as_tensor(self.radii[arms], device=surface.device)
        graph = layout["support_graph"]
        if (not roles["all_roles_retained"] or not valid.all() or gap.min() < .02
                or not graph["geometric_route_found"] or graph["anchor_support_fraction"] < .99
                or not layout["physics_ray_parity_verified"]):
            raise ValueError("Challenge failed physical terrain/role/passage checks")
        self.contract.update(roles=roles, min_arm_terrain_gap_m=float(gap.min()))
        self.layout_checked = True
        return dict(self.contract)


class ArmReset:
    """Change only the 14 arm joints after each normal GRAIL motion reset.

    Original reference tables and desired motion remain unchanged. The frozen
    decoder therefore does not receive a pre-solved avoidance reference.
    """
    def __init__(self, wrapper, admission):
        if not admission.layout_checked:
            raise ValueError("Live layout admission required before reset changes")
        self.motion, self.robot = wrapper.motion_command, wrapper.env.scene["robot"]
        self.targets = admission.targets
        self.indices = [self.robot.joint_names.index(name) for name in self.targets]
        self.values = torch.tensor(list(self.targets.values()), device=wrapper.env.device)
        self.calls, self.rows = 0, []

    def __enter__(self):
        motion = self.motion
        self.had_attribute = "_resample_command" in motion.__dict__
        self.original = motion._resample_command
        def resample(instance, env_ids):
            result = self.original(env_ids)
            ids = torch.as_tensor(env_ids, device=self.values.device, dtype=torch.long)
            if not len(ids):
                return result
            pos = self.robot.data.joint_pos[ids].clone()
            vel = self.robot.data.joint_vel[ids].clone()
            limits = self.robot.data.soft_joint_pos_limits[ids][:, self.indices]
            if ((self.values < limits[..., 0]) | (self.values > limits[..., 1])).any():
                raise ValueError("Reviewed arm target exceeds live joint limits")
            before = pos.clone()
            pos[:, self.indices] = self.values
            vel[:, self.indices] = 0
            self.robot.write_joint_state_to_sim(pos, vel, env_ids=ids)
            protected = [i for i in range(pos.shape[1]) if i not in self.indices]
            if not torch.equal(pos[:, protected], before[:, protected]):
                raise ValueError("Arm reset changed non-arm joint positions")
            self.calls += 1
            self.rows.append(ids.cpu().tolist())
            return result
        motion._resample_command = types.MethodType(resample, motion)
        return self

    def __exit__(self, *args):
        if self.had_attribute:
            self.motion._resample_command = self.original
        else:
            del self.motion._resample_command
