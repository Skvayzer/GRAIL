#!/usr/bin/env python3
"""Bounded arm-posture witness search; no policy execution or training.

Keep root, waist and leg reference motion intact. Smoothly change arm joint
references within limits, then test actual imported body covers against CAT and
terrain and nonlocal capsule pairs. A kinematic witness is NOT dynamic balance
or proof the policy can execute it. Every fixed candidate, including rejects,
is recorded. No original reference/scene files are changed.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from artifacts import ROOT
from cat_scenes import sha256
from cat_roles import role_masks
from reference_postures import reconstruct
from gear_sonic.research.body_envelope import world_probe_centers
from gear_sonic.research.capsule_screen import capsule_gaps, nonlocal_arm_pairs
from gear_sonic.research.cat_geometry import Placement, cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_roles import assess_roles
from gear_sonic.research.scene_audit import check, read_snapshot, save
from gear_sonic.research.terrain_surface import TerrainSurface


def smooth_blend(times, start=1., finish=3.):
    if not 0 <= start < finish or not torch.isfinite(times).all():
        raise ValueError("Finite ordered blend interval required")
    t = ((times-start)/(finish-start)).clamp(0, 1)
    # Symmetric evaluation avoids cancellation/float32 overshoot near one.
    u = torch.minimum(t, 1-t)
    half = u**3*(10+u*(-15+6*u))
    return torch.where(t <= .5, half, 1-half).clamp(0, 1)


def joint_dof_order(model):
    # Upstream actuated_joints_idx may include the free root. The scalar DOF
    # axes/ranges instead follow the non-free MJCF joints in XML traversal order.
    joints = [joint for joint in model.tree.getroot().find("worldbody").iter("joint")
              if joint.get("type", "hinge") != "free"]
    names = [joint.get("name") for joint in joints]
    if (len(names) != model.num_dof or len(set(names)) != len(names) or None in names
            or any(joint.get("type", "hinge") != "hinge" for joint in joints)
            or model.joints_range.shape != (len(names), 2) or model.dof_axis.shape != (len(names), 3)):
        raise ValueError("Cannot bind scalar joint targets to MJCF axes/limits")
    return names


def arm_pose(data, pitch, roll, elbow):
    model, raw = data["model"], data["raw"]
    pose = torch.as_tensor(raw["pose_aa"]).float().clone()
    before = pose.clone()
    blend = smooth_blend(torch.arange(len(pose))/float(raw["fps"]))
    targets, changed = {}, []
    order = joint_dof_order(model)
    joint_bodies = {joint: body for body, joint in model.mjcf_data["body_to_joint"].items()}
    for side, sign in (("left", 1), ("right", -1)):
        for suffix, value in (("shoulder_pitch", pitch), ("shoulder_roll", sign*roll), ("shoulder_yaw", 0.),
                              ("elbow", elbow), ("wrist_roll", 0.), ("wrist_pitch", 0.), ("wrist_yaw", 0.)):
            joint = f"{side}_{suffix}_joint"
            body = model.body_names.index(joint_bodies[joint])
            dof = order.index(joint)
            if not model.joints_range[dof, 0] <= value <= model.joints_range[dof, 1]:
                raise ValueError(f"Candidate exceeds joint limits: {joint}")
            target = model.dof_axis[dof].float()*value
            pose[:, body] = before[:, body]*(1-blend[:, None])+target*blend[:, None]
            targets[joint] = value
            changed.append(body)
    fixed = [i for i in range(pose.shape[1]) if i not in changed]
    if not torch.equal(pose[:, fixed], before[:, fixed]) or not torch.equal(pose[0], before[0]):
        raise ValueError("Posture witness changed root/waist/legs or spawn")
    fk = model.fk_batch(pose[None], torch.as_tensor(raw["root_trans_offset"]).float()[None],
        return_full=True, fps=float(raw["fps"]), target_fps=50, interpolate_data=True)
    return pose, fk, targets


def search(reference_run, scene, placement, device="cuda:0"):
    scene, reference_run = Path(scene).resolve(), Path(reference_run).resolve()
    data = reconstruct(reference_run)
    cv, cf = cat_mesh_arrays(scene)
    clutter = ClosedMeshDistance(cv, cf, device=device)
    v, f, terrain = read_snapshot(reference_run)
    surface = TerrainSurface(v, f, ground_z=terrain["ground"]["height"], device=device)
    masks, meta = role_masks(scene)
    roles = assess_roles(surface, masks, meta["origin_corner"], meta["resolution"], placement)
    layout, arrays, _ = check(reference_run, scene, placement, device=device)
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_posture_witness")
    run.mkdir(parents=True, exist_ok=False)
    save(run/"layout_audit.json", layout, arrays)
    # Do not relabel a conflicted original reference as accepted by its old gate.
    pairs = nonlocal_arm_pairs(data["model"], data["inventory"])
    original_self = capsule_gaps(data["centers"], data["probes"], data["inventory"], pairs)
    arm_indices = [i for i, p in enumerate(data["probes"]) if any(s in p.link for s in ("shoulder", "elbow", "wrist"))]
    report = dict(schema="grail-cat-posture-witness-v1", simulation_only=True, reference=data["provenance"],
        scene=str(scene), scene_sha256=sha256(scene/"scene.json"), placement=asdict(placement),
        blend_seconds=[1., 3.], targets_grid=dict(pitch=[-.5, 0., .5], roll=[.05, .15], elbow=[.7, 1.2, 1.7]),
        role_retention=roles, geometric_route_found=layout["support_graph"]["geometric_route_found"],
        original_reference_clear=layout["reference"]["sampled_reference_clear"],
        original_nonlocal_self_gap_m=float(original_self.min()),
        self_pair_contract="exact imported capsules; excludes same/adjacent collision-bearing ancestors; not PhysX filter equivalence",
        self_pairs=[[data["inventory"][i]["body"], data["inventory"][j]["body"]] for i, j in pairs],
        reference_changed_on_disk=False, policy_connected=False, dynamic_feasibility_verified=False,
        training_ready=False, cases=[], witnesses=[])
    for pitch in (-.5, 0., .5):
        for roll in (.05, .15):
            for elbow in (.7, 1.2, 1.7):
                number = len(report["cases"])
                pose, fk, targets = arm_pose(data, pitch, roll, elbow)
                pos = fk.global_translation[0]
                quat = fk.global_rotation[0][..., [3, 0, 1, 2]]
                centers, radii = world_probe_centers(data["probes"], data["model"].body_names, pos, quat)
                gpu = centers.to(device)
                distance, _, valid = clutter.query(placement.to_local(gpu))
                gap = distance-radii.to(device)
                td, _, tv = surface.distance(gpu[:, arm_indices])
                terrain_gap = td-radii[arm_indices].to(device)
                self_gap = capsule_gaps(centers, data["probes"], data["inventory"], pairs)
                feet = [data["model"].body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
                foot_error = float((pos[:, feet]-data["fk"].global_translation[0, :, feet]).abs().max())
                accepted = (valid.all() and tv.all() and gap.min() >= .03 and terrain_gap.min() >= .02
                            and self_gap.min() >= .005 and foot_error < 1e-6 and roles["all_roles_retained"]
                            and report["geometric_route_found"])
                case = dict(targets=targets, outer_clutter_gap_m=float(gap.min()), arm_terrain_gap_m=float(terrain_gap.min()),
                    nonlocal_self_gap_m=float(self_gap.min()), max_foot_position_change_m=foot_error,
                    kinematic_screen_passed=bool(accepted), dynamic_feasibility_verified=False)
                report["cases"].append(case)
                if accepted:
                    path = run/f"witness_{number:02d}.npz"
                    np.savez_compressed(path, pose_aa=pose.numpy(), root_trans=data["raw"]["root_trans_offset"],
                        body_positions=pos.numpy(), body_quaternions_wxyz=quat.numpy(),
                        centers=centers.numpy(), radii=radii.numpy(), dof=fk.dof_pos[0].numpy())
                    report["witnesses"].append(dict(case=number, file=path.name, sha256=sha256(path)))
    (run/"witness.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(run=str(run), candidates=len(report["cases"]), witnesses=len(report["witnesses"]),
                         roles_retained=roles["all_roles_retained"], route_found=report["geometric_route_found"], training_ready=False)), flush=True)
    return run, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--translation", type=float, nargs=3, required=True)
    parser.add_argument("--yaw", type=float, default=1.5707963267948966)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    search(args.reference_run, args.scene, Placement(tuple(args.translation), args.yaw), args.device)
