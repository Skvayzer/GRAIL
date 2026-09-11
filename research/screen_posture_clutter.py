#!/usr/bin/env python3
"""Fixed CAT placement screen for arm-conflicting, foothold-preserving tasks.

Reads existing CAT meshes and a reproduced reference. Does not alter either,
start simulation, execute a controller or label candidates training-ready.
Actual reference intersection uses spheres INSIDE the imported capsules;
protected-body clearance uses conservative OUTER covers. A viable modified
posture, role/layout/guidance checks and physics review remain separate gates.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import torch

from artifacts import ROOT
from cat_scenes import sha256, verify_scene
from reference_postures import reconstruct
from gear_sonic.research.cat_geometry import Placement, cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance


MOVABLE = ("left_elbow_link", "left_wrist_yaw_link", "right_elbow_link", "right_wrist_yaw_link")
X_OFFSETS = (-.65, -.50, -.35, -.20, 0., .20, .35, .50, .65)
Y_OFFSETS = (-.3, .2, .7, 1.2)
Z_OFFSETS = (0., .2, .4)


def classify_reference(outer_gaps, inner_gaps, links, *, margin=.03, overlap=.005, min_frames=5):
    if (outer_gaps.shape != inner_gaps.shape or outer_gaps.ndim != 2 or outer_gaps.shape[1] != len(links)
            or any(not torch.isfinite(v).all() for v in (outer_gaps, inner_gaps))
            or not 0 < margin <= .1 or not 0 < overlap <= .1 or min_frames < 1):
        raise ValueError("Valid paired outer/inner clearance trajectories required")
    movable = torch.tensor([name in MOVABLE for name in links], device=outer_gaps.device)
    if not movable.any() or movable.all():
        raise ValueError("Both movable arms and protected bodies must be represented")
    protected_gap = float(outer_gaps[:, ~movable].min())
    spawn_gap = float(outer_gaps[:min(50, len(outer_gaps))].min())
    intersection = inner_gaps[:, movable].min(1).values < -overlap
    frames = intersection.nonzero().flatten()
    candidate = protected_gap >= margin and spawn_gap >= margin and len(frames) >= min_frames
    if candidate:
        state = "ARM_CONFLICT_CANDIDATE_NEEDS_POSTURE_WITNESS"
    elif protected_gap < margin:
        state = "REJECTED_PROTECTED_BODY_CONFLICT"
    elif spawn_gap < margin:
        state = "REJECTED_SPAWN_CONFLICT"
    else:
        state = "NO_DEMONSTRATED_ARM_INTERSECTION"
    return dict(state=state, candidate=candidate, protected_cover_gap_m=protected_gap, initial_one_second_gap_m=spawn_gap,
        reference_outer_gap_m=float(outer_gaps.min()), reference_inner_gap_m=float(inner_gaps[:, movable].min()),
        arm_intersection_frames=len(frames), first_intersection_frame=int(frames[0]) if len(frames) else None,
        posture_feasibility_verified=False, dynamic_feasibility_verified=False, training_ready=False)


def screen(reference_run, scene, device="cuda:0"):
    scene = Path(scene).resolve()
    source = verify_scene(scene)
    data = reconstruct(reference_run)
    vertices, faces = cat_mesh_arrays(scene)
    mesh = ClosedMeshDistance(vertices, faces, device=device)
    centers, outer, inner = [data[key].to(device) for key in ("centers", "radii", "inner_radii")]
    links = [p.link for p in data["probes"]]
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_posture_clutter_screen")
    run.mkdir(parents=True, exist_ok=False)
    report = dict(schema="grail-cat-posture-clutter-screen-v1", simulation_only=True,
        reference=data["provenance"], source_scene=str(scene), source_files=source["files"],
        source_manifest_sha256=sha256(scene/"scene.json"), movable_links=list(MOVABLE),
        placement_grid=dict(x=X_OFFSETS, y=Y_OFFSETS, z=Z_OFFSETS, yaw=math.pi/2),
        sampling="fixed finite grid, every placement reported; no retry-until-pass",
        source_geometry_modified=False, policy_connected=False, training_ready=False, cases=[],
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip())
    for x in X_OFFSETS:
        for y in Y_OFFSETS:
            for z in Z_OFFSETS:
                placement = Placement((x, y, z), math.pi/2)
                distance, _, known = mesh.query(placement.to_local(centers))
                if not known.all():
                    raise ValueError("Unknown CAT mesh distance; cannot certify protected bodies")
                result = classify_reference(distance-outer, distance-inner, links)
                result["placement"] = asdict(placement)
                report["cases"].append(result)
    report["candidate_count"] = sum(case["candidate"] for case in report["cases"])
    (run/"screen.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(run=str(run), placements=len(report["cases"]), candidates=report["candidate_count"],
                         training_ready=False)), flush=True)
    return run, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args()
    screen(args.reference_run, args.scene, args.device)
