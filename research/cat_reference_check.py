#!/usr/bin/env python3
"""Check a cached imported-G1 reference sweep against CAT without starting Isaac.

This checks discrete poses and conservative collider covers, not continuous-time
articulated feasibility, terrain support, or the learned policy's actual motion.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cat_scenes import ROOT
from gear_sonic.research.cat_geometry import CatFields, Placement, cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance, reference_clearance_summary


def check(reference_run, directory, placement, device="cpu"):
    source = json.loads((reference_run/"cat_audit.json").read_text())
    if source["schema"] != "grail-cat-reference-audit-v1":
        raise ValueError("Unrecognized reference audit")
    meta = source["reference"]
    if meta["sweep_file"] != "reference_sweep.npz":
        raise ValueError("Unrecognized sweep file")
    path = reference_run/meta["sweep_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sweep_sha256"]:
        raise ValueError("Reference sweep checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        centers = torch.as_tensor(data["centers"], dtype=torch.float32, device=device)
        radii = torch.as_tensor(data["radii"], dtype=torch.float32, device=device)
        links = data["links"].tolist()
    if (centers.ndim != 3 or centers.shape[-1] != 3 or radii.shape != centers.shape[1:2]
            or len(links) != centers.shape[1] or not torch.isfinite(centers).all()
            or not torch.isfinite(radii).all() or not (radii > 0).all()):
        raise ValueError("Invalid reference sweep arrays")
    fields = CatFields(directory, placement, device)
    mesh = ClosedMeshDistance(*cat_mesh_arrays(directory), device=device)
    gaps, valid_fields = [], []
    for batch in centers.split(128):
        distance, _, valid = mesh.query(placement.to_local(batch))
        if not valid.all():
            raise ValueError("Unknown mesh distance")
        gaps.append(distance-radii)
        valid_fields.append(fields.sample(batch)["valid"])
    report = reference_clearance_summary(torch.cat(gaps), links)
    report.update(reference_run=str(reference_run), reference_sweep_sha256=meta["sweep_sha256"],
                  scene_directory=str(directory), source_files=fields.meta["files"], placement=asdict(placement),
                  field_valid_fraction=float(torch.cat(valid_fields).float().mean()),
                  physics_started=False, policy_loaded=False, avoidance_training_ready=False,
                  terrain_support_verified=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--translation", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--yaw", type=float, default=0., help="radians")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    report = check(args.reference_run.resolve(), args.scene.resolve(), Placement(tuple(args.translation), args.yaw), args.device)
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_cat_reference_check")
    run.mkdir(parents=True, exist_ok=False)
    (run/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(report=str(run/"report.json"), sampled_reference_clear=report["sampled_reference_clear"],
                         minimum_clearance_m=report["minimum_clearance_m"], flagged_frames=report["flagged_frames"])))
    raise SystemExit(0 if report["sampled_reference_clear"] else 2)


if __name__ == "__main__":
    main()
