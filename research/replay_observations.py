#!/usr/bin/env python3
"""Replay saved oracle features against checksummed geometry; no Isaac/actor.

Creates a new report, never changes the recorded rollout or original assets.
This verifies geometry/coordinate reproducibility, not sensor realism or a
trained avoidance policy. CPU/CUDA query discrepancies have explicit bounds.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from artifacts import ROOT
from cat_scenes import sha256
from observation_results import audit_observation_shadow
from terrain_guidance import read_layout
from gear_sonic.research.cat_geometry import Placement, cat_mesh_arrays, verify_scene_files
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_observation import GuidanceSampler, ObstacleObservation, ObservationSpec
from gear_sonic.research.scene_audit import read_snapshot
from gear_sonic.research.terrain_surface import TerrainSurface


def replay(run, device="cpu"):
    run = Path(run).resolve()
    audit = audit_observation_shadow(run)
    meta = json.loads((run/"observation_shadow.json").read_text())
    cat = json.loads((run/"cat_audit.json").read_text())
    source = verify_scene_files(cat["source_scene"])
    if source["files"] != meta["cat_files"] or cat["source_files"] != source["files"]:
        raise ValueError("Shadow CAT geometry differs from recorded source")
    vertices, faces, terrain = read_snapshot(run)
    layout, arrays = read_layout(run/"layout_audit.json")
    if (terrain["sha256"] != meta["terrain_sha256"] or layout["grid_sha256"] != meta["grid_sha256"]
            or sha256(run/"layout_audit.json") != meta["layout_sha256"]
            or cat["placement"] != layout["placement"]):
        raise ValueError("Shadow terrain/layout/placement source mismatch")
    p = meta["packet_spec"]
    spec = ObservationSpec(shape=tuple(p["shape"]), spacing=p["spacing"], distance_scale=p["distance_scale"],
                           goal_scale=p["goal_scale"], max_batch=p["max_batch"])
    if json.loads(json.dumps(spec.manifest())) != p:
        raise ValueError("Unknown observation channel/coordinate contract")
    surface = TerrainSurface(vertices, faces, ground_z=terrain["ground"]["height"], device=device)
    clutter = ClosedMeshDistance(*cat_mesh_arrays(cat["source_scene"]), device=device)
    placement = Placement(tuple(cat["placement"]["translation"]), cat["placement"]["yaw"])
    observer = ObstacleObservation(surface, clutter, placement, GuidanceSampler(layout, arrays, device), spec)
    with np.load(run/"observation_shadow.npz", allow_pickle=False) as archive:
        saved = {k: archive[k] for k in archive.files}
    radii = torch.tensor([p["radius"] for p in meta["probe_order"]], dtype=torch.float32, device=device)
    errors = {k: 0. for k in ("volume", "probes", "guidance")}
    for start in range(0, meta["frames"], spec.max_batch):
        section = slice(start, start+spec.max_batch)
        tensors = [torch.as_tensor(saved[k][section], device=device) for k in ("root", "quaternion", "probe_centers")]
        with torch.no_grad():
            result = observer.sample(*tensors, radii)
        if not np.array_equal(result["valid"].cpu().numpy(), saved["valid"][section]):
            raise ValueError(f"Observation validity replay mismatch at frame {start}")
        for key in errors:
            actual, expected = result[key].cpu().numpy(), saved[key][section]
            error = float(np.abs(actual-expected).max())
            errors[key] = max(errors[key], error)
            if error > 2e-5:
                raise ValueError(f"Observation geometry replay mismatch: {key}, frame {start}, error={error}")
    output = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_observation_replay")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="grail-cat-observation-replay-v1", passed=True, simulation_started=False,
        source_run=str(run), source_report_sha256=sha256(run/"observation_shadow.json"),
        source_packet_sha256=meta["packet_sha256"], device=device, frames=meta["frames"],
        max_feature_errors=errors, feature_error_tolerance=2e-5, saved_audit=audit,
        adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    (output/"replay.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), **report)), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    replay(args.run, args.device)
