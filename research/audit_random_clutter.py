#!/usr/bin/env python3
"""Fixed nine-case CAT provenance/placement screen; no retry-until-pass sampling.

Reads a previously captured terrain/reference, never starts simulation or an
actor. Reports every seed, empty scene, unresolved role and rejected placement.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from artifacts import ROOT
from cat_scenes import generate, generator_modules, sha256, verify_scene
from cat_roles import role_masks
from random_cat_roles import trace_random
from gear_sonic.research.obstacle_roles import assess_roles, ground_rigid_placement
from gear_sonic.research.scene_audit import check, read_reference, read_snapshot, save
from gear_sonic.research.terrain_surface import TerrainSurface
from terrain_guidance import export


RECIPES = (("original_counts", 9, 3, 3), ("sparse_lateral", 1, 0, 0), ("floor_only", 0, 1, 0))
SEEDS = (0, 1, 42)
PLACEMENTS_XY = ((0., -1.), (0., .2), (0., 1.2))


def audit(reference_run, device="cpu"):
    reference_run = Path(reference_run).resolve()
    vertices, faces, terrain = read_snapshot(reference_run)
    read_reference(reference_run)  # Validate the reference hash even if every placement rejects early.
    surface = TerrainSurface(vertices, faces, ground_z=terrain["ground"]["height"], device=device)
    _, module, _, _ = generator_modules()
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_random_clutter_audit")
    run.mkdir(parents=True, exist_ok=False)
    report = dict(schema="grail-cat-random-placement-audit-v1", simulation_only=True,
        reference_run=str(reference_run), terrain_manifest_sha256=sha256(reference_run/"terrain_snapshot.json"),
        reference_sweep_sha256=sha256(reference_run/"reference_sweep.npz"), seeds=list(SEEDS),
        recipes=[dict(name=name, n_side=side, n_floor=floor, n_ceiling=ceiling) for name, side, floor, ceiling in RECIPES],
        difficulty=.2, placement_xy=PLACEMENTS_XY, yaw=math.pi/2, resampling=False,
        source_generator_modified=False, policy_connected=False, avoidance_training_ready=False, complete=False, cases=[],
        adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    for name, side, floor, ceiling in RECIPES:
        for seed in SEEDS:
            cfg = module.Cfg(seed=seed, difficulty=.2, n_rect_L=side, n_rect_R=side, n_rect_F=floor, n_rect_C=ceiling)
            occupied, _, arrays, trace = trace_random(cfg, module)
            item = dict(recipe=name, seed=seed, occupied_cells=int(occupied.sum()), trace=trace, placements=[])
            trace_path = run/f"{name}_{seed}_trace.npz"
            np.savez_compressed(trace_path, **arrays, occupied=occupied)
            item.update(trace_file=trace_path.name, trace_sha256=sha256(trace_path))
            report["cases"].append(item)
            if not occupied.any():
                item["state"] = "REJECTED_EMPTY_UPSTREAM_OUTPUT"
            elif not trace["placement_roles_resolved"]:
                item["state"] = "REJECTED_UNRESOLVED_ROLE_PROVENANCE"
            else:
                scene = generate(seed=seed, n_side=side, n_floor=floor, n_ceiling=ceiling)
                source = verify_scene(scene)
                masks, _ = role_masks(scene)
                item.update(state="ROLE_TRACE_RESOLVED_PLACEMENTS_SCREENED", scene=str(scene),
                            scene_manifest_sha256=sha256(scene/"scene.json"), cat_files=source["files"])
                for number, xy in enumerate(PLACEMENTS_XY):
                    trial = dict(xy=xy, yaw=math.pi/2, accepted_for_cached_reference_screen=False,
                                 avoidance_challenge_demonstrated=False, physical_attachment_verified=False)
                    item["placements"].append(trial)
                    try:
                        placement, grounding = ground_rigid_placement(surface, masks, source["origin_corner"], source["resolution"], xy, math.pi/2)
                    except ValueError as error:
                        trial.update(state="REJECTED_RIGID_GROUNDING", reason=str(error))
                        continue
                    roles = assess_roles(surface, masks, source["origin_corner"], source["resolution"], placement)
                    trial.update(placement=asdict(placement), grounding=grounding, role_retention=roles)
                    if not roles["all_roles_retained"]:
                        trial["state"] = "REJECTED_ROLE_RETENTION"
                        continue
                    layout, data, _ = check(reference_run, scene, placement, device=device)
                    path = run/f"{name}_{seed}_placement_{number}_layout.json"
                    save(path, layout, data)
                    trial.update(state="CACHED_REFERENCE_SCREENED", layout_file=path.name, layout_sha256=sha256(path),
                                 accepted_for_cached_reference_screen=layout["geometry_screen_accepted"],
                                 route_found=layout["support_graph"]["geometric_route_found"],
                                 reference_clear=layout["reference"]["sampled_reference_clear"])
                    # Guidance may be inspected even for a reference-conflict rejection;
                    # it never promotes the trial to training/whole-body feasibility.
                    if data["route"].size:
                        guidance, result = export(path)
                        trial.update(guidance_run=str(guidance), guidance_start_cost_m=result["start_cost_m"])
            (run/"random_clutter_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(json.dumps(dict(recipe=name, seed=seed, state=item["state"], occupied=item["occupied_cells"],
                                  unresolved=trace["unresolved_added_cells"], placements=item["placements"])), flush=True)
    report["complete"] = True
    (run/"random_clutter_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"Audit complete: {run}", flush=True)
    return run, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    audit(args.reference_run, args.device)
