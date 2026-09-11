#!/usr/bin/env python3
"""Fixed role-valid CAT placement batch; candidates, not training admission.

Preserve original random CAT output and independently screen terrain grounding,
role retention and actual arm intersection with protected feet/root/torso. No
actor, Isaac startup, robot connection, retries-until-pass or source mutation.
The fixed validation seeds are geometry-development validation, not a final
unseen-policy benchmark. No layout is marked training-ready by this script.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

import torch

from artifacts import ROOT
from cat_roles import role_masks
from cat_scenes import generate, generator_modules, sha256
from random_cat_roles import trace_random
from reference_postures import reconstruct
from screen_posture_clutter import classify_reference
from gear_sonic.research.cat_geometry import cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_roles import assess_roles, ground_rigid_placement
from gear_sonic.research.scene_audit import read_snapshot
from gear_sonic.research.terrain_surface import TerrainSurface


SPLITS = {"development": (0, 2, 3, 42), "validation": (101, 102)}
XS = (-.65, -.5, -.35, -.2, -.1, 0., .1, .2, .35, .5, .65)
YS = (-.4, -.2, 0., .2, .4, .6, .8, 1., 1.2, 1.4, 1.6)
VOXEL_XS = tuple(round(i*.04, 8) for i in range(-16, 17))
VOXEL_YS = tuple(round(i*.04, 8) for i in range(-10, 41))


def screen(reference_run, device="cuda:0", placement_grid="coarse"):
    if placement_grid not in ("coarse", "voxel"):
        raise ValueError("Known fixed placement grid required")
    xs, ys = (XS, YS) if placement_grid == "coarse" else (VOXEL_XS, VOXEL_YS)
    reference_run = Path(reference_run).resolve()
    data = reconstruct(reference_run)
    v, f, terrain = read_snapshot(reference_run)
    surface = TerrainSurface(v, f, ground_z=terrain["ground"]["height"], device=device)
    centers, outer, inner = [data[key].to(device) for key in ("centers", "radii", "inner_radii")]
    links = [p.link for p in data["probes"]]
    _, module, _, _ = generator_modules()
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_training_layout_screen")
    run.mkdir(parents=True, exist_ok=False)
    report = dict(schema="grail-cat-fixed-training-layout-screen-v1", simulation_only=True,
        reference=data["provenance"], splits=SPLITS, x=xs, y=ys, yaw=math.pi/2, placement_grid=placement_grid,
        generation=dict(difficulty=.2, n_side=1, n_floor=0, n_ceiling=0, role_trace="unique-physical-v2"),
        grounding="single rigid placement over a common support datum, no shape deformation",
        source_geometry_modified=False, resampling=False, training_ready=False, complete=False,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(), cases=[])
    def save():
        (run/"screen.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    for split, seeds in SPLITS.items():
        for seed in seeds:
            cfg = module.Cfg(seed=seed, difficulty=.2, n_rect_L=1, n_rect_R=1, n_rect_F=0, n_rect_C=0)
            occupied, _, _, trace = trace_random(cfg, module, "unique-physical-v2")
            item = dict(split=split, seed=seed, placements=[], candidates=[], trace=trace)
            report["cases"].append(item)
            if not occupied.any():
                item["state"] = "REJECTED_EMPTY_SOURCE"
            elif not trace["placement_roles_resolved"]:
                item["state"] = "REJECTED_UNRESOLVED_SOURCE_ROLES"
            else:
                scene = generate(seed=seed, n_side=1, n_floor=0, n_ceiling=0, role_trace="unique-physical-v2")
                masks, meta = role_masks(scene)
                cv, cf = cat_mesh_arrays(scene)
                mesh = ClosedMeshDistance(cv, cf, device=device)
                item.update(state="SCREENED", scene=str(scene), scene_sha256=sha256(scene/"scene.json"))
                for x in xs:
                    for y in ys:
                        trial = dict(x=x, y=y, candidate=False)
                        item["placements"].append(trial)
                        try:
                            placement, grounding = ground_rigid_placement(surface, masks, meta["origin_corner"],
                                                                         meta["resolution"], (x, y), math.pi/2)
                        except ValueError as error:
                            trial.update(state="REJECTED_GROUNDING", reason=str(error))
                            continue
                        roles = assess_roles(surface, masks, meta["origin_corner"], meta["resolution"], placement)
                        trial.update(placement=asdict(placement), grounding=grounding, roles_retained=roles["all_roles_retained"])
                        if not roles["all_roles_retained"]:
                            trial.update(state="REJECTED_ROLE_RETENTION", roles=roles)
                            continue
                        distance, _, known = mesh.query(placement.to_local(centers))
                        if not known.all():
                            raise ValueError("Unknown signed distance; no classification permitted")
                        trial.update(classify_reference(distance-outer, distance-inner, links))
                        if trial["candidate"]:
                            item["candidates"].append(len(item["placements"])-1)
            save()
            print(json.dumps(dict(split=split, seed=seed, state=item["state"], candidates=len(item["candidates"]),
                                  placement_states=dict(Counter(p["state"] for p in item["placements"])))), flush=True)
    report.update(complete=True, candidate_count=sum(len(case["candidates"]) for case in report["cases"]))
    save()
    print(json.dumps(dict(run=str(run), candidate_count=report["candidate_count"], training_ready=False)), flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--placement-grid", choices=("coarse", "voxel"), default="coarse")
    args = parser.parse_args()
    screen(args.reference_run, args.device, args.placement_grid)
