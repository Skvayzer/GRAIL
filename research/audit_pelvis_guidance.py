#!/usr/bin/env python3
"""Read-only old/new pelvis-guidance comparison; no actor or simulator startup.

Checks selected connectors with a separate scalar cell-intersection routine,
then compares CPU/CUDA and batch/single-frame behavior. Saves a new report;
old observation packets are never rewritten as if they used the new contract.
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
from gear_sonic.research.cat_geometry import verify_scene_files
from gear_sonic.research.obstacle_observation import GuidanceSampler, yaw_rotation
from gear_sonic.research.scene_audit import read_snapshot
from gear_sonic.research.terrain_surface import TerrainSurface


def reference_cover(start, end):
    """Scalar line clipping against every cell in the segment bounding box."""
    result = []
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    lower = np.ceil(np.minimum(start, end)-.500001).astype(int)
    upper = np.floor(np.maximum(start, end)+.500001).astype(int)
    if np.prod(upper-lower+1) > 400:
        raise ValueError("Unbounded connector in independent audit")
    for i in range(lower[0], upper[0]+1):
        for j in range(lower[1], upper[1]+1):
            entry, exit = 0., 1.
            for a, b, centre in zip(start, end, (i, j)):
                d = b-a
                if abs(d) < 1e-12:
                    if abs(a-centre) > .500001:
                        entry, exit = 1., 0.
                        break
                else:
                    crossings = sorted(((centre-.5-a)/d, (centre+.5-a)/d))
                    entry, exit = max(entry, crossings[0]), min(exit, crossings[1])
            if entry <= exit+1e-6:
                result.append((i, j))
    return result


def check_connector(root, support, target, arrays, sampler):
    """Independent witness check, not a call to the GPU connector selector."""
    target = tuple(map(int, target))
    h, transit = arrays["support_height"], arrays["transit_valid"]
    if any(v < 0 or v >= n for v, n in zip(target, h.shape)) or not arrays["walkable"][target]:
        raise ValueError("Target is not a supported node")
    if not sampler.attachment.reachable.cpu().numpy()[target]:
        raise ValueError("Target cannot reach the goal")
    origin = sampler.origin.cpu().numpy().astype(float)
    start = (root[:2]-origin)/sampler.resolution
    cells = reference_cover(start, target)
    if not cells:
        raise ValueError("No connector cells")
    for cell in cells:
        if any(v < 0 or v >= n for v, n in zip(cell, h.shape)) or not transit[cell] or not np.isfinite(h[cell]):
            raise ValueError("Connector crosses blocked/out-of-bounds/unknown transit")
    heights = [support, *[float(h[cell]) for cell in cells]]
    if np.ptp(heights) > sampler.attachment.max_step+2e-6:
        raise ValueError("Connector exceeds height-excursion bound")
    distance = np.linalg.norm((np.array(target)-start)*sampler.resolution)
    if distance > sampler.attachment.max_stride+2e-6:
        raise ValueError("Connector exceeds length bound")
    return len(cells)


def capture(meta, arrays, vertices, faces, terrain, saved, device, batch):
    surface = TerrainSurface(vertices, faces, ground_z=terrain["ground"]["height"], device=device)
    sampler = GuidanceSampler(meta, arrays, device)
    gathered = {}
    for start in range(0, len(saved["root"]), batch):
        section = slice(start, start+batch)
        root = torch.tensor(saved["root"][section], device=device)
        quaternion = torch.tensor(saved["quaternion"][section], device=device)
        features, valid = sampler.sample(root, yaw_rotation(quaternion), surface, 5.)
        support, _, _ = surface.support_below(root, max_drop=2.)
        packet = dict(features=features, support=support, **sampler.last_attachment)
        for key, tensor in packet.items():
            gathered.setdefault(key, []).append(tensor.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in gathered.items()}, sampler


def audit(run, require_all_valid=False):
    run = Path(run).resolve()
    old_audit = audit_observation_shadow(run)
    old = json.loads((run/"observation_shadow.json").read_text())
    meta, arrays = read_layout(run/"layout_audit.json")
    vertices, faces, terrain = read_snapshot(run)
    cat = json.loads((run/"cat_audit.json").read_text())
    source = verify_scene_files(cat["source_scene"])
    if (not meta["accepted_for_reference_diagnostic"] or source["files"] != old["cat_files"]
            or old["terrain_sha256"] != terrain["sha256"] or old["grid_sha256"] != meta["grid_sha256"]
            or old["layout_sha256"] != sha256(run/"layout_audit.json") or cat["placement"] != meta["placement"]):
        raise ValueError("Source layout/geometry no longer matches recorded observation")
    with np.load(run/"observation_shadow.npz", allow_pickle=False) as data:
        saved = {k: data[k] for k in ("root", "quaternion", "guidance", "valid")}
    if not torch.cuda.is_available():
        raise RuntimeError("This confirmation audit requires both CPU and CUDA")
    cpu, sampler = capture(meta, arrays, vertices, faces, terrain, saved, "cpu", 16)
    errors = {}
    for device, batch in (("cuda:0", 16), ("cuda:0", 1)):
        candidate, _ = capture(meta, arrays, vertices, faces, terrain, saved, device, batch)
        per_key = {}
        for key in cpu:
            a, b = cpu[key], candidate[key]
            if a.dtype.kind in "biu":
                if not np.array_equal(a, b):
                    raise ValueError(f"{device}/{batch} discrete {key} differs from CPU")
            else:
                error = float(np.abs(a-b).max())
                per_key[key] = error
                if error > 2e-5:
                    raise ValueError(f"{device}/{batch} {key} error={error}")
        errors[f"{device}/batch{batch}"] = per_key
    witnesses = []
    for i in np.flatnonzero(cpu["valid"]):
        count = check_connector(saved["root"][i], cpu["support"][i], cpu["target_cell"][i], arrays, sampler)
        if count != cpu["covered_cells"][i]:
            raise ValueError(f"Independent connector cell count differs at frame {i}")
        witnesses.append(int(i))
    old_valid = saved["guidance"][:, 7] == 1
    if (old_valid & ~cpu["valid"]).any() or (require_all_valid and not cpu["valid"].all()):
        raise ValueError("Guidance regression or required coverage not achieved")
    output = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_pelvis_guidance_audit")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"guidance.npz", root=saved["root"], old_valid=old_valid, **cpu)
    report = dict(schema="grail-cat-pelvis-guidance-audit-v1", passed=True, source_run=str(run),
        source_report_sha256=sha256(run/"observation_shadow.json"), source_packet_sha256=old["packet_sha256"],
        source_layout_sha256=old["layout_sha256"], frames=len(old_valid), old_valid=int(old_valid.sum()),
        new_valid=int(cpu["valid"].sum()), recovered=int((~old_valid & cpu["valid"]).sum()),
        lost_valid=int((old_valid & ~cpu["valid"]).sum()), independent_connectors_checked=len(witnesses),
        max_connector_xy_m=float(cpu["connector_xy"].max()), parity_max_errors=errors,
        contract=sampler.attachment.manifest(), source_audit=old_audit,
        packet_sha256=sha256(output/"guidance.npz"), simulation_started=False, policy_connected=False,
        training_started=False, full_body_feasibility_verified=False,
        adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    (output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(output), **report)), flush=True)
    return output, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--require-all-valid", action="store_true", help="Only for a known traversable reference, not arbitrary scenes")
    args = parser.parse_args()
    torch.set_num_threads(2)  # Offline tool only; never changes the actor process.
    audit(args.run, args.require_all_valid)
