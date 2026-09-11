#!/usr/bin/env python3
"""Screen a rigid CAT placement and export full-height diagnostic oracle fields.

Read-only source terrain, CAT occupancy/fields and released reference. No Isaac,
policy execution, training, ROS, SDK or actuation. A failed retention gate still
writes a labelled rejected diagnostic, never a supposedly valid training scene.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import importlib.metadata
from pathlib import Path
import subprocess

import numpy as np
import torch

from artifacts import ROOT
from cat_roles import role_masks
from cat_scenes import sha256
from gear_sonic.research.cat_geometry import CatFields, Placement, cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_roles import assess_roles, ground_rigid_placement
from gear_sonic.research.scene_audit import check, read_reference, read_snapshot, save
from gear_sonic.research.scene_volume import VolumeSpec, build_volume, sample_distances
from gear_sonic.research.terrain_surface import TerrainSurface


def compose(reference_run, scene, placement, device="cpu", resolution=.04, ground_on_terrain=False):
    reference_run, scene = Path(reference_run).resolve(), Path(scene).resolve()
    masks, source = role_masks(scene)  # fail early for missing explicit role provenance
    grounding = dict(mode="explicit XYZ/yaw placement; no automatic adjustment")
    if ground_on_terrain:
        if placement.translation[2] != 0.:
            raise ValueError("Use Z=0 with --ground-on-terrain; do not mix explicit and computed datum")
        v, f, terrain = read_snapshot(reference_run)
        patch = TerrainSurface(v, f, ground_z=terrain["ground"]["height"], device=device)
        placement, grounding = ground_rigid_placement(patch, masks, source["origin_corner"],
            source["resolution"], placement.translation[:2], placement.yaw)
    layout, layout_arrays, surface = check(reference_run, scene, placement, device=device)
    roles = assess_roles(surface, masks, source["origin_corner"], source["resolution"], placement)
    cv, cf = cat_mesh_arrays(scene)
    clutter = ClosedMeshDistance(cv, cf, device=device)
    world_vertices = placement.to_world(torch.as_tensor(cv.copy(), device=device)).cpu().numpy()
    anchors, centers, radii, _ = read_reference(reference_run)
    spec = VolumeSpec.fit(surface.vertices.cpu().numpy(), world_vertices, centers, radii,
                          ground_z=surface.ground_z, resolution=resolution, margin=max(.12, 2*resolution))
    arrays = build_volume(surface, clutter, placement, spec)
    points = torch.as_tensor(centers, device=device)
    samples = sample_distances(arrays, spec, points)
    exact_c, _, valid_c = clutter.query(placement.to_local(points))
    exact_t, _, valid_t = surface.distance(points)
    coverage = samples["clutter_sdf_valid"] & samples["terrain_unsigned_distance_valid"] & valid_c & valid_t
    if not bool(coverage.all()):
        raise ValueError("Full-height volume does not cover every reference probe")
    errors = dict(clutter_sdf=float((samples["clutter_sdf"]-exact_c).abs().max()),
                  terrain_unsigned_distance=float((samples["terrain_unsigned_distance"]-exact_t).abs().max()))
    # 1-Lipschitz distance interpolation bound; exact queries, not these fields,
    # still decide geometric clearance. No analogous normal bound at mesh edges.
    tolerance = float(np.sqrt(3)*resolution+1e-5)
    if max(errors.values()) > tolerance:
        raise ValueError(f"Distance sampling mismatch {errors}, limit {tolerance}")
    old = CatFields(scene, placement, device).sample(points)["valid"]
    retained = roles["all_roles_retained"]
    accepted = layout["geometry_screen_accepted"] and retained
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_composed_scene")
    run.mkdir(parents=True, exist_ok=False)
    save(run/"layout_audit.json", layout, layout_arrays)
    np.savez_compressed(run/"volume.npz", **arrays)
    report = dict(schema="grail-cat-composed-oracle-v1", simulation_only=True, source_scene=str(scene),
        source_cat_files=source["files"], source_cat_manifest_sha256=sha256(scene/"scene.json"),
        reference_run=str(reference_run), terrain_manifest_sha256=sha256(reference_run/"terrain_snapshot.json"),
        terrain_npz_sha256=sha256(reference_run/"terrain_snapshot.npz"),
        reference_sweep_sha256=sha256(reference_run/"reference_sweep.npz"),
        placement=asdict(placement), grounding=grounding, grid=asdict(spec), sample_origin=spec.sample_origin,
        axis_order="xyz", units="m", up_axis="Z", role_retention=roles,
        placement_accepted_for_cached_reference_screen=accepted,
        original_cat_probe_coverage=float(old.float().mean()), full_height_probe_coverage=float(coverage.float().mean()),
        reference_probe_samples=int(coverage.numel()), max_interpolation_error_m=errors,
        interpolation_error_bound_m=tolerance,
        volume_file="volume.npz", volume_sha256=sha256(run/"volume.npz"),
        source_geometry_modified=False, fields_recomputed_from_exact_mesh=True,
        terrain_solid_inside_outside_known=False, guidance_extended=False, sensor_realism=False,
        physics_validated_for_this_placement=False, avoidance_training_ready=False,
        contact_permissions_applied=False,
        packages={n: importlib.metadata.version(n) for n in ("numpy", "torch", "warp-lang", "trimesh")},
        semantic_contract={
            "clutter_sdf": "signed exact closed CAT mesh distances at cell centres; negative inside",
            "clutter_inside": "CAT occupied cell-centre samples, not terrain occupancy",
            "terrain_unsigned_distance": "unsigned to actual triangles/explicit ground; NOT a solid/free-space test",
            "terrain_authored_normal": "closest face winding normal, NOT unsigned-distance gradient; ambiguous at edges",
            "terrain_surface_band": "distance <= half voxel diagonal; retained at support surfaces",
            "support_candidate": "top-down upward terrain/ground only; NOT reachable foothold/contact permission",
            "validity": "simulation oracle query/domain validity; NOT lidar visibility/known free space",
            "cat_guidance": "unchanged original CAT domain only; no terrain-aware FMM yet"},
        adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    (run/"composition.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    plot(run, spec, arrays, centers, anchors)
    print(json.dumps(dict(run=str(run), accepted=accepted, roles_retained=retained,
        old_coverage=report["original_cat_probe_coverage"], full_height_coverage=report["full_height_probe_coverage"],
        max_error=errors, shape=spec.shape, avoidance_training_ready=False)), flush=True)
    return run, report


def plot(run, spec, arrays, centers, anchors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # YZ slice through the median reference X, plus all reference probe centres
    # near that slice. This is a geometry diagnostic, not a rendered rollout.
    index = int(np.clip(np.rint((np.median(anchors[:, 0])-spec.sample_origin[0])/spec.resolution), 0, spec.shape[0]-1))
    extent = [spec.origin_corner[1], spec.origin_corner[1]+spec.shape[1]*spec.resolution,
              spec.origin_corner[2], spec.origin_corner[2]+spec.shape[2]*spec.resolution]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    im = axes[0].imshow(arrays["clutter_sdf"][index].T, extent=extent, origin="lower", cmap="coolwarm", vmin=-.2, vmax=.5)
    fig.colorbar(im, ax=axes[0], label="Closed CAT signed distance (m)")
    im = axes[1].imshow(arrays["terrain_unsigned_distance"][index].T, extent=extent, origin="lower", cmap="viridis", vmin=0, vmax=.5)
    fig.colorbar(im, ax=axes[1], label="Open terrain unsigned distance (m)")
    near = centers[np.abs(centers[..., 0]-(spec.sample_origin[0]+index*spec.resolution)) < .06][::15]
    for ax in axes:
        ax.plot(near[:, 1], near[:, 2], ".", color="orange", markersize=2, label="Reference probes near slice")
        ax.set(xlabel="World Y (m)", ylabel="World Z (m)")
        ax.legend(fontsize=7)
    fig.suptitle("Full-height geometry oracle — no combined terrain solid / no avoidance policy")
    fig.savefig(run/"volume_slice.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--translation", type=float, nargs=3, default=(0., 0., 0.))
    p.add_argument("--yaw", type=float, default=0.)
    p.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    p.add_argument("--resolution", type=float, default=.04)
    p.add_argument("--ground-on-terrain", action="store_true", help="Infer rigid Z datum from full CAT footprint; reject uneven support")
    args = p.parse_args()
    _, report = compose(args.reference_run, args.scene, Placement(tuple(args.translation), args.yaw), args.device, args.resolution, args.ground_on_terrain)
    raise SystemExit(0 if report["placement_accepted_for_cached_reference_screen"] else 2)


if __name__ == "__main__":
    main()
