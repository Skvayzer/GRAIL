"""Combined scene diagnostic. No geometry modification or policy training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cat_geometry import CatFields, Placement, cat_mesh_arrays
from .layout_validation import LayoutLimits, layout_grid, obstacle_embedding
from .mesh_distance import ClosedMeshDistance, reference_clearance_summary
from .terrain_surface import TerrainSurface


def read_snapshot(run):
    run = Path(run)
    meta = json.loads((run / "terrain_snapshot.json").read_text())
    if (meta["schema"] != "grail-cat-terrain-snapshot-v1" or meta["source_file"] != "terrain_snapshot.npz"
            or meta["units"] != "m" or meta["up_axis"] != "Z"):
        raise ValueError("Unvalidated terrain snapshot")
    path = run / meta["source_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Terrain snapshot checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return data["vertices"], data["faces"], meta


def check(run, directory, placement, device="cpu", limits=LayoutLimits()):
    run, directory = Path(run), Path(directory)
    vertices, faces, terrain = read_snapshot(run)
    surface = TerrainSurface(vertices, faces, ground_z=terrain["ground"]["height"], device=device)
    fields = CatFields(directory, placement, device)
    clutter = ClosedMeshDistance(*cat_mesh_arrays(directory), device=device)
    prior = json.loads((run / "cat_audit.json").read_text())
    if prior["schema"] != "grail-cat-reference-audit-v1":
        raise ValueError("Unvalidated reference snapshot")
    sweep_path = run / "reference_sweep.npz"
    if hashlib.sha256(sweep_path.read_bytes()).hexdigest() != prior["reference"]["sweep_sha256"]:
        raise ValueError("Reference sweep checksum mismatch")
    with np.load(sweep_path, allow_pickle=False) as data:
        anchors, centers, radii, links = data["anchors"], data["centers"], data["radii"], data["links"].tolist()
    gaps = []
    for batch in torch.as_tensor(centers, device=device).split(128):
        d, _, valid = clutter.query(placement.to_local(batch))
        if not valid.all():
            raise ValueError("Missing clutter clearance")
        gaps.append(d-torch.as_tensor(radii, device=device))
    reference = reference_clearance_summary(torch.cat(gaps), links)
    grid, arrays = layout_grid(surface, clutter, placement, anchors, limits)
    embedding = obstacle_embedding(surface, np.load(directory / "obs.npy", allow_pickle=False),
        fields.meta["origin_corner"], fields.meta["resolution"], placement)
    accepted = grid["geometric_route_found"] and grid["anchor_support_fraction"] >= .99 and reference["sampled_reference_clear"]
    report = dict(schema="grail-cat-layout-audit-v1", simulation_only=True,
        snapshot_run=str(run), terrain_snapshot=terrain, cat_scene=str(directory),
        cat_files=fields.meta["files"], placement=dict(translation=list(placement.translation), yaw=placement.yaw),
        reference=reference, support_graph=grid, obstacle_embedding=embedding,
        geometry_screen_accepted=accepted, physics_ray_parity_verified=False,
        requires_placement_review=any(c["requires_placement_review"] for c in embedding),
        avoidance_training_ready=False, continuous_articulated_path_verified=False,
        limitations=["Open terrain: top-surface support, not signed solid occupancy",
            "Top-down single-layer graph does not handle underpasses/stacked floors",
            "Small support patch is not full-foot or balance/contact certification",
            "Trunk graph is approximate; full-body gate applies only to sampled reference poses",
            "Buried/floating components need explicit role/placement review",
            "Terminal contact, terrain-aware 3D fields and avoidance learning remain separate gates"])
    return report, arrays, surface


def save(output, report, arrays):
    output = Path(output)
    array_path = output.with_suffix(".npz")
    np.savez_compressed(array_path, **arrays)
    report["grid_file"] = array_path.name
    report["grid_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    xy = arrays["xy"]
    half = report["support_graph"]["limits"]["resolution"]/2
    extent = [float(xy[0, 0, 0])-half, float(xy[-1, -1, 0])+half,
              float(xy[0, 0, 1])-half, float(xy[-1, -1, 1])+half]
    im = axes[0].imshow(arrays["support_height"].T, origin="lower", extent=extent, cmap="terrain")
    fig.colorbar(im, ax=axes[0], label="Support height (m)")
    axes[0].set_title("Terrain/ground support — CAT excluded")
    axes[1].imshow(arrays["walkable"].T, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Geometric support + trunk-clearance screen")
    for ax in axes:
        anchors = arrays["reference_anchors"]
        ax.plot(anchors[:, 0], anchors[:, 1], color="orange", label="Reference pelvis XY", linewidth=1)
        route = arrays["route"]
        if len(route):
            ax.plot(route[:, 0], route[:, 1], color="cyan", label="Geometric route", linewidth=1)
        ax.set(xlabel="World X (m)", ylabel="World Y (m)")
        ax.legend(fontsize=7)
    fig.suptitle("Scene screening only — not whole-body feasibility or learned avoidance")
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)


def physics_parity(wrapper, surface, arrays, terrain):
    import omni.physx
    from .terrain_snapshot import remap_rays
    query = omni.physx.get_physx_scene_query_interface()
    allowed = {c["prim"] for c in terrain["colliders"]}
    flat = arrays["xy"].reshape(-1, 2)
    picks = np.linspace(0, len(flat)-1, 96).astype(int)
    origins = np.column_stack((flat[picks], np.full(len(picks), float(surface.vertices[:, 2].max())+3.)))
    tensor = torch.as_tensor(origins, dtype=torch.float32, device=surface.device)
    heights, _, known = surface.support_below(tensor)
    origin_offset = wrapper.env.scene.env_origins[0].cpu().numpy()
    # Direct GPU mode does NOT update CPU actor poses used by scene queries.
    # Query the SAME cooked terrain collider in its CPU pose by rigidly mapping
    # rays from the live GPU pose. This verifies shape/scale/transform parity,
    # not CPU readback freshness. Never substitute the CPU pose into the policy.
    cpu = omni.physx.get_physx_interface().get_rigidbody_transformation(
        wrapper.env.scene["object"].root_physx_view.prim_paths[0])
    if not cpu["ret_val"]:
        raise ValueError("CPU scene-query terrain pose unavailable")
    cpu_position = np.asarray(cpu["position"])
    xyzw = np.asarray(cpu["rotation"])
    cpu_quaternion = xyzw[[3, 0, 1, 2]]
    gpu_position = np.asarray(terrain["live_position"])+origin_offset
    directions = np.tile([0., 0., -1.], (len(origins), 1))
    query_origins, query_directions = remap_rays(origins+origin_offset, directions,
        gpu_position, terrain["live_quaternion_wxyz"], cpu_position, cpu_quaternion)
    checks = []
    for point, q_origin, q_direction, expected, valid in zip(origins, query_origins, query_directions, heights.tolist(), known.tolist()):
        hits, all_hits = [], []
        def on_hit(hit):
            all_hits.append(dict(collision=str(hit.collision), rigid_body=str(hit.rigid_body), distance=float(hit.distance)))
            if str(hit.collision) in allowed:
                hits.append(float(hit.distance))
            return True
        query.raycast_all(tuple(q_origin), tuple(q_direction), 10., on_hit)
        # The static ground is in world coordinates, not the terrain body frame.
        def on_ground(hit):
            if str(hit.collision) == terrain["ground"]["prim"]:
                hits.append(float(hit.distance))
            return True
        query.raycast_all(tuple(point+origin_offset), (0., 0., -1.), 10., on_ground)
        measured = float(point[2]-min(hits)) if hits else None
        error = abs(measured-expected) if measured is not None and valid else None
        checks.append(dict(xyz=point.tolist(), expected_height=expected if valid else None,
                           measured_height=measured, error_m=error,
                           all_hits=all_hits,
                           passed=error is not None and error < .005))
    return dict(rays=checks, count=len(checks), passed=all(r["passed"] for r in checks),
        mode="same cooked collider with live-GPU-to-CPU-query-pose ray remapping; static ground separate",
        cpu_query_position=cpu_position.tolist(), cpu_query_quaternion_wxyz=cpu_quaternion.tolist(),
        live_gpu_position=gpu_position.tolist(), live_gpu_quaternion_wxyz=terrain["live_quaternion_wxyz"],
        direct_gpu_cpu_pose_limitation="https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/DirectGPUAPI.html")


def run_live(wrapper, cat_audit, output):
    from .terrain_snapshot import capture
    output = Path(output)
    capture(wrapper, output.parent)
    report, arrays, surface = check(output.parent, cat_audit.fields.directory, cat_audit.placement,
                                    device=str(wrapper.env.device))
    parity = physics_parity(wrapper, surface, arrays, report["terrain_snapshot"])
    report["physics_rays"] = parity
    report["physics_ray_parity_verified"] = parity["passed"]
    report["accepted_for_reference_diagnostic"] = report["geometry_screen_accepted"] and parity["passed"]
    save(output, report, arrays)
    print(f"CAT_LAYOUT_AUDIT {output} accepted={report['accepted_for_reference_diagnostic']}", flush=True)
    if not report["accepted_for_reference_diagnostic"]:
        raise RuntimeError("Combined terrain/CAT layout failed screening; see layout_audit.json. No policy-loop steps allowed.")
