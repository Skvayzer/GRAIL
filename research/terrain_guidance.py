#!/usr/bin/env python3
"""Export diagnostic goal guidance from a checksummed terrain/clutter layout.

No simulator, actor, ROS, SDK or motor commands. Original CAT gf/travel files
remain untouched. This single-layer support graph is not a 3D whole-body planner.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from artifacts import ROOT
from cat_scenes import sha256
from gear_sonic.research.layout_validation import LayoutLimits
from gear_sonic.research.support_guidance import SupportGraph


def read_layout(path):
    path = Path(path).resolve()
    meta = json.loads(path.read_text())
    if meta["schema"] != "grail-cat-layout-audit-v1" or meta["grid_file"] != path.with_suffix(".npz").name:
        raise ValueError("Expected a validated layout snapshot")
    data_path = path.with_suffix(".npz")
    if sha256(data_path) != meta["grid_sha256"]:
        raise ValueError("Layout grid checksum mismatch")
    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    return meta, arrays


def build(meta, arrays, goal=None, start=None):
    limits = LayoutLimits(**meta["support_graph"]["limits"])
    h, xy = arrays["support_height"], arrays["xy"]
    if h.ndim != 2 or list(h.shape) != meta["support_graph"]["shape"] or xy.shape != (*h.shape, 2):
        raise ValueError("Layout coordinate shape mismatch")
    indices = np.stack(np.indices(h.shape), axis=-1)
    expected = indices*limits.resolution+np.array(meta["support_graph"]["xy_origin"])
    if not np.isfinite(xy).all() or not np.allclose(xy, expected, atol=1e-6, rtol=0):
        raise ValueError("Layout is not an axis-aligned world XY grid")
    graph = SupportGraph(h, arrays["walkable"], limits.resolution, limits.max_step,
                         transit=arrays["transit_valid"], max_stride=limits.max_stride)
    endpoints = meta["support_graph"].get("endpoint_grid_indices")
    if endpoints is None:  # Read-only compatibility with older validated snapshots.
        route = arrays["route"]
        endpoints = [None, None]
        if route.ndim != 2 or route.shape[1] != 3:
            raise ValueError("Invalid cached route")
        if len(route):
            endpoints = []
            for point in route[[0, -1]]:
                cell = tuple(np.rint((point[:2]-xy[0, 0])/limits.resolution).astype(int))
                if graph.index(cell) is None or not np.allclose(point, [*xy[cell], h[cell]], atol=1e-5, rtol=0):
                    raise ValueError("Cached route endpoint is not on the support graph")
                endpoints.append(cell)
    start = endpoints[0] if start is None else start
    goal = endpoints[1] if goal is None else goal
    field = graph.goal_field(goal)  # Missing/invalid goal fails; no identity/flat fallback.
    route_cells = graph.route(start, goal, field)
    route = np.array([[*xy[p], h[p]] for p in route_cells], dtype=np.float32).reshape(-1, 3)
    field.update(xy=xy, support_height=h, walkable=arrays["walkable"], route=route,
                 source_route=arrays["route"], reference_anchors=arrays["reference_anchors"])
    metrics = dict(goal_cell=list(map(int, goal)), start_cell=list(map(int, start)) if start is not None else None,
        goal_world=[*map(float, xy[tuple(goal)]), float(h[tuple(goal)])],
        reachable_cells=int(field["reachable"].sum()), walkable_cells=int(graph.walkable.sum()),
        directed_edges=int(graph.matrix.nnz), start_reachable=bool(route_cells),
        start_cost_m=float(field["goal_cost"][tuple(start)]) if route_cells else None,
        route_points=len(route), route_support_height_range_m=[float(route[:, 2].min()), float(route[:, 2].max())] if len(route) else None,
        graph_limits=meta["support_graph"]["limits"])
    return field, metrics


def export(path, goal=None, start=None):
    path = Path(path).resolve()
    meta, arrays = read_layout(path)
    field, metrics = build(meta, arrays, goal, start)
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_terrain_guidance")
    run.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(run/"guidance.npz", **field)
    report = dict(schema="grail-cat-support-guidance-v1", simulation_only=True,
        source_layout=str(path), source_layout_sha256=sha256(path), source_grid_sha256=meta["grid_sha256"],
        source_geometry_screen_accepted=meta["geometry_screen_accepted"],
        placement_role_retention_rechecked=False, field_file="guidance.npz", field_sha256=sha256(run/"guidance.npz"),
        **metrics, frame="world", units="m", axis_order="xy", up_axis="Z",
        cost="Dijkstra shortest sum of 3D endpoint edge lengths; not continuous geodesic distance",
        direction="unit world XYZ direction to next support node; zero at goal; NaN if unreachable",
        next_index="row-major flat support-grid index; -1 at goal and unreachable cells",
        invalid="NaN cost/direction/delta plus reachable=false; never treated as free or zero-cost",
        original_cat_fields_modified=False, full_3d_guidance=False, multi_level_terrain_supported=False,
        whole_body_feasibility_verified=False, sensor_realism=False, policy_connected=False,
        contact_permissions_applied=False, avoidance_training_ready=False,
        adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
        adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)))
    (run/"guidance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    plot(run, field, metrics)
    print(json.dumps(dict(run=str(run), **metrics, policy_connected=False)), flush=True)
    return run, report


def plot(run, field, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xy, route = field["xy"], field["route"]
    half = metrics["graph_limits"]["resolution"]/2
    extent = [xy[0, 0, 0]-half, xy[-1, -1, 0]+half, xy[0, 0, 1]-half, xy[-1, -1, 1]+half]
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = fig.add_subplot(121)
    im = ax.imshow(field["goal_cost"].T, origin="lower", extent=extent, cmap="viridis")
    ax.set_facecolor("#d0d0d0")
    fig.colorbar(im, ax=ax, label="Goal graph distance (m); gray = invalid/unreachable")
    s = (slice(None, None, 5),)*2
    v = field["direction"][s]
    ax.quiver(xy[s][..., 0], xy[s][..., 1], v[..., 0], v[..., 1], color="white", scale=24)
    if len(route):
        ax.plot(route[:, 0], route[:, 1], color="cyan", linewidth=2, label="Graph route")
        ax.legend()
    goal = metrics["goal_world"]
    ax.plot(goal[0], goal[1], "r*", markersize=12)
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", title="Terrain-aware 2.5D guidance")
    ax = fig.add_subplot(122, projection="3d", computed_zorder=False)
    s = (slice(None, None, 2),)*2
    ax.plot_surface(xy[s][..., 0], xy[s][..., 1], field["support_height"][s], cmap="terrain", alpha=.4, zorder=1)
    if len(route):
        ax.plot(route[:, 0], route[:, 1], route[:, 2]+.025, color="blue", linewidth=2, zorder=3,
                marker=".", label="Support-node route (overlay)")
        ax.legend(fontsize=8)
    ax.scatter(*goal, color="red", marker="*", s=65)
    ax.set(xlabel="World X (m)", ylabel="World Y (m)", zlabel="Support Z (m)", title="Support heights, not robot trajectory")
    fig.suptitle("Diagnostic only — no policy input, full-body feasibility or contact permission")
    fig.savefig(run/"guidance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--goal-cell", type=int, nargs=2)
    parser.add_argument("--start-cell", type=int, nargs=2)
    args = parser.parse_args()
    export(args.layout, args.goal_cell, args.start_cell)
