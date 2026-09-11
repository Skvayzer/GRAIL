#!/usr/bin/env python3
"""Render checked frozen-rollout data into laptop-friendly, labeled MP4 demos.

No simulator, policy, optimizer, ROS or service starts here. Geometry/observations
are recorded oracle diagnostics, not sensor data or learned navigation output.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from cat_scenes import sha256
from demo_video import inspect_video
from observation_results import audit_observation_shadow
from terrain_guidance import read_layout
from gear_sonic.research.scene_audit import read_snapshot
from gear_sonic.research.cat_geometry import verify_scene_files


def load_sources(run, comparison):
    audit_observation_shadow(run)
    meta = json.loads((run/"observation_shadow.json").read_text())
    layout, arrays = read_layout(run/"layout_audit.json")
    _, _, terrain = read_snapshot(run)
    cat = json.loads((run/"cat_audit.json").read_text())
    source = verify_scene_files(cat["source_scene"])
    if (meta["cat_files"] != source["files"] or meta["layout_sha256"] != sha256(run/"layout_audit.json")
            or meta["terrain_sha256"] != terrain["sha256"] or meta["grid_sha256"] != layout["grid_sha256"]
            or cat["placement"] != layout["placement"]):
        raise ValueError("Recorded geometry provenance mismatch")
    with np.load(run/"observation_shadow.npz", allow_pickle=False) as data:
        saved = {k: data[k] for k in data.files}
    report = json.loads((comparison/"audit.json").read_text())
    if not report["passed"] or report["packet_sha256"] != sha256(comparison/"guidance.npz"):
        raise ValueError("Guidance comparison is not intact/passed")
    # The comparison uses the pre-fix run. Its grid and moving pelvis must be
    # identical to this post-fix frozen rollout; never align unrelated runs.
    prior = json.loads((Path(report["source_run"])/"observation_shadow.json").read_text())
    if (prior["grid_sha256"] != meta["grid_sha256"] or prior["cat_files"] != meta["cat_files"]
            or prior["terrain_sha256"] != meta["terrain_sha256"]
            or report["source_report_sha256"] != sha256(Path(report["source_run"])/"observation_shadow.json")):
        raise ValueError("Guidance comparison describes different source geometry")
    with np.load(comparison/"guidance.npz", allow_pickle=False) as data:
        compared = {k: data[k] for k in data.files}
    if (saved["root"].shape != compared["root"].shape
            or not np.allclose(saved["root"], compared["root"], rtol=0, atol=2e-5)
            or not np.allclose(saved["guidance"], compared["features"], rtol=0, atol=2e-5)
            or not np.array_equal(saved["valid"], compared["valid"])):
        raise ValueError("Cannot overlay a comparison from a different trajectory")
    return meta, layout, arrays, saved, compared, report


def write_frames(path, fig, update, count, fps=25):
    # Simulation has 50 policy steps per second. Sample every second step and
    # include the terminal sample; render at 25 fps (approximately real time).
    indices = sorted(set(range(0, count, 2)) | {count-1})
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1) as writer:
        for n, i in enumerate(indices):
            update(i)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
            if n == 0:
                imageio.imwrite(path.with_suffix(".png"), frame)
                for _ in range(fps):
                    writer.append_data(frame)
        for _ in range(fps):
            writer.append_data(frame)
    plt.close(fig)
    return inspect_video(path)


def guidance_video(output, layout, arrays, saved, compared):
    xy = arrays["xy"]
    half = layout["support_graph"]["limits"]["resolution"]/2
    extent = [xy[0, 0, 0]-half, xy[-1, -1, 0]+half, xy[0, 0, 1]-half, xy[-1, -1, 1]+half]
    classes = np.where(arrays["walkable"], 2, np.where(arrays["transit_valid"], 1, 0))
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], left=.07, right=.97, top=.85, bottom=.13,
                         wspace=.27, hspace=.55)
    title = fig.suptitle("Guidance correction | recorded frozen-controller trajectory", fontsize=18, y=.98)
    fig.text(.5, .9, "Green: support nodes   |   amber: clear transit   |   dark: blocked/invalid", ha="center", fontsize=11)
    axes = [fig.add_subplot(gs[0, k]) for k in range(2)]
    points, paths, status = [], [], []
    for ax, name in zip(axes, ["Before: nearest support cell", "After: checked transit connection"]):
        ax.imshow(classes.T, origin="lower", extent=extent,
                  cmap=ListedColormap(["#252b36", "#876537", "#28554b"]), vmin=0, vmax=2)
        route = arrays["route"]
        ax.plot(route[:, 0], route[:, 1], "--", c="#a6b0c6", lw=1, label="Geometric graph route")
        ax.scatter(route[-1, 0], route[-1, 1], marker="*", s=130, c="#ffb347", label="Graph goal")
        points.append(ax.plot([], [], "o", c="#60dfff", ms=7)[0])
        paths.append(ax.plot([], [], c="#60dfff", lw=1.5)[0])
        status.append(ax.text(.02, .97, "", transform=ax.transAxes, va="top", fontsize=11,
                              bbox=dict(facecolor="#111827", alpha=.9, edgecolor="none")))
        ax.set(title=name, xlabel="World X (m)", ylabel="World Y (m)")
        ax.legend(loc="lower right", fontsize=8)
    connector = axes[1].plot([], [], c="#ffffff", lw=3)[0]
    target = axes[1].plot([], [], "s", c="#ffffff", ms=5)[0]
    timeline = fig.add_subplot(gs[1, :])
    times = np.arange(len(saved["root"]))*0.02
    timeline.plot(times, compared["old_valid"], color="#ffac69", lw=1, label="Before")
    timeline.plot(times, compared["valid"].astype(float)+.08, color="#57e4b5", lw=2, label="After (+0.08 display offset)")
    cursor = timeline.axvline(0, color="white", lw=1)
    timeline.set(xlabel="Simulation time (s)", ylabel="Valid", ylim=(-.15, 1.25), yticks=[0, 1])
    timeline.legend(loc="lower right", fontsize=8)
    fig.text(.5, .025, "No new control actions. Graph guidance is a geometric screen, not a balance or full-body feasibility certificate.",
             ha="center", fontsize=10)

    def update(i):
        root = saved["root"][i]
        for k in range(2):
            points[k].set_data([root[0]], [root[1]])
            paths[k].set_data(saved["root"][:i+1, 0], saved["root"][:i+1, 1])
        cell = tuple(compared["target_cell"][i].astype(int))
        valid = bool(compared["valid"][i])
        if valid:
            end = xy[cell]
            connector.set_data([root[0], end[0]], [root[1], end[1]])
            target.set_data([end[0]], [end[1]])
        else:
            connector.set_data([], [])
            target.set_data([], [])
        status[0].set_text("VALID" if compared["old_valid"][i] else "INVALID at moving pelvis")
        status[0].set_color("#57e4b5" if compared["old_valid"][i] else "#ff7070")
        status[1].set_text(f"{'VALID' if valid else 'INVALID'} | {compared['connector_xy'][i]*100:.1f} cm connector")
        status[1].set_color("#57e4b5" if valid else "#ff7070")
        cursor.set_xdata([i*.02, i*.02])
        title.set_text(f"Guidance correction | simulation t = {i*.02:.2f} s")
    return write_frames(output/"02_guidance_before_after.mp4", fig, update, len(saved["root"]))


def observation_video(output, meta, saved):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 3, left=.06, right=.98, top=.84, bottom=.13, wspace=.52, hspace=.68,
                         width_ratios=[1, 1, 1.2])
    title = fig.suptitle("Inputs available to the whole-body obstacle adapter", fontsize=18, y=.97)
    fig.text(.5, .905, "Recorded mesh oracle — NOT LiDAR. The adapter is shadow-only; residual = 0.", ha="center", fontsize=12)
    spacing, scale = meta["packet_spec"]["spacing"], meta["packet_spec"]["distance_scale"]
    shape = saved["volume"].shape[2:]
    half = spacing*np.array(shape)/2
    extent = [-half[0], half[0], -half[1], half[1]]
    images = []
    for k, name in enumerate(["CAT clutter: SIGNED distance", "Terrain: UNSIGNED distance"]):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(saved["volume"][0, k, :, :, shape[2]//2].T*scale,
                       origin="lower", extent=extent, cmap="coolwarm_r" if k == 0 else "viridis",
                       vmin=-.3 if k == 0 else 0., vmax=1.5)
        images.append(im)
        ax.plot(0, 0, "+", color="white", ms=10)
        ax.set(title=name, xlabel="Local forward X (m)", ylabel="Local left Y (m)")
        fig.colorbar(im, ax=ax, label="Distance (m)", shrink=.7)
    body = fig.add_subplot(gs[:, 2], projection="3d")
    body.set(xlim=(-.65, .65), ylim=(-.65, .65), zlim=(-1., .7), xlabel="Local X (m)", ylabel="Local Y (m)",
             zlabel="Pelvis-relative Z (m)", title="104 articulated collider-cover centres")
    body.set_box_aspect((1.3, 1.3, 1.7))
    dots = body.scatter([], [], [], s=18, c="#60dfff")
    body.view_init(elev=15, azim=-55)
    ax = fig.add_subplot(gs[1, :2])
    unique = list(dict.fromkeys(p["link"] for p in meta["probe_order"]))
    groups = [[i for i, p in enumerate(meta["probe_order"]) if p["link"] == link] for link in unique]
    bars = ax.bar(np.arange(len(unique)), np.zeros(len(unique)), color="#57d0d9")
    ax.set_xticks(np.arange(len(unique)), [s.replace("_link", "").replace("left_", "L ").replace("right_", "R ").replace("_", " ")
                  for s in unique], rotation=40, ha="right", fontsize=7)
    ax.set(title="Per-body CAT gap: mesh distance minus cover radius (clipped at 2 m)", ylabel="Gap (m)", ylim=(-.1, 2.1))
    ax.axhline(0, c="#ff7070", lw=1)
    fig.text(.5, .025, "13 x 13 x 11 volume at 16 cm | horizontal slice through pelvis | world Z up | support is separate from clutter",
             ha="center", fontsize=10)

    def update(i):
        title.set_text(f"Whole-body obstacle inputs | simulation t = {i*.02:.2f} s | packet {'VALID' if saved['valid'][i] else 'INVALID'}")
        for k, im in enumerate(images):
            distance = saved["volume"][i, k, :, :, shape[2]//2].T*scale
            valid = saved["volume"][i, k+2, :, :, shape[2]//2].T == 1
            im.set_data(np.ma.masked_where(~valid, distance))
        centers = saved["probes"][i, :, :3]*scale
        dots._offsets3d = tuple(centers.T)
        for bar, group in zip(bars, groups):
            gap = float(saved["probes"][i, group, 4].min()*scale)
            bar.set_height(gap)
            bar.set_color("#ff7070" if gap < 0 else "#57d0d9")
    return write_frames(output/"03_whole_body_observations.mp4", fig, update, len(saved["root"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--guidance-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite existing demos")
    args = parser.parse_args()
    run, comparison, output = args.run.resolve(), args.guidance_comparison.resolve(), args.output.resolve()
    meta, layout, arrays, saved, compared, audit = load_sources(run, comparison)
    output.mkdir(parents=True, exist_ok=False)
    plt.style.use("dark_background")
    videos = [guidance_video(output, layout, arrays, saved, compared), observation_video(output, meta, saved)]
    report = dict(schema="grail-cat-review-videos-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(run), source_guidance_comparison=str(comparison),
        source_report_sha256=sha256(run/"observation_shadow.json"), source_packet_sha256=meta["packet_sha256"],
        source_comparison_sha256=sha256(comparison/"audit.json"), simulation_started=False, training_started=False,
        control_source="recorded frozen GRAIL actor; adapter never applied", videos=videos,
        guidance_frames=dict(before=audit["old_valid"], after=audit["new_valid"], total=audit["frames"]),
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)))
    (output/"diagnostic_videos.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
