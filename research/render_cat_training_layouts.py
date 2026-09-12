#!/usr/bin/env python3
"""CPU-only movies of existing generated CAT training fields, not policy rollouts.

Reads and verifies the cached bank; never regenerates or changes training scenes.
No simulator, torch, CUDA, ROS, robot connection, or GPU encoder is imported.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from cat_scenes import occupancy_mesh, sha256
from demo_video import inspect_video


FAMILIES = ("lateral", "low", "overhead", "mixed")
OFFSETS = (1, 18, 47)
START = np.array([0., 0., .75])
COLORS = dict(start="#26d69c", goal="#ff697d")


def select_scenes(bank):
    """Fixed seed offsets sample difficulty .4/.6/.8; no visual cherry-picking."""
    if bank.get("schema") != "cat-generated-parallel-bank-v1" or not bank.get("complete"):
        raise ValueError("A complete generated parallel bank is required")
    groups = {}
    for family in FAMILIES:
        rows = sorted((r for r in bank["scenes"] if r["family"] == family
                       and r["split"] == "train"), key=lambda r: r["seed"])
        # Use requested recipes, not first accepted seed: low seed zero is rejected.
        base = min(r["seed"] for r in bank["recipes"]
                   if r["family"] == family and r["split"] == "train")
        lookup = {r["seed"]: r for r in rows}
        groups[family] = [lookup[base+i] for i in OFFSETS]
    return groups


def load_scene(bank_path, row):
    path = (bank_path.parent / row["file"]).resolve()
    if path.parent != bank_path.parent.resolve() or sha256(path) != row["sha256"]:
        raise ValueError("Scene path/checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        obs = arrays["obs"]
    if (obs.dtype != np.bool_ or list(obs.shape) != row["shape"] or not obs.any()
            or hashlib.sha256(obs.tobytes()).hexdigest() != row["occupancy_sha256"]):
        raise ValueError("Occupancy shape/type/checksum mismatch")
    return obs


def figure_for_scene(row, obs, example):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#101722")
    fig.suptitle(f"CAT generated training layouts  /  {row['family'].upper()}",
                 fontsize=21, x=.045, ha="left", y=.965)
    fig.text(.045, .895,
             f"Example {example}/3   |   seed {row['seed']}   |   difficulty {row['difficulty']:.1f}"
             f"   |   source grid: 4 cm", color="#c5cedb", fontsize=12)
    fig.text(.045, .042, "GEOMETRY PREVIEW ONLY — no robot rollout or training result", fontsize=12)
    fig.text(.045, .015, "Flat-floor stage. Colored volumes = CAT obstacle fields; no added supports or invented route.",
             color="#aeb8c6", fontsize=10)
    ax = fig.add_axes([.01, .13, .67, .72], projection="3d", facecolor="#101722")
    top = fig.add_axes([.73, .55, .22, .28], facecolor="#182536")
    side = fig.add_axes([.73, .17, .22, .27], facecolor="#182536")
    origin = np.array(row["origin"])
    upper = origin + np.array(obs.shape)*row["resolution"]
    goal = np.array(row["goal"])
    vertices, faces = occupancy_mesh(obs, row["resolution"], origin)
    triangles = vertices[faces]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    light = np.array([-.4, -.5, 1.]); light /= np.linalg.norm(light)
    cmap, norm = plt.get_cmap("turbo"), Normalize(0, upper[2])
    colors = cmap(norm(triangles[..., 2].mean(axis=1)))
    colors[:, :3] *= (.6+.4*np.maximum(normal @ light, 0))[:, None]
    mesh = Poly3DCollection(triangles, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(mesh)
    # Only a ground grid: do not turn native floating forbidden volumes into furniture.
    for x in np.arange(origin[0], upper[0]+.01, .5):
        ax.plot([x, x], [origin[1], upper[1]], [0, 0], c="#536072", lw=.6)
    for y in np.arange(origin[1], upper[1]+.01, .5):
        ax.plot([origin[0], upper[0]], [y, y], [0, 0], c="#536072", lw=.6)
    for point, label, marker in ((START, "start", "o"), (goal, "goal", "*")):
        ax.plot([point[0]]*2, [point[1]]*2, [0, point[2]], "--", c=COLORS[label], lw=1.2)
        ax.scatter(*point, s=90, marker=marker, c=COLORS[label], depthshade=False)
        ax.text(point[0], point[1], point[2]+.1, label, color=COLORS[label], fontsize=11)
        top.scatter(point[0], point[1], s=55, marker=marker, c=COLORS[label], edgecolors="black")
        side.scatter(point[0], point[2], s=55, marker=marker, c=COLORS[label], edgecolors="black")
    ax.set(xlim=(origin[0]-.1, upper[0]+.1), ylim=(origin[1]-.1, upper[1]+.1),
           zlim=(0, upper[2]+.1), xlabel="X forward (m)", ylabel="Y (m)", zlabel="Z up (m)")
    ax.set_box_aspect((3.2, 2.2, upper[2]+.1))
    ax.tick_params(labelsize=9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#536072")
        axis._axinfo["grid"]["color"] = "#344355"
    z = origin[2] + (np.arange(obs.shape[2])+.5)*row["resolution"]
    highest = np.where(obs, z[None, None, :], -np.inf).max(axis=2)
    highest = np.ma.masked_where(~obs.any(axis=2), highest)
    top.imshow(highest.T, origin="lower", extent=[origin[0], upper[0], origin[1], upper[1]],
               interpolation="nearest", cmap=cmap, norm=norm)
    projected = np.broadcast_to(z, (obs.shape[0], obs.shape[2]))
    projected = np.ma.masked_where(~obs.any(axis=1), projected)
    side.imshow(projected.T, origin="lower", extent=[origin[0], upper[0], origin[2], upper[2]],
                interpolation="nearest", cmap=cmap, norm=norm)
    top.set(title="Top view: highest occupied Z", ylabel="Y (m)")
    side.set(title="Side projection: all Y combined", xlabel="X forward (m)", ylabel="Z (m)")
    for panel in (top, side):
        panel.tick_params(labelsize=8)
        panel.title.set_fontsize(10)
    colorbar_ax = fig.add_axes([.975, .19, .008, .6])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax)
    colorbar_ax.tick_params(labelsize=7)
    colorbar_ax.set_title("Z / m", fontsize=8)
    return fig, ax


def render_family(job):
    bank_path, output, family, rows, fps, seconds = job
    path = output / f"{FAMILIES.index(family)+1:02d}_{family}.mp4"
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=1,
                            ffmpeg_params=["-threads", "2"]) as writer:
        for example, row in enumerate(rows, 1):
            obs = load_scene(bank_path, row)
            fig, ax = figure_for_scene(row, obs, example)
            for frame_idx in range(round(seconds*fps)):
                fraction = frame_idx/max(1, round(seconds*fps)-1)
                ax.view_init(elev=22+12*np.sin(np.pi*fraction), azim=-120+100*fraction)
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                writer.append_data(frame)
                if frame_idx == round(seconds*fps)//2:
                    imageio.imwrite(output/f"{family}_{row['seed']}.png", frame)
            plt.close(fig)
            print(f"Rendered {family} {example}/3: seed {row['seed']}", flush=True)
    video = inspect_video(path)
    video["file"] = path.name
    video["scenes"] = rows
    return video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds-per-layout", type=float, default=4.)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    if args.fps < 1 or args.seconds_per_layout*args.fps < 25:
        parser.error("At least 25 frames per layout required")
    bank_path, output = args.bank.resolve(), args.output.resolve()
    bank = json.loads(bank_path.read_text())
    groups = select_scenes(bank)
    for rows in groups.values():
        for row in rows:
            load_scene(bank_path, row)
    output.mkdir(parents=True, exist_ok=False)
    jobs = [(bank_path, output, family, rows, args.fps, args.seconds_per_layout)
            for family, rows in groups.items()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        videos = list(pool.map(render_family, jobs))
    report = dict(schema="cat-training-layout-videos-v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  bank=str(bank_path), bank_sha256=sha256(bank_path), renderer_sha256=sha256(Path(__file__)),
                  source=bank["source"], seed_offsets=OFFSETS, videos=videos,
                  rendering="CPU Matplotlib Agg + libx264", simulation_started=False,
                  policy_rollout=False, obstacle_semantics=bank["obstacle_semantics"])
    (output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"README.txt").write_text(
        "CAT GENERATED TRAINING LAYOUT EXAMPLES\n\n"
        "Four 12-second videos at default settings: lateral, low, overhead, mixed.\n"
        "Each contains three distinct cached training layouts, difficulty 0.4 / 0.6 / 0.8.\n"
        "Green circle = nominal start (0,0,0.75 m); red star = goal (2,0,0.75 m).\n"
        "Z points UP. Colors show obstacle height; dimensions are meters.\n"
        "Right panels show occupied projections, NOT a traversable route or costmap.\n\n"
        "These are CPU geometry previews, NOT Isaac recordings or learned-policy rollouts.\n"
        "They show the actual accepted CAT-generated grids used by the new training setup.\n"
        "Clutter is an SDF forbidden field over a physical flat floor in this training stage;\n"
        "solid rendering is for inspection, not a claim of physical obstacle contacts.\n"
        "Floating native-generator volumes are preserved; no supports were invented.\n"
        "The complete bank contains 180 unique training and 45 held-out layouts.\n"
        "Example selection is fixed by seed offsets 1, 18, 47, not cherry-picked by appearance.\n"
        "No robot, GPU simulator, training process, or live controller was started/changed.\n"
        "manifest.json records source and individual scene/video checksums.\n")
    print(f"Verified all videos by decoding every frame: {output}", flush=True)


if __name__ == "__main__":
    main()
