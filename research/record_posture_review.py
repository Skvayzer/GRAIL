#!/usr/bin/env python3
"""Render verified kinematic examples, never a simulator or learned policy.

Actual exported terrain/CAT meshes and imported sphere/capsule dimensions are
shown. Capsule surfaces are tessellated for display; clearance uses the original
signed-mesh query and conservative body cover, not the display tessellation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from cat_scenes import sha256
from reference_postures import reconstruct
from record_review_demos import write_frames
from gear_sonic.research.cat_geometry import Placement, cat_mesh_arrays
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.scene_audit import read_snapshot


def capsule_surface(start, end, radius):
    """Low-poly display of an actual capsule, endpoints are sphere centres."""
    direction = end-start
    length = np.linalg.norm(direction)
    axis = direction/length if length > 1e-8 else np.array([0., 0., 1.])
    helper = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.arange(8)*2*np.pi/8
    ring_direction = np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v
    rings = []
    for centre, angles in ((start, (-np.pi/2, -np.pi/4, 0.)), (end, (0., np.pi/4, np.pi/2))):
        for angle in angles:
            rings.append(centre+radius*(np.sin(angle)*axis+np.cos(angle)*ring_direction))
    return np.array([[rings[j][k], rings[j][(k+1) % 8], rings[j+1][(k+1) % 8], rings[j+1][k]]
                     for j in range(5) for k in range(8)])


def body_surface(centres, probes, inventory):
    faces, colors = [], []
    for index, item in enumerate(inventory):
        group = [i for i, p in enumerate(probes) if p.collision_index == index]
        if not group:
            raise ValueError("Imported primitive has no matching recorded cover")
        polygon = capsule_surface(centres[group[0]], centres[group[-1]], item["radius"])
        arm = any(word in item["body"] for word in ("shoulder", "elbow", "wrist"))
        faces.extend(polygon)
        colors.extend(["#5ce1e6" if arm else "#dae3ef"]*len(polygon))
    return faces, colors


def render(output, name, data, saved, terrain, clutter, gaps, start=0, close=False):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1], left=.055, right=.98, bottom=.15, top=.84,
                         wspace=.08, hspace=.2)
    title = fig.suptitle("CAT + stairs | KINEMATIC EXAMPLE, NOT A POLICY ROLLOUT", y=.98, fontsize=16)
    fig.text(.5, .91, "Same root, waist and legs. Only arm joint references change. Balance is NOT verified.",
             ha="center", fontsize=11)
    root = data["anchors"]
    artists = []
    for k, label in enumerate(("Original reference: arm conflict", "Collision-screened arm posture")):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        ax.add_collection3d(Poly3DCollection(terrain, facecolor="#ae785b", edgecolor="#745544", linewidths=.1))
        ax.add_collection3d(Poly3DCollection(clutter, facecolor="#dd9d36", edgecolor="#b98539", linewidths=.15, alpha=.75))
        faces, colors = body_surface(saved[k][start], data["probes"], data["inventory"])
        body = Poly3DCollection(faces, facecolors=colors, edgecolors="none")
        ax.add_collection3d(body)
        artists.append(body)
        # Draw the trajectory at pelvis height, not a claimed foot-support route.
        ax.plot(root[:, 0], root[:, 1], root[:, 2], "--", lw=1, color="#edfafc", alpha=.5)
        ax.set(title=label, xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        if close:
            ax.set(xlim=(-.8, .8), ylim=(.25, 2.15), zlim=(0, 1.9))
            ax.set_box_aspect((1.6, 1.9, 1.9))
        else:
            ax.set(xlim=(-1.3, 1.3), ylim=(-1.2, 2.3), zlim=(0, 2.9))
            ax.set_box_aspect((2.6, 3.5, 2.9))
        ax.view_init(elev=23, azim=65)
        ax.tick_params(labelsize=7)
    graph = fig.add_subplot(gs[1, :])
    times = np.arange(len(root))*.02
    graph.plot(times, gaps[0], color="#ff7474", label="Original reference")
    graph.plot(times, gaps[1], color="#5ce1e6", label="Changed arm posture")
    graph.axhline(0, color="white", lw=.8)
    graph.axhline(.03, color="#6bd8a5", lw=.8, linestyle=":", label="3 cm geometric screen")
    cursor = graph.axvline(times[start], color="white", lw=1)
    graph.set(xlabel="Reference time (s)", ylabel="CAT cover gap (m)", ylim=(-.1, .4),
              xlim=(times[start], times[-1]))
    graph.legend(loc="upper right", fontsize=8)
    fig.text(.5, .025, "Offline mesh/FK render. Arms differ from spawn; no new policy, physics rollout or training. Dimensions in metres.",
             ha="center", fontsize=10)

    def update(index):
        i = start+index
        for k, artist in enumerate(artists):
            faces, _ = body_surface(saved[k][i], data["probes"], data["inventory"])
            artist.set_verts(faces)
        cursor.set_xdata([times[i], times[i]])
        title.set_text(f"CAT + stairs | KINEMATIC, NOT LEARNED | t = {times[i]:.2f} s")
    update(int(np.argmin(gaps[0][start:])))
    fig.savefig(output/(name+"_conflict.png"), dpi=100)
    # Five diagnostic frames per reference second; duration remains real-time.
    return write_frames(output/(name+".mp4"), fig, update, len(root)-start, fps=5, step=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness-run", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.witness_run.resolve()
    report = json.loads((run/"witness.json").read_text())
    item = next(i for i in report["witnesses"] if i["case"] == args.case)
    path = run/item["file"]
    if (path.parent != run or sha256(path) != item["sha256"]
            or not report["cases"][args.case]["kinematic_screen_passed"]):
        raise ValueError("Intact accepted kinematic example required")
    data = reconstruct(report["reference"]["reference_run"])
    if data["provenance"] != report["reference"]:
        raise ValueError("Reference provenance changed")
    scene = Path(report["scene"])
    if sha256(scene/"scene.json") != report["scene_sha256"]:
        raise ValueError("Scene metadata changed")
    cv, cf = cat_mesh_arrays(scene)
    tv, tf, _ = read_snapshot(report["reference"]["reference_run"])
    placement = Placement(**report["placement"])
    with np.load(path, allow_pickle=False) as loaded:
        witness = loaded["centers"].copy()
        radii = loaded["radii"].copy()
    original = data["centers"].numpy()
    if (witness.shape != original.shape or not np.isfinite(witness).all()
            or not np.array_equal(radii, data["radii"].numpy())):
        raise ValueError("Kinematic packet shape/radii mismatch")
    mesh = ClosedMeshDistance(cv, cf, device="cuda:0")
    gaps = []
    for centres in (original, witness):
        d, _, valid = mesh.query(placement.to_local(torch.tensor(centres, device="cuda:0")))
        if not valid.all():
            raise ValueError("Unknown CAT gap cannot enter review video")
        gaps.append((d-torch.tensor(radii, device="cuda:0")).min(-1).values.cpu().numpy())
    if not np.isclose(gaps[1].min(), report["cases"][args.case]["outer_clutter_gap_m"], atol=1e-6, rtol=0):
        raise ValueError("Recorded witness no longer reproduces clearance")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.style.use("dark_background")
    terrain = tv[tf]
    clutter = placement.to_world(torch.tensor(cv)).numpy()[cf]
    videos = [render(output, "01_stairs_and_clutter", data, (original, witness), terrain, clutter, gaps),
              render(output, "02_arm_clearance_closeup", data, (original, witness), terrain, clutter, gaps,
                     start=max(0, len(original)-200), close=True)]
    metadata = dict(schema="grail-cat-kinematic-review-v1", created_utc=datetime.now(timezone.utc).isoformat(),
        witness_run=str(run), witness_report_sha256=sha256(run/"witness.json"), witness=item,
        scene_sha256=report["scene_sha256"], placement=report["placement"], reference=report["reference"],
        case_metrics=report["cases"][args.case], videos=videos, training_started=False,
        simulator_started=False, dynamic_feasibility_verified=False, initial_arms_changed=report["initial_arm_pose_changed"])
    (output/"provenance.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(output), "videos": videos, "training_started": False}), flush=True)


if __name__ == "__main__":
    main()
