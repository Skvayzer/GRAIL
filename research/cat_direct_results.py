#!/usr/bin/env python3
"""Audit actual direct CAT rollouts, field/mesh transfer, and plot comparison.

Native threshold acceptance is NOT full-body physical collision certification.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

from cat_direct_native import CAT, SOURCES, digest, make_player
from gear_sonic.research.cat_bridge import CatFieldSampler, SITES


def inertia_error(dynamics):
    """PhysX get_inertias is COM-centered but expressed in link axes."""
    rotation = Rotation.from_quat(dynamics["native_inertia_quat"], scalar_first=True).as_matrix()
    native = np.array([np.diag(v) for v in dynamics["native_diaginertia"]])
    imported = np.array(dynamics["imported_inertias"]).reshape(-1, 3, 3)
    return float(np.max(np.abs(imported-rotation@native@rotation.transpose(0, 2, 1))))


def audit(run):
    reports, traces = [], {}
    for scene in ("side1", "hurdle1", "crouch1"):
        player, _, contract = make_player(scene)
        directory = CAT/"data/assets/TypiObs"/scene
        fields = {k: torch.as_tensor(getattr(player, k), dtype=torch.float32) for k in ("gf", "bf", "sdf")}
        sampler = CatFieldSampler(fields, player.pf_origin, player.dx)
        original = trimesh.load(directory/"obs.obj", force="mesh", process=False)
        stage = Usd.Stage.Open(str(run/f"{scene}_imported.usda"))
        shape = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Clutter/Obstacles"))
        matrix = np.array(UsdGeom.Xformable(shape).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        vertices = np.asarray(shape.GetPointsAttr().Get())
        world = np.c_[vertices, np.ones(len(vertices))]@matrix
        expected = original.vertices+player.mj_model.body("scene").pos
        mesh_error = float(np.max(np.abs(world[:, :3]-expected)))
        np.testing.assert_allclose(world[:, :3], expected, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(shape.GetFaceVertexIndicesAttr().Get(), original.faces.ravel())
        scene_report = dict(scene=scene, usd_world_mesh_error_m=mesh_error,
            source_field_origin=player.pf_origin.tolist(), source_resolution=player.dx,
            source_files={n: digest(directory/n) for n in ("obs.obj", "obs.npy", "sdf.npy", "gf.npy", "bf.npy")},
            native_mesh_closed=bool(original.is_watertight), episodes={})
        for engine in ("mujoco", "isaac"):
            meta = json.loads((run/f"{scene}_{engine}.json").read_text())
            path = run/meta["archive"]
            if digest(path) != meta["sha256"] or meta["grail_loaded"] or meta["optimizer_steps"] or meta["robot_actuation"]:
                raise ValueError("Changed or invalid direct-run evidence")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Nonfinite direct-run data")
            error, outside = 0., 0
            for q in arrays["qpos"][::max(1, len(arrays["qpos"])//30)]:
                player.mj_data.qpos[:] = q
                mujoco.mj_forward(player.mj_model, player.mj_data)
                points = np.stack([player.mj_data.site(n).xpos for n in SITES])
                actual = sampler.sample(torch.as_tensor(points, dtype=torch.float32))
                outside += int((~actual["in_domain"]).sum())
                for key in ("gf", "bf", "sdf"):
                    native = player.sample_field(getattr(player, key), points)
                    error = max(error, float(np.max(np.abs(native-actual[key].numpy()))))
                    np.testing.assert_allclose(actual[key], native, atol=1e-5, rtol=1e-5)
            root = arrays["qpos"][:, :3]
            goal = np.array([2., 0.])
            goal_distance = float(np.linalg.norm(root[-1, :2]-goal))
            scene_report["episodes"][engine] = dict(**meta,
                field_port_max_error=error, outside_sampled_site_queries=outside,
                final_distance_to_native_goal_xy_m=goal_distance,
                reached_native_goal_radius_0_2m=goal_distance < .2,
                native_threshold_acceptance=meta["native_clearance_or_height_violation_steps"] == 0)
            traces[scene, engine] = arrays
        dynamics_path = run/f"{scene}_dynamics.json"
        if dynamics_path.exists():
            dynamics = json.loads(dynamics_path.read_text())
            scene_report["dynamics"] = {k: dynamics[k] for k in ("mass_error_kg", "com_position_error_m", "joint_limit_error_rad")}
            scene_report["dynamics"]["inertia_tensor_error_kgm2"] = inertia_error(dynamics)
            if inertia_error(dynamics) > 1e-5:
                raise ValueError("Native/imported body inertia mismatch")
        else:
            scene_report["dynamics"] = "Same imported robot; checked on hurdle1/crouch1 runs"
        reports.append(scene_report)
    report = dict(schema="cat-direct-sim2sim-v1", geometry_and_field_transfer_passed=True,
        all_isaac_native_clearance_checks_passed=all(r["episodes"]["isaac"]["native_threshold_acceptance"] for r in reports),
        cases=reports, source_code_hashes=SOURCES, original_policy_weights_unchanged=True,
        training_started=False, robot_actuation=False, full_body_physical_collision_certified=False,
        scope="Three deterministic native-player fixtures, native SDF-only clutter semantics; NOT full distribution/real sensor validation")
    with (run/"comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    plot(run, traces)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases", "source_code_hashes")}, indent=2))
    return report


def plot(run, traces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    for row, scene in enumerate(("side1", "hurdle1", "crouch1")):
        for engine, color in (("mujoco", "#2471a3"), ("isaac", "#d35400")):
            arrays = traces[scene, engine]
            t = (np.arange(len(arrays["qpos"]))+1)*.02
            axes[row, 0].plot(arrays["qpos"][:, 0], arrays["qpos"][:, 1], color=color, label=engine)
            axes[row, 1].plot(t, arrays["head"][:, 2], color=color, label=engine)
            axes[row, 2].plot(t, arrays["clearance"].min(-1), color=color, label=engine)
        axes[row, 0].set(title=scene+" — root trajectory", xlabel="x (m)", ylabel="y (m)", xlim=(-.1, 2.1))
        axes[row, 0].add_patch(plt.Circle((2, 0), .2, fill=False, color="green", label="goal vicinity"))
        axes[row, 1].set(title="Head height", xlabel="time (s)", ylabel="m")
        axes[row, 2].set(title="Native head/feet/hand site SDF", xlabel="time (s)", ylabel="m")
        axes[row, 2].axhline(-.04, color="red", linestyle="--", label="native violation threshold")
        for ax in axes[row]:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Frozen CAT: actual MuJoCo vs Isaac trajectories — no GRAIL, no training\nSDF-only clutter reproduction; not full-body physical collision certification")
    fig.savefig(run/"comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run)
