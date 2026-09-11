#!/usr/bin/env python3
"""Bounded desktop PhysX contact/field fixtures. No policy and no robot APIs.

Requires the same explicit EULA acceptance as baseline.py. Outputs a new report
under research/runs; isolated temporary logs, no GUI or network services.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-isaac-eula", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("Prepared code only. Use --execute --accept-isaac-eula for desktop physics fixtures.")
        return
    if not args.accept_isaac_eula:
        parser.error("Explicit --accept-isaac-eula is required for this fixture runner")
    root = Path(__file__).resolve().parent
    run = root / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_contact_fixtures")
    run.mkdir(parents=True, exist_ok=False)
    temporary = tempfile.mkdtemp(prefix="grail-contact-", dir="/tmp")
    (run/"tmp").symlink_to(temporary, target_is_directory=True)
    os.environ["TMPDIR"] = temporary
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "Yes"
    # Kit must not see this script's CLI flags.
    import sys
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    # Like upstream's evaluator, exit this single-purpose process after flushing
    # output. Kit 5.1's full extension teardown can spin after a completed test.
    # There are no writers/render jobs here; report is closed before this exit.
    try:
        run_fixtures(run)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_fixtures(run):
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from gear_sonic.research.geometry import ContactPermission, FIELD_VERSION
    from gear_sonic.research.isaac_clutter import solid_cfg
    from gear_sonic.research.scenes import contact_fixture_solids

    dt, radius = .005, .05
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0", gravity=(0., 0., 0.)))
    solids = {s.name: s for s in contact_fixture_solids()}
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.)
    for solid in solids.values():
        setattr(cfg, solid.name, solid_cfg(solid, "{ENV_REGEX_NS}/"+solid.name))
    cfg.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/probe",
        spawn=sim_utils.SphereCfg(radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 2.)),
    )
    cfg.contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/probe", filter_prim_paths_expr=["{ENV_REGEX_NS}/"+s for s in solids],
        update_period=0., history_length=1, track_contact_points=True, max_contact_data_count_per_prim=64,
    )
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.reset()
    # solid, approach center, commanded velocity, representative link/phase, expected permission
    fixtures = [
        ("foot_tread", "tread", (0., 0., .65), (0., 0., -.2), "foot", "stance", True),
        ("shin_riser", "tread", (.65, 0., .25), (-.2, 0., 0.), "shin", "stance", False),
        ("swing_foot_riser", "tread", (.65, 0., .25), (-.2, 0., 0.), "foot", "swing", False),
        ("foot_edge", "tread", (.48, 0., .65), (0., 0., -.2), "foot", "stance", False),
        ("hand_rail", "rail", (3.2, 0., .7), (-.2, 0., 0.), "hand", "transit", False),
        ("head_beam", "beam", (6., 0., .75), (0., 0., .2), "head", "transit", False),
        ("hand_pole", "pole", (9.25, 0., .5), (-.2, 0., 0.), "hand", "transit", False),
    ]
    permission = ContactPermission("tread", ("foot",), ("stance",), (-.45, -.45, .249), (.45, .45, .251))
    results = []
    probe, contacts = scene["probe"], scene["contacts"]
    for name, solid_name, start, velocity, link, phase, expected_allowed in fixtures:
        state = probe.data.default_root_state.clone()
        state[:, :3] = state.new_tensor(start)
        state[:, 7:] = 0
        probe.write_root_pose_to_sim(state[:, :7])
        probe.write_root_velocity_to_sim(state[:, 7:])
        scene.reset()
        forces, first_hit, worst_overlap = [], None, 0.
        target_index = list(solids).index(solid_name)
        solid = solids[solid_name]
        for step in range(180):
            gap_before = float(solid.distance_normal(probe.data.root_pos_w)[0][0]-radius)
            v = state.new_zeros((1, 6))
            v[:, :3] = v.new_tensor(velocity)
            probe.write_root_velocity_to_sim(v)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            force = contacts.data.force_matrix_w[0, 0, target_index]
            magnitude = float(torch.linalg.vector_norm(force))
            if not torch.isfinite(force).all():
                raise ValueError("Nonfinite measured contact")
            gap_after = float(solid.distance_normal(probe.data.root_pos_w)[0][0]-radius)
            worst_overlap = max(worst_overlap, -gap_after)
            forces.append(magnitude)
            if magnitude > .1 and first_hit is None:
                point = contacts.data.contact_pos_w[0, 0, target_index]
                normal = force/torch.linalg.vector_norm(force)
                point_distance, field_normal = solid.distance_normal(point)
                allowed = permission.allows(solid, link, phase, point.tolist(), normal.tolist(),
                                             max(0., -gap_after), magnitude)
                first_hit = dict(step=step, pre_gap_m=gap_before, post_gap_m=gap_after,
                                 contact_point=point.tolist(), normal=normal.tolist(),
                                 force_N=magnitude, permitted=allowed,
                                 surface_error_m=float(point_distance),
                                 normal_agreement=float(torch.dot(normal, field_normal)))
        passed = (first_hit is not None and abs(first_hit["pre_gap_m"]) < .005
                  and abs(first_hit["surface_error_m"]) < .005 and first_hit["normal_agreement"] > .99
                  and first_hit["permitted"] == expected_allowed and worst_overlap < .005
                  and max(forces[:20]) < .1)
        results.append(dict(name=name, passed=passed, first_contact=first_hit,
                            expected_permission=expected_allowed, max_penetration_m=worst_overlap,
                            peak_force_N=max(forces)))
    report = dict(schema=FIELD_VERSION, simulation_only=True, policy_loaded=False, device=sim.device,
                  dt=dt, solids=[asdict(s) for s in solids.values()], fixtures=results,
                  source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                  passed=all(r["passed"] for r in results),
                  scope="primitive sphere contacts only; no articulated contact permission validation yet")
    (run/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(run/"report.json"), "passed": report["passed"]}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Contact/field fixtures failed; inspect report; do not enable RL rewards")


if __name__ == "__main__":
    main()
