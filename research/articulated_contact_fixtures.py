"""Controlled imported-G1 collision diagnostics; imported by physics_contact_check.

Zero gravity, commanded root approach and original joint-position actuators.
Not a walking controller, support/stability test, or reference phase annotation.
All robot state writes are to a new desktop simulation, never hardware.
"""
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import subprocess

import numpy as np


@dataclass(frozen=True)
class Fixture:
    name: str
    link: str = "left_ankle_roll_link"
    approach: str = "down"
    phase: str = "stance_candidate"
    partner: str = "terrain"
    expected: str = "stance_support_candidate"
    yaw: float = 0.
    clear: bool = False

    @property
    def size(self):
        if self.approach == "front":
            return (.08, .10, .08 if "ankle" in self.link else .14)
        if self.approach == "up":
            # A small patch over the forefoot, not a plate through the shin.
            return (.04, .05, .02)
        return (.50, .10, .08)


def fixture_specs():
    return (
        Fixture("left_tread"),
        Fixture("right_tread_yaw", link="right_ankle_roll_link", yaw=math.pi/2),
        Fixture("swing_support", phase="swing_candidate", expected="unexpected_swing_support"),
        Fixture("unknown_phase", phase="unknown", expected="phase_unknown"),
        Fixture("foot_riser", approach="front", expected="forbidden_riser_or_side"),
        Fixture("shin_riser", link="left_knee_link", approach="front", expected="forbidden_nonfoot"),
        Fixture("foot_underside", approach="up", expected="forbidden_riser_or_side"),
        Fixture("foot_cat", partner="cat", expected="forbidden_cat"),
        Fixture("clear_approach", clear=True, expected="no_contact"),
    )


def station_geometry(spec, capsule_centers, capsule_radii, gap=.04):
    """Place a box from actual capsule extrema in the robot-yaw frame.

    Endpoints plus true radii give exact directional extrema for capsules.
    A clear control stops before the same surface, without moving it away.
    """
    c, s = math.cos(spec.yaw), math.sin(spec.yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    centers, radii = np.asarray(capsule_centers) @ rotation, np.asarray(capsule_radii)[:, None]
    if (centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers), 1)
            or not len(centers) or not np.isfinite(centers).all() or not np.isfinite(radii).all()
            or (radii <= 0).any() or not math.isfinite(gap) or gap <= 0):
        raise ValueError("Invalid fixture capsule geometry")
    lo, hi = (centers-radii).min(0), (centers+radii).max(0)
    box = (lo+hi)/2
    normal = np.zeros(3)
    if spec.approach == "down":
        box[2] = lo[2]-gap-spec.size[2]/2
        normal[2] = 1
    elif spec.approach == "up":
        box[0] = centers[:, 0].max()-.01
        box[2] = hi[2]+gap+spec.size[2]/2
        normal[2] = -1
    elif spec.approach == "front":
        # A pitched shin's forward extremum is near an endpoint, not the
        # capsule midpoint. Centre the face there to avoid an unintended edge.
        box[1:] = centers[np.argmax(centers[:, 0]+radii[:, 0]), 1:]
        box[0] = hi[0]+gap+spec.size[0]/2
        normal[0] = -1
    else:
        raise ValueError("Unknown fixture approach")
    return box @ rotation.T, normal @ rotation.T


def box_normal_cone_agreement(point, normal, center, size, yaw, tolerance=1e-5):
    """Compare with the convex box's outward normal cone at a boundary point.

    At an edge the SDF gradient is not unique: choosing argmax's first face
    incorrectly rejects a valid normal from the other face. Project onto the
    cone of active outward face normals; do not flip or replace the measurement.
    The 10 micrometre boundary tolerance covers float32 point rounding only.
    """
    point, normal, center, size = [np.asarray(v, dtype=float) for v in (point, normal, center, size)]
    if (any(v.shape != (3,) or not np.isfinite(v).all() for v in (point, normal, center, size))
            or (size <= 0).any() or not math.isfinite(yaw) or not math.isfinite(tolerance)
            or tolerance <= 0 or tolerance >= size.min()/2 or abs(np.linalg.norm(normal)-1.) > .002):
        raise ValueError("Invalid box contact geometry")
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local, n = (point-center) @ rotation, normal @ rotation
    boundary_error = np.abs(local)-size/2
    if (boundary_error > tolerance).any():
        return 0.
    active = np.abs(boundary_error) <= tolerance
    outward = n*np.where(local >= 0, 1., -1.)
    projected = np.where(active, np.maximum(outward, 0.), 0.)
    return float(np.linalg.norm(projected)/np.linalg.norm(normal))


def run_articulated_fixtures(run):
    # All Isaac/USD imports follow AppLauncher in the caller.
    import torch
    import omni.physics.tensors as tensors
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from gear_sonic.envs.manager_env.robots.g1 import G1_CYLINDER_CFG
    from gear_sonic.research.body_envelope import URDF, world_probe_centers
    from gear_sonic.research.contact_accounting import (
        ContactLimits, bind_contact_names, classify_contact, sole_regions, unpack_contacts,
    )
    from gear_sonic.research.geometry import Solid
    from gear_sonic.research.isaac_clutter import solid_cfg
    from gear_sonic.research.terrain_snapshot import rotation_wxyz
    from gear_sonic.research.usd_envelope import imported_collision_probes

    dt, speed, steps = .005, .10, 130
    limits = ContactLimits()
    specs = fixture_specs()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0", gravity=(0., 0., 0.)))
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.)
    cfg.robot = G1_CYLINDER_CFG.copy()
    cfg.robot.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.robot.spawn.asset_path = str(URDF)
    # Avoid modifying the default asset converter cache or any upstream USD.
    cfg.robot.spawn.usd_dir = str(run/"imported_robot")
    cfg.robot.init_state.pos = (0., 0., 2.)
    for spec in specs:
        setattr(cfg, spec.name, solid_cfg(Solid(spec.name, "box", (20., 0., 0.), spec.size),
                                          "{ENV_REGEX_NS}/"+spec.name))
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    if robot.num_joints != 29:
        raise ValueError("Expected the original 29-joint G1 articulation")
    probes, inventory = imported_collision_probes(get_current_stage(), "/World/envs/env_0/Robot", robot.body_names)
    sole = sole_regions(probes, inventory, limits)
    links = sorted({p.link for p in probes})
    paths = ["/World/envs/env_0/Robot/"+link for link in links]
    filters = ["/World/envs/env_0/"+spec.name for spec in specs]
    physics = tensors.create_simulation_view("torch")
    physics.set_subspace_roots("/")
    view = physics.create_rigid_contact_view(paths, [filters]*len(paths), max_contact_data_count=8192)
    if view.sensor_count != len(paths) or view.filter_count != len(specs):
        raise ValueError("Unexpected fixture contact view dimensions")
    sensor_links = bind_contact_names(list(view.sensor_names), list(view.filter_names), paths, filters)
    records, results, unexpected = [], [], []
    peak_error = 0.
    for fi, spec in enumerate(specs):
        # Only reset at fixture boundaries. During an approach the articulation
        # responds dynamically; no body-pose/joint teleporting through contact.
        for other in specs:
            box = scene[other.name]
            away = box.data.default_root_state.clone()
            away[:, :3] = away.new_tensor([20.+list(specs).index(other), 0., 0.])
            box.write_root_pose_to_sim(away[:, :7])
        state = robot.data.default_root_state.clone()
        state[:, 3:7] = state.new_tensor([math.cos(spec.yaw/2), 0., 0., math.sin(spec.yaw/2)])
        robot.write_root_pose_to_sim(state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:]))
        joints = robot.data.default_joint_pos.clone()
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.set_joint_position_target(joints)
        scene.reset()
        for _ in range(20):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
        selected = [p for p in probes if p.link == spec.link]
        if any(inventory[p.collision_index]["type"] != "Capsule" for p in selected):
            raise ValueError("Fixture placement currently validates capsules only")
        centers, _ = world_probe_centers(selected, robot.body_names, robot.data.body_pos_w, robot.data.body_quat_w)
        radii = np.array([inventory[p.collision_index]["radius"] for p in selected])
        center, normal = station_geometry(spec, centers[0].cpu().numpy(), radii)
        solid = Solid(spec.name, "box", tuple(center), spec.size, yaw=spec.yaw)
        box = scene[spec.name]
        pose = box.data.default_root_state[:, :7].clone()
        pose[:, :3] = pose.new_tensor(center)
        pose[:, 3:7] = pose.new_tensor([math.cos(spec.yaw/2), 0., 0., math.sin(spec.yaw/2)])
        box.write_root_pose_to_sim(pose)
        initial_body_pos = robot.data.body_pos_w[0, robot.body_names.index(spec.link)].cpu().numpy().copy()
        first_step, observed, wrong_pairs, samples = None, Counter(), 0, []
        for step in range(30 if spec.clear else steps):
            velocity = state.new_zeros((1, 6))
            velocity[:, :3] = velocity.new_tensor(-speed*normal)
            robot.write_root_velocity_to_sim(velocity)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            raw = [a.cpu().numpy().copy() for a in view.get_contact_data(dt)]
            data, error = unpack_contacts(raw, view.get_contact_force_matrix(dt).cpu().numpy().copy())
            peak_error = max(peak_error, error)
            if len(data["force"]):
                distance, field_normal = solid.distance_normal(torch.as_tensor(data["point"], device=sim.device))
                distance, field_normal = distance.cpu().numpy(), field_normal.cpu().numpy()
            sample_count = 0
            for k, force in enumerate(data["force"]):
                if force <= limits.force_threshold:
                    continue
                si, pi = int(data["sensor"][k]), int(data["partner"][k])
                link = sensor_links[si]
                if link != spec.link or pi != fi:
                    wrong_pairs += 1
                    unexpected.append(dict(fixture=spec.name, step=step, link=link,
                        partner_fixture=specs[pi].name, force_N=float(force), point=data["point"][k].tolist()))
                    continue
                bi = robot.body_names.index(link)
                body_pos = robot.data.body_pos_w[0, bi].cpu().numpy()
                quat = robot.data.body_quat_w[0, bi].cpu().numpy()
                local = (data["point"][k]-body_pos) @ rotation_wxyz(quat)
                lo, hi = sole.get(link, (np.zeros(3), np.zeros(3)))
                sdf_agreement = float(np.dot(data["normal"][k], field_normal[k]))
                agreement = box_normal_cone_agreement(data["point"][k], data["normal"][k],
                                                       solid.center, solid.size, solid.yaw)
                code = classify_contact(spec.partner, link in sole, spec.phase, local, lo, hi,
                    data["normal"][k], float(force), float(data["separation"][k]), float(distance[k]), agreement, limits)
                record = dict(fixture=spec.name, step=step, link=link, phase=spec.phase,
                    force_N=float(force), point=data["point"][k].tolist(), normal=data["normal"][k].tolist(),
                    separation_m=float(data["separation"][k]), local_point=local.tolist(),
                    body_position=body_pos.tolist(), body_quaternion_wxyz=quat.tolist(),
                    surface_error_m=float(distance[k]), normal_agreement=agreement,
                    sdf_single_normal_agreement=sdf_agreement, classification=code)
                records.append(record)
                samples.append(record)
                observed[code] += 1
                sample_count += 1
            if sample_count and first_step is None:
                first_step = step
            # Retain 10 substeps from onset; do not drive through an obstacle.
            if first_step is not None and step >= first_step+9:
                break
        final_body_pos = robot.data.body_pos_w[0, robot.body_names.index(spec.link)].cpu().numpy()
        approached = float(np.dot(initial_body_pos-final_body_pos, normal))
        # A fixture must physically reach the named collider, not pass on empty
        # labels. Surface checks remain independent of classifier precedence.
        geometry_ok = all(abs(r["surface_error_m"]) < .005 and r["normal_agreement"] > .98
                          and np.dot(r["normal"], normal) > .98 and r["separation_m"] > -.005 for r in samples)
        passed = (not samples and wrong_pairs == 0 and .005 < approached < .03) if spec.clear else (
            bool(samples) and set(observed) == {spec.expected} and wrong_pairs == 0 and geometry_ok
            and first_step >= 20 and .02 < approached < .08)
        results.append(dict(**asdict(spec), passed=passed, steps=step+1, first_contact_step=first_step,
            actual_approach_m=approached, classifications=dict(observed), other_pair_contacts=wrong_pairs,
            geometry_checks_passed=geometry_ok, solid=asdict(solid), contact_count=len(samples)))
        print(json.dumps(results[-1]), flush=True)
    detail = run/"articulated_fixture_contacts.json"
    detail.write_text(json.dumps(records, indent=2, allow_nan=False)+"\n")
    report = dict(schema="grail-cat-articulated-fixtures-v1", simulation_only=True, policy_loaded=False,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        robot_urdf=str(URDF), robot_urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
        imported_colliders=inventory, joint_names=robot.joint_names, physics_dt=dt, gravity=[0., 0., 0.],
        imposed_root_speed_m_s=speed, limits=asdict(limits), fixtures=results,
        sole_regions={k: [a.tolist() for a in v] for k, v in sole.items()},
        force_reconstruction_error_N=peak_error, contact_file=detail.name,
        unexpected_contacts=unexpected,
        contact_sha256=hashlib.sha256(detail.read_bytes()).hexdigest(),
        fixture_phase_source="explicit scenario label, independent of actual force; not a learned/reference phase detector",
        normal_check="exact convex-box outward normal cone, boundary tolerance 1e-5 m; measured normals unchanged",
        phase_truth_verified=False, contact_permissions_granted=False, avoidance_training_ready=False,
        scope="isolated articulated geometry/semantic cases; no walking, stability, self-contact or reward validation",
        passed=all(r["passed"] for r in results))
    (run/"articulated_fixture_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(run/"articulated_fixture_report.json"), "passed": report["passed"]}), flush=True)
    if not report["passed"]:
        raise RuntimeError("Articulated contact fixtures failed; inspect retained report")
    from articulated_fixture_results import audit_articulated_fixtures
    print(json.dumps({"saved_artifact_verification": audit_articulated_fixtures(run)}), flush=True)
