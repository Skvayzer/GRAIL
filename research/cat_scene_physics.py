"""PhysX ray/contact validation of CAT-generated USD; import only after Kit."""
import json

import numpy as np
import torch
import omni.physx
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

from cat_scenes import verify_scene
from voxel_fixtures import axis_ray_fixtures, planar_contact_patch


def run_cat_scene_check(run, directory):
    meta = verify_scene(directory)
    occupied = np.load(directory/"obs.npy", allow_pickle=False)
    voxel, origin = meta["resolution"], meta["origin_corner"]
    dt, radius = .005, .025
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device="cuda:0", gravity=(0., 0., 0.), enable_scene_query_support=True))
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=6.)
    cfg.cat = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/CAT", spawn=sim_utils.UsdFileCfg(usd_path=str(directory/"scene.usda")))
    cfg.probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/probe",
        spawn=sim_utils.SphereCfg(radius=radius, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100., 100., 100.)),
    )
    cfg.contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/probe", update_period=0., history_length=1)
    scene = InteractiveScene(cfg)
    sim.reset()
    scene.reset()
    sim.step(render=False)
    scene.update(dt)
    query = omni.physx.get_physx_scene_query_interface()
    results = []
    for fixture in axis_ray_fixtures(occupied, voxel, origin):
        hit = query.raycast_closest(tuple(fixture["start"]), tuple(fixture["direction"]), fixture["max_distance"])
        measured = float(hit["distance"]) if hit["hit"] else None
        expected = fixture["expected_distance"]
        error = abs(measured-expected) if measured is not None and expected is not None else None
        passed = (measured is None and expected is None) or (error is not None and error < .005)
        results.append(dict(**fixture, measured_distance=measured, error_m=error, passed=passed))
    # A mesh query is not proof of physical response: drive a dynamic sphere
    # into a planar occupied column and require a measured stopping contact.
    # Select a genuinely planar 3x3-cell patch from OCCUPANCY, not a random
    # point ray. A finite sphere can hit a neighboring bevel before that ray.
    # Do not loosen the normal check to hide that fixture geometry mismatch.
    if radius > voxel:
        raise ValueError("Contact probe exceeds the validated patch size")
    boundary = planar_contact_patch(occupied, voxel, origin)
    start = boundary.copy()
    start[0] -= radius+.1
    probe = scene["probe"]
    state = probe.data.default_root_state.clone()
    state[:, :3] = torch.tensor(start, device=sim.device)
    state[:, 7:] = 0
    probe.write_root_pose_to_sim(state[:, :7])
    probe.write_root_velocity_to_sim(state[:, 7:])
    scene.reset()
    first_contact = None
    for step in range(160):
        velocity = state.new_zeros((1, 6))
        velocity[:, 0] = .2
        probe.write_root_velocity_to_sim(velocity)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        force = scene["contacts"].data.net_forces_w[0, 0]
        magnitude = float(torch.linalg.vector_norm(force))
        if magnitude > .1 and first_contact is None:
            first_contact = dict(step=step, force_N=magnitude, normal=(force/magnitude).tolist(),
                                 position=probe.data.root_pos_w[0].tolist(),
                                 surface_gap_m=float(boundary[0]-probe.data.root_pos_w[0, 0]-radius))
    contact_passed = (first_contact is not None and abs(first_contact["surface_gap_m"]) < .01
                      and first_contact["normal"][0] < -.9)
    from isaaclab.sim.utils.stage import get_current_stage
    mesh = get_current_stage().GetPrimAtPath("/World/envs/env_0/CAT/Obstacles")
    approximation = UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()
    report = dict(simulation_only=True, policy_loaded=False, source_scene=str(directory),
                  source_files=meta["files"], mesh_approximation=approximation,
                  ray_count=len(results), rays=results, physical_sphere_contact=first_contact,
                  contact_passed=contact_passed,
                  passed=all(r["passed"] for r in results) and contact_passed and approximation == "none")
    (run/"cat_mesh_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"report": str(run/"cat_mesh_report.json"), "passed": report["passed"],
                      "rays": len(results)}), flush=True)
    if not report["passed"]:
        raise RuntimeError("CAT mesh/occupancy physics agreement failed")
