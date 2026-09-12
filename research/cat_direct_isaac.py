#!/usr/bin/env python3
"""Run native CAT or opt-in CAT/GRAIL student in Isaac/PhysX; no hardware.

One CPU-physics robot, native 500Hz explicit PD / 50Hz policy. Original CAT
player methods compute observations from a kinematic-only MuJoCo mirror.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def supervise(args):
    """Bound this process group only; verify explicit outcome, not Kit exit code."""
    import signal
    import subprocess
    stem = f"{args.scene}_{time.time_ns()}"
    outcome = args.run/(stem+"_worker.json")
    log = args.run/(stem+".log")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker-outcome", str(outcome)]
    print(f"Direct CAT worker log: {log}", flush=True)
    with log.open("x") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline, finished = time.monotonic()+(300 if args.render_replay else 90), None
        try:
            while child.poll() is None:
                if outcome.exists() and finished is None:
                    finished = time.monotonic()
                if time.monotonic() > deadline or finished is not None and time.monotonic()-finished > 10:
                    break
                time.sleep(.2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    result = json.loads(outcome.read_text()) if outcome.exists() else dict(complete=False, error="Worker timeout/early exit")
    print(json.dumps(dict(**result, log=str(log))), flush=True)
    if not result["complete"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scene", choices=["side1", "hurdle1", "crouch1"], default="side1")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--render-replay", action="store_true", help="Render saved MuJoCo/Isaac paths in a common Isaac scene; no physics steps")
    parser.add_argument("--accept-isaac-eula", action="store_true")
    parser.add_argument("--worker-outcome", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--distill-checkpoint", type=Path, help="Opt-in whole-body CAT/GRAIL student evaluation; no updates")
    parser.add_argument("--distill-run", type=Path, help="Hash-verified flat distillation dataset directory")
    parser.add_argument("--distill-seed", type=int, default=6)
    parser.add_argument("--distill-collect", action="store_true", help="Save CAT labels on student-visited flat states")
    parser.add_argument("--distill-teacher-fraction", type=float, default=0.)
    parser.add_argument("--distill-full-horizon", action="store_true", help="Continue after x exit to check student stability")
    args = parser.parse_args()
    if bool(args.distill_checkpoint) != bool(args.distill_run) or (args.distill_checkpoint and (args.render_replay or args.inspect_only)):
        parser.error("Student evaluation needs both dataset and checkpoint, no replay/inspection")
    if not 0 <= args.distill_seed <= 1000 or not 0 <= args.distill_teacher_fraction <= 1:
        parser.error("Invalid seed or teacher fraction")
    if args.distill_collect and (not args.distill_checkpoint or args.distill_seed >= 6 or args.scene != "side1"):
        parser.error("DAgger collection requires flat side1 training seeds 0..5, not held-out tests")
    if args.distill_teacher_fraction and not args.distill_collect:
        parser.error("Teacher assistance is for explicitly labelled DAgger collection only")
    if args.distill_full_horizon and not args.distill_checkpoint:
        parser.error("Full-horizon switch is for explicit student diagnostics")
    if not args.accept_isaac_eula or not 1 <= args.steps <= 2000:
        parser.error("Explicit EULA acceptance and 1..2000 steps required")
    args.run = args.run.resolve()
    args.run.mkdir(parents=True, exist_ok=True)
    if args.worker_outcome is None:
        supervise(args)
        return
    temporary = args.run/(args.scene+"_tmp")
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    import tempfile
    tempfile.tempdir = str(temporary)
    if not args.render_replay and not args.inspect_only and (args.run/f"{args.scene}_isaac.npz").exists():
        raise FileExistsError("Isaac episode already recorded")
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "Yes"
    sys.argv = sys.argv[:1]
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cpu", livestream=0, enable_cameras=args.render_replay).app
    try:
        run(args)
        args.worker_outcome.write_text(json.dumps(dict(complete=True, error=None,
            mode="render_replay" if args.render_replay else "inspect" if args.inspect_only else "physics",
            pid=os.getpid()))+"\n")
    except BaseException as error:
        import traceback
        traceback.print_exc()
        (args.run/f"{args.scene}_isaac_error_{time.time_ns()}.json").write_text(
            json.dumps(dict(error=repr(error), complete=False))+"\n")
        args.worker_outcome.write_text(json.dumps(dict(complete=False, error=repr(error), pid=os.getpid()))+"\n")
        raise
    finally:
        # Outputs/video writers are closed before this point. Kit's full CPU
        # teardown can spin indefinitely; do not leave completed workers alive.
        app.close(skip_cleanup=True)


def run(args):
    import mujoco
    import numpy as np
    import torch
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import IdealPDActuatorCfg
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from isaacsim.core.utils.extensions import enable_extension
    from cat_direct_native import (CAT, act, digest, make_player, prepare_robot_xml,
                                   save_episode, snapshot, teacher)
    from cat_scenes import export_usd

    torch.set_num_threads(2)
    enable_extension("isaacsim.asset.importer.mjcf")
    player, constants, contract = make_player(args.scene)
    if args.distill_checkpoint:
        from cat_distill import reset_player
        reset_player(player, constants, args.distill_seed)
    model = player.mj_model
    native_names = [model.actuator(i).name for i in range(model.nu)]
    xml = args.run/"cat_robot_v2.xml"
    if not xml.exists():
        xml = prepare_robot_xml(args.run)
    asset = MjcfConverter(MjcfConverterCfg(asset_path=str(xml), usd_dir=str(args.run/"robot_usd_v2"),
        usd_file_name="cat_robot.usd", fix_base=False, make_instanceable=False,
        import_inertia_tensor=True, self_collision=True))
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.002, device="cpu",
        enable_scene_query_support=True, render_interval=10,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.),
        physx=sim_utils.PhysxCfg(solver_type=1)))
    ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1., dynamic_friction=1., restitution=0.))
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1800.)
    light.func("/World/Light", light)
    # Use the actual native OBJ and native scene translation, not a regenerated
    # occupancy surface whose half-cell offset might change this comparison.
    mesh_path = CAT/"data/assets/TypiObs"/args.scene/"obs.obj"
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    scene_position = model.body("scene").pos.copy()
    mesh_usd = args.run/f"{args.scene}_native_mesh.usda"
    if not mesh_usd.exists():
        export_usd(mesh_usd, np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32))
    spawn = sim_utils.UsdFileCfg(usd_path=str(mesh_usd))
    spawn.func("/World/Clutter", spawn, translation=tuple(scene_position))
    gains = lambda values: dict(zip(native_names, map(float, values)))
    cfg = ArticulationCfg(prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=asset.usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0., angular_damping=0.,
                max_linear_velocity=100., max_angular_velocity=100., max_depenetration_velocity=10.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                solver_position_iteration_count=8, solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(constants.DEFAULT_QPOS[:3]),
            rot=tuple(constants.DEFAULT_QPOS[3:7]), joint_pos=gains(constants.DEFAULT_QPOS[7:]), joint_vel={".*": 0.}),
        actuators={"native_torque": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0.,
            effort_limit=1000., effort_limit_sim=1000., velocity_limit_sim=1000.,
            armature=gains(model.dof_armature[6:]), friction=0., dynamic_friction=0., viscous_friction=0.)})
    robot = Articulation(cfg)
    stage = sim.stage
    # MJCF importer adds an empty world articulation alongside the free robot.
    # It has no native counterpart/joints; keep only the pelvis articulation.
    world = stage.GetPrimAtPath("/World/Robot/worldBody")
    if world and world.HasAPI(UsdPhysics.ArticulationRootAPI):
        world.SetActive(False)
    # The importer correctly leaves contype=conaffinity=0 visual-only, but does
    # not reconstruct native <contact><pair>. Add exactly those named primitives.
    add_native_pair_shapes(stage, model)
    stage.Export(str(args.run/f"{args.scene}_imported.usda"))
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    inventory = [str(p.GetPath()) for p in colliders]
    (args.run/f"{args.scene}_import_inventory.json").write_text(json.dumps(dict(colliders=inventory,
        native_pairs=[[model.geom(int(a)).name, model.geom(int(b)).name] for a, b in zip(model.pair_geom1, model.pair_geom2)],
        native_robot_dynamic_contact_masks=sorted(set(zip(model.geom_contype.tolist(), model.geom_conaffinity.tolist()))),
        source_mesh_sha256=digest(mesh_path), mesh_translation=scene_position.tolist()), indent=2))
    print("DIRECT_CAT_IMPORT", json.dumps(inventory), flush=True)
    if args.inspect_only:
        return
    configure_contacts(stage, model, colliders)
    camera = None
    if args.render_replay:
        from isaaclab.sensors import Camera, CameraCfg
        camera = Camera(CameraCfg(prim_path="/World/Camera", height=480, width=640,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24.,
                horizontal_aperture=24., clipping_range=(.05, 100.))))
    sim.reset()
    robot.reset()
    ids = [robot.joint_names.index(name) for name in native_names]
    order = [native_names.index(name) for name in robot.joint_names]
    if args.render_replay:
        render_replay(args, sim, robot, camera, order)
        return
    native_body_ids = [model.body(name).id for name in robot.body_names]
    view = robot.root_physx_view
    masses, coms = view.get_masses().numpy()[0], view.get_coms().numpy()[0]
    inertias = view.get_inertias().numpy()[0]
    mass_error = float(np.max(np.abs(masses-model.body_mass[native_body_ids])))
    com_error = float(np.max(np.abs(coms[:, :3]-model.body_ipos[native_body_ids])))
    joint_limit_error = float(np.max(np.abs(view.get_dof_limits().numpy()[0]-model.jnt_range[1:][order])))
    from scipy.spatial.transform import Rotation
    native_rotation = Rotation.from_quat(model.body_iquat[native_body_ids], scalar_first=True).as_matrix()
    native_inertia = np.array([np.diag(v) for v in model.body_inertia[native_body_ids]])
    # PhysX Tensor API get_inertias is already COM-centered, expressed in the
    # LINK frame (not its principal-inertia frame). Do not rotate it twice.
    inertia_error = float(np.max(np.abs(inertias.reshape(-1, 3, 3)
        - native_rotation@native_inertia@native_rotation.transpose(0, 2, 1))))
    dynamics = dict(body_names=robot.body_names, joint_names=robot.joint_names,
        mass_error_kg=mass_error, com_position_error_m=com_error, joint_limit_error_rad=joint_limit_error,
        inertia_tensor_error_kgm2=inertia_error,
        imported_masses=masses.tolist(), imported_coms=coms.tolist(), imported_inertias=inertias.tolist(),
        native_diaginertia=model.body_inertia[native_body_ids].tolist(),
        native_inertia_quat=model.body_iquat[native_body_ids].tolist(),
        kp=constants.KPs.tolist(), kd=constants.KDs.tolist(),
        torque_limits=model.actuator_ctrlrange.tolist(), joint_armature=model.dof_armature[6:].tolist(),
        frictionloss=model.dof_frictionloss[6:].tolist(), damping=model.dof_damping[6:].tolist())
    (args.run/f"{args.scene}_dynamics.json").write_text(json.dumps(dynamics, indent=2)+"\n")
    if mass_error > 1e-4 or com_error > 1e-4 or joint_limit_error > 1e-4 or inertia_error > 1e-5:
        raise ValueError(f"Native/imported dynamics mismatch: mass={mass_error}, COM={com_error}, limits={joint_limit_error}")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
    root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros((1, 6)))
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))
    robot.update(.002)
    player_state = player.reset()
    policy = teacher()
    student = None
    dagger_rows = []
    mix_rng = np.random.default_rng(args.distill_seed+1000)
    if args.distill_checkpoint:
        from cat_distill_evaluate import StudentController
        student = StudentController(args.distill_run, args.distill_checkpoint, native_names)
    rows, started, max_fk_error = [], time.monotonic(), 0.
    # Explicit torque computation includes native passive damping; passive dry
    # friction is configured using PhysX 5's joint friction model below.
    robot.write_joint_friction_coefficient_to_sim(tensor(model.dof_frictionloss[6:][order]),
        tensor(model.dof_frictionloss[6:][order]), tensor(model.dof_damping[6:][order]))
    kp, kd = tensor(constants.KPs[order]), tensor(constants.KDs[order])
    limits = tensor(model.actuator_ctrlrange[:, 1][order])
    pelvis_id = robot.body_names.index("pelvis")

    def sync_mirror(forward=True):
        from scipy.spatial.transform import Rotation
        data = robot.data
        quat = data.body_quat_w[0, pelvis_id].numpy()
        rotation = Rotation.from_quat(quat, scalar_first=True).as_matrix()
        player.mj_data.qpos[:3] = data.body_pos_w[0, pelvis_id].numpy()
        player.mj_data.qpos[3:7] = quat
        player.mj_data.qpos[7:] = data.joint_pos[0, ids].numpy()
        vel = data.body_link_vel_w[0, pelvis_id].numpy()
        player.mj_data.qvel[:3] = vel[:3]
        player.mj_data.qvel[3:6] = rotation.T@vel[3:]
        player.mj_data.qvel[6:] = data.joint_vel[0, ids].numpy()
        if forward:
            mujoco.mj_forward(model, player.mj_data)

    sync_mirror()
    for name in robot.body_names:
        error = np.linalg.norm(robot.data.body_pos_w[0, robot.body_names.index(name)].numpy()-player.mj_data.body(name).xpos)
        max_fk_error = max(max_fk_error, float(error))
    if max_fk_error > .001:
        raise ValueError(f"Imported robot frame error {max_fk_error}m")
    for step in range(args.steps):
        if student is None:
            action = act(policy, player_state)
            targets = constants.DEFAULT_QPOS[7:].copy()
            targets[player.action_joint_ids] = np.clip(player_state.info["motor_targets"][player.action_joint_ids]
                +action*player._config.action_scale, player._soft_lowers[player.action_joint_ids], player._soft_uppers[player.action_joint_ids])
        else:
            targets = student.targets(player, player_state)
            if args.distill_collect:
                label, valid = student.label(player, player_state, policy)
                if valid:
                    dagger_rows.append(dict(**label, episode=np.int64(300+args.distill_seed), validation=np.bool_(False)))
                if mix_rng.random() < args.distill_teacher_fraction:
                    targets[player.action_joint_ids] = label["target"][student.model.legs]
            action = (targets[player.action_joint_ids]-player_state.info["motor_targets"][player.action_joint_ids])/.5
        player_state.info["motor_targets"] = targets.copy()
        target = tensor(targets[order])
        for substep in range(10):
            if substep == 9:
                # Native mj_step leaves position/sensor caches at pre-integration
                # state while qpos/qvel advance; preserve that 2ms convention.
                sync_mirror()
            effort = (kp*(target-robot.data.joint_pos)-kd*robot.data.joint_vel).clamp(-limits, limits)
            robot.set_joint_effort_target(effort)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(.002)
        sync_mirror(forward=False)
        player_state = player.observe_after_physics(player_state, action)
        rows.append(snapshot(player, player_state, action))
        if step % 100 == 0:
            print("DIRECT_CAT_STEP", step, player.mj_data.qpos[:3].tolist(), flush=True)
        if rows[-1]["head"][2] < .7 or (player.mj_data.qpos[0] >= 1.9 and not args.distill_full_horizon):
            break
    result = save_episode(args.run, args.scene, "isaac", rows, player, time.monotonic()-started,
        dict(policy="GRAIL frozen 29-joint decoder + CAT-trained motor-token adapter", grail_loaded=True)
        if student is not None else None)
    result.update(imported_body_fk_max_error_m=max_fk_error, physics_device="cpu", control_dt=.02, physics_dt=.002,
        source_mesh_sha256=digest(mesh_path), mesh_translation=scene_position.tolist(),
        contact_mode="native explicit pairs; clutter is SDF-only as in original CAT")
    if student is not None:
        result.update(policy="GRAIL frozen 29-joint decoder + CAT-trained motor-token adapter", grail_loaded=True,
            checkpoint_sha256=student.checkpoint_hash, checkpoint_updates=student.step, applied_joint_count=29,
            soft_clipped_target_fraction=student.clip_count/student.target_count,
            scope="native CAT dynamics in Isaac; not full GRAIL terrain task")
        result.update(teacher_fraction=args.distill_teacher_fraction, dagger_collection=args.distill_collect)
        result.update(full_horizon=args.distill_full_horizon,
            final_goal_distance_xy=float(np.linalg.norm(player.mj_data.qpos[:2]-[2., 0.])))
    if args.distill_collect:
        if not dagger_rows:
            raise ValueError("No admitted Isaac student-state teacher labels")
        packet = args.run/"dagger.npz"
        np.savez_compressed(packet, **{k: np.stack([r[k] for r in dagger_rows]) for k in dagger_rows[0]})
        (args.run/"dagger.json").write_text(json.dumps(dict(sha256=digest(packet),
            source_dataset_sha256=digest(args.distill_run/"dataset.npz"), checkpoint_sha256=student.checkpoint_hash,
            training_seeds=[args.distill_seed], frames=len(dagger_rows), teacher_fraction=args.distill_teacher_fraction,
            physics="Isaac CPU PhysX", support_boundary="native kinematic ground-contact check",
            robot_actuation=False), indent=2)+"\n")
    (args.run/f"{args.scene}_isaac.json").write_text(json.dumps(result, indent=2)+"\n")


def configure_contacts(stage, model, colliders):
    from pxr import UsdPhysics
    # Importer's collider names are checked, never silently assume that every
    # visual mesh should collide (native model uses explicit contact pairs).
    pair_names = {model.geom(int(i)).name for ids in (model.pair_geom1, model.pair_geom2) for i in ids}
    by_name = {}
    for prim in colliders:
        names = [part.removeprefix("native_") for part in str(prim.GetPath()).split("/") if part.startswith("native_")]
        name = names[0] if len(names) == 1 else prim.GetName()
        if str(prim.GetPath()).startswith("/World/Ground"):
            by_name["floor"] = prim
        elif str(prim.GetPath()).startswith("/World/Robot") and name in pair_names:
            by_name[name] = prim
        else:
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    if set(by_name) != pair_names:
        raise ValueError(f"Unmapped native contact colliders: {pair_names-set(by_name)}")
    allowed = {frozenset((model.geom(int(a)).name, model.geom(int(b)).name))
               for a, b in zip(model.pair_geom1, model.pair_geom2)}
    for name, prim in by_name.items():
        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
        rel = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        rel.SetTargets([other.GetPath() for other_name, other in by_name.items()
                        if name != other_name and frozenset((name, other_name)) not in allowed])


def add_native_pair_shapes(stage, model):
    import mujoco
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdPhysics
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)
              and str(p.GetPath()).startswith("/World/Robot/")}
    required = set(model.pair_geom1.tolist()+model.pair_geom2.tolist())
    for geom_id in required:
        geom = model.geom(geom_id)
        if geom.name == "floor":
            continue
        body = bodies[model.body(int(geom.bodyid[0])).name]
        common = dict(collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001, rest_offset=0.),
                      physics_material=sim_utils.RigidBodyMaterialCfg(
                          static_friction=1. if geom.name.endswith("foot") else 0.,
                          dynamic_friction=1. if geom.name.endswith("foot") else 0., restitution=0.))
        kind = int(geom.type[0])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            cfg = sim_utils.CuboidCfg(size=tuple(2*geom.size), **common)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            cfg = sim_utils.CapsuleCfg(radius=float(geom.size[0]), height=float(2*geom.size[1]), axis="Z", **common)
        else:
            raise ValueError(f"Native contact shape not implemented: {geom.name}, {kind}")
        path = str(body.GetPath())+"/native_"+geom.name
        cfg.func(path, cfg, translation=tuple(geom.pos), orientation=tuple(geom.quat))
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()


def render_replay(args, sim, robot, camera, order):
    """Same renderer/camera for recorded trajectories, not new policy rollouts."""
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    camera.set_world_poses_from_view(torch.tensor([[2.8, -3.5, 2.0]]), torch.tensor([[1., 0., .7]]))
    videos = {}
    for engine in ("mujoco", "isaac"):
        path = args.run/f"{args.scene}_{engine}.npz"
        with np.load(path, allow_pickle=False) as archive:
            poses = archive["qpos"]
        output = args.run/f"{args.scene}_{engine}_isaac_render.mp4"
        if output.exists():
            raise FileExistsError(output)
        writer = imageio.get_writer(output, fps=25, codec="libx264", quality=7)
        try:
            for step in range(0, min(len(poses), 400), 2):
                q = poses[step]
                robot.write_root_pose_to_sim(torch.tensor(q[:7], dtype=torch.float32)[None])
                robot.write_root_velocity_to_sim(torch.zeros((1, 6)))
                robot.write_joint_state_to_sim(torch.tensor(q[7:][order], dtype=torch.float32)[None],
                                               torch.zeros_like(robot.data.joint_pos))
                sim.forward()
                for _ in range(2 if step == 0 else 1):
                    sim.render()
                camera.update(.04, force_recompute=True)
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                frame = Image.fromarray(rgb)
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 43), fill="black")
                draw.text((10, 5), f"CAT / {args.scene} / {engine.upper()} physics / {(step+1)*.02:.2f}s", fill="white")
                draw.text((10, 23), "Recorded poses replayed in Isaac renderer; no new dynamics", fill="white")
                writer.append_data(np.array(frame))
        finally:
            writer.close()
        videos[engine] = str(output)
        print("DIRECT_CAT_VIDEO", engine, output, flush=True)
    import subprocess
    comparison = args.run/f"{args.scene}_comparison.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", videos["mujoco"], "-i", videos["isaac"],
        "-filter_complex", "[0:v][1:v]hstack=inputs=2:shortest=1[v]", "-map", "[v]", "-c:v", "libx264",
        "-crf", "20", str(comparison)], check=True)


if __name__ == "__main__":
    main()
