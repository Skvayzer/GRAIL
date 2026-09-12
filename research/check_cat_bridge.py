#!/usr/bin/env python3
"""CPU MuJoCo/ONNX versus named-body PyTorch teacher bridge parity.

Run with Click-and-Traverse/.venv/bin/python. Only the original CAT policy drives
MuJoCo; the new teacher merely labels those SAME states. Does not start Isaac,
train a policy, contact a robot, or import deployment modules.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def native_contract(env, cat_repo):
    from gear_sonic.research.cat_bridge import JOINTS, OBS_JOINTS, SITES
    model = env.mj_model
    names = [model.actuator(i).name for i in range(model.nu)]
    sites = []
    for name in SITES:
        site = model.site(name)
        if not (abs(site.quat[0]-1) < 1e-8 and (abs(site.quat[1:]) < 1e-8).all()):
            raise ValueError("Nonidentity site rotation needs explicit support")
        sites.append(dict(site=name, body=model.body(int(site.bodyid[0])).name, position=site.pos.tolist()))
    source_files = ["cat_ppo/envs/g1/env_cat.py", "cat_ppo/envs/g1/play_cat.py",
                    "cat_ppo/envs/g1/constants.py", "data/assets/unitree_g1/g1_mjx_feetonly_torque.xml"]
    return dict(schema="cat-observation-action-contract-v1", action_joints=JOINTS,
        observation_joints=OBS_JOINTS, sites=sites, action_scale=float(env._config.action_scale),
        default_joint_positions=dict(zip(names, env._default_qpos.tolist())),
        soft_lower=env._soft_lowers[env.action_joint_ids].tolist(),
        soft_upper=env._soft_uppers[env.action_joint_ids].tolist(),
        action_semantics="clip(previous_applied_target + action_scale * action, soft limits)",
        sources={s: hashlib.sha256((cat_repo/s).read_bytes()).hexdigest() for s in source_files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cat-repo", type=Path, default=ROOT.parent.parent/"Click-and-Traverse")
    parser.add_argument("--teacher", type=Path, default=ROOT/"artifacts/cat_teacher_v2")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--scenes", nargs="+", default=["side1", "hurdle1", "crouch1"])
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000 or any(Path(s).name != s for s in args.scenes):
        parser.error("Use 1..1000 steps and typical scene basenames")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["GLI_PATH"] = str((args.cat_repo/"cat_ppo").resolve())
    os.environ.setdefault("WANDB_PROJECT", "cat_bridge_cpu_check")
    os.environ["WANDB_MODE"] = "offline"
    os.environ["SWANLAB_MODE"] = "disabled"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(args.cat_repo))
    sys.path.insert(0, str(ROOT.parent/"imports/SONIC"))
    import mujoco
    import numpy as np
    import onnxruntime as ort
    import torch
    from ml_collections import ConfigDict
    from cat_ppo.envs.g1.play_cat import PlayG1CatEnv
    from gear_sonic.research.cat_teacher import CatTeacher
    from gear_sonic.research.cat_bridge import (CatFieldSampler, CatObservationBridge, GROUPS, SITES,
                                               sample_teacher_fields)
    torch.set_num_threads(2)
    fixture = json.loads((ROOT/"tests/fixtures/cat_sampler.json").read_text())
    if hashlib.sha256((args.cat_repo/fixture["file"]).read_bytes()).hexdigest() != fixture["sha256"]:
        raise ValueError("Native CAT training source differs from pinned reference")
    tensor = lambda a: torch.as_tensor(np.array(a), dtype=torch.float32)
    meta = json.loads((args.teacher/"verification.json").read_text())
    if hashlib.sha256((args.teacher/"weights.npz").read_bytes()).hexdigest() != meta["weights_sha256"]:
        raise ValueError("Teacher checksum differs from verified export")
    teacher = CatTeacher(args.teacher/"weights.npz")
    command_errors = check_native_command(args.cat_repo)
    # Independent released ONNX actor, not another call to our PyTorch export.
    onnx = args.cat_repo/"data/models/generalist_v1/policy.onnx"
    if hashlib.sha256(onnx.read_bytes()).hexdigest() != "17372d1d7b1c6759a2eb2eea09b9589f55746def198d744517efd05ac8665f62":
        raise ValueError("Unexpected native ONNX checkpoint")
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 2
    native = ort.InferenceSession(str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
    config = json.loads((ROOT/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json").read_text())
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_cat_bridge")
    run.mkdir(parents=True, exist_ok=False)
    reports, observations, labels, sample_ids = [], [], [], []
    np.random.seed(42)
    for scene_index, scene in enumerate(args.scenes):
        cfg = ConfigDict(config["env_config"])
        cfg.pf_config.path = str((args.cat_repo/"data/assets/TypiObs"/scene).resolve())
        env = PlayG1CatEnv(config=cfg, headless=True)
        contract = native_contract(env, args.cat_repo)
        if scene_index == 0:
            (run/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        elif json.loads((run/"contract.json").read_text()) != json.loads(json.dumps(contract)):
            raise ValueError("Scenes must use the same robot contract")
        model, data = env.mj_model, env.mj_data
        # Reverse both arrays to ensure no accidental MuJoCo/PhysX index assumption.
        joint_ids = list(reversed(range(model.nu)))
        body_ids = list(reversed(range(1, model.nbody)))
        joint_names = [model.actuator(i).name for i in joint_ids]
        body_names = [model.body(i).name for i in body_ids]
        bridge = CatObservationBridge(contract, joint_names, body_names)
        fields = {k: tensor(getattr(env, k)) for k in ("gf", "bf", "sdf")}
        vectorized = CatFieldSampler(fields, env.pf_origin, env.dx)
        state = env.reset()
        errors = dict(observation=0., action=0., target=0., sites=0., sampler=0.)
        resets, outside, saved = 0, 0, []
        for step in range(args.steps):
            # MuJoCo mj_step leaves position-dependent sensor data from before
            # the final integration. Player get_obs calls mj_forward AFTER
            # packing. Refresh first so both labels refer to one physical state.
            mujoco.mj_forward(model, data)
            state.obs = env.get_obs(state.info)
            velocities = []
            for body in body_ids:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
                velocities.append(velocity[:3])
            physical = SimpleNamespace(joint_pos=tensor(data.qpos[7:][joint_ids])[None],
                joint_vel=tensor(data.qvel[6:][joint_ids])[None],
                body_pos_w=tensor(data.xpos[body_ids])[None],
                body_quat_w=tensor(data.xquat[body_ids])[None],
                body_ang_vel_w=tensor(velocities)[None])
            inputs = bridge.read_articulation(physical, torch.zeros(1, 3))
            sites = inputs.pop("sites")
            native_sites = data.site_xpos[[model.site(s).id for s in SITES]]
            np.testing.assert_allclose(sites[0], native_sites, atol=3e-6, rtol=1e-5)
            errors["sites"] = max(errors["sites"], float(np.max(np.abs(sites[0].numpy()-native_sites))))
            sampled = sample_teacher_fields(fields, sites, env.pf_origin, env.dx)
            fast = vectorized.sample(sites)
            for key in sampled:
                torch.testing.assert_close(fast[key], sampled[key], atol=1e-6, rtol=1e-5)
            outside += int((~sampled["in_domain"]).sum())
            for key in ("gf", "bf", "sdf"):
                expected_field = env.sample_field(getattr(env, key), sites[0].numpy())
                np.testing.assert_allclose(sampled[key][0], expected_field, atol=8e-6, rtol=2e-5)
                errors["sampler"] = max(errors["sampler"], float(np.max(np.abs(sampled[key][0].numpy()-expected_field))))
            info = state.info
            # Preserve the native player's provided field normalization/history.
            # Training may delay these; the bridge must not recompute them here.
            pf = {key: tensor(np.concatenate([info[g+suffix] for g in GROUPS]))[None]
                  for key, suffix in (("gf", "gf"), ("bf", "bf"), ("sdf", "df"))}
            command = tensor(np.r_[info["last_flags"][1], info["command"]])[None]
            previous = tensor(info["motor_targets"][env.action_joint_ids])[None]
            obs = bridge.pack(**inputs, **pf, command_world=command,
                foot_height=tensor([[info["foot_height"]]]), phase=tensor(info["phase"])[None],
                last_action=tensor(info["last_act"])[None], previous_targets=previous)
            expected_obs = state.obs["state"].astype(np.float32)[None]
            np.testing.assert_allclose(obs, expected_obs, atol=1e-5, rtol=2e-5)
            with torch.no_grad():
                action = teacher(obs)
            expected_action = native.run(["continuous_actions"], {"obs": expected_obs})[0]
            np.testing.assert_allclose(action, expected_action, atol=6e-5, rtol=2e-5)
            targets = bridge.leg_targets(action, previous)
            expected_targets = np.clip(previous.numpy()+expected_action*cfg.action_scale,
                env._soft_lowers[env.action_joint_ids], env._soft_uppers[env.action_joint_ids])
            np.testing.assert_allclose(targets, expected_targets, atol=4e-5, rtol=2e-5)
            for key, actual, expected in (("observation", obs, expected_obs),
                                         ("action", action, expected_action), ("target", targets, expected_targets)):
                errors[key] = max(errors[key], float(np.max(np.abs(actual.numpy()-expected))))
            observations.append(obs.numpy()[0])
            labels.append(targets.numpy()[0])
            sample_ids.append((scene_index, step, resets))
            saved.append((inputs, pf, command, previous, info["phase"].copy(), info["foot_height"], info["last_act"].copy()))
            # Only native released actor controls this CPU simulation.
            state = env.step(state, expected_action[0])
            np.testing.assert_allclose(state.info["motor_targets"][env.action_joint_ids],
                                       targets.numpy()[0], atol=4e-5, rtol=2e-5)
            if not bool(state.info.get("succ", True)):
                resets += 1
                state = env.reset()
        reports.append(dict(scene=scene, states=args.steps, maximum_errors=errors,
                            native_failure_resets=resets, outside_field_queries=outside))
        # Validate original TRAINING _get_obs too, noise disabled; no MJX compile.
        check_training_packer(args.cat_repo, saved, bridge)
        print(json.dumps(reports[-1]), flush=True)
    np.savez_compressed(run/"same_state_labels.npz", observations=np.stack(observations),
                        leg_targets=np.stack(labels), scene_step_episode=np.array(sample_ids, dtype=np.int64))
    report = dict(schema="cat-teacher-bridge-parity-v1", passed=True, scenes=reports,
        teacher_sha256=meta["weights_sha256"], native_training_packer_noiseless_parity=True,
        physics="CPU MuJoCo, native ONNX actions only; NOT Isaac validation",
        destination_tensor_interface="Isaac-style tensors reconstructed from MuJoCo link state, reordered by name",
        sim_steps=len(observations), optimizer_steps=0, robot_actuation=False,
        label_file="same_state_labels.npz",
        label_sha256=hashlib.sha256((run/"same_state_labels.npz").read_bytes()).hexdigest(),
        native_training_source_sha256=fixture["sha256"],
        native_command_parity=command_errors,
        whole_body_distilled=False, randomized_training_mdp_ported=False)
    (run/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(report=str(run/"report.json"), **report), indent=2))


def check_training_packer(cat_repo, saved, bridge):
    """Execute the pinned upstream _get_obs itself on recorded finite inputs."""
    import jax
    import jax.numpy as jp
    import numpy as np
    import torch
    from gear_sonic.research.cat_bridge import GROUPS, GROUP_SIZES, heading_matrix
    source = cat_repo/"cat_ppo/envs/g1/env_cat.py"
    body = ast.parse(source.read_text()).body
    method = next(n for c in body if isinstance(c, ast.ClassDef)
                  for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "_get_obs")
    rotation = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == "world_to_navi_vel")
    namespace = dict(jax=jax, jp=jp)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), rotation, method],
                      type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), namespace)
    jarray = lambda t: jp.array(t.detach().numpy() if isinstance(t, torch.Tensor) else t)
    for inputs, pf, command, previous, phase, height, last in saved[::max(1, len(saved)//8)]:
        obj = SimpleNamespace(_pelvis_imu_site_id=0, _default_qpos=jarray([
            bridge.contract["default_joint_positions"][n] for n in bridge.joint_names]),
            obs_joint_ids=jp.array(bridge.obs_ids), action_joint_ids=jp.array(bridge.action_ids),
            _config=SimpleNamespace(noise_config=SimpleNamespace(level=0., scales=SimpleNamespace(
                gyro=0., gravity=0., joint_pos=0., joint_vel=0.))),
            get_gyro=lambda data, frame: jarray(inputs["gyro"][0]),
            get_local_linvel=lambda data, frame: jp.zeros(3))
        data = SimpleNamespace(qpos=jp.r_[jp.zeros(7), jarray(inputs["joint_pos"][0])],
            qvel=jp.r_[jp.zeros(6), jarray(inputs["joint_vel"][0])],
            site_xmat=jarray(inputs["pelvis_rotation"]))
        targets = jp.zeros(len(bridge.joint_names)).at[obj.action_joint_ids].set(jarray(previous[0]))
        pose = jp.eye(4).at[:3, :3].set(jarray(heading_matrix(inputs["pelvis_rotation"])[0]))
        info = dict(rng=jax.random.PRNGKey(1), phase=jarray(phase), navi2world_pose=pose,
                    command=jarray(command[0]).at[1:].add(2.), command_delay=jarray(command[0]),
                    last_act=jarray(last), motor_targets=targets, foot_height=jp.array([height]),
                    navi_torso_rpy=jp.zeros(3), gait_mask=jp.zeros(2),
                    kp_scale=jp.ones(1), kd_scale=jp.ones(1), rfi_lim_scale=jp.ones(1))
        start = 0
        for group, count in zip(GROUPS, GROUP_SIZES):
            for key, suffix in (("gf", "gf"), ("bf", "bf"), ("sdf", "df")):
                info[group+suffix+"_delay"] = jarray(pf[key][0, start:start+count])
                # Intentionally distinguish current and delayed PF/command so
                # using the wrong branch cannot accidentally pass this test.
                info[group+suffix] = info[group+suffix+"_delay"]*.17+.01
            info[group+"_pos"] = jp.zeros((count, 3))
            info[group+"_vel"] = jp.zeros((count, 3))
            start += count
        expected = namespace["_get_obs"](obj, data, info, jp.zeros(2))["state"]
        actual = bridge.pack(**inputs, **pf, command_world=command, previous_targets=previous,
            last_action=torch.tensor(last, dtype=torch.float32)[None],
            phase=torch.tensor(phase, dtype=torch.float32)[None], foot_height=torch.tensor([[height]], dtype=torch.float32))
        np.testing.assert_allclose(actual[0], np.array(expected), atol=1e-5, rtol=2e-5)


def check_native_command(cat_repo):
    """Direct source-method comparison; do not substitute our own reference math."""
    import jax.numpy as jp
    import numpy as np
    import torch
    from gear_sonic.research.cat_command import field_command, gait_step
    source = cat_repo/"cat_ppo/envs/g1/env_cat.py"
    methods = [n for c in ast.parse(source.read_text()).body if isinstance(c, ast.ClassDef)
               for n in c.body if isinstance(n, ast.FunctionDef)
               and n.name in ("compute_cmd_from_rtf", "_update_phase")]
    namespace = dict(jp=jp)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    obj = SimpleNamespace(_stop_cmd=jp.zeros(4), _stance_phase=jp.zeros(2), _gait_bound=.6)
    rng = np.random.default_rng(49)
    gf, bf = [rng.normal(size=(20, 11, 3)).astype(np.float32) for _ in range(2)]
    gf[0] = bf[0] = 0
    expected = np.stack([namespace["compute_cmd_from_rtf"](obj, g[1], g[[0, 3, 4, 5, 6]], b[[0, 3, 4, 5, 6]])
                         for g, b in zip(gf, bf)])
    actual = field_command(torch.from_numpy(gf), torch.from_numpy(bf)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
    phase_error = 0.
    for stop_value in (100, 51, 50, 20, 1, 0):
        for flag in (0., .75, 1.):
            command = np.array([flag, .3*flag, -.2*flag, 0], np.float32)
            last = np.array([1., .2, 0, 0], np.float32)
            phase = np.array([1.9, -.7], np.float32)
            info = dict(command=jp.array(command), last_command=jp.array(last), phase=jp.array(phase),
                        phase_dt=jp.array(.17593), stop_timestep=jp.array(stop_value))
            namespace["_update_phase"](obj, SimpleNamespace(info=info))
            values = gait_step(torch.tensor(command)[None], torch.tensor(last)[None], torch.tensor(phase)[None],
                               torch.tensor([stop_value]), .17593)
            for value, key in zip(values, ("command", "phase", "stop_timestep")):
                np.testing.assert_allclose(value[0], np.array(info[key]), atol=2e-6, rtol=1e-5)
                phase_error = max(phase_error, float(np.max(np.abs(value[0].numpy()-np.array(info[key])))))
    return dict(command_max_error=float(np.max(np.abs(actual-expected))), gait_max_error=phase_error,
                field_cases=20, stop_transition_cases=18)


if __name__ == "__main__":
    main()
