"""Live Isaac same-state CAT labels; the released GRAIL actor retains control.

Records original policy inputs and ACTUAL pre-reset joint targets. No optimizer,
SDK, robot connection or teacher action application. Stair samples are diagnostic:
CAT's original fields describe clutter only, not combined stair traversability.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .cat_bridge import (AppliedTargetHistory, CatFieldSampler, CatObservationBridge, finite_shape,
                         indices, normalize_fields)
from .cat_command import delayed_sites, field_command, gait_step
from .cat_teacher import CatTeacher
from .observation_shadow import state_hash
from .pre_reset_capture import PreResetCapture


def config_json(config):
    def convert(value):
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported resolved configuration object: {type(value).__name__}")
    return json.dumps(config, indent=2, default=convert)+"\n"


class CatTeacherShadow:
    def __init__(self, wrapper, cat_audit, policy, output, contract_path, teacher_dir):
        self.wrapper, self.env, self.cat, self.policy = wrapper, wrapper.env, cat_audit, policy
        if policy.training or wrapper.action_transform_module is not None:
            raise ValueError("Shadow needs released joint-action GRAIL evaluation")
        self.robot = self.env.scene["robot"]
        self.term = self.env.action_manager.get_term("joint_pos")
        self.output = Path(output)
        contract = json.loads(Path(contract_path).read_text())
        self.bridge = CatObservationBridge(contract, self.robot.joint_names, self.robot.body_names)
        self.sampler = CatFieldSampler(self.cat.fields.fields,
            self.cat.fields.meta["origin_corner"], self.cat.fields.meta["resolution"])
        self.term_leg_ids = indices(self.term._joint_names, contract["action_joints"])
        self.term_robot_ids = indices(self.robot.joint_names, self.term._joint_names)
        meta = json.loads((Path(teacher_dir)/"verification.json").read_text())
        weights = Path(teacher_dir)/"weights.npz"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != meta["weights_sha256"]:
            raise ValueError("Teacher weights differ from verified export")
        self.teacher = CatTeacher(weights).to(self.env.device)
        self.initial_hash = state_hash(policy.actor_module)
        self.teacher_hash = state_hash(self.teacher)
        # First sample has no executed history. It is marked explicitly and
        # excluded from paired-history claims, not invented as a prior action.
        initial = self.robot.data.joint_pos[:, self.bridge.action_ids]
        self.history = AppliedTargetHistory(initial, contract["action_scale"])
        self.ready = torch.zeros(self.env.num_envs, device=self.env.device, dtype=torch.bool)
        self.steps = torch.zeros_like(self.ready, dtype=torch.long)
        self.phase = initial.new_tensor([0., math.pi]).expand(self.env.num_envs, -1).clone()
        self.command = initial.new_zeros((self.env.num_envs, 4))
        self.stop = torch.full_like(self.steps, 100)
        self.cached_root = self.cached_rotation = None
        # Deterministic teacher-query phase in the released 1.3..1.5Hz range.
        # Domain-randomized training reset/noise are NOT claimed by this recorder.
        self.dt = float(self.env.step_dt)
        if abs(self.dt-.02) > 1e-8:
            raise ValueError("CAT teacher shadow currently requires the native 50Hz control rate")
        self.phase_dt = 2*math.pi*self.dt*1.4
        self.rows, self.current = [], None
        self.tap = PreResetCapture(self.env, self._pre_reset)
        self.report = dict(schema="grail-cat-teacher-shadow-v1", motion_enabled=False,
            physics_action_source="unchanged released GRAIL actor only", optimizer_steps=0,
            original_observations_changed=False, original_rewards_changed=False,
            original_terminations_changed=False, robot_actuation=False,
            joint_names=list(self.robot.joint_names), body_names=list(self.robot.body_names),
            action_joint_names=list(self.term._joint_names), contract=contract,
            wrapper_action_clip=wrapper.config.get("action_clip_value", None),
            contract_sha256=hashlib.sha256(Path(contract_path).read_bytes()).hexdigest(),
            source_fields=self.cat.fields.meta, source_scene=str(self.cat.fields.directory),
            placement=dict(translation=self.cat.placement.translation, yaw=self.cat.placement.yaw),
            teacher_sha256=meta["weights_sha256"], base_backbone_sha256=self.initial_hash,
            teacher_settings=dict(frequency_hz=1.4, foot_height_m=.07, noise=False,
                odometry_period_steps=5, reset_phase="left first", control_dt=self.dt),
            training_admitted=False,
            limitation="Original CAT clutter fields only; terrain-relative teacher semantics and whole-body training pending")
        # Save the RESOLVED policy dimensions after Isaac assembled observations;
        # the released YAML contains empty dimension dictionaries before setup.
        config = dict(env_config=OmegaConf.to_container(policy.env_config, resolve=True),
                      algo_config=OmegaConf.to_container(policy.algo_config, resolve=True))
        config_path = self.output.with_name("cat_teacher_policy_config.json")
        config_path.write_text(config_json(config))
        self.report.update(policy_config_file=config_path.name,
                           policy_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())

    def __enter__(self):
        self.tap.__enter__()
        return self

    def _fields(self, sites):
        local = self.cat.placement.to_local(sites)
        packet = self.sampler.sample(local)
        for key in ("gf", "bf"):
            packet[key] = self.cat.placement.vectors_to_world(packet[key])
        return packet

    @torch.no_grad()
    def sample(self, buffered_obs, live_actions):
        if self.current is not None or len(self.rows) >= 500:
            raise ValueError("Each teacher sample must have exactly one bounded physics outcome")
        # Clone everything being saved BEFORE the next physics step mutates it.
        state = self.bridge.read_articulation(self.robot.data, self.env.scene.env_origins)
        sites = state.pop("sites")
        root = self.robot.data.body_pos_w[:, self.bridge.pelvis_id]-self.env.scene.env_origins
        rotation = state["pelvis_rotation"]
        first = self.steps == 0
        if self.cached_root is None:
            self.cached_root, self.cached_rotation = root.clone(), rotation.clone()
        refresh = first | ((self.steps-1) % 5 == 0)
        self.cached_root[refresh], self.cached_rotation[refresh] = root[refresh], rotation[refresh]
        delayed = delayed_sites(sites, root, rotation, self.cached_root, self.cached_rotation)
        now, delayed_pf = self._fields(sites), self._fields(delayed)
        # Native reset computes its command from normalized fields; step computes
        # the current command before normalization, then updates gait.
        reset_gf, reset_bf = normalize_fields(now["gf"], now["bf"], torch.ones_like(self.steps))
        raw_command = field_command(now["gf"], now["bf"])
        cmd, phase, stop = gait_step(raw_command, self.command, self.phase, self.stop, self.phase_dt)
        self.command = torch.where(first[:, None], field_command(reset_gf, reset_bf), cmd)
        self.phase = torch.where(first[:, None], self.phase, phase)
        self.stop = torch.where(first, self.stop, stop)
        gf, bf = normalize_fields(delayed_pf["gf"], delayed_pf["bf"], self.command[:, 0])
        gf = torch.where(first[:, None, None], reset_gf, gf)
        bf = torch.where(first[:, None, None], reset_bf, bf)
        command_delay = field_command(gf, bf)
        command_delay = torch.where(first[:, None], self.command, command_delay)
        command_delay[:, 0] = self.command[:, 0]
        obs = self.bridge.pack(**state, last_action=self.history.last_action,
            previous_targets=self.history.targets, command_world=command_delay,
            foot_height=root.new_full((len(root), 1), .07), phase=self.phase,
            gf=gf, bf=bf, sdf=delayed_pf["sdf"])
        # Preserve the caller's math mode; source checkpoint parity used FP32.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            action = self.teacher(obs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        labels = self.bridge.leg_targets(action, self.history.targets)
        values = dict(cat_obs=obs, teacher_action=action, teacher_leg_targets=labels,
            previous_leg_targets=self.history.targets, last_action=self.history.last_action,
            history_ready=self.ready, in_domain=delayed_pf["in_domain"],
            grail_actions=live_actions, joint_pos=state["joint_pos"], joint_vel=state["joint_vel"],
            body_pos=self.robot.data.body_pos_w-self.env.scene.env_origins[:, None],
            body_quat=self.robot.data.body_quat_w, sites=sites, step=self.steps,
            reference_step=self.wrapper.motion_command.time_steps,
            action_scale=torch.broadcast_to(torch.as_tensor(self.term._scale, device=root.device), live_actions.shape),
            action_offset=torch.broadcast_to(torch.as_tensor(self.term._offset, device=root.device), live_actions.shape))
        for key, value in buffered_obs.items():
            values["grail_obs__"+key] = value
        self.current = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}

    @torch.no_grad()
    def _pre_reset(self, reward):
        if self.current is None:
            raise ValueError("Missing pre-action teacher sample")
        # Read what Isaac actually sent to PD, after wrapper clipping/scale/offset
        # and BEFORE reset callbacks can replace targets for the next episode.
        targets = self.term.processed_actions.detach().clone()
        finite_shape(targets, (self.env.num_envs, len(self.term._joint_names)))
        actuator_targets = self.robot.data.joint_pos_target[:, self.term_robot_ids]
        if not torch.allclose(targets, actuator_targets, atol=1e-6, rtol=0):
            raise ValueError("Processed actions differ from actual actuator position targets")
        return dict(applied_targets=targets, terminated=self.env.reset_terminated.clone(),
                    truncated=self.env.reset_time_outs.clone())

    @torch.no_grad()
    def outcome(self, dones):
        result = self.tap.take()
        done = dones.reshape(-1).bool()
        if not torch.equal(done, result["terminated"] | result["truncated"]):
            raise ValueError("Captured reset flags differ from wrapper outcome")
        for key, value in result.items():
            self.current[key] = value.cpu().numpy().copy()
        self.rows.append(self.current)
        self.current = None
        if len(self.rows) % 50 == 0:
            print(f"CAT_TEACHER_PROGRESS frames={len(self.rows)} history_paired={int(self.ready.sum())}", flush=True)
        targets = result["applied_targets"][:, self.term_leg_ids].clone()
        targets[done] = self.robot.data.joint_pos[done][:, self.bridge.action_ids]
        self.history.commit(targets, reset=done)
        self.ready = ~done
        self.steps = torch.where(done, 0, self.steps+1)
        self.phase[done] = self.phase.new_tensor([0., math.pi])
        self.command[done] = 0
        self.stop[done] = 100

    def __exit__(self, kind, error, traceback):
        self.tap.__exit__(kind, error, traceback)
        unchanged = state_hash(self.policy.actor_module) == self.initial_hash
        teacher_unchanged = state_hash(self.teacher) == self.teacher_hash
        complete = bool(self.rows and self.current is None and unchanged and teacher_unchanged
                        and error is None and self.tap.calls == len(self.rows))
        if self.rows:
            arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
            path = self.output.with_suffix(".npz")
            np.savez_compressed(path, **arrays)
            self.report.update(packet_file=path.name, packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                all_sites_in_domain_frames=int(arrays["in_domain"].all(-1).sum()),
                paired_history_frames=int(arrays["history_ready"].sum()))
        self.report.update(complete=complete, frames=len(self.rows), base_backbone_unchanged=unchanged,
                           teacher_unchanged=teacher_unchanged, error=str(error) if error else None)
        self.output.write_text(json.dumps(self.report, indent=2, allow_nan=False)+"\n")
        print(f"CAT_TEACHER_SHADOW {self.output} complete={complete}", flush=True)
        if error is None and not complete:
            raise ValueError("Incomplete teacher shadow record")
        return False
