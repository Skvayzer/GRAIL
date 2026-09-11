"""Reference-conditioned learner state with explicit, resettable physical history.

This is a PRIVILEGED M2 teacher interface, not the later deployable student.
It reads simulator state/motion tables without invoking observation managers,
advancing motion commands, consuming RNG or changing the frozen actor's history.
"""
import math
import torch

from .obstacle_observation import rotate_vectors, yaw_rotation


def environment_selection(mask, count):
    if mask is None:
        return slice(None)
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (count,) or not mask.any():
        raise ValueError("Nonempty boolean environment selection required")
    return mask


def quaternion_matrix(q):
    if (q.shape[-1] != 4 or not torch.isfinite(q).all()
            or not torch.allclose(q.norm(dim=-1), torch.ones_like(q[..., 0]), atol=1e-4, rtol=0)):
        raise ValueError("Normalized finite WXYZ rotations required")
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def future_steps(motion, phase_offset=0):
    if type(phase_offset) is not int or phase_offset not in (0, 1):
        raise ValueError("Only current or one-step-ahead reference queries supported")
    return torch.minimum(motion.future_time_steps_init+motion.time_steps[:, None]
        + motion.motion_start_time_steps[:, None]+phase_offset, motion.motion_num_steps[:, None]-1).flatten().long()


class PhysicalHistory:
    """Oldest-to-newest history; reset fills every slot with the initial sample."""
    def __init__(self, initial, length=10):
        if initial.ndim != 2 or not 1 <= length <= 32 or not torch.isfinite(initial).all():
            raise ValueError("Finite B,D state and bounded history required")
        self.data = initial[:, None].expand(-1, length, -1).clone()

    def candidate(self, current, reset=None, env_mask=None):
        history = self.data[environment_selection(env_mask, len(self.data))]
        if current.shape != (len(history), self.data.shape[-1]) or not torch.isfinite(current).all():
            raise ValueError("Physical state shape/finite contract changed")
        result = torch.cat((history[:, 1:], current[:, None]), 1)
        if reset is not None:
            if reset.shape != (len(history),) or reset.dtype != torch.bool:
                raise ValueError("Boolean per-environment reset mask required")
            result = torch.where(reset[:, None, None], current[:, None], result)
        return result

    def commit(self, current, reset=None):
        self.data = self.candidate(current, reset).detach().clone()


class LearnerStateSampler:
    def __init__(self, wrapper, history_length=10):
        self.wrapper, self.env, self.motion = wrapper, wrapper.env, wrapper.motion_command
        self.robot = self.env.scene["robot"]
        self.anchor = self.robot.body_names.index(self.motion.cfg.anchor_body)
        self.joints = list(self.robot.joint_names)
        if len(self.joints) != 29 or len(set(self.joints)) != 29:
            raise ValueError("This research embodiment requires 29 uniquely named body joints")
        self.reference_names = list(self.env.cfg.isaaclab_to_mujoco_mapping["isaaclab_joints"])
        self.bodies = list(self.motion.cfg.body_names)
        self.reference_indices = [self.reference_names.index(n) for n in self.bodies]
        self.history = PhysicalHistory(self.physical(), history_length)
        self.frames = int(self.motion.num_future_frames)

    def physical(self, env_mask=None):
        data = self.robot.data
        selected = environment_selection(env_mask, self.env.num_envs)
        # Use the named pelvis, not the articulation root/FAST-LIO body frame.
        rotation = quaternion_matrix(data.body_quat_w[selected, self.anchor]).transpose(1, 2)
        gravity = torch.zeros_like(data.body_pos_w[selected, self.anchor])
        gravity[:, 2] = -1
        values = [
            (data.joint_pos[selected]-data.default_joint_pos[selected])/math.pi,
            data.joint_vel[selected]/10.,
            rotate_vectors(rotation, data.body_lin_vel_w[selected, self.anchor])/5.,
            rotate_vectors(rotation, data.body_ang_vel_w[selected, self.anchor])/5.,
            rotate_vectors(rotation, gravity),
            self.env.action_manager.action[selected]/20.,
        ]
        result = torch.cat(values, -1)
        if result.shape != (len(rotation), 96) or not torch.isfinite(result).all():
            raise ValueError("Invalid physical learner state (no sanitizing fallback)")
        return result

    def reference(self, phase_offset=0, env_mask=None):
        motion, data = self.motion, self.robot.data
        selected = environment_selection(env_mask, self.env.num_envs)
        ids = motion.future_motion_ids.reshape(self.env.num_envs, self.frames)[selected].flatten()
        steps = future_steps(motion, phase_offset)
        if phase_offset == 0 and not torch.equal(steps, motion.future_time_steps):
            raise ValueError("Learner reference indexing differs from GRAIL")
        steps = steps.reshape(self.env.num_envs, self.frames)[selected].flatten()
        pos = motion.motion_lib.get_body_pos_w_full(ids, steps)
        quat = motion.motion_lib.get_body_quat_w_full(ids, steps)
        if pos.shape[1] != len(self.reference_names):
            raise ValueError("Reference body-name map differs from motion data")
        b, f, k = len(ids)//self.frames, self.frames, len(self.bodies)
        pos = pos[:, self.reference_indices].reshape(b, f*k, 3)
        quat = quat[:, self.reference_indices].reshape(b, f*k, 4)
        root = data.body_pos_w[selected, self.anchor]-self.env.scene.env_origins[selected]
        yaw_inverse = yaw_rotation(data.body_quat_w[selected, self.anchor]).transpose(1, 2)
        delta = rotate_vectors(yaw_inverse, pos-root[:, None])/2.
        matrices = quaternion_matrix(quat)
        # Two matrix columns, explicitly rotated to avoid actor TF32 precision.
        first = rotate_vectors(yaw_inverse, matrices[..., :, 0])
        second = rotate_vectors(yaw_inverse, matrices[..., :, 1])
        features = torch.cat((delta, first, second), -1).flatten(1)
        phase = ((motion.time_steps+motion.motion_start_time_steps+phase_offset).float()
                 / (motion.motion_num_steps-1).clamp_min(1)).clamp(0, 1)
        return torch.cat((features, phase[selected, None]), -1)

    def state(self, *, phase_offset=0, next_physical=None, reset=None, env_mask=None):
        selected = environment_selection(env_mask, self.env.num_envs)
        history = self.history.data[selected] if next_physical is None else self.history.candidate(next_physical, reset, env_mask)
        result = torch.cat((history.flatten(1), self.reference(phase_offset, env_mask)), -1)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite reference-conditioned state")
        return result

    def manifest(self):
        return dict(schema="grail-cat-privileged-state-v1", deployable=False,
            body_joint_order=self.joints, reference_body_order=self.bodies,
            history_length=self.history.data.shape[1], physical_dim=96,
            history="oldest to newest; repeat initial sample at each per-env reset",
            physical_features=["joint_pos_minus_default / pi", "joint_vel / 10",
                "pelvis_linear_velocity_in_pelvis / 5", "pelvis_angular_velocity_in_pelvis / 5",
                "gravity_in_pelvis", "last_executed_action / 20"],
            future_frames=self.frames, future_offsets_steps=self.motion.future_time_steps_init[0].tolist(),
            reference="future named-body position relative to live pelvis / 2; yaw-relative matrix columns 0 and 1; normalized clip phase",
            future_boundary="clamp to final valid reference frame, matching GRAIL",
            terminal_query="next physical state with old history; query reference at +1 WITHOUT advancing commands",
            state_dim=self.history.data.shape[1]*96+self.frames*len(self.bodies)*9+1)
