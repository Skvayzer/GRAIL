"""Read-only state/terminal-capture checks; no simulator or optimizer needed."""
import math
from types import SimpleNamespace as NS
import unittest

import torch

from gear_sonic.research.learner_state import PhysicalHistory, future_steps, quaternion_matrix, LearnerStateSampler
from gear_sonic.research.pre_reset_capture import PreResetCapture


class HistoryTests(unittest.TestCase):
    def test_candidate_is_read_only_and_reset_is_per_environment(self):
        history = PhysicalHistory(torch.tensor([[1., 2.], [3., 4.]]), 3)
        before = history.data.clone()
        current = torch.tensor([[5., 6.], [7., 8.]])
        reset = torch.tensor([False, True])
        expected = torch.tensor([[[1., 2.], [1., 2.], [5., 6.]], [[7., 8.], [7., 8.], [7., 8.]]])
        torch.testing.assert_close(history.candidate(current, reset), expected, rtol=0, atol=0)
        torch.testing.assert_close(history.data, before, rtol=0, atol=0)
        history.commit(current, reset)
        current.zero_()
        torch.testing.assert_close(history.data, expected, rtol=0, atol=0)

    def test_invalid_history_rejected(self):
        with self.assertRaises(ValueError):
            PhysicalHistory(torch.tensor([[float("nan")]]))
        history = PhysicalHistory(torch.ones(2, 3))
        with self.assertRaises(ValueError):
            history.commit(torch.ones(2, 3), torch.tensor([1, 0]))
        with self.assertRaises(ValueError):
            history.candidate(torch.ones(3, 3))

    def test_future_cursor_query_clamps_without_advancing(self):
        motion = NS(future_time_steps_init=torch.tensor([[0, 2, 4], [0, 2, 4]]),
            time_steps=torch.tensor([1, 3]), motion_start_time_steps=torch.tensor([2, 0]),
            motion_num_steps=torch.tensor([7, 5]))
        torch.testing.assert_close(future_steps(motion), torch.tensor([3, 5, 6, 3, 4, 4]))
        torch.testing.assert_close(future_steps(motion, 1), torch.tensor([4, 6, 6, 4, 4, 4]))
        torch.testing.assert_close(motion.time_steps, torch.tensor([1, 3]))
        for offset in (-1, 2, True, 1.):
            with self.assertRaises(ValueError):
                future_steps(motion, offset)

    def test_rotations_are_wxyz_and_sign_invariant(self):
        q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]])
        rotation = quaternion_matrix(q)
        torch.testing.assert_close(rotation, torch.tensor([[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(rotation, quaternion_matrix(-q), atol=0, rtol=0)
        with self.assertRaises(ValueError):
            quaternion_matrix(q*2)


class StateTests(unittest.TestCase):
    def test_environment_origins_cancel_and_reference_query_is_read_only(self):
        origins = torch.tensor([[0., 0., 0.], [8., -6., 0.]])
        local_pos = torch.tensor([[[0., 0., 1.], [.1, .2, 1.3]]]).repeat(2, 1, 1)
        quat = torch.tensor([[[1., 0., 0., 0.]]]).expand(2, 2, 4).clone()
        data = NS(body_pos_w=local_pos+origins[:, None], body_quat_w=quat,
            joint_pos=torch.zeros(2, 29), default_joint_pos=torch.zeros(2, 29), joint_vel=torch.zeros(2, 29),
            body_lin_vel_w=torch.zeros(2, 2, 3), body_ang_vel_w=torch.zeros(2, 2, 3))
        robot = NS(body_names=["pelvis", "hand"], joint_names=[f"joint_{i}" for i in range(29)], data=data)
        class Scene(dict):
            env_origins = origins
        class Library:
            def get_body_pos_w_full(self, ids, steps):
                return local_pos[:1].expand(len(ids), -1, -1)+steps[:, None, None]*.01
            def get_body_quat_w_full(self, ids, steps):
                return quat[:1].expand(len(ids), -1, -1)
        motion = NS(cfg=NS(anchor_body="pelvis", body_names=["pelvis", "hand"]),
            future_motion_ids=torch.zeros(4, dtype=torch.long), num_future_frames=2,
            future_time_steps_init=torch.tensor([[0, 2], [0, 2]]), time_steps=torch.ones(2, dtype=torch.long),
            motion_start_time_steps=torch.zeros(2, dtype=torch.long), motion_num_steps=torch.full((2,), 20),
            motion_lib=Library())
        motion.future_time_steps = future_steps(motion)
        env = NS(num_envs=2, scene=Scene(robot=robot), action_manager=NS(action=torch.zeros(2, 29)),
            cfg=NS(isaaclab_to_mujoco_mapping={"isaaclab_joints": ["pelvis", "hand"]}))
        sampler = LearnerStateSampler(NS(env=env, motion_command=motion), history_length=2)
        before_rng = torch.get_rng_state().clone()
        initial = sampler.state()
        self.assertEqual(initial.shape, (2, sampler.manifest()["state_dim"]))
        torch.testing.assert_close(initial[0], initial[1], rtol=0, atol=1e-6)
        next_state = sampler.state(phase_offset=1, next_physical=sampler.physical())
        torch.testing.assert_close(sampler.state(), initial, rtol=0, atol=0)
        self.assertFalse(torch.equal(initial, next_state))
        motion.time_steps += 1
        motion.future_time_steps = future_steps(motion)
        sampler.history.commit(sampler.physical())
        torch.testing.assert_close(sampler.state(), next_state, rtol=0, atol=0)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        class Rewards:
            def __init__(self):
                self.calls = 0
            def compute(self, dt):
                self.calls += 1
                return torch.tensor([dt])
        self.env = NS(reward_manager=Rewards(), position=torch.tensor([4.]))

    def test_captures_before_reset_and_restores_original_method(self):
        manager = self.env.reward_manager
        with PreResetCapture(self.env, lambda reward: dict(position=self.env.position.clone(), reward=reward.clone())) as tap:
            reward = manager.compute(dt=.02)
            self.env.position.zero_()  # simulated reset, after reward
            sample = tap.take()
            self.assertEqual(sample["position"].item(), 4.)
            self.assertTrue(torch.equal(sample["reward"], reward))
            self.assertEqual(tap.calls, 1)
            with self.assertRaisesRegex(ValueError, "Missing"):
                tap.take()
        self.assertNotIn("compute", vars(manager))
        self.assertEqual(manager.calls, 1)

    def test_unconsumed_capture_is_rejected_before_reward_runs_again(self):
        with PreResetCapture(self.env, lambda reward: reward.clone()) as tap:
            self.env.reward_manager.compute(.02)
            with self.assertRaisesRegex(ValueError, "not consumed"):
                self.env.reward_manager.compute(.02)
            self.assertEqual(self.env.reward_manager.calls, 1)
            tap.take()

    def test_exception_restores_method_and_existing_owner_is_preserved(self):
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with PreResetCapture(self.env, lambda reward: reward):
                raise RuntimeError("fixture")
        self.assertNotIn("compute", vars(self.env.reward_manager))
        sentinel = lambda dt: None
        self.env.reward_manager.compute = sentinel
        with self.assertRaisesRegex(ValueError, "override"):
            with PreResetCapture(self.env, lambda reward: reward):
                pass
        self.assertIs(self.env.reward_manager.compute, sentinel)

    def test_none_callback_cannot_masquerade_as_capture(self):
        with PreResetCapture(self.env, lambda reward: None):
            with self.assertRaisesRegex(ValueError, "explicit sample"):
                self.env.reward_manager.compute(.02)


if __name__ == "__main__":
    unittest.main()
