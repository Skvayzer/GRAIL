"""No simulator, network, optimizer steps or robot operations in these tests."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2_train import parse_args
from gear_sonic.research.training_admission import ArmReset, ChallengeAdmission, digest, artifact_path
from gear_sonic.research.training_runtime import EvaluationCollector, identical_tree, tracking_metrics
from gear_sonic.research.residual_learning import ResidualActorCritic
from test_residual_learning import packet


def targets():
    return {f"{side}_{joint}_joint": .2 for side in ("left", "right") for joint in
            ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")}


class LauncherTests(unittest.TestCase):
    def test_default_is_prepare_and_no_optimizer(self):
        args, config = parse_args([])
        self.assertFalse(args.execute)
        self.assertFalse(args.approve_optimizer)
        self.assertEqual(args.mode, "collect")
        self.assertEqual(args.wandb_mode, "disabled")
        self.assertEqual(config.horizon, 32)

    def test_approvals_and_bounds_are_explicit(self):
        for args in (["--mode", "train"], ["--approve-optimizer"], ["--iterations", "0"],
                     ["--iterations", "10001"], ["--num-envs", "17"], ["--horizon", "2", "--minibatch-size", "32"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args)
        args, _ = parse_args(["--mode", "train", "--approve-optimizer"])
        self.assertFalse(args.execute)  # approval alone must not start anything

    def test_retention_cannot_train(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--mode", "train", "--approve-optimizer", "--retention-family", "curb"])

    def test_relocated_evidence_preserves_relative_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory)/"research/runs/source/reference_sweep.npz"
            self.assertEqual(artifact_path("/old/machine/research/runs/source/reference_sweep.npz", directory), expected)
            self.assertEqual(artifact_path("research/runs/source/reference_sweep.npz", directory), expected)
            for name in ("/etc/passwd", "research/runs/../../../escape", "research/artifacts/file"):
                with self.assertRaises(ValueError):
                    artifact_path(name, directory)

    def test_optimizer_moment_comparison_and_tracking_scalars(self):
        a = {"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.ones(2)}}}
        b = {"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.zeros(2)}}}
        self.assertTrue(identical_tree(a, a))
        self.assertFalse(identical_tree(a, b))
        values = tracking_metrics(dict(iteration=1, update=dict(kl_early_stop=True, optimizer_steps=1,
            batches=[dict(loss=1., gradient_norm=3.), dict(loss=2.)])))
        self.assertEqual(values["ppo/loss"], 1.5)
        self.assertEqual(values["ppo/gradient_norm"], 3.)


class AdmissionTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)/"witness.json"
        packet_path = path.with_name("witness_17.npz")
        np.savez(packet_path, centers=np.zeros((2, 1, 3), np.float32), radii=np.ones(1, np.float32))
        report = dict(schema="grail-cat-posture-witness-v1", arm_mode="constant", initial_arm_pose_changed=True,
                      role_retention={"all_roles_retained": True}, geometric_route_found=True,
                      dynamic_feasibility_verified=False, cases=[{} for _ in range(18)],
                      witnesses=[dict(case=17, file=packet_path.name, sha256=digest(packet_path))])
        report["cases"][17] = dict(kinematic_screen_passed=True, targets=targets())
        path.write_text(json.dumps(report))
        return path, report

    def test_report_and_packet_must_match_review_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = self.fixture(directory)
            admission = ChallengeAdmission(path, digest(path))
            self.assertFalse(admission.reference_checked or admission.layout_checked)
            self.assertFalse(admission.contract["reference_arms_changed"])
            with self.assertRaisesRegex(ValueError, "checksum"):
                ChallengeAdmission(path, "0"*64)
            report["cases"][17]["targets"]["left_knee_joint"] = .1
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "14"):
                ChallengeAdmission(path, digest(path))


class ArmResetTests(unittest.TestCase):
    def fixture(self):
        t = targets()
        names = list(t)+[f"protected_{i}" for i in range(15)]
        class Robot:
            joint_names = names
            def __init__(self):
                self.data = NS(joint_pos=torch.zeros(2, 29), joint_vel=torch.zeros(2, 29),
                               soft_joint_pos_limits=torch.tensor([-1., 1.]).repeat(2, 29, 1))
            def write_joint_state_to_sim(self, pos, vel, env_ids):
                self.data.joint_pos[env_ids] = pos
                self.data.joint_vel[env_ids] = vel
        robot = Robot()
        class Motion:
            def _resample_command(self, ids):
                robot.data.joint_pos[ids] = .1
                robot.data.joint_vel[ids] = .3
        wrapper = NS(motion_command=Motion(), env=NS(device="cpu", scene={"robot": robot}))
        return wrapper, NS(layout_checked=True, targets=t)

    def test_reset_changes_only_selected_environment_arms(self):
        wrapper, admission = self.fixture()
        robot = wrapper.env.scene["robot"]
        with ArmReset(wrapper, admission) as reset:
            wrapper.motion_command._resample_command(torch.tensor([1]))
            self.assertTrue(torch.equal(robot.data.joint_pos[0], torch.zeros(29)))
            torch.testing.assert_close(robot.data.joint_pos[1, :14], torch.full((14,), .2))
            torch.testing.assert_close(robot.data.joint_pos[1, 14:], torch.full((15,), .1))
            torch.testing.assert_close(robot.data.joint_vel[1, :14], torch.zeros(14))
            torch.testing.assert_close(robot.data.joint_vel[1, 14:], torch.full((15,), .3))
            self.assertEqual(reset.rows, [[1]])
        self.assertNotIn("_resample_command", vars(wrapper.motion_command))
        wrapper.motion_command._resample_command(torch.tensor([0]))
        torch.testing.assert_close(robot.data.joint_pos[0], torch.full((29,), .1))

    def test_joint_limits_reject_and_hook_is_restored_on_failure(self):
        wrapper, admission = self.fixture()
        admission.targets[next(iter(admission.targets))] = 5.
        with self.assertRaisesRegex(ValueError, "limits"):
            with ArmReset(wrapper, admission):
                wrapper.motion_command._resample_command(torch.tensor([0]))
        self.assertNotIn("_resample_command", vars(wrapper.motion_command))
        admission.layout_checked = False
        with self.assertRaisesRegex(ValueError, "admission"):
            ArmReset(wrapper, admission)


class EvaluationTests(unittest.TestCase):
    def test_deterministic_evaluation_does_not_sample_or_produce_ppo_data(self):
        model = ResidualActorCritic(2, 7).eval()
        with torch.no_grad():
            model.obstacle.head.bias.fill_(.4)
        collector = EvaluationCollector(model, 2)
        before = torch.get_rng_state().clone()
        outputs = []
        for _ in range(2):
            value = collector.sample(packet(2), torch.zeros(2, 7))
            outputs.append(value)
            collector.outcome(value, torch.ones(2), torch.zeros(2, dtype=bool),
                              torch.zeros(2, dtype=bool), {}, None)
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertTrue(torch.count_nonzero(outputs[0]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(set(collector.finish()), {"reward"})
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
