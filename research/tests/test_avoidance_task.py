"""Counterfactual reward/termination checks without training or robot actions."""
from pathlib import Path
import sys
import unittest

import torch
from omegaconf import OmegaConf
from hydra.core.override_parser.overrides_parser import OverridesParser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluation_command
from gear_sonic.research.avoidance_task import AvoidanceSpec, avoidance_signals, foot_world_error, failure_event_rate, LOWER, FEET
from gear_sonic.research.avoidance_config import avoidance_overrides


def signals(gaps, links=("arm", "leg"), peaks=None, previous=None, current=None):
    gaps = torch.tensor(gaps, dtype=torch.float32).reshape(1, -1)
    return avoidance_signals(gaps, links, torch.zeros(1, len(set(links))) if peaks is None else torch.tensor([peaks]),
        torch.zeros(1, 3) if previous is None else torch.tensor([previous], dtype=torch.float32),
        torch.zeros(1, 3) if current is None else torch.tensor([current], dtype=torch.float32),
        torch.tensor([[1., 0., 0.]]), .02)


class AvoidanceMathTests(unittest.TestCase):
    def test_clearance_incentivizes_arm_adaptation_and_counts_links_not_probes(self):
        clear = signals([.2, .2])
        close = signals([.04, .2])
        touching = signals([0., .2])
        self.assertEqual(float(clear["clearance_penalty"]), 0.)
        self.assertGreater(float(touching["clearance_penalty"]), float(close["clearance_penalty"]))
        repeated = signals([.04, .04, .04, .2], links=("arm", "arm", "arm", "leg"))
        torch.testing.assert_close(close["clearance_penalty"], repeated["clearance_penalty"], atol=0, rtol=0)

    def test_foot_cat_contacts_are_not_exempted_as_support(self):
        result = signals([.2, .2], links=("left_ankle_roll_link", "torso_link"), peaks=[21., 0.])
        self.assertTrue(bool(result["contact_failure"]))
        self.assertEqual(float(result["contact_penalty"]), .5)
        self.assertFalse(bool(signals([-.02, .2])["contact_failure"]))  # geometry cover != actual contact

    def test_progress_is_root_only_and_telescopes(self):
        forward = signals([.2, .2], current=[.1, 0., 0.])
        back = signals([.2, .2], previous=[.1, 0., 0.], current=[0., 0., 0.])
        torch.testing.assert_close(forward["root_progress"]+back["root_progress"], torch.zeros(1), atol=0, rtol=0)
        # Changing only clearance cannot generate locomotion progress.
        self.assertEqual(float(signals([.01, .2])["root_progress"]), 0.)
        self.assertEqual(float(signals([.2, .2])["root_progress"]), 0.)

    def test_world_foot_check_catches_translated_support_trajectory(self):
        reference = torch.tensor([[[0., .1, 1.], [0., -.1, 1.]]])
        moved = reference+torch.tensor([.25, 0., 0.])
        self.assertAlmostEqual(float(foot_world_error(reference, moved)), .25)
        self.assertGreater(float(foot_world_error(reference, moved)), AvoidanceSpec().foot_world_failure_m)

    def test_terminal_event_cost_not_scaled_by_action_frequency(self):
        flags = torch.tensor([False, True])
        for dt in (.01, .02, .04):
            torch.testing.assert_close(failure_event_rate(flags, dt)*dt, torch.tensor([0., 1.]), atol=0, rtol=0)

    def test_invalid_contact_or_geometry_is_not_silently_free(self):
        for gaps, peaks in (([float("nan"), 1.], [0., 0.]), ([1., 1.], [-1., 0.]), ([1.], [0., 0.])):
            with self.assertRaises(ValueError):
                signals(gaps, peaks=peaks)
        with self.assertRaises(ValueError):
            AvoidanceSpec(contact_failure_threshold_n=.5)


class ConfigTests(unittest.TestCase):
    def test_overlay_leaves_actor_inputs_and_retention_source_unchanged(self):
        root = Path(__file__).resolve().parents[2]
        config_path = root/"research/artifacts/checkpoint/SONIC/models/terrain_release/config.yaml"
        if not config_path.is_file():
            self.skipTest("Pinned terrain checkpoint config not downloaded")
        source = OmegaConf.load(config_path)
        before = OmegaConf.to_container(source, resolve=False)
        changed = OmegaConf.create(before)
        for key, value in avoidance_overrides().items():
            self.assertTrue(key.startswith(("manager_env.rewards.", "manager_env.terminations.")))
            OmegaConf.update(changed, key, value, merge=False, force_add=True)
        self.assertEqual(OmegaConf.to_container(source, resolve=False), before)
        self.assertEqual(changed.manager_env.observations, source.manager_env.observations)
        self.assertEqual(changed.manager_env.commands, source.manager_env.commands)
        self.assertEqual(changed.manager_env.actions, source.manager_env.actions)
        self.assertIsNone(changed.manager_env.rewards.tracking_vr_5point_local)
        self.assertEqual(list(changed.manager_env.terminations.ee_body_pos.params.body_names), list(FEET))
        for name in ("tracking_relative_body_pos", "tracking_relative_body_ori", "tracking_body_linvel", "tracking_body_angvel"):
            self.assertEqual(list(changed.manager_env.rewards[name].params.body_names), list(LOWER))
        for name in ("action_rate_l2", "joint_limit", "feet_acc", "tracking_anchor_pos", "tracking_anchor_ori"):
            self.assertEqual(changed.manager_env.rewards[name], source.manager_env.rewards[name])

    def test_task_profile_is_opt_in_no_training_and_hydra_parseable(self):
        options = dict(cat_scene=Path("/cat"), layout_audit=True, residual_preflight=True)
        ordinary = evaluation_command(Path("/run"), Path("/data"), "fixture", 1, **options)
        self.assertNotIn("++research_avoidance_task=true", ordinary)
        command = evaluation_command(Path("/run"), Path("/data"), "fixture", 1, avoidance_task=True, **options)
        self.assertIn("++research_avoidance_task=true", command)
        self.assertNotIn("train_agent", " ".join(command))
        OverridesParser.create().parse_overrides([arg for arg in command if arg.startswith("++")])
        with self.assertRaises(ValueError):
            evaluation_command(Path("/run"), Path("/data"), "fixture", 1, avoidance_task=True)


if __name__ == "__main__":
    unittest.main()
