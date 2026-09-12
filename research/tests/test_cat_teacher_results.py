"""Saved shadow pairing catches off-by-one and post-reset target substitution."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_teacher_results import audit_teacher_shadow
from replay_cat_teacher_grail import load_policy_config
from gear_sonic.research.cat_teacher import WIDTHS
from test_cat_bridge import ALL_JOINTS, contract


class PairingTests(unittest.TestCase):
    def fixture(self, directory):
        run = Path(directory)
        weights = {f"actor_{i}_{key}": np.zeros(shape, np.float32)
            for i, (a, b) in enumerate(zip(WIDTHS, WIDTHS[1:]))
            for key, shape in (("kernel", (a, b)), ("bias", (b,)))}
        np.savez(run/"weights.npz", **weights)
        actions = np.broadcast_to(np.array([.2, .4, -.2], np.float32)[:, None, None], (3, 1, 29)).copy()
        targets = np.broadcast_to(np.array([0., .1, .2], np.float32)[:, None, None], (3, 1, 12)).copy()
        last = np.broadcast_to(np.array([0., .2, .2], np.float32)[:, None, None], (3, 1, 12)).copy()
        obs = np.zeros((3, 1, 162), np.float32)
        obs[..., 52:64], obs[..., 64:76] = last, targets
        arrays = dict(cat_obs=obs, last_action=last, previous_leg_targets=targets,
            teacher_action=np.zeros_like(targets), teacher_leg_targets=targets.copy(),
            applied_targets=actions*.5, grail_actions=actions,
            action_scale=np.full_like(actions, .5), action_offset=np.zeros_like(actions),
            terminated=np.zeros((3, 1), bool), truncated=np.array([[False], [False], [True]]),
            history_ready=np.array([[False], [True], [True]]), step=np.arange(3)[:, None],
            in_domain=np.ones((3, 1, 11), bool), grail_obs__actor_obs=np.zeros((3, 1, 1, 8), np.float32))
        meta = dict(schema="grail-cat-teacher-shadow-v1", complete=True, base_backbone_unchanged=True,
            teacher_unchanged=True, robot_actuation=False, optimizer_steps=0, training_admitted=False,
            frames=3, wrapper_action_clip=20, action_joint_names=ALL_JOINTS,
            joint_names=ALL_JOINTS, contract=contract(), packet_file="cat_teacher_shadow.npz",
            teacher_sha256=hashlib.sha256((run/"weights.npz").read_bytes()).hexdigest())
        return run, arrays, meta

    def save(self, run, arrays, meta):
        np.savez(run/"cat_teacher_shadow.npz", **arrays)
        meta["packet_sha256"] = hashlib.sha256((run/"cat_teacher_shadow.npz").read_bytes()).hexdigest()
        (run/"cat_teacher_shadow.json").write_text(json.dumps(meta))

    def test_valid_pair_and_shifted_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, arrays, meta = self.fixture(tmp)
            self.save(run, arrays, meta)
            self.assertTrue(audit_teacher_shadow(run, run)["passed"])
            arrays["previous_leg_targets"][2] = arrays["previous_leg_targets"][1]
            arrays["cat_obs"][..., 64:76] = arrays["previous_leg_targets"]
            self.save(run, arrays, meta)
            with self.assertRaises(AssertionError):
                audit_teacher_shadow(run, run)

    def test_post_reset_targets_are_not_accepted_as_executed_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, arrays, meta = self.fixture(tmp)
            arrays["applied_targets"][-1] = 0
            self.save(run, arrays, meta)
            with self.assertRaises(AssertionError):
                audit_teacher_shadow(run, run)


class PolicyConfigTests(unittest.TestCase):
    def test_explicit_recovery_requires_matching_actor_and_config_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, source = root/"old", root/"new"
            target.mkdir()
            source.mkdir()
            config = source/"cat_teacher_policy_config.json"
            config.write_text('{"env_config": {}, "algo_config": {}}')
            meta = dict(base_backbone_sha256="frozen-actor", action_joint_names=list(ALL_JOINTS),
                        wrapper_action_clip=20, complete=True, base_backbone_unchanged=True,
                        policy_config_file=config.name,
                        policy_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            (source/"cat_teacher_shadow.json").write_text(json.dumps(meta))
            value, path, digest = load_policy_config(target, meta, source)
            self.assertEqual(value, {"env_config": {}, "algo_config": {}})
            self.assertEqual(path, str(config))
            self.assertEqual(digest, meta["policy_config_sha256"])
            with self.assertRaisesRegex(ValueError, "base_backbone_sha256"):
                load_policy_config(target, {**meta, "base_backbone_sha256": "different"}, source)
            with self.assertRaisesRegex(ValueError, "action_joint_names"):
                load_policy_config(target, {**meta, "action_joint_names": []}, source)
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError, "config changed"):
                load_policy_config(target, meta, source)


if __name__ == "__main__":
    unittest.main()
