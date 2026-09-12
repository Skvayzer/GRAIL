"""Recipe and frozen teacher contract tests; no simulator/network/updates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_style_recipe import recipe
from gear_sonic.research.cat_teacher import CatTeacher, WIDTHS, JOINTS


class RecipeTests(unittest.TestCase):
    def test_only_hardware_fields_change_and_missing_scenes_are_not_replaced(self):
        config = dict(policy_config=dict(num_envs=65536, batch_size=2048,
            max_devices_per_host=8, entropy_cost=.003, learning_rate=.0003, unroll_length=32,
            num_minibatches=64, discounting=.98),
            env_config=dict(pf_config=dict(paths=["data/assets/RandObs/D8G2L3O2S13"])))
        before = json.dumps(config, sort_keys=True)
        with tempfile.TemporaryDirectory() as d:
            out = recipe(config, d, 512)
        self.assertEqual(json.dumps(config, sort_keys=True), before)
        policy = out["configuration"]["policy_config"]
        self.assertEqual(policy["num_envs"], 512)
        self.assertEqual(policy["batch_size"], 16)
        changed = {"num_envs", "batch_size", "max_devices_per_host"}
        self.assertEqual({k: v for k, v in policy.items() if k not in changed},
                         {k: v for k, v in config["policy_config"].items() if k not in changed})
        self.assertFalse(out["training_started"] or out["grail_teacher_bridge_ready"])
        self.assertFalse(out["scene_inventory"][0]["available"])


class TeacherTests(unittest.TestCase):
    def test_exact_dimensions_joint_order_zero_fixture_and_frozen_weights(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"weights.npz"
            values = {f"actor_{i}_{k}": np.zeros(shape, np.float32)
                for i, (a, b) in enumerate(zip(WIDTHS, WIDTHS[1:]))
                for k, shape in (("kernel", (a, b)), ("bias", (b,)))}
            np.savez(p, **values)
            before = torch.get_rng_state().clone()
            model = CatTeacher(p)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertFalse(model.training)
            out = model(torch.ones(3, 162))
            self.assertEqual(out.shape, (3, 12))
            self.assertEqual(int(torch.count_nonzero(out)), 0)
            self.assertEqual(len(JOINTS), 12)
            self.assertEqual(JOINTS[0], "left_hip_pitch_joint")
            self.assertEqual(JOINTS[6], "right_hip_pitch_joint")
            for bad in (torch.zeros(3, 2221), torch.full((3, 162), float("nan"))):
                with self.assertRaises(ValueError):
                    model(bad)


if __name__ == "__main__":
    unittest.main()
