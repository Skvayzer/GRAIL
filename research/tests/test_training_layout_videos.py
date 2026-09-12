from pathlib import Path
import hashlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_parallel_bank import recipes
from render_cat_training_layouts import FAMILIES, load_scene, select_scenes


class TrainingLayoutVideoTests(unittest.TestCase):
    def test_fixed_selection_preserves_split_and_missing_low_seed_zero(self):
        requested = recipes()
        accepted = [r for r in requested if not (r["family"] == "low" and r["seed"] % 4 == 0)]
        bank = dict(schema="cat-generated-parallel-bank-v1", complete=True,
                    recipes=requested, scenes=accepted)
        groups = select_scenes(bank)
        self.assertEqual(tuple(groups), FAMILIES)
        for i, family in enumerate(FAMILIES):
            rows = groups[family]
            self.assertEqual([r["seed"] for r in rows], [310000+10000*i+k for k in (1, 18, 47)])
            self.assertEqual([r["difficulty"] for r in rows], [.4, .6, .8])
            self.assertTrue(all(r["split"] == "train" for r in rows))
        bank["complete"] = False
        with self.assertRaises(ValueError):
            select_scenes(bank)

    def test_scene_is_verified_not_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            obs = np.zeros((3, 4, 5), dtype=bool)
            obs[1, 2, 3] = True
            path = root/"scene.npz"
            np.savez_compressed(path, obs=obs)
            before = path.read_bytes()
            row = dict(file=path.name, shape=list(obs.shape),
                       sha256=hashlib.sha256(before).hexdigest(),
                       occupancy_sha256=hashlib.sha256(obs.tobytes()).hexdigest())
            np.testing.assert_array_equal(load_scene(root/"bank.json", row), obs)
            self.assertEqual(path.read_bytes(), before)
            row["occupancy_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "Occupancy"):
                load_scene(root/"bank.json", row)
            row["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_scene(root/"bank.json", row)
