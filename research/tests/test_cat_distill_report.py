"""Closed-loop success must survive the full evaluation, not just the exit."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_distill_report import full_horizon_success


class DistillationReportTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(full_horizon=True, simulated_s=5., fell=False,
            native_clearance_or_height_violation_steps=0, teacher_fraction=0.,
            dagger_collection=False)

    def test_unassisted_complete_clearance_safe_goal_is_success(self):
        self.assertTrue(full_horizon_success(self.meta, .1))

    def test_early_exit_fall_clearance_or_assistance_is_not_success(self):
        for key, value in (("full_horizon", False), ("simulated_s", 2.),
                ("fell", True), ("native_clearance_or_height_violation_steps", 1),
                ("teacher_fraction", .25), ("dagger_collection", True)):
            with self.subTest(key=key):
                self.assertFalse(full_horizon_success(dict(self.meta, **{key: value}), .1))

    def test_goal_miss_or_invalid_distance_is_not_success(self):
        for distance in (.2, 1., float("nan"), float("inf")):
            with self.subTest(distance=distance):
                self.assertFalse(full_horizon_success(self.meta, distance))


if __name__ == "__main__":
    unittest.main()
