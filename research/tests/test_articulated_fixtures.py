from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import math
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from articulated_contact_fixtures import Fixture, box_normal_cone_agreement, fixture_specs, station_geometry
from articulated_fixture_results import audit_articulated_fixtures
from review_contacts import contact_intervals
from gear_sonic.research.contact_accounting import ContactLimits, sole_regions


class FixtureGeometryTests(unittest.TestCase):
    def test_suite_has_positive_negative_clear_and_rotated_cases(self):
        specs = fixture_specs()
        self.assertEqual(len({s.name for s in specs}), len(specs))
        self.assertTrue(any(s.clear for s in specs))
        self.assertTrue(any(s.yaw != 0 for s in specs))
        self.assertEqual({s.expected for s in specs}, {"stance_support_candidate", "unexpected_swing_support",
            "phase_unknown", "forbidden_riser_or_side", "forbidden_nonfoot", "forbidden_cat", "no_contact"})

    def test_tread_gap_uses_actual_radius_and_yaw(self):
        c = np.array([[.1, .1, .3], [.2, .1, .3]])
        radii = [.01, .02]
        for yaw in (0., math.pi/2):
            r = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            spec = Fixture("fixture", yaw=yaw)
            box, normal = station_geometry(spec, c @ r.T, radii)
            np.testing.assert_allclose(normal, [0, 0, 1])
            self.assertAlmostEqual(box[2]+spec.size[2]/2, .3-.02-.04)
            np.testing.assert_allclose((box @ r)[:2], [.155, .1])

    def test_pitched_shin_front_face_covers_extremum_not_midpoint(self):
        spec = Fixture("shin", link="left_knee_link", approach="front")
        box, normal = station_geometry(spec, [[0, .1, .3], [.1, .1, .5]], [.05, .05])
        self.assertAlmostEqual(box[2], .5)
        self.assertAlmostEqual(box[0]-spec.size[0]/2, .15+.04)
        np.testing.assert_allclose(normal, [-1, 0, 0])

    def test_underside_patch_is_over_toe_not_through_ankle(self):
        spec = Fixture("up", approach="up")
        box, normal = station_geometry(spec, [[-.05, 0, 0], [.13, 0, 0]], [.01, .01])
        self.assertGreater(box[0]-spec.size[0]/2, .09)
        self.assertAlmostEqual(box[2]-spec.size[2]/2, .01+.04)
        np.testing.assert_allclose(normal, [0, 0, -1])

    def test_bad_capsule_data_is_rejected(self):
        for radii in ([0.], [float("nan")], [-.1]):
            with self.assertRaises(ValueError):
                station_geometry(Fixture("bad"), [[0, 0, 0]], radii)
        with self.assertRaises(ValueError):
            station_geometry(Fixture("bad", approach="sideways"), [[0, 0, 0]], [.01])

    def test_shared_sole_bounds_do_not_use_cover_radius(self):
        probes = [SimpleNamespace(link=side+"_ankle_roll_link", collision_index=0, offset=p, radius=99.)
                  for side in ("left", "right") for p in ((-.05, 0., -.025), (.13, 0., -.025))]
        inventory = [dict(type="Capsule", radius=.01)]
        regions = sole_regions(probes, inventory)
        for lo, hi in regions.values():
            np.testing.assert_allclose(lo, [-.057, -.007, -.047])
            np.testing.assert_allclose(hi, [.137, .007, -.023])
        with self.assertRaises(ValueError):
            sole_regions(probes, [dict(type="Cylinder", radius=.01)])
        with self.assertRaises(ValueError):
            sole_regions(probes, [dict(type="Capsule", radius=float("nan"))])

    def test_box_face_edge_and_corner_normals_not_arbitrary_sdf_gradient(self):
        def score(p, n):
            return box_normal_cone_agreement(p, n, [0, 0, 0], [2, 2, 2], 0.)
        self.assertAlmostEqual(score([0, 0, -1], [0, 0, -1]), 1.)
        self.assertAlmostEqual(score([-1, 0, -1], [0, 0, -1]), 1.)
        self.assertAlmostEqual(score([-1, 0, -1], [-1, 0, 0]), 1.)
        self.assertAlmostEqual(score([-1, 0, -1], [-2**-.5, 0, -2**-.5]), 1.)
        self.assertAlmostEqual(score([-1, -1, -1], [-3**-.5]*3), 1.)
        self.assertEqual(score([-1, 0, -1], [0, 0, 1]), 0.)  # inward is never flipped
        self.assertEqual(score([-1, 0, -1], [0, 1, 0]), 0.)  # edge tangent isn't outward
        self.assertEqual(score([0, 0, -.99], [0, 0, -1]), 0.)  # no surface at this point
        self.assertEqual(score([-1.001, 0, -1], [0, 0, -1]), 0.)  # not a wide tolerance

    def test_box_normal_cone_yaw_and_invalid_normal(self):
        score = box_normal_cone_agreement([0, -1, -1], [0, -1, 0], [0, 0, 0], [2, 2, 2], math.pi/2)
        self.assertAlmostEqual(score, 1.)
        with self.assertRaises(ValueError):
            box_normal_cone_agreement([0, 0, 1], [0, 0, -2], [0, 0, 0], [2, 2, 2], 0.)


class ReviewTests(unittest.TestCase):
    def fixture(self):
        report = dict(physics_dt=.005, limits=dict(force_threshold=.1),
            classification_order=["forbidden_riser_or_side", "stance_support_candidate", "phase_unknown"],
            sensor_link_order=["left", "right"], partner_order=["terrain", "cat"])
        rows = np.zeros((8, 19), dtype=np.float32)
        rows[:, 5] = 10.
        rows[:, 11] = 1.
        rows[:, 18] = 1.
        rows[:, 0] = [0, 0, 1, 3, 3, 3, 3, 4]
        rows[4, 3] = 1  # other foot, same label/time
        rows[5, 4] = 1  # other partner, same foot/time
        rows[6, 13] = 1  # stance candidate omitted
        rows[7, 5] = .01  # low force must not bridge separated intervals
        return rows, report

    def test_review_groups_consecutive_steps_not_point_records(self):
        rows, report = self.fixture()
        events = contact_intervals(rows, report)
        self.assertEqual(len(events), 4)
        first = events[0]
        self.assertEqual(first["physics_samples"], 2)
        self.assertEqual(first["contact_point_records"], 3)
        self.assertAlmostEqual(first["first_sample_time_s"], .005)
        self.assertAlmostEqual(first["last_sample_time_s"], .010)
        self.assertAlmostEqual(first["sampled_duration_s"], .010)
        for e in events:
            self.assertEqual(e["review_status"], "unreviewed")
            self.assertIsNone(e["verified_phase"])
            self.assertIsNone(e["reward_label"])

    def test_review_does_not_bridge_gaps_or_labels(self):
        rows, report = self.fixture()
        rows[2, 13] = 2
        events = contact_intervals(rows, report)
        self.assertEqual(len(events), 5)
        self.assertEqual(events[0]["physics_samples"], 1)
        self.assertEqual(events[1]["classification"], "phase_unknown")
        self.assertEqual(contact_intervals(rows[:0], report), [])
        for bad in (0., float("nan")):
            with self.assertRaises(ValueError):
                contact_intervals(rows, dict(report, physics_dt=bad))


class SavedFixturesTests(unittest.TestCase):
    def make_fixture(self, run):
        results, contacts = [], []
        for spec in fixture_specs():
            size = spec.size
            point, normal = {
                "down": ([0., 0., size[2]/2], [0., 0., 1.]),
                "up": ([0., 0., -size[2]/2], [0., 0., -1.]),
                "front": ([-size[0]/2, 0., 0.], [-1., 0., 0.]),
            }[spec.approach]
            row = dict(fixture=spec.name, step=79, link=spec.link, phase=spec.phase,
                point=point, normal=normal, body_position=point, body_quaternion_wxyz=[1., 0., 0., 0.],
                local_point=[0., 0., 0.], force_N=1., separation_m=0., surface_error_m=0.,
                normal_agreement=1., classification=spec.expected)
            if not spec.clear:
                contacts.append(row)
            results.append(dict(**asdict(spec), passed=True, contact_count=0 if spec.clear else 1,
                classifications={} if spec.clear else {spec.expected: 1}, other_pair_contacts=0,
                geometry_checks_passed=True, first_contact_step=None if spec.clear else 79,
                actual_approach_m=.015 if spec.clear else .04,
                steps=30 if spec.clear else 89, solid=dict(center=[0., 0., 0.], size=size)))
        report = dict(schema="grail-cat-articulated-fixtures-v1", passed=True, simulation_only=True,
            policy_loaded=False, phase_truth_verified=False, contact_permissions_granted=False,
            avoidance_training_ready=False, contact_file="articulated_fixture_contacts.json",
            unexpected_contacts=[], fixtures=results, limits=asdict(ContactLimits()),
            sole_regions={side+"_ankle_roll_link": [[-.1]*3, [.1]*3] for side in ("left", "right")})
        def save():
            path = run/report["contact_file"]
            path.write_text(json.dumps(contacts))
            report["contact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (run/"articulated_fixture_report.json").write_text(json.dumps(report))
        return report, contacts, save

    def test_saved_fixture_reverification_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            report, contacts, save = self.make_fixture(run)
            save()
            self.assertEqual(audit_articulated_fixtures(run)["fixtures"], 9)
            path = run/report["contact_file"]
            path.write_text(path.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_articulated_fixtures(run)
            save()
            contacts[0]["local_point"][0] = .01
            save()
            with self.assertRaisesRegex(ValueError, "geometry"):
                audit_articulated_fixtures(run)

    def test_saved_fixture_missing_case_and_wrong_label_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            report, contacts, save = self.make_fixture(run)
            contacts[0]["classification"] = "phase_unknown"
            save()
            with self.assertRaisesRegex(ValueError, "counts"):
                audit_articulated_fixtures(run)
            report["fixtures"].pop()
            save()
            with self.assertRaisesRegex(ValueError, "Missing"):
                audit_articulated_fixtures(run)


if __name__ == "__main__":
    unittest.main()
