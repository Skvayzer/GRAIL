from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import hashlib
import json
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluation_command
from contact_results import audit_contact_capture
from gear_sonic.research.contact_accounting import (
    ContactLimits, PhysicsStepTap, bind_contact_names, classify_contact, reference_phase, unpack_contacts,
)


class TapTests(unittest.TestCase):
    def test_every_physics_step_including_terminal_is_read_before_reset(self):
        class Scene:
            def __init__(self):
                self.force, self.calls = 0., 0
            def update(self, dt, force_recompute=False):
                self.calls += 1
                self.force = 100.*self.calls
                return dt
        scene = Scene()
        before = vars(scene).copy()
        observed = []
        def read(dt, **kwargs):
            observed.append(scene.force)
        with PhysicsStepTap(scene, read):
            for _ in range(4):
                self.assertEqual(scene.update(.005, force_recompute=False), .005)
            # Mirror Isaac's reset after the last physics update, before return.
            scene.force = 0.
        self.assertEqual(observed, [100., 200., 300., 400.])
        self.assertEqual(scene.force, 0.)
        self.assertEqual(scene.calls, 4)  # no added step or overwritten simulator state
        self.assertEqual(set(vars(scene)), set(before))

    def test_exception_restores_local_method_and_double_owner_is_rejected(self):
        events = []
        original = lambda dt: events.append("original")
        scene = SimpleNamespace(update=original)
        def fail(dt):
            events.append("observer")
            raise RuntimeError("fixture")
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with PhysicsStepTap(scene, fail):
                with self.assertRaisesRegex(RuntimeError, "already"):
                    with PhysicsStepTap(scene, fail):
                        pass
                scene.update(.005)
        self.assertIs(scene.update, original)
        self.assertEqual(events, ["original", "observer"])
        self.assertFalse(hasattr(scene, "_research_physics_tap"))


class BufferTests(unittest.TestCase):
    def test_physx_leaf_names_bind_by_identity_not_assumed_order(self):
        paths = ["/World/Robot/left", "/World/Robot/right"]
        partners = ["/World/Object", "/World/Ground"]
        self.assertEqual(bind_contact_names(["right", "left"], [["Object", "Ground"]]*2, paths, partners), ["right", "left"])
        self.assertEqual(bind_contact_names(paths, [partners]*2, paths, partners), ["left", "right"])
        with self.assertRaises(ValueError):
            bind_contact_names(["right", "left"], [["Ground", "Object"]]*2, paths, partners)
        with self.assertRaises(ValueError):
            bind_contact_names(["left", "left"], [partners]*2, paths, partners)
        with self.assertRaises(ValueError):
            bind_contact_names(["left", "right"], [partners]*2, ["/a/left", "/b/left"], partners)

    def fixture(self):
        # Deliberately non-contiguous, unordered offsets. Empty tails are NaN.
        f, p, n, s = np.full((8, 1), np.nan), np.full((8, 3), np.nan), np.full((8, 3), np.nan), np.full((8, 1), np.nan)
        f[[1, 2, 5], 0] = [2., 3., 7.]
        p[[1, 2, 5]] = [[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]]
        n[[1, 2, 5]] = [[0., 0., 1.], [0., 0., 1.], [1., 0., 0.]]
        s[[1, 2, 5], 0] = [-.001, 0., .002]
        count = np.array([[0, 1], [2, 0]], np.int32)
        start = np.array([[7, 5], [1, 7]], np.int32)
        matrix = np.array([[[0., 0., 0.], [7., 0., 0.]], [[0., 0., 5.], [0., 0., 0.]]])
        return [f, p, n, s, count, start], matrix

    def test_six_buffers_and_force_reconstruction_no_normal_flipping(self):
        buffers, matrix = self.fixture()
        packed, error = unpack_contacts(buffers, matrix)
        self.assertEqual(error, 0.)
        self.assertEqual(packed["sensor"].tolist(), [0, 1, 1])
        self.assertEqual(packed["partner"].tolist(), [1, 0, 0])
        self.assertEqual(packed["force"].tolist(), [7., 2., 3.])
        self.assertEqual(packed["separation"].tolist(), [.002, -.001, 0.])
        buffers[2][5] *= -1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            unpack_contacts(buffers, matrix)

    def test_saturation_bad_ranges_overlap_and_nonfinite_fail_closed(self):
        for problem, message in (("saturation", "saturation"), ("bounds", "outside"),
                                 ("overlap", "Overlapping"), ("nan", "Nonfinite")):
            b, m = self.fixture()
            if problem == "saturation":
                b[4][0, 0] = 8
            elif problem == "bounds":
                b[5][0, 1] = 8
            elif problem == "overlap":
                b[5][0, 1] = 1
            else:
                b[1][1, 0] = float("nan")
            with self.subTest(problem=problem), self.assertRaisesRegex(ValueError, message):
                unpack_contacts(b, m)

    def test_empty_contact_set_is_valid_only_with_zero_aggregate(self):
        b, m = self.fixture()
        b[4][:] = 0
        data, _ = unpack_contacts(b, np.zeros_like(m))
        self.assertEqual(len(data["force"]), 0)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            unpack_contacts(b, m)


class SemanticTests(unittest.TestCase):
    def classify(self, **changes):
        args = dict(partner="terrain", is_foot=True, phase="stance_candidate",
            local_point=(.03, 0., -.035), sole_lower=(-.05, -.03, -.047), sole_upper=(.14, .03, -.023),
            normal=(0., 0., 1.), force=100., separation=-.001, surface_error=.001, normal_agreement=1.)
        args.update(changes)
        return classify_contact(**args)

    def test_stance_is_only_a_candidate_and_swing_support_is_not_allowed(self):
        self.assertEqual(self.classify(), "stance_support_candidate")
        self.assertEqual(self.classify(phase="swing_candidate"), "unexpected_swing_support")
        self.assertEqual(self.classify(phase="unknown"), "phase_unknown")
        self.assertEqual(self.classify(partner="ground"), "stance_support_candidate")

    def test_riser_body_clutter_and_sole_region_are_not_support_exemptions(self):
        cases = [(dict(normal=(1., 0., 0.)), "forbidden_riser_or_side"),
                 (dict(normal=(0., 0., -1.)), "forbidden_riser_or_side"),
                 (dict(is_foot=False), "forbidden_nonfoot"),
                 (dict(partner="cat"), "forbidden_cat"),
                 (dict(local_point=(.03, 0., 0.)), "outside_sole_region"),
                 (dict(local_point=(.03, .08, -.035)), "outside_sole_region"),
                 (dict(separation=-.03), "excess_penetration"),
                 (dict(force=3000.), "excess_point_force"),
                 (dict(normal_agreement=.5), "unverified_support_surface"),
                 (dict(surface_error=float("nan")), "invalid_geometry")]
        for changes, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.classify(**changes), expected)

    def test_reference_phase_does_not_use_actual_contact_force(self):
        self.assertEqual(reference_phase(.005, .1, True), "stance_candidate")
        self.assertEqual(reference_phase(.1, .1, True), "swing_candidate")
        self.assertEqual(reference_phase(.01, .8, True), "swing_candidate")
        self.assertEqual(reference_phase(.05, .3, True), "unknown")
        self.assertEqual(reference_phase(.0, .0, False), "unknown")
        self.assertEqual(reference_phase(-.03, .0, True), "unknown")
        with self.assertRaises(ValueError):
            ContactLimits(max_penetration=float("nan"))

    def test_contact_launch_opt_in_requires_existing_headless_layout_gate(self):
        args = (Path("/run"), Path("/data"), "fixture", 1)
        command = evaluation_command(*args, cat_scene=Path("/cat"), layout_audit=True, contact_audit=True)
        self.assertIn('++research_contact_output="/run/contact_audit.json"', command)
        self.assertIn('++research_layout_output="/run/layout_audit.json"', command)
        for kwargs in (dict(), dict(gui=True, layout_audit=True)):
            with self.assertRaises(ValueError):
                evaluation_command(*args, cat_scene=Path("/cat"), contact_audit=True, **kwargs)


class SavedCaptureTests(unittest.TestCase):
    def test_saved_terminal_capture_counts_and_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            data = np.zeros((2, 19), dtype=np.float32)
            data[:, 0] = [0, 1]
            data[:, 5] = [2., 3.]
            data[:, 11] = 1.
            np.savez_compressed(run/"contact_audit.npz", contacts=data)
            report = dict(schema="grail-cat-articulated-contact-audit-v1", contact_file="contact_audit.npz",
                capture_complete=True, error=None, completed=True, detailed_contacts=2,
                contact_sha256=hashlib.sha256((run/"contact_audit.npz").read_bytes()).hexdigest(),
                decimation=2, physics_steps=2, policy_steps=1, terminal_physics_samples=2,
                samples=[dict(physics_counter=i+1, contact_count=1) for i in range(2)],
                policy_outcomes=[dict(first_physics_sample=0, physics_samples=2, terminal=True)],
                sensor_link_order=["foot"], partner_order=["terrain"], classification_order=["phase_unknown"],
                classifications={"phase_unknown": 2})
            def save():
                (run/"contact_audit.json").write_text(json.dumps(report))
            save()
            self.assertTrue(audit_contact_capture(run)["capture_complete"])
            report["terminal_physics_samples"] = 0
            save()
            with self.assertRaisesRegex(ValueError, "terminal"):
                audit_contact_capture(run)
            report["terminal_physics_samples"] = 2
            report["samples"][1]["physics_counter"] = 1
            save()
            with self.assertRaisesRegex(ValueError, "counter"):
                audit_contact_capture(run)
            report["samples"][1]["physics_counter"] = 2
            save()
            (run/"contact_audit.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_contact_capture(run)


if __name__ == "__main__":
    unittest.main()
