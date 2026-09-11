import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_scenes import SOURCE, generate, generator_modules, verify_scene
from cat_roles import role_masks
from random_cat_roles import ROLE_BITS, morphology_provenance, trace_random, verify_trace


class MorphologyTests(unittest.TestCase):
    def test_replay_matches_scipy_with_padding_and_role_overlap(self):
        rng = np.random.default_rng(13)
        masks = {k: rng.random((13, 14, 15)) > .96 for k in ROLE_BITS}
        for iters, kernel in ((1, 3), (2, 3), (1, 5)):
            occ, bits = morphology_provenance(masks, iters, kernel)
            pad = max(1, (kernel//2)*iters)
            se = np.ones((kernel,)*3, bool)
            x = np.pad(np.logical_or.reduce(list(masks.values())), pad, constant_values=True)
            x = ndimage.binary_opening(ndimage.binary_closing(x, structure=se, iterations=iters), structure=se)
            np.testing.assert_array_equal(occ, x[(slice(pad, -pad),)*3])
            self.assertTrue((bits[~occ] == 0).all())
            self.assertTrue((bits[occ] > 0).all())

    def test_no_spurious_labels_for_interior_single_role(self):
        masks = {k: np.zeros((25,)*3, bool) for k in ROLE_BITS}
        masks["floor"][8:17, 8:17, 8:17] = True
        masks["floor"][12, 12, 12] = False
        occ, bits = morphology_provenance(masks)
        self.assertTrue(occ[12, 12, 12])
        self.assertEqual(bits[12, 12, 12], ROLE_BITS["floor"])
        masks["overhead"][14:17, 8:17, 8:17] = True
        _, bits = morphology_provenance(masks)
        self.assertTrue((bits & ROLE_BITS["overhead"]).any())

    def test_unbounded_or_wrong_masks_reject(self):
        masks = {k: np.zeros((5,)*3, bool) for k in ROLE_BITS}
        for kwargs in (dict(iters=0), dict(iters=4), dict(iters=1.5), dict(kernel=4)):
            with self.assertRaises(ValueError):
                morphology_provenance(masks, **kwargs)
        masks["floor"] = masks["floor"].astype(np.uint8)
        with self.assertRaises(ValueError):
            morphology_provenance(masks)


@unittest.skipUnless((SOURCE/"LICENSE").is_file(), "fetch pinned CAT source first")
class RandomRoleTests(unittest.TestCase):
    def test_original_outputs_and_module_are_unchanged(self):
        _, module, _, _ = generator_modules()
        original_build, original_morph = module.build_occ_from_masks_thick_xyxz, module.closing_opening_padded
        cfg = module.Cfg(seed=42, difficulty=.2)
        expected, *_ = module.generate_and_save(cfg, save=False)
        occ, axes, arrays, summary = trace_random(cfg, module)
        np.testing.assert_array_equal(occ, expected)
        self.assertEqual(summary["unresolved_added_cells"], 395)
        self.assertFalse(summary["placement_roles_resolved"])
        self.assertIs(module.build_occ_from_masks_thick_xyxz, original_build)
        self.assertIs(module.closing_opening_padded, original_morph)
        with patch("random_cat_roles.morphology_provenance", side_effect=ValueError("injected")):
            with self.assertRaisesRegex(ValueError, "injected"):
                trace_random(cfg, module)
        self.assertIs(module.build_occ_from_masks_thick_xyxz, original_build)
        self.assertIs(module.closing_opening_padded, original_morph)
        covered = np.logical_or.reduce([arrays[k] for k in ROLE_BITS]) | arrays["unresolved_added"]
        np.testing.assert_array_equal(covered, expected)

    def test_trace_replay_rejects_tampered_source_even_if_union_unchanged(self):
        _, module, _, _ = generator_modules()
        cfg = module.Cfg(seed=0, difficulty=.2, n_rect_L=1, n_rect_R=1, n_rect_F=0, n_rect_C=0)
        occ, axes, arrays, summary = trace_random(cfg, module)
        meta = dict(seed=0, difficulty=.2, n_side=1, n_floor=0, n_ceiling=0,
                    role_provenance=summary, resolution=cfg.voxel, sample_origin=[axis[0] for axis in axes],
                    origin_corner=cfg.origin_w.tolist(), shape=list(occ.shape), axis_order="xyz", units="m")
        verify_trace(meta, occ, arrays, module)
        altered = copy.deepcopy(arrays)
        altered["source_floor"][0, 0, 0] = not altered["source_floor"][0, 0, 0]
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            verify_trace(meta, occ, altered, module)
        meta["role_provenance"]["role_cells"]["floor"] += 1
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            verify_trace(meta, occ, arrays, module)

    def test_export_hashes_roles_and_ambiguous_placement_rejection(self):
        with tempfile.TemporaryDirectory() as directory, patch("cat_scenes.ROOT", Path(directory)), \
                patch("cat_scenes.subprocess.check_output", side_effect=lambda command, **kwargs:
                      "" if "status" in command else "test-revision\n"), contextlib.redirect_stdout(io.StringIO()):
            for seed, resolved in ((0, True), (1, False)):
                scene = generate(seed=seed, n_side=1, n_floor=0, n_ceiling=0)
                meta = verify_scene(scene)
                self.assertIn("role_trace.npz", meta["files"])
                self.assertEqual(meta["role_provenance"]["placement_roles_resolved"], resolved)
                if resolved:
                    masks, _ = role_masks(scene)
                    self.assertEqual(set(masks), {"lateral"})
                else:
                    with self.assertRaisesRegex(ValueError, "Unresolved"):
                        role_masks(scene)
                trace = scene/"role_trace.npz"
                trace.write_bytes(trace.read_bytes()+b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    verify_scene(scene)


if __name__ == "__main__":
    unittest.main()
