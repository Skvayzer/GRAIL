import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_scenes import SOURCE, export_usd, generator_modules, occupancy_mesh, verify_source
from voxel_fixtures import axis_ray_fixtures, planar_contact_patch


class CatMeshTests(unittest.TestCase):
    def test_ray_fixtures_are_serializable_and_do_not_guess_contact_normals(self):
        occ = np.zeros((5, 6, 7), bool)
        occ[2:4, 1:5, 1:6] = True
        fixtures = axis_ray_fixtures(occ, .04, (-.5, -1., 0.))
        json.dumps(fixtures, allow_nan=False)
        self.assertTrue(any(r["expected_distance"] is None for r in fixtures))
        self.assertTrue(any(r["expected_distance"] is not None for r in fixtures))
        np.testing.assert_allclose(planar_contact_patch(occ, .04, (-.5, -1., 0.)), [-.42, -.9, .1])
        occ[:] = False
        occ[2, 2, 2] = True
        with self.assertRaises(ValueError):
            planar_contact_patch(occ, .04, (0., 0., 0.))

    def test_voxel_center_world_offset_boundary_closure_and_winding(self):
        occ = np.zeros((4, 5, 6), dtype=bool)
        occ[0:2, 1:4, 2:5] = True  # touches lower X grid boundary
        v, f = occupancy_mesh(occ, .1, (-.5, -1., 0.))
        np.testing.assert_allclose(v.min(0), [-.5, -.9, .2], atol=1e-6)
        np.testing.assert_allclose(v.max(0), [-.3, -.6, .5], atol=1e-6)
        mesh = trimesh.Trimesh(v, f, process=False)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0.)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"scene.usda"
            export_usd(path, v, f)
            stage = Usd.Stage.Open(str(path))
            prim = stage.GetPrimAtPath("/CAT/Obstacles")
            self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), "none")
            np.testing.assert_allclose(np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get()), v)
            with self.assertRaises(FileExistsError):
                export_usd(path, v, f)

    def test_empty_occupancy_rejected(self):
        with self.assertRaises(ValueError):
            occupancy_mesh(np.zeros((3, 3, 3), bool), .04, (0., 0., 0.))


@unittest.skipUnless((SOURCE/"LICENSE").is_file(), "run cat_scenes.py fetch for pinned upstream integration tests")
class CatGeneratorTests(unittest.TestCase):
    def test_upstream_is_unchanged_and_deterministic(self):
        manifest = verify_source()
        _, random, typical, pf = generator_modules()
        cfg = random.Cfg(seed=42, difficulty=.2)
        a, x, y, z = random.generate_and_save(cfg, save=False)
        b, *_ = random.generate_and_save(cfg, save=False)
        c, *_ = random.generate_and_save(random.Cfg(seed=43, difficulty=.2), save=False)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(a.shape, (75, 50, 38))
        self.assertEqual(int(a.sum()), 16966)
        self.assertEqual(hashlib.sha256(a.tobytes()).hexdigest(),
                         "237c442b738826bc734757b746df65327b847a96f9d840f29d95a14478a7e242")
        expected_first = cfg.origin_w+cfg.voxel/2
        np.testing.assert_allclose([x[0], y[0], z[0]], expected_first)
        # Byte hashes are the unchanged upstream source contract, not a ported generator.
        self.assertEqual(len(manifest["files"]), 6)
        sdf = pf.make_sdf(a, cfg.voxel)
        self.assertTrue(np.isfinite(sdf).all())
        self.assertTrue((sdf[a] < 0).all())
        self.assertTrue((sdf[~a] > 0).all())
        side = typical.build_obstacles("side0", np.meshgrid(x, y, z, indexing="ij"))
        self.assertEqual(int(side.sum()), 4560)


if __name__ == "__main__":
    unittest.main()
