import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_roles import role_masks
from cat_scenes import SOURCE, generator_modules, verify_source
from gear_sonic.research.cat_geometry import Placement
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_roles import assess_roles, ground_rigid_placement
from gear_sonic.research.scene_volume import VolumeSpec, build_volume, sample_distances
from gear_sonic.research.terrain_surface import TerrainSurface


def flat(z=0.):
    return np.array([[-1., -1., z], [1., -1., z], [1., 1., z], [-1., 1., z]], np.float32), np.array([[0, 1, 2], [0, 2, 3]])


class UnsignedTerrainTests(unittest.TestCase):
    def test_open_surface_has_no_fake_inside_outside_or_normal_flip(self):
        v, f = flat(.5)
        p = torch.tensor([[.2, .1, .8], [.2, .1, .2], [3., 0., .5]])
        d, n, valid = TerrainSurface(v, f).distance(p)
        self.assertTrue(valid.all())
        torch.testing.assert_close(d, torch.tensor([.3, .3, 2.]))
        torch.testing.assert_close(n, torch.tensor([[0., 0., 1.]]).expand(3, -1))
        rd, rn, _ = TerrainSurface(v, f[:, ::-1]).distance(p)
        torch.testing.assert_close(rd, d)
        torch.testing.assert_close(rn, -n)

    def test_ground_is_separate_and_misses_stay_unknown(self):
        p = torch.tensor([[2., 0., .1], [2., 0., -.1], [float("nan"), 0., .1]])
        d, n, known = TerrainSurface(*flat(.5), ground_z=0.).distance(p, limit=.2)
        self.assertEqual(known.tolist(), [True, True, False])
        torch.testing.assert_close(d[:2], torch.tensor([.1, .1]))
        self.assertTrue(torch.isnan(d[2]))
        self.assertEqual(TerrainSurface(*flat(.5)).distance(p, limit=.2)[2].tolist(), [False]*3)
        with self.assertRaises(ValueError):
            TerrainSurface(*flat()).distance(p, limit=float("nan"))


class RoleTests(unittest.TestCase):
    def test_rigid_grounding_moves_only_datum_and_rejects_stair_edges(self):
        masks = {"floor": np.ones((5, 5, 2), bool)}
        # Uniform raised landing supports the entire unchanged feature.
        place, report = ground_rigid_placement(TerrainSurface(*flat(.4), ground_z=0.), masks,
            (-.1, -.1, 0.), .04, (0., 0.), math.pi/2)
        self.assertAlmostEqual(place.translation[2], .4, delta=1e-6)  # float32 ray arithmetic
        self.assertEqual(place.yaw, math.pi/2)
        self.assertFalse(report["geometry_modified"])
        # Straddling the landing/ground edge cannot be fixed by one Z offset.
        with self.assertRaisesRegex(ValueError, "uneven"):
            ground_rigid_placement(TerrainSurface(*flat(.4), ground_z=0.), masks,
                (-.1, -.1, 0.), .04, (1., 0.), 0.)
        with self.assertRaisesRegex(ValueError, "missing"):
            ground_rigid_placement(TerrainSurface(*flat(.4)), masks,
                (-.1, -.1, 0.), .04, (3., 0.), 0.)

    def test_joined_roles_detect_lost_hurdle_but_dont_guess_from_wall(self):
        shape = (2, 8, 30)
        floor = np.zeros(shape, bool)
        floor[:, :, :2] = True
        side = np.zeros(shape, bool)
        side[:, :2, :] = True
        side[:, -2:, :] = True
        masks = dict(floor=floor, lateral=side)  # connected, overlapping by design
        args = (masks, (-.04, -.16, 0.), .04)
        flat_result = assess_roles(TerrainSurface(*flat()), *args, Placement())
        self.assertTrue(flat_result["all_roles_retained"])
        buried = assess_roles(TerrainSurface(*flat(.2)), *args, Placement())
        floor_report = next(r for r in buried["roles"] if r["role"] == "floor")
        self.assertEqual(floor_report["minimum_exposed_height_fraction"], 0.)
        self.assertFalse(buried["all_roles_retained"])
        restored = assess_roles(TerrainSurface(*flat(.2)), *args, Placement((0., 0., .2)))
        self.assertTrue(restored["all_roles_retained"])
        floating = assess_roles(TerrainSurface(*flat()), *args, Placement((0., 0., .2)))
        self.assertFalse(floating["all_roles_retained"])

    def test_explicit_overhead_is_not_a_floating_floor_or_support_permission(self):
        mask = np.zeros((2, 2, 35), bool)
        mask[:, :, 28:] = True
        good = assess_roles(TerrainSurface(*flat()), {"overhead": mask}, (0., 0., 0.), .04, Placement())
        self.assertTrue(good["all_roles_retained"])
        self.assertFalse(good["roles"][0]["contact_permission"])
        bad = assess_roles(TerrainSurface(*flat(.5)), {"overhead": mask}, (0., 0., 0.), .04, Placement())
        self.assertFalse(bad["all_roles_retained"])
        inverted_v, inverted_f = flat()
        inverted = assess_roles(TerrainSurface(inverted_v, inverted_f[:, ::-1]), {"overhead": mask}, (0., 0., 0.), .04, Placement())
        self.assertFalse(inverted["all_roles_retained"])

    @unittest.skipUnless((SOURCE/"LICENSE").is_file(), "requires pinned CAT source")
    def test_explicit_replay_exact_union_and_random_provenance_rejected(self):
        _, random, typical, _ = generator_modules()
        cfg = random.Cfg()
        axes = random.make_axes(cfg)
        for scene in ("side-hurdle2", "side-hurdle-crouch3", "crouch0", "side0"):
            meta = dict(source=verify_source(), sample_origin=[a[0] for a in axes], shape=[len(a) for a in axes],
                        resolution=cfg.voxel, scene=scene)
            obs = typical.build_obstacles(scene, np.meshgrid(*axes, indexing="ij"))
            with patch("cat_roles.verify_scene_files", return_value=meta), patch("numpy.load", return_value=obs):
                masks, _ = role_masks(Path("/unused"))
                np.testing.assert_array_equal(np.logical_or.reduce(list(masks.values())), obs)
            with patch("cat_roles.verify_scene_files", return_value=meta), patch("numpy.load", return_value=~obs):
                with self.assertRaisesRegex(ValueError, "exactly"):
                    role_masks(Path("/unused"))
        meta["scene"] = "random"
        with patch("cat_roles.verify_scene_files", return_value=meta):
            with self.assertRaisesRegex(ValueError, "provenance"):
                role_masks(Path("/unused"))


class VolumeTests(unittest.TestCase):
    def fixture(self, device="cpu"):
        box = trimesh.creation.box(extents=(.2, .4, .6))
        box.apply_translation((.3, -.2, .3))
        surface = TerrainSurface(*flat(), ground_z=0., device=device)
        clutter = ClosedMeshDistance(box.vertices, box.faces, device=device)
        return surface, clutter

    def test_full_sphere_bounds_not_just_centres_and_resource_guard(self):
        surface, clutter = self.fixture()
        centers = np.array([[[0., 0., 2.8], [.1, -.1, 1.]]], np.float32)
        radii = np.array([.2, .1], np.float32)
        spec = VolumeSpec.fit(surface.vertices.numpy(), clutter.vertices.numpy(), centers, radii, resolution=.1)
        self.assertTrue((np.array(spec.sample_origin) < (centers-radii[None, :, None]).min((0, 1))).all())
        last = np.array(spec.sample_origin)+(np.array(spec.shape)-1)*spec.resolution
        self.assertTrue((last > (centers+radii[None, :, None]).max((0, 1))).all())
        self.assertGreater(last[2], 3.)
        for bad in ((float("nan"), (3, 3, 3)), (.04, (2000, 2000, 2000)), (.04, (1, 2, 3)), (.04, (2.5, 3, 3))):
            with self.assertRaises(ValueError):
                VolumeSpec((0., 0., 0.), bad[1], bad[0])

    def test_sampling_xyz_rotation_boundary_unknown_and_terrain_retained(self):
        surface, clutter = self.fixture()
        placement = Placement((.1, .2, 0.), math.pi/2)
        spec = VolumeSpec((-.4, -.4, -.2), (16, 18, 23), .06)
        arrays = build_volume(surface, clutter, placement, spec, batch_size=819)
        self.assertTrue(arrays["terrain_surface_band"].any())
        self.assertTrue(arrays["support_candidate"].all())
        self.assertTrue(arrays["clutter_inside"].any())
        self.assertNotIn("terrain_inside", arrays)
        p = torch.tensor([[.13, .29, .17], [-.2, .4, 1.], spec.sample_origin,
                          tuple(np.array(spec.sample_origin)+(np.array(spec.shape)-1)*spec.resolution),
                          [4., 0., 0.], [float("nan"), 0., 0.]], dtype=torch.float32)
        result = sample_distances(arrays, spec, p)
        for name, expected in (("clutter_sdf", clutter.query(placement.to_local(p))[0]),
                               ("terrain_unsigned_distance", surface.distance(p)[0])):
            self.assertEqual(result[name+"_valid"].tolist(), [True, True, True, True, False, False])
            self.assertLess(float((result[name][:4]-expected[:4]).abs().max()), math.sqrt(3)*spec.resolution)
            self.assertTrue(torch.isnan(result[name][4:]).all())
        arrays["terrain_distance_valid"][:] = False
        invalid = sample_distances(arrays, spec, p[:1])
        self.assertFalse(invalid["terrain_unsigned_distance_valid"].any())
        self.assertTrue(torch.isnan(invalid["terrain_unsigned_distance"]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA parity requires GPU")
    def test_cpu_cuda_unsigned_and_volume_distances_agree(self):
        spec = VolumeSpec((-.4, -.4, -.1), (12, 13, 14), .07)
        a = build_volume(*self.fixture(), Placement(), spec)
        b = build_volume(*self.fixture("cuda:0"), Placement(), spec)
        for name in ("terrain_unsigned_distance", "clutter_sdf"):
            np.testing.assert_allclose(a[name], b[name], atol=2e-6)
        for name in ("terrain_distance_valid", "clutter_distance_valid", "terrain_surface_band", "support_known"):
            np.testing.assert_array_equal(a[name], b[name])


if __name__ == "__main__":
    unittest.main()
