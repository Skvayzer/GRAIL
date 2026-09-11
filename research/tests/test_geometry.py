import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from gear_sonic.research.geometry import (
    ContactPermission, Solid, sample_xyz_grid, sphere_clearances, support_below,
)
from gear_sonic.research.body_envelope import collision_probes, self_pair_mask, world_probe_centers
from gear_sonic.research.scenes import contact_fixture_solids


class FieldTests(unittest.TestCase):
    def test_signed_distance_radius_margin_and_gradient(self):
        solid = Solid("box", "box", (0., 0., 0.), (2., 2., 2.))
        p = torch.tensor([[1.5, 0., 0.], [0., 0., 0.], [2., 2., 0.]], requires_grad=True)
        distance, normal = solid.distance_normal(p)
        torch.testing.assert_close(distance, torch.tensor([.5, -1., math.sqrt(2)]))
        torch.testing.assert_close(torch.autograd.grad(distance.sum(), p)[0][[0, 2]], normal[[0, 2]])
        gap = sphere_clearances(p, torch.tensor([.2, .3, .4]), [solid], margin=.05)
        torch.testing.assert_close(gap[:, 0], distance-torch.tensor([.25, .35, .45]))

    def test_rotation_and_finite_cylinder_caps(self):
        box = Solid("box", "box", (2., 3., 0.), (2., 1., 1.), yaw=math.pi/2)
        d, n = box.distance_normal(torch.tensor([[2., 4.3, 0.]], dtype=torch.float64))
        torch.testing.assert_close(d, torch.tensor([.3], dtype=torch.float64))
        torch.testing.assert_close(n, torch.tensor([[0., 1., 0.]], dtype=torch.float64))
        pole = Solid("pole", "cylinder", (0., 0., 0.), (1., 1., 2.))
        p = torch.tensor([[.8, 0., 0.], [0., 0., 1.3], [.8, 0., 1.4]], requires_grad=True)
        d, n = pole.distance_normal(p)
        torch.testing.assert_close(d, torch.tensor([.3, .3, .5]))
        torch.testing.assert_close(torch.autograd.grad(d.sum(), p)[0], n)

    def test_empty_scene_and_invalid_geometry(self):
        self.assertEqual(sphere_clearances(torch.zeros(2, 3), torch.ones(2), []).shape, (2, 0))
        with self.assertRaises(ValueError):
            Solid("bad", "box", (float("nan"), 0., 0.), (1., 1., 1.))
        with self.assertRaises(ValueError):
            sphere_clearances(torch.zeros(2, 3), torch.tensor([-1., 1.]), [], margin=0)

    def test_support_is_not_removed_from_obstacles(self):
        tread = contact_fixture_solids()[0]
        roof = Solid("roof", "box", (0., 0., 1.5), (2., 2., .2))
        p = torch.tensor([[0., 0., .6], [.51, 0., .6], [0., 0., 2.0], [0., 0., .2]])
        h, valid = support_below(p, [tread, roof], max_drop=.8, edge_margin=.02)
        self.assertEqual(valid.tolist(), [True, False, False, False])
        self.assertAlmostEqual(float(h[0]), .5)
        self.assertTrue(torch.isnan(h[~valid]).all())
        self.assertLess(float(sphere_clearances(p[:1], torch.tensor([.15]), [tread])[0, 0]), 0)

    def test_xyz_legacy_parity_and_intentional_correction(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/cat_sampler.json").read_text())
        x, y, z = torch.meshgrid(*[torch.arange(4.)]*3, indexing="ij")
        field = torch.stack((x, y, z, x+10*y+100*z), -1)
        idx = torch.tensor(fixture["indices"])
        positions = torch.tensor(fixture["origin"])+idx*fixture["resolution"]
        old, old_valid = sample_xyz_grid(field, positions, fixture["origin"], .25, legacy_cat=True)
        torch.testing.assert_close(old, torch.tensor(fixture["legacy_values"]), atol=2e-5, rtol=1e-5)
        new, valid = sample_xyz_grid(field, positions, fixture["origin"], .25)
        self.assertEqual(valid.tolist(), [True, True, True, True, False])
        self.assertEqual(old_valid.tolist(), valid.tolist())
        torch.testing.assert_close(new[:4, :3], idx[:4], atol=1e-6, rtol=1e-5)
        self.assertAlmostEqual(float(new[0, 3]), 84.2, places=4)
        self.assertTrue(torch.isnan(new[4]).all())
        torch.testing.assert_close(new[1:3], old[1:3])  # equal at shared lattice/midpoint cases

    def test_unknown_invalid_and_final_grid_boundary(self):
        field = torch.ones(3, 3, 3, 1)
        known = torch.ones(3, 3, 3, dtype=torch.bool)
        known[1, 1, 1] = False
        p = torch.tensor([[.1, .2, .3], [2., 2., 2.], [-.01, 0., 0.], [float("nan"), 0., 0.]])
        values, valid = sample_xyz_grid(field, p, (0., 0., 0.), 1., known)
        self.assertEqual(valid.tolist(), [False, True, False, False])
        self.assertTrue(torch.isnan(values[~valid]).all())
        field[0, 0, 0] = float("nan")
        values, valid = sample_xyz_grid(field, torch.tensor([[1., 0., 0.]]), (0., 0., 0.), 1.)
        self.assertTrue(bool(valid[0]))  # zero-weight NaN corner must not contaminate the sample
        self.assertEqual(float(values[0, 0]), 1.)


class PermissionTests(unittest.TestCase):
    def test_link_phase_region_normal_depth_and_force(self):
        tread = contact_fixture_solids()[0]
        rule = ContactPermission("tread", ("foot",), ("stance",), (-.45, -.45, .249), (.45, .45, .251))
        def allowed(**kw):
            args = dict(solid=tread, link="foot", phase="stance", point=(0., 0., .5),
                        normal=(0., 0., 1.), penetration=.001, force=50.)
            args.update(kw)
            return rule.allows(**args)
        self.assertTrue(allowed())
        for change in (dict(link="shin"), dict(link="hand"), dict(phase="swing"),
                       dict(normal=(1., 0., 0.)), dict(point=(.49, 0., .5)),
                       dict(penetration=.03), dict(force=1001.), dict(force=float("nan")),
                       dict(solid=contact_fixture_solids()[1])):
            self.assertFalse(allowed(**change))

    def test_grasp_permission_is_not_whole_object_whitelist(self):
        target = Solid("target", "box", (0., 0., 0.), (1., 1., 1.))
        grasp = ContactPermission("target", ("finger",), ("grasp",), (.49, -.1, -.1), (.51, .1, .1),
                                  min_up_normal=-1., max_force=20.)
        args = (target, "finger", "grasp", (.5, 0., 0.), (1., 0., 0.), 0., 10.)
        self.assertTrue(grasp.allows(*args))
        self.assertFalse(grasp.allows(target, "forearm", *args[2:]))
        self.assertFalse(grasp.allows(target, "finger", "transit", *args[3:]))


class BodyTests(unittest.TestCase):
    def test_full_collision_inventory_name_mapping_and_rotation(self):
        probes = collision_probes()
        links = sorted({p.link for p in probes}, reverse=True)
        self.assertEqual(len(probes), 76)
        self.assertEqual(len(links), 14)
        for name in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
                     "left_elbow_link", "right_knee_link"):
            self.assertIn(name, links)
        pos = torch.zeros(1, len(links), 3)
        pos[:, :, 0] = 3.
        q = torch.zeros(1, len(links), 4)
        q[:, :, 3] = 1.  # 180 degree yaw, WXYZ
        centers, r = world_probe_centers(probes, links, pos, q)
        offsets = torch.tensor([p.offset for p in probes])
        expected = offsets*torch.tensor([-1., -1., 1.])+torch.tensor([3., 0., 0.])
        torch.testing.assert_close(centers[0], expected)
        self.assertTrue((r > 0).all())
        with self.assertRaises(ValueError):
            world_probe_centers(probes, links[:-1], pos, q)

    def test_cylinder_spheres_cover_surface_including_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"robot.urdf"
            path.write_text('<robot><link name="test"><collision><geometry><cylinder radius=".1" length=".23"/></geometry></collision></link></robot>')
            probes = collision_probes(path)
            c = torch.tensor([p.offset for p in probes])
            r = torch.tensor([p.radius for p in probes])
            z = torch.linspace(-.115, .115, 1001)
            surface = torch.stack((torch.full_like(z, .1), torch.zeros_like(z), z), -1)
            gap = torch.cdist(surface, c)-r
            self.assertLessEqual(float(gap.amin(-1).amax()), 1e-6)

    def test_self_pairs_keep_hands_and_arm_torso(self):
        probes = collision_probes()
        mask = self_pair_mask(probes)
        def included(a, b):
            i = next(i for i, p in enumerate(probes) if p.link == a)
            j = next(i for i, p in enumerate(probes) if p.link == b)
            return bool(mask[min(i, j), max(i, j)])
        self.assertTrue(included("left_wrist_yaw_link", "right_wrist_yaw_link"))
        self.assertTrue(included("left_elbow_link", "torso_link"))
        self.assertFalse(included("left_elbow_link", "left_shoulder_yaw_link"))


if __name__ == "__main__":
    unittest.main()
