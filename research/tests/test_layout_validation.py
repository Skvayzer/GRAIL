import math
from pathlib import Path
import sys
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluation_command
from gear_sonic.research.cat_geometry import Placement
from gear_sonic.research.layout_validation import LayoutLimits, layout_grid, obstacle_embedding, support_path
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.terrain_snapshot import remap_rays, rigid_collision_mesh, rotation_wxyz, collision_triangles
from gear_sonic.research.terrain_surface import TerrainSurface


def flat(size=2., z=0.):
    return np.array([[-size/2, -size/2, z], [size/2, -size/2, z],
                     [size/2, size/2, z], [-size/2, size/2, z]], np.float32), np.array([[0, 1, 2], [0, 2, 3]])


class SurfaceTests(unittest.TestCase):
    def test_open_surface_upside_down_and_misses_are_not_free_support(self):
        v, f = flat()
        p = torch.tensor([[.2, .1, 1.], [4., 4., 1.]])
        height, normal, valid = TerrainSurface(v, f).support_below(p)
        self.assertEqual(valid.tolist(), [True, False])
        self.assertEqual(float(height[0]), 0.)
        self.assertTrue(torch.isnan(height[1]))
        torch.testing.assert_close(normal[0], torch.tensor([0., 0., 1.]))
        _, n, _ = TerrainSurface(v, f[:, ::-1]).support_below(p)
        self.assertLess(float(n[0, 2]), -.99)

    def test_explicit_ground_only_and_finite_query_limits(self):
        v, f = flat(z=.5)
        surface = TerrainSurface(v, f, ground_z=0.)
        p = torch.tensor([[0., 0., 1.], [3., 0., 1.], [3., 0., -1.], [float("nan"), 0., 1.]])
        h, _, valid = surface.support_below(p)
        self.assertEqual(valid.tolist(), [True, True, False, False])
        torch.testing.assert_close(h[:2], torch.tensor([.5, 0.]))
        self.assertFalse(surface.support_below(p[:1], max_drop=.1)[2].any())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA parity requires GPU")
    def test_cpu_cuda_rays_agree(self):
        v, f = flat()
        p = torch.tensor([[0., 0., 1.], [.3, -.4, 2.], [2., 2., 1.]])
        a, b = TerrainSurface(v, f).support_below(p), TerrainSurface(v, f, device="cuda:0").support_below(p.cuda())
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y.cpu(), equal_nan=True)


class SnapshotTests(unittest.TestCase):
    def test_cpu_query_frame_remapping_preserves_local_ray_and_distance(self):
        live_q = [math.cos(math.pi/4), math.sin(math.pi/4), 0., 0.]
        cpu_q = [1., 0., 0., 0.]
        local_origin, local_direction = np.array([[.2, 1., .3]]), np.array([[0., -1., 0.]])
        live_origin = local_origin @ rotation_wxyz(live_q).T + [1., 2., 3.]
        live_direction = local_direction @ rotation_wxyz(live_q).T
        origin, direction = remap_rays(live_origin, live_direction, [1., 2., 3.], live_q,
                                      [8., 9., 10.], cpu_q)
        np.testing.assert_allclose(origin, local_origin + [8., 9., 10.], atol=1e-6)
        np.testing.assert_allclose(direction, local_direction, atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(direction[0]), 1.)

    def fixture(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, "Z")
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, "/Terrain")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        root.AddTranslateOp().Set(Gf.Vec3d(8., 7., 6.))
        root.AddRotateZOp().Set(90.)
        root.AddScaleOp().Set(Gf.Vec3f(2., 2., 2.))
        mesh = UsdGeom.Mesh.Define(stage, "/Terrain/collision")
        v, f = flat()
        mesh.CreatePointsAttr(v)
        mesh.CreateFaceVertexCountsAttr([3, 3])
        mesh.CreateFaceVertexIndicesAttr(f.ravel())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        # A visual mesh must not be included as physical support.
        UsdGeom.Mesh.Define(stage, "/Terrain/visual")
        return stage

    def test_live_pose_replaces_authored_pose_once_and_preserves_scale(self):
        stage = self.fixture()
        quat = [math.cos(math.pi/4), math.sin(math.pi/4), 0., 0.]
        vertices, faces, inventory = rigid_collision_mesh(stage, "/Terrain", [1., 2., 3.], quat)
        original, _ = flat()
        np.testing.assert_allclose(vertices, original*2 @ rotation_wxyz(quat).T + [1., 2., 3.], atol=1e-6)
        self.assertEqual(len(inventory), 1)
        self.assertEqual(faces.shape, (2, 3))

    def test_only_convex_planar_quads_are_split_without_surface_changes(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        faces = collision_triangles(vertices, [4], [0, 1, 2, 3])
        np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])
        normals = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                           vertices[faces[:, 2]]-vertices[faces[:, 0]])
        np.testing.assert_allclose(normals, [[0, 0, 1], [0, 0, 1]])
        nonplanar = vertices.copy()
        nonplanar[3, 2] = .1
        with self.assertRaisesRegex(ValueError, "Nonplanar"):
            collision_triangles(nonplanar, [4], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Nonconvex"):
            collision_triangles(vertices, [4], [0, 2, 1, 3])
        with self.assertRaises(ValueError):
            collision_triangles(vertices, [3], [0, 0, 1])

    def test_usd_quad_terrain_retains_exact_live_pose(self):
        stage = self.fixture()
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Terrain/collision"))
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        mesh.GetPointsAttr().Set(vertices)
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        actual, faces, _ = rigid_collision_mesh(stage, "/Terrain", [1., 2., 3.], [1., 0., 0., 0.])
        np.testing.assert_allclose(actual, vertices*2+[1, 2, 3], atol=1e-6)
        np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])
        mesh.CreateHoleIndicesAttr([0])
        with self.assertRaisesRegex(ValueError, "holes"):
            rigid_collision_mesh(stage, "/Terrain", [1., 2., 3.], [1., 0., 0., 0.])

    def test_reject_convexification_and_wrong_units(self):
        stage = self.fixture()
        UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath("/Terrain/collision")).GetApproximationAttr().Set("convexHull")
        with self.assertRaisesRegex(ValueError, "approximation"):
            rigid_collision_mesh(stage, "/Terrain", [0., 0., 0.], [1., 0., 0., 0.])
        UsdGeom.SetStageMetersPerUnit(stage, .01)
        with self.assertRaisesRegex(ValueError, "metre"):
            rigid_collision_mesh(stage, "/Terrain", [0., 0., 0.], [1., 0., 0., 0.])

    def test_mirrored_root_is_not_silently_reoriented(self):
        stage = self.fixture()
        stage.GetPrimAtPath("/Terrain").GetAttribute("xformOp:scale").Set(Gf.Vec3f(-2., 2., 2.))
        with self.assertRaisesRegex(ValueError, "Mirrored"):
            rigid_collision_mesh(stage, "/Terrain", [0., 0., 0.], [1., 0., 0., 0.])


class PassageTests(unittest.TestCase):
    def test_stair_risers_gaps_and_cliffs(self):
        h = np.tile(np.arange(6)[:, None]*.1, (1, 3))
        walkable = np.ones_like(h, dtype=bool)
        self.assertTrue(support_path(h, walkable, (0, 1), (5, 1), .1, .15))
        self.assertFalse(support_path(h, walkable, (0, 1), (5, 1), .1, .05))
        walkable[3] = False
        self.assertFalse(support_path(h, walkable, (0, 1), (5, 1), .1, .15))

    def test_no_diagonal_corner_cutting_or_unknown_support(self):
        h = np.zeros((2, 2))
        self.assertFalse(support_path(h, np.eye(2, dtype=bool), (0, 0), (1, 1), .1, .2))
        h[0, 0] = np.nan
        with self.assertRaises(ValueError):
            support_path(h, np.ones((2, 2), bool), (0, 0), (1, 1), .1, .2)

    def test_actual_clutter_blocks_route_and_clear_placement_passes(self):
        surface = TerrainSurface(*flat())
        box = trimesh.creation.box(extents=(.2, 3., 2.))
        box.apply_translation([0., 0., 1.])
        clutter = ClosedMeshDistance(box.vertices, box.faces)
        anchors = np.array([[-.6, 0., .8], [.6, 0., .8], [-.6, 0., .8]], np.float32)
        blocked, _ = layout_grid(surface, clutter, Placement(), anchors)
        self.assertFalse(blocked["geometric_route_found"])
        clear, arrays = layout_grid(surface, clutter, Placement((0., 3., 0.)), anchors)
        self.assertTrue(clear["geometric_route_found"])
        self.assertEqual(clear["endpoint_reference_frames"], [0, 1])
        self.assertGreater(len(arrays["route"]), 3)

    def test_stride_crosses_tread_edge_but_not_a_hole_or_wall(self):
        h = np.zeros((7, 3))
        h[3:] = .1
        foothold, transit = np.ones_like(h, bool), np.ones_like(h, bool)
        foothold[2:4] = False  # a footprint straddling the edge is invalid
        self.assertTrue(support_path(h, foothold, (0, 1), (6, 1), .04, .2,
                                    transit=transit, max_stride=.20))
        transit[3] = False
        self.assertFalse(support_path(h, foothold, (0, 1), (6, 1), .04, .2,
                                     transit=transit, max_stride=.20))

    def test_buried_and_floating_components_flagged(self):
        surface = TerrainSurface(*flat(z=.2), ground_z=0.)
        occupied = np.ones((2, 2, 2), dtype=bool)
        buried = obstacle_embedding(surface, occupied, [0., 0., 0.], .04, Placement())[0]
        floating = obstacle_embedding(surface, occupied, [0., 0., 0.], .04, Placement((0., 0., 1.)))[0]
        self.assertEqual(buried["below_support_envelope_fraction"], 1.)
        self.assertTrue(buried["requires_placement_review"])
        self.assertEqual(floating["floating_base_columns"], floating["base_columns"])

    def test_visible_gap_is_not_enough_for_trunk_width(self):
        surface = TerrainSurface(*flat())
        anchors = np.array([[-.6, 0., .8], [.6, 0., .8]], np.float32)
        for gap, expected in ((.3, False), (.8, True)):
            walls = []
            for sign in (-1., 1.):
                wall = trimesh.creation.box(extents=(.2, 1.5, 2.))
                wall.apply_translation([0., sign*(gap/2+.75), 1.])
                walls.append(wall)
            mesh = trimesh.util.concatenate(walls)
            report, _ = layout_grid(surface, ClosedMeshDistance(mesh.vertices, mesh.faces), Placement(), anchors)
            self.assertEqual(report["geometric_route_found"], expected)

    def test_layout_launch_is_opt_in_and_retains_preflight(self):
        cmd = evaluation_command(Path("/run"), Path("/data"), "fixture", 1,
                                 cat_scene=Path("/cat"), layout_audit=True)
        self.assertIn('++research_layout_output="/run/layout_audit.json"', cmd)
        self.assertIn("++manager_env.config.research_layout_audit=true", cmd)
        self.assertIn("++max_render_steps=501", cmd)
        with self.assertRaises(ValueError):
            evaluation_command(Path("/run"), Path("/data"), "fixture", 2, cat_scene=Path("/cat"), layout_audit=True)
        with self.assertRaises(ValueError):
            LayoutLimits(trunk_radius=float("nan"))


if __name__ == "__main__":
    unittest.main()
