import math
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluation_command
from gear_sonic.research.cat_geometry import Placement, CatFields
from gear_sonic.research.mesh_distance import ClosedMeshDistance, reference_clearance_summary


class CatIntegrationTests(unittest.TestCase):
    def test_cat_field_vectors_rotate_and_outside_is_unknown_not_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sdf = np.indices((4, 4, 4))[0].astype(np.float32)
            vector = np.zeros((4, 4, 4, 3), np.float32)
            vector[..., 0] = 1.
            for name, array in dict(obs=sdf < 1, sdf=sdf, bf=vector, gf=vector, travel=sdf).items():
                np.save(directory/(name+'.npy'), array, allow_pickle=False)
            (directory/'scene.usda').write_text('fixture not opened for array queries')
            meta = dict(schema='cat-isaac-scene-v1', axis_order='xyz', units='m', resolution=1.,
                        shape=[4, 4, 4], origin_corner=[0., 0., 0.], sample_origin=[.5, .5, .5])
            meta['files'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
            (directory/'scene.json').write_text(json.dumps(meta))
            placement = Placement((3., -2., 1.), math.pi/2)
            field = CatFields(directory, placement)
            points = placement.to_world(torch.tensor([[1.5, 1.5, 1.5], [-.1, 0., 0.]]))
            sampled = field.sample(points)
            self.assertEqual(sampled['valid'].tolist(), [True, False])
            self.assertAlmostEqual(float(sampled['sdf'][0]), 1.)
            torch.testing.assert_close(sampled['gf'][0], torch.tensor([0., 1., 0.]))
            self.assertTrue(torch.isnan(sampled['gf'][1]).all())
            self.assertTrue(torch.isnan(sampled['sdf'][1]))
            (directory/'gf.npy').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                CatFields(directory)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA distance parity needs a GPU")
    def test_cuda_distance_matches_cpu(self):
        cube = trimesh.creation.box(extents=(1., 2., 3.))
        points = torch.tensor([[2., .1, .1], [.4, .1, .1], [0., 0., 0.], [-2., -3., 4.]])
        a = ClosedMeshDistance(cube.vertices, cube.faces).query(points)
        b = ClosedMeshDistance(cube.vertices, cube.faces, 'cuda:0').query(points.cuda())
        torch.testing.assert_close(a[0], b[0].cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(a[2], b[2].cpu())
        # At the exact box centre, +X and -X are equally close. The signed
        # distance agrees, but a closest-surface normal is not unique there.
        torch.testing.assert_close(a[1][[0, 1, 3]], b[1][[0, 1, 3]].cpu(), atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(b[1][2])), 1., places=5)

    def test_placement_rotates_vectors_and_roundtrips_without_modifying_input(self):
        p = Placement((3., -2., 1.), math.pi/2)
        x = torch.tensor([[1., 2., 3.], [-1., .3, .4]])
        saved = x.clone()
        torch.testing.assert_close(p.to_world(x)[0], torch.tensor([1., -1., 4.]))
        torch.testing.assert_close(p.to_local(p.to_world(x)), x)
        torch.testing.assert_close(x, saved)
        torch.testing.assert_close(p.vectors_to_world(torch.tensor([1., 0., 0.])), torch.tensor([0., 1., 0.]))
        with self.assertRaises(ValueError):
            Placement((float("nan"), 0., 0.))

    def test_mesh_distances_signed_outside_grid_and_missing_not_free(self):
        cube = trimesh.creation.box(extents=(1., 1., 1.))
        mesh = ClosedMeshDistance(cube.vertices, cube.faces)
        points = torch.tensor([[2., .1, .1], [.4, .1, .1], [.5, .1, .1], [float("nan"), 0., 0.]])
        distances, normals, valid = mesh.query(points)
        torch.testing.assert_close(distances[:3], torch.tensor([1.5, -.1, 0.]), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(normals[:3], torch.tensor([[1., 0., 0.]]*3), atol=1e-5, rtol=1e-5)
        self.assertEqual(valid.tolist(), [True, True, True, False])
        self.assertTrue(torch.isnan(distances[-1]))
        d, _, valid = mesh.query(points[:1], max_distance=.1)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isnan(d).all())
        with self.assertRaises(ValueError):
            ClosedMeshDistance(cube.vertices, cube.faces[:-1])
        with self.assertRaises(ValueError):
            ClosedMeshDistance(cube.vertices, cube.faces[:, ::-1])

    def test_reference_gate_rejects_limbs_even_when_root_clears(self):
        gaps = torch.tensor([[.5, .2, .3], [.5, -.1, .2]])
        report = reference_clearance_summary(gaps, ["pelvis", "hand", "foot"])
        self.assertFalse(report["sampled_reference_clear"])
        self.assertEqual(report["flagged_frames"], 1)
        self.assertLess(report["minimum_by_link_m"]["hand"], 0)
        self.assertTrue(reference_clearance_summary(gaps+1., ["pelvis", "hand", "foot"])["sampled_reference_clear"])
        with self.assertRaises(ValueError):
            reference_clearance_summary(torch.tensor([[float("nan")]]), ["pelvis"])

    def test_cat_launch_is_explicit_and_does_not_change_policy_inputs(self):
        cmd = evaluation_command(Path("/run"), Path("/run/data"), "fixture", 2,
            cat_scene=Path("/scene"), cat_translation=(1., 2., 3.), cat_yaw=math.pi/2)
        self.assertIn('++manager_env.config.research_cat_scene="/scene"', cmd)
        self.assertIn("++max_render_steps=501", cmd)
        self.assertIn("++run_once=true", cmd)
        self.assertFalse(any("observations" in a or "rewards" in a or "train_agent" in a for a in cmd))
        normal = evaluation_command(Path("/run"), Path("/run/data"), "fixture", 1)
        self.assertFalse(any("research_cat" in a for a in normal))


if __name__ == "__main__":
    unittest.main()
