from dataclasses import asdict
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import evaluation_command
from gear_sonic.research.cat_geometry import Placement
from gear_sonic.research.layout_validation import LayoutLimits
from gear_sonic.research.mesh_distance import ClosedMeshDistance
from gear_sonic.research.obstacle_adapter import ObstacleAdapter
from gear_sonic.research.obstacle_observation import GuidanceSampler, ObstacleObservation, ObservationSpec, packed_distance, yaw_rotation
from gear_sonic.research.terrain_surface import TerrainSurface


def terrain(device="cpu"):
    return TerrainSurface(np.array([[-3., -3., 0.], [3., -3., 0.], [3., 3., 0.], [-3., 3., 0.]], np.float32),
                          np.array([[0, 1, 2], [0, 2, 3]]), ground_z=0., device=device)


def guidance_fixture(device="cpu"):
    h = np.zeros((9, 9), np.float32)
    xy = np.stack(np.indices(h.shape), -1)*.1-.4
    meta = dict(support_graph=dict(limits=asdict(LayoutLimits(resolution=.1)), shape=[9, 9],
                                  xy_origin=[-.4, -.4], endpoint_grid_indices=[[0, 4], [8, 4]]))
    arrays = dict(support_height=h, xy=xy, walkable=np.ones_like(h, bool), transit_valid=np.ones_like(h, bool))
    return GuidanceSampler(meta, arrays, device)


class ConstantGuidance:
    def sample(self, root, rotation, surface, scale):
        value = torch.zeros((len(root), 9), device=root.device)
        value[:, 7] = 1
        return value, torch.ones(len(root), device=root.device, dtype=torch.bool)


def observer(placement=Placement(), device="cpu", spec=ObservationSpec(shape=(5, 5, 5))):
    box = trimesh.creation.box(extents=(.2, 1., 1.))
    box.apply_translation([.5, 0., .6])
    return ObstacleObservation(terrain(device), ClosedMeshDistance(box.vertices, box.faces, device),
                               placement, ConstantGuidance(), spec)


def pose(device="cpu"):
    return (torch.tensor([[0., 0., .8]], device=device), torch.tensor([[1., 0., 0., 0.]], device=device),
            torch.tensor([[[.32, 0., .8], [0., .2, .2]]], device=device), torch.tensor([.1, .05], device=device))


class ObservationTests(unittest.TestCase):
    def test_packet_shape_terrain_unsigned_and_clutter_cover_gap(self):
        source = observer()
        packet = source.sample(*pose())
        self.assertEqual(packet["volume"].shape, (1, 4, 5, 5, 5))
        self.assertEqual(packet["probes"].shape, (1, 2, 8))
        self.assertTrue(packet["valid"].all())
        # Probe at x=.32: 8cm to box, 10cm cover radius -> -2cm clearance.
        self.assertAlmostEqual(float(packet["probes"][0, 0, 4]), -.02/2, places=6)
        self.assertAlmostEqual(float(packet["probes"][0, 0, 5]), .8/2, places=6)
        args = list(pose())
        args[0][:, 2] = .1  # volume now extends below the floor
        packet = source.sample(*args)
        self.assertTrue((packet["volume"][:, 1] >= 0).all())
        self.assertAlmostEqual(float(packet["volume"][0, 1, 2, 2, 0]), .22/2, places=6)

    def test_joint_rigid_yaw_translation_equivariance(self):
        a = pose()
        placement = Placement((2., -1., 0.), math.pi/2)
        b = (placement.to_world(a[0]), torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]]),
             placement.to_world(a[2]), a[3])
        first, second = observer().sample(*a), observer(placement).sample(*b)
        for key in ("volume", "probes", "guidance", "valid"):
            torch.testing.assert_close(first[key], second[key], atol=3e-6, rtol=1e-5)

    def test_yaw_keeps_world_up_and_rejects_bad_or_vertical_forward(self):
        roll = torch.tensor([[math.cos(.2), math.sin(.2), 0., 0.]])
        torch.testing.assert_close(yaw_rotation(roll), torch.eye(3)[None])
        for q in (torch.zeros((1, 4)), torch.tensor([[float("nan"), 0., 0., 0.]]),
                  torch.tensor([[math.sqrt(.5), 0., math.sqrt(.5), 0.]])):
            with self.assertRaises(ValueError):
                yaw_rotation(q)

    def test_unknown_channels_and_nonfinite_known_distances(self):
        distance = torch.tensor([.2, float("nan")])
        mask = torch.tensor([True, False])
        torch.testing.assert_close(packed_distance(distance, mask, 2.), torch.tensor([.1, 0.]))
        with self.assertRaises(ValueError):
            packed_distance(distance, torch.ones(2, dtype=bool), 2.)
        with self.assertRaises(ValueError):
            packed_distance(-torch.ones(2), torch.ones(2, dtype=bool), 2., unsigned=True)

    def test_guidance_world_yaw_mask_and_no_outside_clamp(self):
        sampler, surface = guidance_fixture(), terrain()
        root = torch.tensor([[0., 0., .8], [4., 0., .8], [.4, 0., .8]])
        q = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]]).expand(3, -1)
        data, valid = sampler.sample(root, yaw_rotation(q), surface, 5.)
        self.assertEqual(valid.tolist(), [True, False, True])
        self.assertAlmostEqual(float(data[0, 1]), -.4/5., places=6)
        self.assertLess(float(data[0, 4]), -.99)
        torch.testing.assert_close(data[1], torch.zeros(9))
        torch.testing.assert_close(data[2, 3:6], torch.zeros(3))
        self.assertEqual(float(data[2, 8]), 1.)
        sampler.reachable[4, 4] = False
        self.assertFalse(sampler.sample(root[:1], yaw_rotation(q[:1]), surface, 5.)[1].any())

    def test_packet_missing_query_masks_gate_adapter(self):
        source = observer()
        original = source.clutter.query
        def missing(points):
            d, n, v = original(points)
            d[:, 0] = float("nan")
            v[:, 0] = False
            return d, n, v
        source.clutter.query = missing
        packet = source.sample(*pose())
        self.assertFalse(packet["valid"].any())
        self.assertEqual(float(packet["volume"][0, 0, 0, 0, 0]), 0.)
        self.assertEqual(float(packet["volume"][0, 2, 0, 0, 0]), 0.)
        net = ObstacleAdapter(2, 64)
        with torch.no_grad():
            net.head.bias.fill_(1.)
        self.assertEqual(int(torch.count_nonzero(net(packet))), 0)

    def test_specs_and_geometry_bounds(self):
        for kwargs in (dict(shape=(4, 5, 5)), dict(shape=(99,)*3), dict(spacing=-1), dict(max_batch=17)):
            with self.assertRaises(ValueError):
                ObservationSpec(**kwargs)
        args = list(pose())
        args[-1][0] = -1
        with self.assertRaises(ValueError):
            observer().sample(*args)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_cuda_observation_parity(self):
        cpu = observer().sample(*pose())
        gpu = observer(device="cuda:0").sample(*pose("cuda:0"))
        for name in cpu:
            torch.testing.assert_close(cpu[name], gpu[name].cpu(), atol=3e-6, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_non_axis_yaw_is_independent_of_tf32_and_batch_size(self):
        # The upstream Isaac evaluator enables TF32 for the actor. Geometry
        # must not inherit reduced-precision matmul or change that global flag.
        root = torch.tensor([[.0217, -.0474, .81]])
        quaternion = torch.tensor([[math.cos(.1732), 0., 0., math.sin(.1732)]])
        centers = torch.linspace(-.3173, .4317, 104*3).reshape(1, 104, 3)+root[:, None]
        radii = torch.full((104,), .0317)
        spec = ObservationSpec()
        cpu_source = observer(spec=spec)
        cpu_source.guidance = guidance_fixture()
        expected = cpu_source.sample(root, quaternion, centers, radii)
        gpu_source = observer(device="cuda:0", spec=spec)
        gpu_source.guidance = guidance_fixture("cuda:0")
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            for tf32 in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                for batch in (1, 16):
                    args = [t.expand(batch, *t.shape[1:]).contiguous().cuda() for t in (root, quaternion, centers)]
                    result = gpu_source.sample(*args, radii.cuda())
                    self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)
                    for key in expected:
                        torch.testing.assert_close(result[key].cpu(), expected[key].expand(batch, *expected[key].shape[1:]),
                                                   atol=3e-6, rtol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


class AdapterTests(unittest.TestCase):
    def test_zero_initialization_rng_and_first_head_gradient(self):
        rng = torch.get_rng_state().clone()
        net = ObstacleAdapter(2, 64)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        packet = observer().sample(*pose())
        result = net(packet)
        self.assertEqual(int(torch.count_nonzero(result)), 0)
        gradient = torch.autograd.grad(result.sum(), net.head.weight)[0]
        self.assertGreater(float(gradient.norm()), 0.)
        self.assertTrue(torch.isfinite(gradient).all())
        with torch.no_grad():
            net.head.bias.fill_(20.)
        self.assertLessEqual(float(net(packet).abs().max()), .100001)

    def test_packet_corruption_and_wrong_count_reject(self):
        net, packet = ObstacleAdapter(2, 64), observer().sample(*pose())
        packet["volume"][0, 2, 0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "validity"):
            net(packet)
        packet["volume"][0, 2, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            net(packet)
        with self.assertRaises(ValueError):
            ObstacleAdapter(0, 64)

    def test_shadow_launch_opt_in_and_has_no_action_override(self):
        args = (Path("/run"), Path("/data"), "fixture", 1)
        with self.assertRaises(ValueError):
            evaluation_command(*args, observation_shadow=True)
        command = evaluation_command(*args, cat_scene=Path("/cat"), layout_audit=True, observation_shadow=True)
        self.assertIn('++research_observation_shadow_output="/run/observation_shadow.json"', command)
        self.assertFalse(any("latent_residual=" in a or "train_agent" in a or "rewards" in a for a in command))
        default = evaluation_command(*args, cat_scene=Path("/cat"), layout_audit=True)
        self.assertFalse(any("observation_shadow" in a for a in default))
        with self.assertRaises(ValueError):
            evaluation_command(*args, cat_scene=Path("/cat"), layout_audit=True, observation_shadow=True, gui=True)


if __name__ == "__main__":
    unittest.main()
