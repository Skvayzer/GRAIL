from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.layout_validation import LayoutLimits
from gear_sonic.research.pelvis_guidance import PelvisGraphAttachment, segment_cell_cover
from gear_sonic.research.support_guidance import SupportGraph
from audit_pelvis_guidance import reference_cover, check_connector


def fixture(kind="flat", goal=(22, 8), device="cpu"):
    h = np.zeros((25, 17), np.float32)
    walk, transit = np.ones_like(h, bool), np.ones_like(h, bool)
    if kind == "stairs":
        h[10:] = .16
        walk[9:11] = False
    elif kind in ("hole", "wall", "cliff"):
        if kind == "cliff":
            h[12:] = .5
        else:
            transit[12] = walk[12] = False
            if kind == "hole":
                h[12] = np.nan
    limits = LayoutLimits()
    graph = SupportGraph(h, walk, .04, .20, transit=transit, max_stride=.28)
    field = graph.goal_field(goal)
    return PelvisGraphAttachment(h, transit, field["reachable"], field["goal_cost"], [0., 0.], goal, limits, device), graph


def query(sampler, xy, support=None, known=None):
    xy = torch.tensor(xy, dtype=torch.float32, device=sampler.device).reshape(-1, 2)
    support = torch.zeros(len(xy), device=sampler.device) if support is None else torch.tensor(support, dtype=torch.float32, device=sampler.device)
    known = torch.ones(len(xy), dtype=bool, device=sampler.device) if known is None else torch.tensor(known, device=sampler.device)
    return sampler.sample(torch.cat((xy, (support+.8)[:, None]), -1), support, known)


class PelvisGuidanceTests(unittest.TestCase):
    def test_tread_transit_both_directions_without_changing_support_nodes(self):
        xy = [[x*.04, .32] for x in (9., 9.3, 9.8, 10., 10.4, 10.9)]
        for goal, sign in (((22, 8), 1), ((2, 8), -1)):
            sampler, graph = fixture("stairs", goal)
            before = graph.walkable.copy()
            result = query(sampler, xy, [.0, .0, .16, .16, .16, .16])
            self.assertTrue(result["valid"].all())
            self.assertTrue((result["direction"][:, 0]*sign > 0).all())
            self.assertTrue((result["connector_xy"] <= .280001).all())
            self.assertTrue((result["covered_cells"] > 1).all())
            self.assertFalse(graph.walkable[9:11].any())
            np.testing.assert_array_equal(before, graph.walkable)
            for cell in result["target_cell"].tolist():
                self.assertIsNotNone(graph.index(cell))

    def test_holes_walls_and_excessive_risers_cannot_be_skipped(self):
        for kind in ("hole", "wall", "cliff"):
            sampler, _ = fixture(kind)
            result = query(sampler, [[11*.04, .32]])
            self.assertFalse(result["valid"].any(), kind)
            self.assertEqual(result["target_cell"].tolist(), [[-1, -1]])
            self.assertEqual(int(torch.count_nonzero(result["direction"])), 0)

    def test_actual_position_unknown_height_and_bounds_reject(self):
        sampler, _ = fixture()
        for xy, height, known in (([.4, .32], .6, True), ([.4, .32], 0., False),
                                  ([-.0001, .32], 0., True), ([.9601, .32], 0., True)):
            self.assertFalse(query(sampler, [xy], [height], [known])["valid"].any())
        # Root's own occupied cell cannot be escaped by snapping to a neighbour.
        sampler.transit[10, 8] = False
        self.assertFalse(query(sampler, [[.4, .32]])["valid"].any())

    def test_blocked_corner_and_line_on_boundary_are_both_covered(self):
        cells = torch.tensor([[[0., 0.], [0., 1.], [1., 0.], [1., 1.], [2., 0.]]])
        cover = segment_cell_cover(torch.tensor([[0., 0.]]), torch.tensor([[[1., 1.]]]), cells)
        self.assertEqual(cover.tolist(), [[[True, True, True, True, False]]])
        cover = segment_cell_cover(torch.tensor([[.5, 0.]]), torch.tensor([[[.5, 1.]]]), cells)
        self.assertEqual(cover.tolist(), [[[True, True, True, True, False]]])
        cover = segment_cell_cover(torch.tensor([[0., 0.]]), torch.tensor([[[0., 0.]]]), cells)
        self.assertEqual(cover.tolist(), [[[True, False, False, False, False]]])

    def test_independent_scalar_cover_matches_batched_random_segments(self):
        rng = np.random.default_rng(42)
        starts = rng.uniform(-.49, .49, (16, 2)).astype(np.float32)
        offsets = np.stack(np.meshgrid(np.arange(-4, 5), np.arange(-4, 5), indexing="ij"), -1).reshape(-1, 2)
        ends = rng.integers(-3, 4, (16, 5, 2)).astype(np.float32)
        actual = segment_cell_cover(torch.tensor(starts), torch.tensor(ends),
                                    torch.tensor(offsets, dtype=torch.float32)[None].expand(16, -1, -1))
        for i in range(16):
            for j in range(5):
                self.assertEqual(set(map(tuple, offsets[actual[i, j].numpy()])),
                                 set(reference_cover(starts[i], ends[i, j])))

    def test_independent_witness_rejects_blocked_connector_and_unknown_target(self):
        from types import SimpleNamespace
        attachment, graph = fixture()
        sampler = SimpleNamespace(attachment=attachment, origin=attachment.origin, resolution=.04)
        arrays = dict(support_height=graph.height.copy(), transit_valid=np.ones_like(graph.walkable),
                      walkable=graph.walkable.copy())
        root, target = np.array([.4, .32, .8]), (14, 8)
        self.assertGreater(check_connector(root, 0., target, arrays, sampler), 1)
        arrays["transit_valid"][12, 8] = False
        with self.assertRaisesRegex(ValueError, "blocked"):
            check_connector(root, 0., target, arrays, sampler)
        arrays["transit_valid"][12, 8] = True
        arrays["support_height"][12, 8] = .5
        with self.assertRaisesRegex(ValueError, "height"):
            check_connector(root, 0., target, arrays, sampler)
        with self.assertRaisesRegex(ValueError, "supported"):
            check_connector(root, 0., (-1, -1), arrays, sampler)

    def test_continuous_root_does_not_steer_back_to_its_cell_centre(self):
        sampler, _ = fixture()
        costs = []
        for x in np.linspace(.16, .83, 51):
            result = query(sampler, [[x, .32]])
            self.assertTrue(result["valid"].all())
            self.assertGreater(float(result["direction"][0, 0]), .999)
            costs.append(float(result["cost"][0]))
        self.assertTrue((np.diff(costs) < 0).all())

    def test_goal_is_zero_but_invalid_goal_cell_is_not_arrival(self):
        sampler, _ = fixture()
        result = query(sampler, [[.881, .32]])
        self.assertTrue(result["valid"].all() and result["at_goal"].all())
        self.assertEqual(float(result["cost"][0]), 0.)
        self.assertEqual(int(torch.count_nonzero(result["direction"])), 0)
        self.assertEqual(result["target_cell"].tolist(), [[22, 8]])
        sampler.transit[22, 8] = False
        self.assertFalse(query(sampler, [[.881, .32]])["valid"].any())

    def test_resource_and_corruption_bounds(self):
        sampler, _ = fixture()
        with self.assertRaises(ValueError):
            query(sampler, [[.4, .32]]*17)
        with self.assertRaises(ValueError):
            query(sampler, [[float("nan"), .32]])
        with self.assertRaises(ValueError):
            query(sampler, [[.4, .32]], [float("nan")], [True])
        with self.assertRaisesRegex(ValueError, "budget"):
            PelvisGraphAttachment(np.zeros((2, 2)), np.ones((2, 2), bool), np.ones((2, 2), bool),
                np.zeros((2, 2)), [0, 0], [0, 0], replace(LayoutLimits(), max_stride=3.))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cpu_cuda_single_batch_and_tf32_parity(self):
        cpu, _ = fixture("stairs")
        gpu, _ = fixture("stairs", device="cuda:0")
        xy = [[x*.04, .3217] for x in np.linspace(8.31, 11.03, 16)]
        height = [0. if x[0] < .4 else .16 for x in xy]
        expected = query(cpu, xy, height)
        previous = torch.backends.cuda.matmul.allow_tf32
        try:
            for tf32 in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                actual = query(gpu, xy, height)
                for key in expected:
                    torch.testing.assert_close(actual[key].cpu(), expected[key], atol=2e-6, rtol=1e-5)
                for i in range(16):
                    single = query(gpu, xy[i:i+1], height[i:i+1])
                    for key in actual:
                        torch.testing.assert_close(single[key], actual[key][i:i+1], atol=2e-6, rtol=1e-5)
                self.assertEqual(torch.backends.cuda.matmul.allow_tf32, tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous


if __name__ == "__main__":
    unittest.main()
