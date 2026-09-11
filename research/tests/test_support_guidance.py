import heapq
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gear_sonic.research.support_guidance import SupportGraph
from terrain_guidance import build, read_layout
from gear_sonic.research.layout_validation import LayoutLimits
from dataclasses import asdict
from cat_scenes import sha256


class GuidanceTests(unittest.TestCase):
    def test_stairs_goal_field_descends_and_path_stays_on_edges(self):
        h = np.tile(np.repeat(np.arange(3)*.1, 4)[:, None], (1, 4))
        walk = np.ones_like(h, bool)
        walk[3:5] = False
        graph = SupportGraph(h, walk, .04, .15, transit=np.ones_like(walk), max_stride=.20)
        field = graph.goal_field((11, 2))
        route = graph.route((0, 2), (11, 2), field)
        self.assertGreater(len(route), 2)
        self.assertGreater(field["direction"][..., 2][field["reachable"]].max(), 0)
        for a, b in zip(route, route[1:]):
            self.assertGreater(graph.matrix[graph.index(a), graph.index(b)], 0)
            self.assertGreater(field["goal_cost"][a], field["goal_cost"][b])
        self.assertEqual(field["goal_cost"][11, 2], 0)
        self.assertEqual(field["next_index"][11, 2], -1)
        np.testing.assert_array_equal(field["direction"][11, 2], 0)
        self.assertTrue(np.isnan(field["direction"][~walk]).all())

    def test_unknown_holes_and_cliffs_do_not_get_guidance(self):
        for kind in ("hole", "cliff", "wall"):
            h, walk = np.zeros((9, 5)), np.ones((9, 5), bool)
            transit = walk.copy()
            if kind == "cliff":
                h[4] = .5
            else:
                walk[4] = False
                transit[4] = False
                if kind == "hole":
                    h[4] = np.nan
            graph = SupportGraph(h, walk, .04, .2, transit=transit, max_stride=.2)
            field = graph.goal_field((8, 2))
            self.assertFalse(field["reachable"][0, 2], kind)
            self.assertTrue(np.isnan(field["goal_cost"][0, 2]))
            self.assertTrue(np.isnan(field["direction"][0, 2]).all())
            self.assertEqual(graph.route((0, 2), (8, 2), field), [])

    def test_no_corner_cut_and_no_goal_snapping(self):
        graph = SupportGraph(np.zeros((2, 2)), np.eye(2, dtype=bool), .1, .2)
        self.assertFalse(graph.goal_field((1, 1))["reachable"][0, 0])
        for goal in ((-1, 0), (2, 1), (0, 1), (0., 0), (False, 0), None):
            with self.assertRaises(ValueError):
                graph.goal_field(goal)

    def test_budgets_and_bad_support_reject_before_allocation(self):
        h, walk = np.zeros((5, 5)), np.ones((5, 5), bool)
        for kwargs in (dict(max_stride=100), dict(max_stride=float("nan"))):
            with self.assertRaises(ValueError):
                SupportGraph(h, walk, .04, .2, **kwargs)
        with self.assertRaises(ValueError):
            SupportGraph(h, walk, .04, .2, transit=np.zeros_like(walk))
        from unittest.mock import patch
        with patch.object(SupportGraph, "MAX_DIRECTED_CANDIDATES", 10):
            with self.assertRaisesRegex(ValueError, "budget"):
                SupportGraph(h, walk, .04, .2)

    def test_independent_eight_neighbour_dijkstra_parity(self):
        # Independent Python edge/corner checks, not the new vectorized graph.
        rng = np.random.default_rng(7)
        for _ in range(4):
            h = rng.integers(0, 4, (9, 8))*.08
            w = rng.random(h.shape) > .2
            goal = (8, 7)
            w[goal] = True
            queue, costs = [(0., goal)], {goal: 0.}
            while queue:
                cost, a = heapq.heappop(queue)
                if costs[a] != cost:
                    continue
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        b = (a[0]+dx, a[1]+dy)
                        if not (dx or dy) or not (0 <= b[0] < 9 and 0 <= b[1] < 8):
                            continue
                        cover = {(a[0]+i, a[1]+j) for i in (0, dx) for j in (0, dy)}
                        if not all(w[p] for p in cover) or np.ptp([h[p] for p in cover]) > .2:
                            continue
                        value = cost+np.sqrt((dx*.1)**2+(dy*.1)**2+(h[b]-h[a])**2)
                        if value < costs.get(b, np.inf):
                            costs[b] = value
                            heapq.heappush(queue, (value, b))
            graph = SupportGraph(h, w, .1, .2)
            actual = graph.goal_field(goal)
            self.assertEqual(int(actual["reachable"].sum()), len(costs))
            for p, cost in costs.items():
                self.assertAlmostEqual(actual["goal_cost"][p], cost, places=12)
            np.testing.assert_array_equal(graph.matrix.toarray(), graph.matrix.T.toarray())

    def fixture(self):
        h = np.zeros((5, 5))
        xy = np.stack(np.indices(h.shape), -1)*.04+[-1., 2.]
        route = np.array([[*xy[0, 0], 0.], [*xy[4, 4], 0.]])
        data = dict(support_height=h, xy=xy, route=route, walkable=np.ones_like(h, bool),
                    transit_valid=np.ones_like(h, bool), reference_anchors=route+[0., 0., .8])
        meta = dict(schema="grail-cat-layout-audit-v1", grid_file="layout.npz",
                    support_graph=dict(limits=asdict(LayoutLimits()), shape=[5, 5], xy_origin=[-1., 2.]))
        return meta, data

    def test_legacy_layout_world_coordinates_and_integrity(self):
        meta, data = self.fixture()
        field, metrics = build(meta, data)
        self.assertEqual(metrics["goal_cell"], [4, 4])
        np.testing.assert_allclose(metrics["goal_world"], [-.84, 2.16, 0.])
        self.assertTrue(metrics["start_reachable"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"layout.json"
            np.savez_compressed(path.with_suffix(".npz"), **data)
            meta["grid_sha256"] = sha256(path.with_suffix(".npz"))
            path.write_text(json.dumps(meta))
            read_layout(path)
            meta["grid_sha256"] = "0"*64
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_layout(path)
        data["xy"][0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, "axis-aligned"):
            build(meta, data)

    def test_explicit_invalid_goal_and_missing_legacy_route_reject(self):
        meta, data = self.fixture()
        data["route"] = np.empty((0, 3))
        with self.assertRaises(ValueError):
            build(meta, data)
        _, metrics = build(meta, data, goal=(4, 4), start=(0, 0))
        self.assertTrue(metrics["start_reachable"])
        with self.assertRaises(ValueError):
            build(meta, data, goal=(99, 99))


if __name__ == "__main__":
    unittest.main()
