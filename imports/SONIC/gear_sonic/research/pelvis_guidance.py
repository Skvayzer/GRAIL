"""Bounded virtual-root connections to the unchanged support graph.

The moving pelvis need not project onto a stance patch. It may connect to a
goal-reachable patch only through known, trunk-clear transit cells, with the
same stride/height bounds as support edges. This is grid-level geometric
screening, NOT foot placement, full-body feasibility or contact permission.
"""
import math

import numpy as np
import torch


def segment_cell_cover(start, end, cells):
    """Closed line/closed grid-cell intersections, including corner touches.

    Inputs in grid coordinates: start Bx2, end BxCx2, cells BxKx2. Cells
    represent centre +/- 0.5. Exact slab intersection instead of sampling a
    few points along a segment (which could miss a blocked corner cell).
    """
    enter = start.new_zeros((len(start), end.shape[1], cells.shape[1]))
    leave = torch.ones_like(enter)
    for axis in range(2):
        delta = (end[..., axis]-start[:, None, axis])[:, :, None]
        lower = cells[:, None, :, axis]-.5-start[:, None, None, axis]
        upper = lower+1.
        parallel = delta == 0
        divisor = torch.where(parallel, 1., delta)
        a, b = lower/divisor, upper/divisor
        near, far = torch.minimum(a, b), torch.maximum(a, b)
        inside = (lower <= 1e-6) & (upper >= -1e-6)
        near = torch.where(parallel, torch.where(inside, -torch.inf, torch.inf), near)
        far = torch.where(parallel, torch.where(inside, torch.inf, -torch.inf), far)
        enter, leave = torch.maximum(enter, near), torch.minimum(leave, far)
    return enter <= leave+1e-6


class PelvisGraphAttachment:
    VERSION = "bounded-transit-root-v2"
    MAX_REACH = 8
    MAX_BATCH = 16
    COST_TIE_M = 1e-5

    def __init__(self, height, transit, reachable, cost, origin, goal, limits, device="cpu"):
        self.device = torch.device(device)
        self.resolution, self.max_step, self.max_stride = limits.resolution, limits.max_step, limits.max_stride
        reach = math.ceil(self.max_stride/self.resolution-1e-12)+1
        if reach > self.MAX_REACH or reach < 2:
            raise ValueError("Pelvis connector exceeds bounded local search budget")
        h, t, r, c = map(np.asarray, (height, transit, reachable, cost))
        if (h.ndim != 2 or min(h.shape) < 2 or h.size > 1_048_576
                or any(x.shape != h.shape for x in (t, r, c)) or t.dtype != bool or r.dtype != bool
                or (r & ~t).any() or not np.isfinite(h[t]).all()
                or not np.isfinite(c[r]).all() or (c[r] < 0).any()
                or np.shape(origin) != (2,) or not np.isfinite(origin).all()):
            raise ValueError("Invalid support/transit/goal field")
        if (len(goal) != 2 or any(type(i) not in (int, np.int64, np.int32) for i in goal)
                or any(i < 0 or i >= n for i, n in zip(goal, h.shape))
                or not r[tuple(goal)] or c[tuple(goal)] != 0):
            raise ValueError("Expected an explicit reachable zero-cost goal")
        self.height = torch.tensor(h.copy(), dtype=torch.float32, device=device)
        self.transit = torch.tensor(t.copy(), device=device)
        self.reachable = torch.tensor(r.copy(), device=device)
        self.cost = torch.tensor(c.copy(), dtype=torch.float32, device=device)
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.goal = torch.tensor(goal, device=device)
        axes = torch.arange(-reach, reach+1, device=device)
        self.offsets = torch.stack(torch.meshgrid(axes, axes, indexing="ij"), -1).reshape(-1, 2)

    def manifest(self):
        return dict(version=self.VERSION, max_connector_xy_m=self.max_stride, max_height_range_m=self.max_step,
            cover="closed segment versus all intersected grid cells; corner/boundary contacts included",
            target="goal-reachable supported node; moving root requires transit, not a stance patch",
            selection="connector 3D endpoint length + graph cost; longest lookahead within cost tie; stable index",
            cost_tie_m=self.COST_TIE_M, stateful=False, unknown_fallback=False,
            contact_permission=False, continuous_geometry_certified=False)

    def sample(self, root, support, known):
        b = len(root)
        if (not 1 <= b <= self.MAX_BATCH or root.shape != (b, 3) or support.shape != (b,) or known.shape != (b,)
                or root.dtype != torch.float32 or support.dtype != torch.float32 or known.dtype != torch.bool
                or any(x.device != self.device for x in (root, support, known)) or not torch.isfinite(root).all()
                or not torch.isfinite(support[known]).all()):
            raise ValueError("Bounded finite root and explicit known-support mask required")
        q = (root[:, :2]-self.origin)/self.resolution
        shape = self.goal.new_tensor(self.height.shape)
        # No clamp-to-edge fallback; out-of-domain queries remain invalid.
        bounded = ((q >= 0) & (q <= shape-1)).all(-1)
        base = q.round().long()
        local = q-base
        cells = base[:, None]+self.offsets
        in_bounds = ((cells >= 0) & (cells < shape)).all(-1)
        safe = torch.minimum(cells.clamp_min(0), shape-1)
        i, j = safe.unbind(-1)
        transit = self.transit[i, j] & in_bounds
        reachable = self.reachable[i, j] & in_bounds
        heights = self.height[i, j]
        costs = self.cost[i, j]
        ends = self.offsets.to(root.dtype)[None].expand(b, -1, -1)
        cover = segment_cell_cover(local, ends, ends)
        clear = ~(cover & ~transit[:, None]).any(-1)
        low = torch.where(cover, heights[:, None], torch.inf).amin(-1)
        high = torch.where(cover, heights[:, None], -torch.inf).amax(-1)
        low, high = torch.minimum(low, support[:, None]), torch.maximum(high, support[:, None])
        delta_xy = (ends-local[:, None])*self.resolution
        distance_xy = torch.linalg.vector_norm(delta_xy, dim=-1)
        delta = torch.cat((delta_xy, (heights-support[:, None])[..., None]), -1)
        distance = torch.linalg.vector_norm(delta, dim=-1)
        at_goal_cell = (base == self.goal).all(-1)
        goal_target = (cells == self.goal).all(-1)
        # Never guide back into the current cell centre. Prefer a forward graph
        # connection instead; arrival retains the existing goal-cell convention.
        other_cell = (self.offsets != 0).any(-1)[None]
        valid = (bounded & known)[:, None] & reachable & clear
        valid &= (high-low <= self.max_step+1e-6) & (distance_xy <= self.max_stride+1e-6)
        valid &= other_cell | (at_goal_cell[:, None] & goal_target)
        total = torch.where(valid, distance+costs, torch.inf)
        best = total.amin(-1)
        near_best = valid & (total <= best[:, None]+self.COST_TIE_M)
        longest = torch.where(near_best, distance_xy, -torch.inf).amax(-1)
        candidates = near_best & (distance_xy >= longest[:, None]-1e-6)
        # argmax returns the first (lexicographically fixed) tied candidate.
        pick = candidates.int().argmax(-1)
        row = torch.arange(b, device=self.device)
        arrival = at_goal_cell & (valid & goal_target).any(-1)
        pick = torch.where(arrival, goal_target.int().argmax(-1), pick)
        selected = valid[row, pick]
        direction = delta[row, pick]/distance[row, pick, None].clamp_min(1e-8)
        direction = torch.where((selected & ~arrival)[:, None], direction, 0.)
        result_cost = torch.where(arrival, 0., total[row, pick])
        return dict(valid=selected, at_goal=arrival, direction=direction,
            cost=torch.where(selected, result_cost, 0.),
            target_cell=torch.where(arrival[:, None], self.goal,
                torch.where(selected[:, None], cells[row, pick], -1)),
            connector_xy=torch.where(selected & ~arrival, distance_xy[row, pick], 0.),
            covered_cells=torch.where(selected, cover[row, pick].sum(-1), 0),
            candidate_count=valid.sum(-1))
