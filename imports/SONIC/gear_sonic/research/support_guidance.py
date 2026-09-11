"""Bounded 2.5D geometric guidance; no policy, commands, or balance guarantee.

Nodes are supported trunk-clear patches. Edges require continuous known transit
support, trunk clearance and a bounded height excursion. A stride over a tread
edge is allowed; a stride across unknown ground or a wall is not. No guidance is
invented for unreachable cells, and no interpolation across invalid cells occurs.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class SupportGraph:
    MAX_CELLS = 1_048_576
    MAX_DIRECTED_CANDIDATES = 8_000_000
    MAX_REACH = 16

    def __init__(self, height, walkable, resolution, max_step, *, transit=None, max_stride=None):
        self.height, self.walkable = np.asarray(height, dtype=np.float64), np.asarray(walkable)
        h, w = self.height, self.walkable
        if h.ndim != 2 or h.shape != w.shape or w.dtype != bool or not h.size:
            raise ValueError("Expected nonempty matching height/bool grids")
        if h.size > self.MAX_CELLS:
            raise ValueError("Support graph exceeds cell budget")
        if not np.isfinite([resolution, max_step]).all() or min(resolution, max_step) <= 0:
            raise ValueError("Invalid graph dimensions")
        t = w if transit is None else np.asarray(transit)
        if t.shape != w.shape or t.dtype != bool or not np.isfinite(h[w | t]).all():
            raise ValueError("Walkable/transit masks need known finite support")
        if (w & ~t).any():
            raise ValueError("Walkable support must also be valid transit")
        stride = math.sqrt(2)*resolution if max_stride is None else max_stride
        if not math.isfinite(stride) or stride < resolution or stride/resolution > self.MAX_REACH:
            raise ValueError("Invalid/unbounded stride")
        self.resolution = resolution
        reach = math.ceil(stride/resolution)
        offsets = [(dx, dy) for dx in range(reach+1) for dy in range(-reach, reach+1)
                   if (dx > 0 or dy > 0) and math.hypot(dx, dy)*resolution <= stride+1e-8]
        if 2*len(offsets)*h.size > self.MAX_DIRECTED_CANDIDATES:
            raise ValueError("Support graph exceeds edge candidate budget")
        rows, cols, weights = [], [], []
        flat = np.arange(h.size, dtype=np.int32).reshape(h.shape)
        for dx, dy in offsets:
            lower = (max(0, -dx), max(0, -dy))
            upper = (min(h.shape[0], h.shape[0]-dx), min(h.shape[1], h.shape[1]-dy))
            if any(a >= b for a, b in zip(lower, upper)):
                continue
            def view(array, i=0, j=0):
                return array[lower[0]+i:upper[0]+i, lower[1]+j:upper[1]+j]
            valid = view(w) & view(w, dx, dy)
            cells = set()
            for fraction in np.linspace(0., 1., math.ceil(math.hypot(dx, dy)*2)+1):
                x, y = fraction*dx, fraction*dy
                # Avoid adding a spurious cell due to near-integer roundoff.
                if abs(x-round(x)) < 1e-12:
                    x = round(x)
                if abs(y-round(y)) < 1e-12:
                    y = round(y)
                cells.update((i, j) for i in (math.floor(x), math.ceil(x))
                             for j in (math.floor(y), math.ceil(y)))
            low, high = view(h).copy(), view(h).copy()
            for i, j in sorted(cells):
                valid &= view(t, i, j)
                low = np.minimum(low, view(h, i, j))
                high = np.maximum(high, view(h, i, j))
            valid &= high-low <= max_step
            a, b = view(flat)[valid], view(flat, dx, dy)[valid]
            distance = np.sqrt((resolution*dx)**2+(resolution*dy)**2+
                               (view(h, dx, dy)[valid]-view(h)[valid])**2)
            rows.extend((a, b))
            cols.extend((b, a))
            weights.extend((distance, distance))
        self.matrix = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(h.size, h.size)) if rows else csr_matrix((h.size, h.size))
        self.matrix.sort_indices()

    def index(self, cell):
        if cell is None or len(cell) != 2 or any(isinstance(v, (bool, np.bool_)) or
                                               not isinstance(v, (int, np.integer)) for v in cell):
            return None
        cell = tuple(cell)
        if any(v < 0 or v >= n for v, n in zip(cell, self.height.shape)) or not self.walkable[cell]:
            return None
        return int(np.ravel_multi_index(cell, self.height.shape))

    def goal_field(self, goal):
        index = self.index(goal)
        if index is None:
            raise ValueError("Goal must be an in-bounds walkable support cell")
        distance, predecessor = dijkstra(self.matrix, directed=True, indices=index, return_predecessors=True)
        reachable = np.isfinite(distance) & self.walkable.ravel()
        next_index = np.where(reachable, predecessor, -1).astype(np.int32)
        next_index[index] = -1  # no edge/velocity at the goal
        next_index[~reachable] = -1
        points = np.argwhere(np.ones(self.height.shape, bool))
        delta = np.full((self.height.size, 3), np.nan, dtype=np.float64)
        moving = reachable & (next_index >= 0)
        src, dst = np.flatnonzero(moving), next_index[moving]
        delta[src, :2] = (points[dst]-points[src])*self.resolution
        delta[src, 2] = self.height.ravel()[dst]-self.height.ravel()[src]
        delta[index] = 0.
        direction = delta.copy()
        direction[moving] /= np.linalg.norm(direction[moving], axis=1)[:, None]
        if not np.all(distance[dst] < distance[src]):
            raise ValueError("Guidance must strictly descend goal cost")
        shape = self.height.shape
        return dict(goal_cost=np.where(reachable, distance, np.nan).reshape(shape),
                    reachable=reachable.reshape(shape), next_index=next_index.reshape(shape),
                    direction=direction.reshape(*shape, 3), next_delta=delta.reshape(*shape, 3))

    def route(self, start, goal, field=None):
        source, target = self.index(start), self.index(goal)
        if source is None or target is None:
            return []
        field = self.goal_field(goal) if field is None else field
        if not field["reachable"].ravel()[source]:
            return []
        result = [source]
        for _ in range(self.height.size):
            if source == target:
                return [tuple(np.unravel_index(p, self.height.shape)) for p in result]
            source = int(field["next_index"].ravel()[source])
            if source < 0:
                break
            result.append(source)
        raise ValueError("Broken/cyclic guidance route")
