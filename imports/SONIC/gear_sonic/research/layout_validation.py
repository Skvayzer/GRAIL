"""Conservative *geometric screening*, not articulated feasibility certification.

Requires separate terrain support and CAT forbidden geometry. A support graph
can reject sealed passages, cliffs and unsupported gaps; its acceptance is not
a balance/reachability guarantee. The unchanged full-body reference must also
pass its independent collider-cover gate before a diagnostic rollout.
"""
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np
from scipy import ndimage
import torch


@dataclass(frozen=True)
class LayoutLimits:
    resolution: float = .04
    max_step: float = .20
    max_stride: float = .28
    min_up_normal: float = .70710678
    support_patch_half_width: float = .04
    support_patch_height_range: float = .06
    trunk_radius: float = .20
    trunk_bottom: float = .45  # sphere-centre range above support, not pelvis height
    trunk_top: float = 1.35
    clearance: float = .03
    endpoint_snap: float = .12

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Layout limits must be finite and positive")
        if self.trunk_top <= self.trunk_bottom or self.min_up_normal > 1.:
            raise ValueError("Invalid trunk/normal limits")


def support_path(height, walkable, start, goal, resolution, max_step, *, transit=None, max_stride=None):
    """Dijkstra over support patches with bounded risers/stride and swept screening.

    The default is an 8-neighbour graph. With explicit transit cells, a stride may
    cross a tread edge that cannot hold a patch, but NEVER unknown support, a
    blocked trunk envelope or a height excursion greater than max_step. A
    conservative line-cell cover also prevents diagonal corner cutting.
"""
    height, walkable = np.asarray(height), np.asarray(walkable)
    if height.ndim != 2 or height.shape != walkable.shape or walkable.dtype != bool:
        raise ValueError("Expected matching height/bool grids")
    if resolution <= 0 or max_step <= 0 or not np.isfinite([resolution, max_step]).all():
        raise ValueError("Invalid graph dimensions")
    if not np.isfinite(height[walkable]).all():
        raise ValueError("Walkable cells must have known support")
    transit = walkable if transit is None else np.asarray(transit)
    if transit.shape != walkable.shape or transit.dtype != bool or not np.isfinite(height[transit]).all():
        raise ValueError("Transit mask needs known finite support")
    stride = math.sqrt(2)*resolution if max_stride is None else max_stride
    if not math.isfinite(stride) or stride < resolution:
        raise ValueError("Invalid stride bound")
    offsets = []
    reach = math.ceil(stride/resolution)
    for dx in range(-reach, reach+1):
        for dy in range(-reach, reach+1):
            length = math.hypot(dx, dy)
            if not length or length*resolution > stride+1e-8:
                continue
            cells = set()
            for fraction in np.linspace(0., 1., math.ceil(length*2)+1):
                x, y = fraction*dx, fraction*dy
                for i in (math.floor(x), math.ceil(x)):
                    for j in (math.floor(y), math.ceil(y)):
                        cells.add((i, j))
            offsets.append((dx, dy, tuple(cells)))
    def valid(p):
        return 0 <= p[0] < height.shape[0] and 0 <= p[1] < height.shape[1] and walkable[p]
    def edge(a, b):
        return valid(a) and valid(b) and abs(height[a]-height[b]) <= max_step
    start, goal = tuple(start), tuple(goal)
    if not valid(start) or not valid(goal):
        return []
    queue, cost, parent = [(0., start)], {start: 0.}, {}
    while queue:
        distance, a = heapq.heappop(queue)
        if distance != cost[a]:
            continue
        if a == goal:
            path = [a]
            while a in parent:
                a = parent[a]
                path.append(a)
            return path[::-1]
        for dx, dy, cells in offsets:
            b = a[0]+dx, a[1]+dy
            if not edge(a, b):
                continue
            swept = [(a[0]+i, a[1]+j) for i, j in cells]
            if any(not transit[p] for p in swept):
                continue
            heights = [height[p] for p in swept]
            if max(heights)-min(heights) > max_step:
                continue
            value = distance + math.sqrt((resolution*dx)**2 + (resolution*dy)**2 + (height[b]-height[a])**2)
            if value < cost.get(b, float("inf")):
                cost[b], parent[b] = value, a
                heapq.heappush(queue, (value, b))
    return []


def layout_grid(surface, clutter, placement, anchors, limits=LayoutLimits()):
    anchors = torch.as_tensor(anchors, dtype=torch.float32, device=surface.device)
    if anchors.ndim != 2 or anchors.shape[-1] != 3 or len(anchors) < 2 or not torch.isfinite(anchors).all():
        raise ValueError("Expected a finite anchor trajectory")
    bounds = torch.cat((surface.vertices, anchors))
    lo = torch.floor((bounds[:, :2].amin(0)-.5)/limits.resolution)*limits.resolution
    hi = bounds[:, :2].amax(0)+.5
    shape = torch.ceil((hi-lo)/limits.resolution).int().tolist()
    if max(shape) > 1024 or min(shape) < 2:
        raise ValueError("Unbounded/invalid layout extent")
    ii, jj = torch.meshgrid(*(torch.arange(n, device=surface.device) for n in shape), indexing="ij")
    xy = torch.stack((ii, jj), -1)*limits.resolution+lo
    z_top = float(bounds[:, 2].amax())+2.
    points = torch.cat((xy, torch.full((*shape, 1), z_top, device=surface.device)), -1)
    height, normal, known = surface.support_below(points)
    support = known & (normal[..., 2] >= limits.min_up_normal)
    # Small support patch, not a whole-foot or double-support guarantee.
    samples = []
    for dx in (-limits.support_patch_half_width, 0., limits.support_patch_half_width):
        for dy in (-limits.support_patch_half_width, 0., limits.support_patch_half_width):
            h, n, v = surface.support_below(points+points.new_tensor([dx, dy, 0.]))
            support &= v & (n[..., 2] >= limits.min_up_normal)
            samples.append(h)
    patch = torch.stack(samples)
    support &= (patch.amax(0)-patch.amin(0)) <= limits.support_patch_height_range
    # A sphere chain conservatively covers this chosen upright trunk envelope.
    # Legs, arms, bent postures are covered by the separate reference gate.
    levels = torch.linspace(limits.trunk_bottom, limits.trunk_top,
        math.ceil((limits.trunk_top-limits.trunk_bottom)/.08)+1, device=surface.device)
    dz = float(levels[1]-levels[0])
    radius = math.hypot(limits.trunk_radius, dz/2)
    centers = points[..., None, :].expand(*shape, len(levels), 3).clone()
    centers[..., 2] = height[..., None]+levels
    gap, _, valid = clutter.query(placement.to_local(centers))
    gap = (gap-radius).amin(-1)
    transit = known & valid.all(-1) & (gap >= limits.clearance)
    walkable = support & transit
    h_np, w_np = height.cpu().numpy(), walkable.cpu().numpy()
    xy_np = xy.cpu().numpy()
    # A reference may return to its starting region; use its furthest XY
    # excursion as the second endpoint rather than accepting a trivial loop.
    endpoint = int(torch.linalg.vector_norm(anchors[:, :2]-anchors[0, :2], dim=-1).argmax())
    indices, snaps = [], []
    for anchor in anchors[[0, endpoint]]:
        distance = torch.linalg.vector_norm(xy-anchor[:2], dim=-1)
        support_height, _, support_known = surface.support_below(anchor[None])
        candidates = walkable & (distance <= limits.endpoint_snap)
        candidates &= support_known[0] & ((height-support_height[0]).abs() <= limits.max_step)
        if not candidates.any():
            indices.append(None)
            snaps.append(None)
        else:
            index = int(torch.where(candidates, distance, float("inf")).argmin())
            indices.append(np.unravel_index(index, shape))
            snaps.append(float(distance[indices[-1]]))
    path = support_path(h_np, w_np, *indices, limits.resolution, limits.max_step,
        transit=transit.cpu().numpy(), max_stride=limits.max_stride) if all(p is not None for p in indices) else []
    route = np.array([[*xy_np[p], h_np[p]] for p in path], dtype=np.float32).reshape(-1, 3)
    # Anchor support check catches a scene which is upright but misplaced below
    # the recorded motion. It is deliberately NOT labelled foot-contact proof.
    anchor_h, anchor_n, anchor_known = surface.support_below(anchors, max_drop=1.3)
    anchor_drop = anchors[:, 2]-anchor_h
    # The pelvis need not project onto a stance foothold (especially crossing
    # a riser). Up-normal/patch constraints belong to the stance graph above.
    anchor_supported = anchor_known & (anchor_drop >= .35)
    report = dict(limits=asdict(limits), shape=shape, xy_origin=lo.tolist(),
        support_cells=int(support.sum()), traversable_screen_cells=int(walkable.sum()),
        endpoint_reference_frames=[0, endpoint], endpoint_snap_m=snaps,
        geometric_route_found=len(route) > 1, route_points=len(route),
        route_length_m=float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()) if len(route) else None,
        anchor_support_fraction=float(anchor_supported.float().mean()),
        anchor_over_upward_surface_fraction=float((anchor_known & (anchor_n[:, 2] >= limits.min_up_normal)).float().mean()),
        supported_anchor_height_range_m=[float(anchor_drop[anchor_supported].min()), float(anchor_drop[anchor_supported].max())] if anchor_supported.any() else None,
        whole_body_dynamic_feasibility_verified=False,
        support_contract="upward terrain/ground, small patch, bounded step graph; NOT contact/balance certification",
        passage_contract="upright trunk screen plus separately required sampled whole-body reference check",
        multi_level_terrain_supported=False)
    arrays = dict(support_height=h_np, support_valid=support.cpu().numpy(), transit_valid=transit.cpu().numpy(), walkable=w_np,
        xy=xy_np, clutter_gap=gap.cpu().numpy(), route=route, reference_anchors=anchors.cpu().numpy())
    return report, arrays


def obstacle_embedding(surface, occupied, origin, resolution, placement):
    """Flag burial/floating relative to terrain top, not signed intersection.

    Overhead obstacles need an explicit semantic review; they are never treated
    as footholds. The open stair mesh does not justify solid inside/outside claims.
    """
    labels, count = ndimage.label(occupied, ndimage.generate_binary_structure(3, 1))
    result = []
    for label in range(1, count+1):
        indices = np.argwhere(labels == label)
        points = torch.as_tensor(np.array(origin)+(indices+.5)*resolution,
                                 dtype=torch.float32, device=surface.device)
        world = placement.to_world(points)
        tops = world.clone()
        tops[:, 2] = max(float(surface.vertices[:, 2].max()), float(world[:, 2].max()))+2.
        height, _, valid = surface.support_below(tops)
        buried = valid & (world[:, 2]+resolution/2 < height-.01)
        # Columns' lowest cells identify floating bases, not an arbitrary top cell.
        _, first = np.unique(indices[:, :2], axis=0, return_index=True)
        base_gap = world[first, 2]-resolution/2-height[first]
        result.append(dict(component=label, occupied_voxels=len(indices),
            below_support_envelope_fraction=float(buried.float().mean()),
            exposed_voxels=int((valid & ~buried).sum()), unknown_support_voxels=int((~valid).sum()),
            floating_base_columns=int((base_gap > .08).sum()), base_columns=len(first),
            requires_placement_review=bool(buried.any() or (base_gap > .08).any())))
    return result
