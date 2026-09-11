"""Explicit simulation-oracle obstacle packet; never a sensor/free-space claim.

Yaw-aligned, pelvis-centred, world Z up. Terrain remains UNSIGNED and separate
from signed closed clutter. Unknown numeric values are zero only alongside a
zero validity channel; they never silently become free space or contact rights.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .layout_validation import LayoutLimits
from .support_guidance import SupportGraph


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple = (13, 13, 11)
    spacing: float = .16
    distance_scale: float = 2.
    goal_scale: float = 5.
    max_batch: int = 16

    def __post_init__(self):
        if (len(self.shape) != 3 or any(type(n) is not int or n < 3 or n % 2 != 1 for n in self.shape)
                or math.prod(self.shape) > 32768 or not all(math.isfinite(v) and v > 0 for v in
                (self.spacing, self.distance_scale, self.goal_scale)) or type(self.max_batch) is not int
                or not 1 <= self.max_batch <= 16):
            raise ValueError("Bounded odd XYZ dimensions and positive finite observation scales required")

    def manifest(self):
        return dict(**asdict(self), frame="pelvis origin / yaw only / world Z up", quaternion="WXYZ",
            volume_order="B,C,X,Y,Z", volume_channels=["clutter_signed_distance/scale", "terrain_unsigned_distance/scale",
                                                       "clutter_valid", "terrain_valid"],
            probe_channels=["local_x/scale", "local_y/scale", "local_z/scale", "radius/scale",
                            "clutter_cover_gap/scale", "terrain_unsigned_center_distance/scale",
                            "clutter_valid", "terrain_valid"],
            guidance_channels=["goal_dx/goal_scale", "goal_dy/goal_scale", "goal_support_dz/goal_scale",
                               "next_direction_x", "next_direction_y", "next_direction_z",
                               "graph_cost/goal_scale", "valid", "at_goal_cell"],
            query="exact mesh oracle at sparse sample points; not downsampled occupancy or LiDAR",
            invalid="numeric zero accompanied by validity zero; adapter output gate false",
            support_permission=False, terrain_solid_sign_known=False)


def yaw_rotation(quaternion):
    if (quaternion.ndim != 2 or quaternion.shape[1] != 4 or not torch.isfinite(quaternion).all()
            or not torch.allclose(torch.linalg.vector_norm(quaternion, dim=1), torch.ones_like(quaternion[:, 0]), atol=1e-4, rtol=0)):
        raise ValueError("Finite normalized batch WXYZ quaternions required")
    w, x, y, z = quaternion.unbind(-1)
    # Project the body's forward X axis. Reject vertical forward axes instead of
    # inventing a heading from an ill-conditioned yaw decomposition.
    fx, fy = 1-2*(y*y+z*z), 2*(w*z+x*y)
    norm = torch.sqrt(fx*fx+fy*fy)
    if (norm < 1e-3).any():
        raise ValueError("Body forward axis is vertical; yaw frame is undefined")
    c, s = fx/norm, fy/norm
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)


def packed_distance(distance, valid, scale, *, unsigned=False):
    if distance.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Distance/mask mismatch")
    if not torch.isfinite(distance[valid]).all() or (unsigned and (distance[valid] < 0).any()):
        raise ValueError("Invalid known distance; do not mask corrupt geometry")
    return torch.where(valid, (distance/scale).clamp(0. if unsigned else -1., 1.), 0.)


class GuidanceSampler:
    def __init__(self, meta, arrays, device="cpu"):
        limits = LayoutLimits(**meta["support_graph"]["limits"])
        self.resolution, self.max_step = limits.resolution, limits.max_step
        self.origin = torch.tensor(meta["support_graph"]["xy_origin"], dtype=torch.float32, device=device)
        xy, h = arrays["xy"], arrays["support_height"]
        expected = np.stack(np.indices(h.shape), -1)*self.resolution+self.origin.cpu().numpy()
        if (list(h.shape) != meta["support_graph"]["shape"] or xy.shape != (*h.shape, 2)
                or not np.allclose(xy, expected, atol=1e-6, rtol=0)):
            raise ValueError("Guidance world XY grid mismatch")
        graph = SupportGraph(h, arrays["walkable"], limits.resolution, limits.max_step,
                             transit=arrays["transit_valid"], max_stride=limits.max_stride)
        goal = meta["support_graph"]["endpoint_grid_indices"][1]
        field = graph.goal_field(goal)
        self.goal_cell = torch.tensor(goal, device=device)
        self.goal = torch.tensor([*xy[tuple(goal)], h[tuple(goal)]], dtype=torch.float32, device=device)
        self.height = torch.as_tensor(h.copy(), dtype=torch.float32, device=device)
        self.reachable = torch.as_tensor(field["reachable"], device=device)
        self.direction = torch.as_tensor(field["direction"], dtype=torch.float32, device=device)
        self.cost = torch.as_tensor(field["goal_cost"], dtype=torch.float32, device=device)

    def sample(self, root, rotation, surface, scale):
        index = torch.round((root[:, :2]-self.origin)/self.resolution).long()
        shape = index.new_tensor(self.height.shape)
        in_bounds = ((index >= 0) & (index < shape)).all(-1)
        # Clamped indices are used only for safe gathering; mask remains false
        # out of bounds. There is NO nearest-walkable-cell search or fallback.
        safe = torch.minimum(index.clamp_min(0), shape-1)
        i, j = safe.unbind(-1)
        support, _, known = surface.support_below(root, max_drop=2.)
        valid = in_bounds & known & self.reachable[i, j]
        valid &= (support-self.height[i, j]).abs() <= self.max_step
        delta = self.goal-root
        delta[:, 2] = self.goal[2]-support
        local_delta = torch.einsum("bij,bj->bi", rotation.transpose(1, 2), delta)
        local_direction = torch.einsum("bij,bj->bi", rotation.transpose(1, 2), self.direction[i, j])
        data = torch.cat(((local_delta/scale).clamp(-1., 1.), local_direction,
                          (self.cost[i, j]/scale).clamp(0., 1.)[:, None], valid[:, None],
                          (valid & (index == self.goal_cell).all(-1))[:, None]), -1)
        data = torch.where(valid[:, None], data, 0.)
        if not torch.isfinite(data).all():
            raise ValueError("Nonfinite valid guidance")
        return data, valid


class ObstacleObservation:
    def __init__(self, surface, clutter, placement, guidance, spec=ObservationSpec()):
        if surface.device != clutter.device:
            raise ValueError("Geometry devices must match")
        self.surface, self.clutter, self.placement, self.guidance, self.spec = surface, clutter, placement, guidance, spec
        axes = [(torch.arange(n, device=surface.device)-(n-1)/2)*spec.spacing for n in spec.shape]
        self.local_grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)

    def sample(self, root, quaternion, centers, radii):
        b, spec = len(root), self.spec
        if (not 1 <= b <= spec.max_batch or root.shape != (b, 3) or quaternion.shape != (b, 4)
                or centers.ndim != 3 or centers.shape[0] != b or centers.shape[-1] != 3
                or not 1 <= centers.shape[1] <= 512 or radii.shape != (centers.shape[1],)
                or any(t.dtype != torch.float32 or t.device != self.surface.device or not torch.isfinite(t).all()
                       for t in (root, quaternion, centers, radii)) or (radii <= 0).any()):
            raise ValueError("Finite bounded float32 root/pose/probe geometry on the query device required")
        rotation = yaw_rotation(quaternion)
        grid = torch.einsum("bij,nj->bni", rotation, self.local_grid)+root[:, None, :]
        points = torch.cat((grid, centers), 1)
        cd, _, cv = self.clutter.query(self.placement.to_local(points))
        td, _, tv = self.surface.distance(points)
        signed = packed_distance(cd, cv, spec.distance_scale)
        unsigned = packed_distance(td, tv, spec.distance_scale, unsigned=True)
        n = len(self.local_grid)
        volume = torch.stack((signed[:, :n], unsigned[:, :n], cv[:, :n], tv[:, :n]), 1).reshape(b, 4, *spec.shape)
        local_centers = torch.einsum("bij,bnj->bni", rotation.transpose(1, 2), centers-root[:, None, :])
        probes = torch.cat(((local_centers/spec.distance_scale).clamp(-1., 1.),
            (radii/spec.distance_scale).clamp(0., 1.).expand(b, -1)[..., None],
            packed_distance(cd[:, n:]-radii, cv[:, n:], spec.distance_scale)[..., None],
            unsigned[:, n:, None], cv[:, n:, None], tv[:, n:, None]), -1)
        guidance, gv = self.guidance.sample(root, rotation, self.surface, spec.goal_scale)
        valid = cv.all(-1) & tv.all(-1) & gv
        if not all(torch.isfinite(v).all() for v in (volume, probes, guidance)):
            raise ValueError("Nonfinite packed observation")
        return dict(volume=volume, probes=probes, guidance=guidance, valid=valid)
