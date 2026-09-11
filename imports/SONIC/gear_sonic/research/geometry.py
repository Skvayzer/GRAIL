"""Versioned ideal-geometry fields. Metres, Z up, XYZ grids, WXYZ quaternions.

These are diagnostic/supervision primitives, not yet policy observations. Unknown
grid samples are NaN plus an explicit validity mask, never silently free space.
"""
from dataclasses import dataclass
import itertools
import math

import torch

FIELD_VERSION = "grail-cat-geometry-v1"


@dataclass(frozen=True)
class Solid:
    name: str
    shape: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]  # full XYZ extents; cylinder diameter, diameter, height
    yaw: float = 0.0
    support_top: bool = False

    def __post_init__(self):
        if not self.name.isidentifier() or self.shape not in ("box", "cylinder"):
            raise ValueError("Solid needs an identifier and supported shape")
        if len(self.center) != 3 or len(self.size) != 3:
            raise ValueError("Expected XYZ center/size")
        if not all(math.isfinite(x) for x in (*self.center, *self.size, self.yaw)):
            raise ValueError("Nonfinite geometry")
        if min(self.size) <= 0 or (self.shape == "cylinder" and self.size[0] != self.size[1]):
            raise ValueError("Positive extents and circular cylinders required")
        if self.support_top and self.shape != "box":
            raise ValueError("Only box top support is implemented in v1")

    def local(self, points):
        p = points - points.new_tensor(self.center)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return torch.stack((c*p[..., 0] + s*p[..., 1],
                            -s*p[..., 0] + c*p[..., 1], p[..., 2]), -1)

    def distance_normal(self, points):
        """Exact point SDF and outward unit normal; tie subgradients are arbitrary."""
        p = self.local(points)
        if self.shape == "box":
            q = p.abs() - p.new_tensor(self.size)/2
            outside = q.clamp_min(0)
            length = torch.linalg.vector_norm(outside, dim=-1)
            distance = length + q.amax(-1).clamp_max(0)
            sign = torch.where(p >= 0, 1.0, -1.0)
            inside_n = torch.nn.functional.one_hot(q.argmax(-1), 3).to(p) * sign
            normal = torch.where((length > 0)[..., None],
                                 outside * sign / length.clamp_min(1e-12)[..., None], inside_n)
        else:
            radial = torch.linalg.vector_norm(p[..., :2], dim=-1)
            q = torch.stack((radial-self.size[0]/2, p[..., 2].abs()-self.size[2]/2), -1)
            outside = q.clamp_min(0)
            length = torch.linalg.vector_norm(outside, dim=-1)
            distance = length + q.amax(-1).clamp_max(0)
            weights = torch.where((length > 0)[..., None],
                                  outside / length.clamp_min(1e-12)[..., None],
                                  torch.nn.functional.one_hot(q.argmax(-1), 2).to(p))
            radial_n = p[..., :2] / radial.clamp_min(1e-12)[..., None]
            radial_n = torch.where((radial > 0)[..., None], radial_n, p.new_tensor([1., 0.]))
            normal = torch.cat((weights[..., :1]*radial_n,
                                weights[..., 1:] * torch.where(p[..., 2:] >= 0, 1., -1.)), -1)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        normal_w = torch.stack((c*normal[..., 0]-s*normal[..., 1],
                                s*normal[..., 0]+c*normal[..., 1], normal[..., 2]), -1)
        return distance, normal_w


def sphere_clearances(points, radii, solids, margin=0.0):
    """[..., probe, solid] signed gap, including radius and uncertainty margin.

    Support solids remain in this field. Contact permissions are a SEPARATE
    decision and must never delete an entire stair/object from occupancy.
    """
    if not math.isfinite(margin) or margin < 0 or not torch.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("Invalid clearance margin/radii")
    if not solids:
        return points.new_empty((*points.shape[:-1], 0))
    return torch.stack([s.distance_normal(points)[0] for s in solids], -1) - radii[..., None] - margin


def support_below(points, solids, max_drop=1.0, edge_margin=0.0):
    """Highest designated top below each point within max_drop, not traversability.

    No nearby top returns NaN/False; a roof/rail is not a support by height alone.
    This query does NOT infer foothold reachability, slope continuity or stability.
    """
    if not math.isfinite(max_drop) or max_drop < 0 or not math.isfinite(edge_margin) or edge_margin < 0:
        raise ValueError("Invalid support bounds")
    height = points.new_full(points.shape[:-1], -float("inf"))
    for solid in solids:
        if not solid.support_top:
            continue
        local = solid.local(points)
        half = local.new_tensor(solid.size[:2])/2 - edge_margin
        top = solid.center[2] + solid.size[2]/2
        valid = (local[..., :2].abs() <= half).all(-1)
        valid &= (points[..., 2] >= top) & (points[..., 2]-top <= max_drop)
        height = torch.where(valid, torch.maximum(height, height.new_tensor(top)), height)
    valid = torch.isfinite(height) & torch.isfinite(points).all(-1)
    return torch.where(valid, height, float("nan")), valid


@dataclass(frozen=True)
class ContactPermission:
    solid: str
    links: tuple[str, ...]
    phases: tuple[str, ...]
    region_min: tuple[float, float, float]  # solid-local contact-point bounds
    region_max: tuple[float, float, float]
    min_up_normal: float = 0.9  # outward solid normal; v1 uses yaw-only solids
    max_penetration: float = 0.01
    max_force: float = 1000.0

    def __post_init__(self):
        if len(self.region_min) != 3 or len(self.region_max) != 3:
            raise ValueError("Expected XYZ permission region")
        if not all(math.isfinite(v) for v in (*self.region_min, *self.region_max,
                                             self.min_up_normal, self.max_penetration, self.max_force)):
            raise ValueError("Nonfinite permission limits")
        if any(a > b for a, b in zip(self.region_min, self.region_max)) or not -1 <= self.min_up_normal <= 1:
            raise ValueError("Invalid permission region/normal")
        if self.max_penetration < 0 or self.max_force < 0 or not self.links or not self.phases:
            raise ValueError("Invalid permission limits/links/phases")

    def allows(self, solid, link, phase, point, normal, penetration, force):
        if solid.name != self.solid or link not in self.links or phase not in self.phases:
            return False
        values = [*point, *normal, penetration, force]
        if not all(math.isfinite(float(v)) for v in values):
            return False
        n = torch.as_tensor(normal, dtype=torch.float64)
        if abs(float(torch.linalg.vector_norm(n))-1.0) > 1e-3:
            return False
        p = solid.local(torch.as_tensor(point, dtype=torch.float64))
        in_region = all(lo <= float(x) <= hi for x, lo, hi in zip(p, self.region_min, self.region_max))
        return in_region and float(n[2]) >= self.min_up_normal and (
            0 <= penetration <= self.max_penetration and 0 <= force <= self.max_force)


def sample_xyz_grid(field, positions, origin, resolution, known=None, *, legacy_cat=False):
    """Trilinear XYZ sampling, returning (values, valid).

    v1 uses matched corner/weight ordering and includes the final grid interval.
    legacy_cat reproduces env_cat.py's swapped X/Z fractions and [0,N-2]
    clipping ONLY for parity tests. Neither mode treats out-of-bounds as valid.
    origin is the location of sample [0,0,0], not an implicitly shifted cell edge.
    """
    if field.ndim != 4 or min(field.shape[:3]) < 2 or positions.shape[-1] != 3:
        raise ValueError("Expected XYZC field with axes >=2 and XYZ positions")
    if resolution <= 0 or not math.isfinite(resolution):
        raise ValueError("Invalid resolution")
    if known is not None and (known.shape != field.shape[:3] or known.dtype != torch.bool):
        raise ValueError("known must be an XYZ boolean mask")
    original_shape = positions.shape[:-1]
    idx = ((positions - positions.new_tensor(origin))/resolution).reshape(-1, 3)
    upper = idx.new_tensor(field.shape[:3])-1
    valid = torch.isfinite(idx).all(-1) & ((idx >= 0) & (idx <= upper)).all(-1)
    clipped = torch.minimum(torch.maximum(torch.nan_to_num(idx), idx.new_tensor(0)), upper-(1 if legacy_cat else 0))
    base = torch.minimum(clipped.floor(), upper-1).long()
    frac = clipped-base
    # Offsets and weights are constructed together; never flatten different orders.
    out = field.new_zeros((idx.shape[0], field.shape[-1]))
    for corner in itertools.product((0, 1), repeat=3):
        offset = base.new_tensor(corner)
        index = base+offset
        value = field[index[:, 0], index[:, 1], index[:, 2]]
        f = frac.flip(-1) if legacy_cat else frac
        weight = torch.where(offset.bool(), f, 1-f).prod(-1)
        relevant = weight > 0
        out += torch.where(relevant[:, None], value, 0)*weight[:, None]
        valid &= ~relevant | torch.isfinite(value).all(-1)
        if known is not None:
            valid &= ~relevant | known[index[:, 0], index[:, 1], index[:, 2]]
    if not legacy_cat:
        out = torch.where(valid[:, None], out, float("nan"))
    return out.reshape(*original_shape, field.shape[-1]), valid.reshape(original_shape)
