"""Full-height simulation-oracle features for composed terrain/CAT scenes.

Deliberately NOT a combined signed occupancy grid: GRAIL terrain can be an open
surface. Its unsigned distance/band and support candidates remain separate from
closed CAT signed distance. Neither a support label nor an absent surface band
licenses contact or certifies free space inside/below terrain.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from .geometry import sample_xyz_grid


@dataclass(frozen=True)
class VolumeSpec:
    origin_corner: tuple
    shape: tuple
    resolution: float = .04

    def __post_init__(self):
        if (len(self.origin_corner) != 3 or not all(math.isfinite(v) for v in self.origin_corner)
                or len(self.shape) != 3 or any(not isinstance(v, (int, np.integer)) or v < 2 for v in self.shape)
                or not math.isfinite(self.resolution) or self.resolution <= 0
                or math.prod(self.shape) > 8_000_000):
            raise ValueError("Finite XYZ grid, axes >=2 and at most 8M cells required")

    @property
    def sample_origin(self):
        return tuple(v+self.resolution/2 for v in self.origin_corner)

    @classmethod
    def fit(cls, terrain_vertices, clutter_vertices_world, centers, radii, *, ground_z=None, resolution=.04, margin=.12):
        """Cover every reference sphere's bounding box, not just pelvis/centres."""
        if not math.isfinite(margin) or not math.isfinite(resolution) or resolution <= 0 or margin < resolution:
            raise ValueError("Positive voxel size and at least one-voxel finite fitting margin required")
        centers, radii = np.asarray(centers), np.asarray(radii)
        if (centers.ndim != 3 or centers.shape[-1] != 3 or not centers.size
                or radii.shape != (centers.shape[1],) or not np.isfinite(centers).all()
                or not np.isfinite(radii).all() or (radii <= 0).any()):
            raise ValueError("Nonempty finite frame/probe XYZ and positive probe radii required")
        meshes = [np.asarray(v) for v in (terrain_vertices, clutter_vertices_world)]
        if any(v.ndim != 2 or v.shape[-1] != 3 or not len(v) or not np.isfinite(v).all() for v in meshes):
            raise ValueError("Finite terrain and world-clutter vertices required")
        lower = np.min([*(v.min(0) for v in meshes), (centers-radii[None, :, None]).min((0, 1))], axis=0)
        upper = np.max([*(v.max(0) for v in meshes), (centers+radii[None, :, None]).max((0, 1))], axis=0)
        if ground_z is not None:
            if not math.isfinite(ground_z):
                raise ValueError("Invalid ground plane")
            lower[2] = min(lower[2], ground_z)
            upper[2] = max(upper[2], ground_z)
        origin = np.floor((lower-margin)/resolution)*resolution
        end = np.ceil((upper+margin)/resolution)*resolution
        return cls(tuple(origin.tolist()), tuple(np.rint((end-origin)/resolution).astype(int).tolist()), resolution)


def build_volume(surface, clutter, placement, spec, batch_size=65536):
    if surface.device != clutter.device or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Shared query device and positive integer batch required")
    size = math.prod(spec.shape)
    arrays = {k: np.empty((size, *channels), dtype=dtype) for k, channels, dtype in (
        ("clutter_sdf", (), np.float32), ("clutter_outward_normal", (3,), np.float32),
        ("terrain_unsigned_distance", (), np.float32), ("terrain_authored_normal", (3,), np.float32),
        ("clutter_distance_valid", (), bool), ("terrain_distance_valid", (), bool))}
    for start in range(0, size, batch_size):
        end = min(size, start+batch_size)
        index = np.column_stack(np.unravel_index(np.arange(start, end), spec.shape))
        points = torch.as_tensor(np.asarray(spec.sample_origin)+index*spec.resolution, dtype=torch.float32, device=surface.device)
        cd, cn, cv = clutter.query(placement.to_local(points))
        td, tn, tv = surface.distance(points)
        values = dict(clutter_sdf=cd, clutter_outward_normal=placement.vectors_to_world(cn),
            terrain_unsigned_distance=td, terrain_authored_normal=tn,
            clutter_distance_valid=cv, terrain_distance_valid=tv)
        for name, value in values.items():
            arrays[name][start:end] = value.cpu().numpy()
    arrays = {name: value.reshape(*spec.shape, *value.shape[1:]) for name, value in arrays.items()}
    # Conservative surface neighbourhood; explicitly NOT terrain-solid occupancy.
    arrays["terrain_surface_band"] = arrays["terrain_distance_valid"] & (arrays["terrain_unsigned_distance"] <= math.sqrt(3)*spec.resolution/2)
    arrays["clutter_inside"] = arrays["clutter_distance_valid"] & (arrays["clutter_sdf"] <= 0.)
    xy = np.stack(np.meshgrid(*(spec.sample_origin[i]+np.arange(spec.shape[i])*spec.resolution for i in range(2)), indexing="ij"), -1)
    query_top = max(float(surface.vertices[:, 2].max()), surface.ground_z if surface.ground_z is not None else -np.inf)+1.
    rays = np.concatenate((xy, np.full((*xy.shape[:-1], 1), query_top)), -1)
    height, normal, known = surface.support_below(torch.as_tensor(rays, dtype=torch.float32, device=surface.device), max_drop=100.)
    arrays.update(support_height=height.cpu().numpy(), support_authored_normal=normal.cpu().numpy(),
                  support_known=known.cpu().numpy(), support_candidate=(known & (normal[..., 2] >= .70710678)).cpu().numpy())
    return arrays


def sample_distances(arrays, spec, points):
    """Interpolated diagnostic distances only; no extrapolation or contact rule.

    Interpolated normal vectors/binary semantics are intentionally not exposed.
    Use exact mesh queries for gates; grid distances have voxel interpolation
    error and do not certify sphere/whole-body clearance.
    """
    result = {}
    for name, known in (("clutter_sdf", "clutter_distance_valid"), ("terrain_unsigned_distance", "terrain_distance_valid")):
        field = torch.as_tensor(arrays[name], dtype=torch.float32, device=points.device)
        mask = torch.as_tensor(arrays[known], dtype=torch.bool, device=points.device)
        values, valid = sample_xyz_grid(field[..., None], points, spec.sample_origin, spec.resolution, mask)
        result[name], result[name+"_valid"] = values[..., 0], valid
    return result
