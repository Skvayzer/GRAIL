"""Simulator-independent occupancy-derived validation fixtures (not training tasks)."""
import numpy as np


def axis_ray_fixtures(occupied, voxel, origin, samples_per_axis=32):
    """Deterministic occupied and empty columns, cell-centered rays both ways."""
    rng = np.random.default_rng(73)
    fixtures = []
    origin = np.asarray(origin, dtype=float)
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        columns = np.moveaxis(occupied, axis, 0)
        for has_obstacle in (False, True):
            candidates = np.argwhere(columns.any(axis=0) == has_obstacle)
            if not len(candidates):
                continue
            selected = candidates[rng.choice(len(candidates), min(samples_per_axis, len(candidates)), replace=False)]
            for ij in selected:
                cells = np.flatnonzero(columns[:, ij[0], ij[1]])
                for sign in (-1, 1):
                    start = origin.copy()
                    start[other] += (ij+.5)*voxel
                    start[axis] += -.2 if sign == 1 else occupied.shape[axis]*voxel+.2
                    direction = np.zeros(3)
                    direction[axis] = sign
                    expected = None
                    if len(cells):
                        boundary = origin[axis]+(cells[0] if sign == 1 else cells[-1]+1)*voxel
                        expected = float((boundary-start[axis])*sign)
                    fixtures.append(dict(axis=axis, sign=sign, start=start.tolist(), direction=direction.tolist(),
                                         max_distance=occupied.shape[axis]*voxel+.4, expected_distance=expected))
    return fixtures


def planar_contact_patch(occupied, voxel, origin):
    """Find a front-facing 3x3-cell occupied patch with no earlier neighboring obstruction.

    For a sphere with radius <= voxel: avoids mistaking bevel contact for contact
    on the plane intersected by a zero-radius point ray. Fails if none exists.
    """
    for j, k in np.argwhere(occupied.any(axis=0)):
        if j < 1 or k < 1 or j >= occupied.shape[1]-1 or k >= occupied.shape[2]-1:
            continue
        i = int(np.flatnonzero(occupied[:, j, k])[0])
        if occupied[i, j-1:j+2, k-1:k+2].all() and not occupied[:i, j-1:j+2, k-1:k+2].any():
            return np.asarray(origin)+np.array([i, j+.5, k+.5])*voxel
    raise ValueError("No planar patch for this sphere-contact fixture; add a general mesh-normal fixture")
