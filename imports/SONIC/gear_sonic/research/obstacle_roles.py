"""Terrain-relative CAT feature retention. No inference from connected components.

Masks must come from explicit generator roles; their union is checked separately
against the unchanged cached occupancy. Role overlap (e.g. a bar joining posts)
is allowed. This is geometric screening, not a construction/support simulation.
"""
import math

import numpy as np
import torch

from .cat_geometry import Placement


def ground_rigid_placement(surface, masks, origin, resolution, xy, yaw, max_spread=.04):
    """Place the unchanged CAT floor datum onto one nearly level support patch.

    No per-column warp, feature removal, pitch/roll, or hidden terrain burying.
    Query every footprint cell centre AND corner (a bounding-box centre alone
    could miss a stair edge/hole). Uneven/missing/downward support rejects the
    proposal instead of creating floating obstacles. This is not a guarantee
    about sub-voxel cracks, load bearing or the robot's passage.
    """
    if (not masks or not set(masks) <= {"floor", "lateral", "overhead"}
            or np.shape(origin) != (3,) or not np.isfinite(origin).all()
            or not math.isfinite(resolution) or resolution <= 0
            or not math.isfinite(max_spread) or max_spread < 0 or len(xy) != 2):
        raise ValueError("Invalid explicit roles/grounding bounds")
    shapes = [m.shape for m in masks.values()]
    if any(m.dtype != np.bool_ or m.ndim != 3 or m.shape != shapes[0] or not m.any() for m in masks.values()):
        raise ValueError("Nonempty, equally shaped XYZ boolean role masks required")
    footprint = np.logical_or.reduce(list(masks.values())).any(2)
    cells = np.argwhere(footprint)
    offsets = np.array([[.5, .5], [0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    local_xy = np.unique((cells[:, None, :]+offsets[None, :, :]).reshape(-1, 2), axis=0)*resolution+np.asarray(origin)[:2]
    initial = Placement((xy[0], xy[1], 0.), yaw)
    rays = initial.to_world(torch.as_tensor(np.column_stack((local_xy, np.zeros(len(local_xy)))),
                                           dtype=torch.float32, device=surface.device))
    rays[:, 2] = max(float(surface.vertices[:, 2].max()), surface.ground_z if surface.ground_z is not None else -np.inf)+2.
    h, n, valid = surface.support_below(rays, max_drop=100.)
    if not bool((valid & (n[:, 2] >= .70710678)).all()):
        raise ValueError("Cannot ground rigid CAT: missing or downward/steep support in footprint")
    spread = float(h.max()-h.min())
    if spread > max_spread+1e-6:
        raise ValueError(f"Cannot ground rigid CAT across stair edges/uneven support ({spread:.4f} m); choose another placement")
    datum = float(h.median())
    return Placement((xy[0], xy[1], datum-origin[2]), yaw), dict(
        mode="rigid CAT floor datum to footprint support; no deformation",
        support_height_m=datum, footprint_queries=len(local_xy), support_spread_m=spread,
        maximum_allowed_spread_m=max_spread, geometry_modified=False,
        footprint_continuously_verified=False)


def assess_roles(surface, masks, origin, resolution, placement, tolerance=.04):
    if (not masks or not set(masks) <= {"floor", "lateral", "overhead"}
            or not math.isfinite(resolution) or resolution <= 0
            or not math.isfinite(tolerance) or tolerance < 0
            or np.shape(origin) != (3,) or not np.isfinite(origin).all()):
        raise ValueError("Explicit CAT roles, finite origin/resolution/tolerance required")
    roles = []
    shape = next(iter(masks.values())).shape
    for role, mask in sorted(masks.items()):
        if mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != shape or not mask.any():
            raise ValueError("Nonempty, equally shaped XYZ boolean role masks required")
        indices = np.argwhere(mask)
        columns, inverse = np.unique(indices[:, :2], axis=0, return_inverse=True)
        bottom = np.full(len(columns), np.inf)
        top = np.full(len(columns), -np.inf)
        np.minimum.at(bottom, inverse, indices[:, 2])
        np.maximum.at(top, inverse, indices[:, 2]+1)
        bottom = origin[2]+bottom*resolution
        top = origin[2]+top*resolution
        local = np.column_stack((np.asarray(origin)[:2]+(columns+.5)*resolution, bottom))
        world = placement.to_world(torch.as_tensor(local, dtype=torch.float32, device=surface.device))
        rays = world.clone()
        rays[:, 2] = max(float(surface.vertices[:, 2].max()), float(world[:, 2].max()),
                        surface.ground_z if surface.ground_z is not None else -np.inf)+2.
        height, normal, valid = surface.support_below(rays, max_drop=100.)
        height, normal, valid = height.cpu().numpy(), normal.cpu().numpy(), valid.cpu().numpy()
        # The source CAT floor datum maps onto terrain/ground, not world Z=0.
        # No silent deform/lift: an unsuitable placement is rejected.
        relative_bottom = world[:, 2].cpu().numpy()-height
        desired_bottom = bottom-origin[2]
        retention = np.clip((top+placement.translation[2]-np.maximum(height, bottom+placement.translation[2]))
                            / (top-bottom), 0., 1.)
        error = relative_bottom-desired_bottom
        upright_support = valid & (normal[:, 2] >= .70710678)
        if role == "overhead":
            # Explicit suspended CAT fixture. Do not misclassify it as a
            # grounded obstacle or grant the beam top permission for foot contact.
            retained = upright_support & (np.abs(error) <= tolerance+1e-6)
        else:
            retained = upright_support & (np.abs(error) <= tolerance+1e-6) & (retention >= .95)
        good = valid & np.isfinite(relative_bottom)
        roles.append(dict(role=role, columns=len(columns), occupied_cells=int(mask.sum()),
            role_retained=bool(retained.all()), invalid_columns=int((~retained).sum()),
            known_support_columns=int(valid.sum()), up_normal_columns=int(upright_support.sum()),
            minimum_exposed_height_fraction=float(retention[good].min()) if good.any() else None,
            minimum_bottom_clearance_m=float(relative_bottom[good].min()) if good.any() else None,
            maximum_bottom_clearance_m=float(relative_bottom[good].max()) if good.any() else None,
            maximum_datum_error_m=float(np.abs(error[good]).max()) if good.any() else None,
            attachment="explicit suspended static fixture; mounting not verified" if role == "overhead"
                       else "terrain-relative grounded feature",
            contact_permission=False))
    return dict(schema="grail-cat-role-retention-v1", roles=roles, tolerance_m=tolerance,
        minimum_exposed_height_fraction=.95, all_roles_retained=all(r["role_retained"] for r in roles),
        semantics="column-centre top-envelope screen, not exact solid intersection or full-foot traversal",
        geometry_modified=False, avoidance_training_ready=False)
