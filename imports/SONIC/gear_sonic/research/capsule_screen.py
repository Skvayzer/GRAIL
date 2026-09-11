"""Exact capsule centre-line distances for nonlocal kinematic self screening.

This is not a replacement for PhysX contacts or their articulation filters.
Imported sphere/capsule geometry only; unknown primitives fail explicitly.
"""
import torch


def segment_distance(a, b, c, d):
    if a.shape != b.shape or a.shape != c.shape or a.shape != d.shape or a.shape[-1] != 3:
        raise ValueError("Matching segment endpoints required")
    if not all(torch.isfinite(v).all() for v in (a, b, c, d)):
        raise ValueError("Nonfinite capsule geometry")
    def point_segment(p, first, last):
        line = last-first
        t = ((p-first)*line).sum(-1)/line.square().sum(-1).clamp_min(1e-20)
        return torch.linalg.vector_norm(p-first-t.clamp(0, 1)[..., None]*line, dim=-1)
    u, v, w = b-a, d-c, a-c
    uu, uv, vv, uw, vw = u.square().sum(-1), (u*v).sum(-1), v.square().sum(-1), (u*w).sum(-1), (v*w).sum(-1)
    det = uu*vv-uv.square()
    nonparallel = det > 1e-10*(uu*vv).clamp_min(1e-20)
    denom = torch.where(nonparallel, det, torch.ones_like(det))
    s, t = (uv*vw-vv*uw)/denom, (uu*vw-uv*uw)/denom
    interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    candidate = torch.linalg.vector_norm(w+s[..., None]*u-t[..., None]*v, dim=-1)
    candidate = torch.where(interior, candidate, torch.full_like(candidate, float("inf")))
    return torch.stack((candidate, point_segment(a, c, d), point_segment(b, c, d),
                        point_segment(c, a, b), point_segment(d, a, b)), -1).amin(-1)


def nonlocal_arm_pairs(model, inventory):
    """Exclude same/adjacent COLLISION-BEARING ancestors, not entire arm chains."""
    names = [item["body"] for item in inventory]
    parents = {}
    for name in names:
        index = model.body_names.index(name)
        parent = int(model._parents[index])
        while parent >= 0 and model.body_names[parent] not in names:
            parent = int(model._parents[parent])
        parents[name] = model.body_names[parent] if parent >= 0 else None
    arm = lambda name: any(part in name for part in ("shoulder", "elbow", "wrist"))
    pairs = [(i, j) for i, a in enumerate(names) for j, b in enumerate(names) if i < j and a != b
             and (arm(a) or arm(b)) and parents[a] != b and parents[b] != a]
    if not pairs:
        raise ValueError("No nonlocal arm/body pairs")
    return pairs


def capsule_gaps(centers, probes, inventory, pairs):
    segments = []
    for index, item in enumerate(inventory):
        indices = [i for i, probe in enumerate(probes) if probe.collision_index == index]
        if item["type"] not in ("Sphere", "Capsule") or not indices:
            raise ValueError("Validated imported sphere/capsule endpoint probes required")
        segments.append((centers[:, indices[0]], centers[:, indices[-1]], item["radius"]))
    ia, ib = [i for i, _ in pairs], [j for _, j in pairs]
    a = torch.stack([segments[i][0] for i in ia], 1)
    b = torch.stack([segments[i][1] for i in ia], 1)
    c = torch.stack([segments[i][0] for i in ib], 1)
    d = torch.stack([segments[i][1] for i in ib], 1)
    radii = a.new_tensor([segments[i][2]+segments[j][2] for i, j in pairs])
    return segment_distance(a, b, c, d)-radii
