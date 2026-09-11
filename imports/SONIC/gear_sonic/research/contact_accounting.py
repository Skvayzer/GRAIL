"""Pure contact semantics and an instance-local, pre-reset physics observer.

No Isaac imports, state writes, control logic or reward modifications here.
Reference phase is a *candidate* label, never inferred from actual contact force.
"""
from dataclasses import dataclass
import math

import numpy as np


class PhysicsStepTap:
    """Observe AFTER scene.update, BEFORE RL post-step/reset, without extra steps.

    Scoped to one scene instance. No dependency-file or global-class monkeypatch.
    Preserves the original call/return and restores the instance on exceptions.
    The observer must be read-only. It may raise to abort an invalid diagnostic.
    """
    def __init__(self, scene, callback):
        self.scene, self.callback, self.installed = scene, callback, False

    def __enter__(self):
        if self.installed or getattr(self.scene, "_research_physics_tap", False):
            raise RuntimeError("A research physics tap is already installed")
        self.had_local = "update" in vars(self.scene)
        self.previous = self.scene.update
        def update(*args, **kwargs):
            result = self.previous(*args, **kwargs)
            self.callback(*args, **kwargs)
            return result
        self.replacement = update
        self.scene.update = update
        self.scene._research_physics_tap = True
        self.installed = True
        return self

    def __exit__(self, *exception):
        if self.installed:
            if self.scene.update is not self.replacement:
                raise RuntimeError("Another owner replaced the active physics observer")
            if self.had_local:
                self.scene.update = self.previous
            else:
                del self.scene.update
            del self.scene._research_physics_tap
            self.installed = False


def unpack_contacts(buffers, matrix):
    """Validate/resolve the six PhysX buffers and verify aggregate-force parity.

    Count/start are per sensor/filter, not a dense contact array. Unused slots
    may be NaN and are ignored; missing/overlapping/saturated data fails closed.
    Returns packed sensor/filter/force/point/normal/separation arrays and error.
    """
    if len(buffers) != 6:
        raise ValueError("Expected all six PhysX contact buffers")
    force, point, normal, sep, count, start = map(np.asarray, buffers)
    matrix = np.asarray(matrix)
    capacity = len(force)
    if (force.shape != (capacity, 1) or sep.shape != force.shape
            or point.shape != (capacity, 3) or normal.shape != point.shape
            or count.ndim != 2 or start.shape != count.shape or matrix.shape != (*count.shape, 3)
            or not np.issubdtype(count.dtype, np.integer) or not np.issubdtype(start.dtype, np.integer)
            or not np.isfinite(matrix).all() or (count < 0).any() or (start < 0).any()):
        raise ValueError("Malformed contact buffer layout")
    if count.sum(dtype=np.int64) >= capacity:
        raise ValueError("Contact buffer saturation: increase capacity; do not discard contacts")
    sensor, partner, ids = [], [], []
    for i, j in np.ndindex(count.shape):
        n, offset = int(count[i, j]), int(start[i, j])
        if n and offset+n > capacity:
            raise ValueError("Contact buffer range outside allocation")
        sensor.extend([i]*n)
        partner.extend([j]*n)
        ids.extend(range(offset, offset+n))
    if len(ids) != len(set(ids)):
        raise ValueError("Overlapping contact buffer ranges")
    ids, sensor, partner = np.array(ids, dtype=np.int64), np.array(sensor, dtype=np.int64), np.array(partner, dtype=np.int64)
    f, p, n, s = force[ids, 0], point[ids], normal[ids], sep[ids, 0]
    if (not all(np.isfinite(v).all() for v in (f, p, n, s)) or (f < -1e-5).any()
            or (np.abs(np.linalg.norm(n, axis=1)-1) > .002).any()):
        raise ValueError("Nonfinite/invalid referenced contact payload")
    aggregate = np.zeros_like(matrix)
    np.add.at(aggregate, (sensor, partner), f[:, None]*n)
    error = float(np.max(np.abs(aggregate-matrix))) if matrix.size else 0.
    if not np.allclose(aggregate, matrix, atol=.01, rtol=2e-4):
        raise ValueError(f"Detailed contact/force-matrix mismatch ({error:.6f} N)")
    return dict(sensor=sensor, partner=partner, force=f, point=p, normal=n, separation=s), error


def bind_contact_names(sensor_names, filter_names, sensor_paths, filter_paths):
    """Bind exact requested prims to PhysX's returned leaf names (or full paths).

    The pinned PhysX runtime returns leaf names, not prim paths. Ambiguous leaf
    names are rejected, and caller must also verify sensor/filter counts. No
    wildcard/multi-environment binding is supported by this diagnostic.
    """
    sensor_leaf = [p.rsplit("/", 1)[-1] for p in sensor_paths]
    filter_leaf = [p.rsplit("/", 1)[-1] for p in filter_paths]
    if (len(set(sensor_leaf)) != len(sensor_leaf) or len(set(filter_leaf)) != len(filter_leaf)
            or len(sensor_names) != len(sensor_paths) or len(set(sensor_names)) != len(sensor_names)):
        raise ValueError("Ambiguous/missing contact names")
    mapping = dict(zip(sensor_paths, sensor_leaf)) | dict(zip(sensor_leaf, sensor_leaf))
    if any(n not in mapping for n in sensor_names) or {mapping[n] for n in sensor_names} != set(sensor_leaf):
        raise ValueError("Unexpected contact sensor names")
    if len(filter_names) != len(sensor_names) or any(row not in (filter_paths, filter_leaf) for row in filter_names):
        raise ValueError("Unexpected contact filter order")
    return [mapping[n] for n in sensor_names]


@dataclass(frozen=True)
class ContactLimits:
    force_threshold: float = .1
    min_up_normal: float = .70710678
    max_penetration: float = .01
    max_point_force: float = 2000.
    surface_tolerance: float = .015
    min_normal_agreement: float = .9
    stance_gap: float = .04
    swing_gap: float = .07
    stance_speed: float = .25
    swing_speed: float = .50
    sole_edge_inset: float = .003
    sole_half_band: float = .012

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("Finite positive diagnostic contact limits required")
        if not (self.stance_gap < self.swing_gap and self.stance_speed < self.swing_speed
                and self.min_up_normal <= 1 and self.min_normal_agreement <= 1):
            raise ValueError("Invalid contact phase/normal bounds")


def reference_phase(gap, speed, known, limits=ContactLimits()):
    """Independent kinematic reference candidate, NOT annotated stance truth."""
    if not known or not math.isfinite(gap) or not math.isfinite(speed) or speed < 0:
        return "unknown"
    if gap > limits.swing_gap or speed > limits.swing_speed:
        return "swing_candidate"
    if -limits.max_penetration <= gap <= limits.stance_gap and speed <= limits.stance_speed:
        return "stance_candidate"
    return "unknown"


def sole_regions(probes, inventory, limits=ContactLimits()):
    """Actual capsule sole bounds shared by rollout audit and physical fixtures.

    Imported cover probes include capsule endpoints. Inflate by actual primitive
    radius, never by the conservative sphere-cover radius used for clearance.
    """
    regions = {}
    for link in ("left_ankle_roll_link", "right_ankle_roll_link"):
        selected = [p for p in probes if p.link == link]
        if not selected or any(inventory[p.collision_index]["type"] != "Capsule" for p in selected):
            raise ValueError("Sole region derivation validated for imported G1 foot capsules only")
        offsets = np.asarray([p.offset for p in selected], dtype=float)
        radii = np.asarray([inventory[p.collision_index]["radius"] for p in selected], dtype=float)[:, None]
        if not np.isfinite(offsets).all() or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError("Invalid imported capsule dimensions")
        lo, hi = (offsets-radii).min(0), (offsets+radii).max(0)
        floor = lo[2]
        lo[:2] += limits.sole_edge_inset
        hi[:2] -= limits.sole_edge_inset
        lo[2], hi[2] = floor-limits.sole_half_band, floor+limits.sole_half_band
        if (lo >= hi).any():
            raise ValueError("Invalid imported sole region")
        regions[link] = lo, hi
    return regions


def classify_contact(partner, is_foot, phase, local_point, sole_lower, sole_upper,
                     normal, force, separation, surface_error, normal_agreement, limits=ContactLimits()):
    values = [*local_point, *sole_lower, *sole_upper, *normal, force, separation, surface_error, normal_agreement]
    if not all(math.isfinite(float(v)) for v in values):
        return "invalid_geometry"
    if abs(np.linalg.norm(normal)-1) > .002 or force < 0:
        return "invalid_geometry"
    if force <= limits.force_threshold:
        return "below_force_threshold"
    if partner == "cat":
        return "forbidden_cat"
    if partner not in ("terrain", "ground"):
        return "unknown_partner"
    if not is_foot:
        return "forbidden_nonfoot"
    if normal[2] < limits.min_up_normal:
        return "forbidden_riser_or_side"
    if separation < -limits.max_penetration:
        return "excess_penetration"
    if force > limits.max_point_force:
        return "excess_point_force"
    if any(x < lo or x > hi for x, lo, hi in zip(local_point, sole_lower, sole_upper)):
        return "outside_sole_region"
    if abs(surface_error) > limits.surface_tolerance or normal_agreement < limits.min_normal_agreement:
        return "unverified_support_surface"
    if phase == "swing_candidate":
        return "unexpected_swing_support"
    if phase != "stance_candidate":
        return "phase_unknown"
    return "stance_support_candidate"  # NOT a granted permission / reward exemption
