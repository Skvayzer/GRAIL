"""Trace pinned CAT random generation without changing its code or random draws.

Binary morphology is not distributive over semantic roles. Preserve original
memberships where voxels survive; trace causal role bits for newly added cells.
Mixed-role/padding dependencies are explicitly unresolved for placement, not
invented exclusive labels. The physical occupancy always stays upstream-exact.
"""
import copy
import hashlib
import types

import numpy as np
from scipy import ndimage

ROLE_BITS = {"floor": 1, "lateral": 2, "overhead": 4}
PADDING_BIT = 8


def morphology_provenance(masks, iters=1, kernel=3):
    """Exact upstream closing/opening occupancy, with conservative dependency bits."""
    if (set(masks) != set(ROLE_BITS) or type(iters) is not int or not 1 <= iters <= 3
            or type(kernel) is not int or kernel not in (3, 5, 7)):
        raise ValueError("Expected all explicit roles and pinned morphology bounds")
    shape = masks["floor"].shape
    if (len(shape) != 3 or min(shape) < 1 or np.prod(shape) > 1_000_000
            or any(m.shape != shape or m.dtype != np.bool_ for m in masks.values())):
        raise ValueError("Expected equally shaped XYZ boolean role masks")
    bits = np.zeros(shape, dtype=np.uint8)
    for role, bit in ROLE_BITS.items():
        bits[masks[role]] |= bit
    pad = max(1, (kernel//2)*iters)
    labels = np.pad(bits, pad, constant_values=PADDING_BIT)
    occupied = labels != 0
    structure = np.ones((kernel,)*3, dtype=bool)
    # Mirrors binary_closing(iterations=iters), then binary_opening(iterations=1).
    for operation, count in (("dilate", iters), ("erode", iters), ("erode", 1), ("dilate", 1)):
        for _ in range(count):
            next_labels = np.zeros_like(labels)
            for bit in (*ROLE_BITS.values(), PADDING_BIT):
                inherited = ndimage.maximum_filter((labels & bit) != 0, size=kernel, mode="constant", cval=0)
                next_labels[inherited] |= bit
            function = ndimage.binary_dilation if operation == "dilate" else ndimage.binary_erosion
            occupied = function(occupied, structure=structure, iterations=1, border_value=0)
            labels = np.where(occupied, next_labels, 0).astype(np.uint8)
    crop = (slice(pad, -pad),)*3
    return occupied[crop].copy(), labels[crop].copy()


def trace_random(cfg, module):
    """Private function-global copy wraps two observation points; module untouched."""
    original_build = module.build_occ_from_masks_thick_xyxz
    original_morph = module.closing_opening_padded
    captured = {}
    count = 0

    def build(config, left, right, floor, ceiling, rng, j_left, j_right):
        nonlocal count
        occupied = original_build(config, left, right, floor, ceiling, rng, j_left, j_right)
        count += 1
        if count == 3:
            zero_xz, zero_xy = np.zeros_like(left), np.zeros_like(floor)
            inputs = {"lateral": (left, right, zero_xy, zero_xy),
                      "floor": (zero_xz, zero_xz, floor, zero_xy),
                      "overhead": (zero_xz, zero_xz, zero_xy, ceiling)}
            masks = {name: original_build(config, *values, copy.deepcopy(rng), j_left, j_right)
                     for name, values in inputs.items()}
            if not np.array_equal(np.logical_or.reduce(list(masks.values())), occupied):
                raise ValueError("Upstream rotated role union mismatch")
            captured["rotated_roles"] = masks
        return occupied

    def morph(occupied, iters=3, kernel=3):
        if "before" in captured or count != 3:
            raise ValueError("Unexpected upstream stage order")
        masks = {k: v & occupied for k, v in captured["rotated_roles"].items()}
        if not np.array_equal(np.logical_or.reduce(list(masks.values())), occupied):
            raise ValueError("Start/goal carving added unexplained occupied cells")
        result = original_morph(occupied, iters=iters, kernel=kernel)
        replay, dependencies = morphology_provenance(masks, iters, kernel)
        if not np.array_equal(replay, result):
            raise ValueError("Morphology occupancy replay mismatch")
        captured.update(before=occupied.copy(), masks=masks, after=result.copy(), dependencies=dependencies)
        return result

    private = dict(module.generate_and_save.__globals__)
    private.update(build_occ_from_masks_thick_xyxz=build, closing_opening_padded=morph)
    function = types.FunctionType(module.generate_and_save.__code__, private,
                                  module.generate_and_save.__name__, module.generate_and_save.__defaults__)
    occupied, *axes = function(cfg, save=False)
    if count != 3 or "after" not in captured or np.any(occupied & ~captured["after"]):
        raise ValueError("Unexpected upstream final stage; widening must only remove cells")
    # A separate, unwrapped invocation verifies full parity, including RNG order.
    baseline, *baseline_axes = module.generate_and_save(cfg, save=False)
    if not np.array_equal(occupied, baseline) or any(not np.array_equal(a, b) for a, b in zip(axes, baseline_axes)):
        raise ValueError("Role tracing changed upstream random generation")
    before = captured["before"]
    added = occupied & ~before
    dependencies = np.where(added, captured["dependencies"], 0).astype(np.uint8)
    pure = np.isin(dependencies, list(ROLE_BITS.values())) & added
    unresolved = added & ~pure
    arrays = {"before_morphology": before, "after_morphology": captured["after"],
              "morphology_added_final": added, "added_dependency_bits": dependencies,
              "unresolved_added": unresolved}
    for name, bit in ROLE_BITS.items():
        arrays["source_"+name] = captured["masks"][name]
        arrays[name] = (captured["masks"][name] & occupied) | (pure & (dependencies == bit))
    covered = np.logical_or.reduce([arrays[k] for k in ROLE_BITS]) | unresolved
    if not np.array_equal(covered, occupied):
        raise ValueError("Final role trace does not cover upstream occupancy")
    summary = dict(schema="cat-random-role-trace-v1", file="role_trace.npz", role_bits=ROLE_BITS,
        padding_bit=PADDING_BIT, upstream_parity_verified=True, source_geometry_modified=False,
        morphology_added_final=int(added.sum()), morphology_removed=int((before & ~captured["after"]).sum()),
        widening_removed=int((captured["after"] & ~occupied).sum()),
        unresolved_added_cells=int(unresolved.sum()), placement_roles_resolved=not bool(unresolved.any()),
        role_cells={k: int(arrays[k].sum()) for k in ROLE_BITS},
        semantics="surviving source memberships; additions inherit only a single causal role without padding; mixed/padding additions unresolved",
        occupied_bytes_sha256=hashlib.sha256(occupied.tobytes()).hexdigest())
    return occupied, axes, arrays, summary


def verify_trace(meta, occupied, arrays, module):
    """Replay pinned generation and every trace array, not just final role union."""
    cfg = module.Cfg(seed=meta["seed"], difficulty=meta["difficulty"],
                     n_rect_L=meta["n_side"], n_rect_R=meta["n_side"],
                     n_rect_F=meta["n_floor"], n_rect_C=meta["n_ceiling"])
    expected, axes, trace, summary = trace_random(cfg, module)
    if (occupied.dtype != np.bool_ or not np.array_equal(occupied, expected)
            or set(arrays) != set(trace) or summary != meta["role_provenance"]
            or meta["resolution"] != cfg.voxel
            or meta["axis_order"] != "xyz" or meta["units"] != "m"
            or tuple(meta["shape"]) != occupied.shape
            or not np.allclose(meta["origin_corner"], cfg.origin_w, atol=1e-8, rtol=0)
            or not np.allclose(meta["sample_origin"], [axis[0] for axis in axes], atol=1e-8, rtol=0)):
        raise ValueError("Random role trace metadata/occupancy replay mismatch")
    for name, expected in trace.items():
        if arrays[name].dtype != expected.dtype or not np.array_equal(arrays[name], expected):
            raise ValueError(f"Random role trace array replay mismatch: {name}")
    return {k: arrays[k] for k in ROLE_BITS}, arrays["unresolved_added"]
