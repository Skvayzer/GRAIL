"""Explicit role replay for selected pinned CAT fixed scenes (not new generation).

No connected-component guessing: joining posts and hurdles retain their roles.
Random scenes have no stored role provenance after CAT morphology; reject them
here until a traceable role-preserving generator adapter is implemented.
"""
import numpy as np

from cat_scenes import generator_modules, verify_source
from gear_sonic.research.cat_geometry import verify_scene_files


def role_masks(directory):
    meta = verify_scene_files(directory)
    if meta["source"] != verify_source():
        raise ValueError("Role replay requires the exact pinned CAT generator")
    _, _, typical, _ = generator_modules()
    axes = [meta["sample_origin"][a]+np.arange(meta["shape"][a])*meta["resolution"] for a in range(3)]
    grids = np.meshgrid(*axes, indexing="ij")
    name = meta["scene"]
    masks = {}
    # Each recipe is directly from the pinned build_obstacles dispatch. Verify
    # exact occupancy equality below, so a changed generator/grid cannot drift.
    recipes = {
        "side0": (None, .4, None), "side1": (None, .25, None),
        "hurdle0": (0, None, None), "hurdle1": (1, None, None), "hurdle2": (2, None, None),
        "crouch0": (None, None, 1.15), "crouch1": (None, None, 1.),
        "hurdle-crouch0": (0, None, 1.15), "hurdle-crouch1": (1, None, 1.),
        "side-crouch0": (None, .9, 1.15), "side-crouch1": (None, .9, 1.),
        "side-hurdle0": (0, .5, None), "side-hurdle1": (0, .3, None),
        "side-hurdle2": (0, .9, None), "side-hurdle3": (1, .9, None),
        "side-hurdle4": (2, .9, None),
        "side-hurdle-crouch0": (0, .9, 1.15), "side-hurdle-crouch1": (1, .9, 1.05),
        "side-hurdle-crouch2": (0, .5, 1.15), "side-hurdle-crouch3": (1, .5, 1.05),
    }
    if name not in recipes:
        raise ValueError(f"No explicit role provenance for CAT scene {name!r}; do not infer roles")
    hurdle, gap, overhead = recipes[name]
    if hurdle is not None:
        masks["floor"] = getattr(typical, f"obs_hurdle{hurdle}")(*grids)
    if gap is not None:
        masks["lateral"] = typical.obs_side(*grids, gap_width=gap)
    if overhead is not None:
        masks["overhead"] = typical.obs_crouch(*grids, z_low=overhead)
    from pathlib import Path
    cached = np.load(Path(directory)/"obs.npy", allow_pickle=False)
    if cached.dtype != np.bool_ or not np.array_equal(np.logical_or.reduce(list(masks.values())), cached):
        raise ValueError("Explicit CAT roles do not reproduce cached occupancy exactly")
    return masks, meta
