#!/usr/bin/env python3
"""Reuse pinned CAT occupancy/FMM generators and export physical Isaac USD.

No MuJoCo, training, policy or robot connection. Source downloads and generated
scenes go to ignored research directories; originals remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "cat_generator_manifest.json"
SOURCE = ROOT / "artifacts/cat_generator"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source=SOURCE):
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        path = source/name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing/changed CAT source: {path}; run fetch, do not bypass pin")
    return manifest


def fetch(source_repo=None):
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        target = SOURCE/name
        if target.exists():
            if sha256(target) != expected:
                raise ValueError(f"Refusing to overwrite changed source {target}")
            continue
        if source_repo is not None:
            content = (source_repo/name).read_bytes()
        else:
            url = ("https://raw.githubusercontent.com/GalaxyGeneralRobotics/Click-and-Traverse/"
                   +manifest["revision"]+"/"+name)
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"CAT source checksum mismatch: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    verify_source()
    print(f"Verified unchanged CAT generator at {SOURCE}")


def generator_modules():
    verify_source()
    directory = SOURCE/"procedural_obstacle_generation"
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(directory))
    try:
        modules = []
        for name in ("grid_config", "random_obstacle", "typical_obstacle", "pf_modular"):
            module = importlib.import_module(name)
            if Path(module.__file__).resolve().parent != directory.resolve():
                raise ValueError(f"Conflicting module: {name}")
            modules.append(module)
    finally:
        sys.path.pop(0)
    return modules


def occupancy_mesh(occupied, voxel, origin):
    """Closed marching-cubes surface of occupied cells in WORLD coordinates.

    CAT's axes locate values at voxel centers. Pad by one free cell to close
    grid-edge obstacles, subtract the padding and apply origin + half-cell.
    This fixes export offsets; the original occupancy/FMM arrays are unchanged.
    """
    import numpy as np
    from skimage.measure import marching_cubes
    if occupied.ndim != 3 or occupied.dtype != np.bool_ or not occupied.any():
        raise ValueError("Expected nonempty XYZ boolean occupancy")
    if not np.isfinite(voxel) or voxel <= 0 or np.shape(origin) != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid voxel/origin")
    padded = np.pad(occupied.astype(np.uint8), 1, constant_values=0)
    vertices, faces, _, _ = marching_cubes(padded, level=.5, spacing=(voxel,)*3,
                                          gradient_direction="ascent", allow_degenerate=False)
    vertices = vertices + np.asarray(origin)+voxel/2-voxel
    return vertices.astype(np.float32), faces.astype(np.int32)


def export_usd(path, vertices, faces):
    """Static, non-convex physical mesh: convex hull would seal CAT's openings."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
    if path.exists():
        raise FileExistsError(path)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/CAT")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/CAT/Obstacles")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(.75, .32, .12)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    # No RigidBodyAPI: this is a static concave triangle mesh, not a dynamic hull.
    stage.GetRootLayer().Save()


def generate(scene="random", seed=42, difficulty=.2, n_side=9, n_floor=3, n_ceiling=3):
    import numpy as np
    grid, random, typical, pf = generator_modules()
    source_manifest = verify_source()
    cfg = random.Cfg(seed=seed, difficulty=difficulty, n_rect_L=n_side, n_rect_R=n_side,
                     n_rect_F=n_floor, n_rect_C=n_ceiling)
    if scene == "random":
        occupied, xv, yv, zv = random.generate_and_save(cfg, save=False)
    else:
        xv, yv, zv = random.make_axes(cfg)
        occupied = typical.build_obstacles(scene, np.meshgrid(xv, yv, zv, indexing="ij"))
    if not occupied.any() or occupied.all():
        raise ValueError("Empty/full CAT scene; no suitable field interface")
    pf_cfg = pf.PFConfig()
    sdf = pf.make_sdf(occupied, cfg.voxel)
    bf = pf.grad3(sdf, cfg.voxel)
    travel, gf = pf.make_guidance_field_progressive(
        pf_cfg, np.meshgrid(xv, yv, zv, indexing="ij"), occupied, cfg.goal_w, bf, sdf)
    for value in (sdf, bf, travel, gf):
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite upstream field; do not export a partially valid scene")
    vertices, faces = occupancy_mesh(occupied, cfg.voxel, cfg.origin_w)
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_cat_scene_"+scene)
    run.mkdir(parents=True, exist_ok=False)
    arrays = {"obs": occupied, "sdf": sdf, "bf": bf, "gf": gf, "travel": travel}
    for name, value in arrays.items():
        np.save(run/(name+".npy"), value, allow_pickle=False)
    export_usd(run/"scene.usda", vertices, faces)
    versions = {n: importlib.metadata.version(n) for n in
                ("numpy", "scipy", "scikit-image", "scikit-fmm", "torch", "PyYAML")}
    manifest = dict(schema="cat-isaac-scene-v1", source=source_manifest, scene=scene, seed=seed,
                    difficulty=difficulty, n_side=n_side, n_floor=n_floor, n_ceiling=n_ceiling,
                    generator_parameters_effective=(scene == "random"),
                    resolution=cfg.voxel, shape=list(occupied.shape), axis_order="xyz", units="m",
                    origin_corner=cfg.origin_w.tolist(), sample_origin=[float(x[0]) for x in (xv, yv, zv)],
                    effective_size=(np.asarray(occupied.shape)*cfg.voxel).tolist(),
                    start_w=cfg.start_w.tolist(), goal_w=cfg.goal_w.tolist(),
                    occupied_cells=int(occupied.sum()), vertices=len(vertices), triangles=len(faces),
                    physical_collision="static triangle mesh; no convexification",
                    mesh_export_changes=["occupied solid winding", "closed boundary padding", "world origin and half-voxel center offset"],
                    support_semantics="CAT clutter is forbidden; no stair/support labels inferred",
                    sensor_realism=False, policy_loaded=False, simulation_only=True, packages=versions,
                    adapter_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
                    adapter_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT.parent, text=True)),
                    reproducibility_contract="pinned source, recorded dependencies and cached array hashes; no cross-runtime bitwise guarantee",
                    whole_body_path_feasibility_verified=False)
    manifest["files"] = {p.name: sha256(p) for p in run.iterdir() if p.is_file()}
    (run/"scene.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"scene_directory": str(run), "occupied_cells": manifest["occupied_cells"],
                      "triangles": len(faces), "source_unchanged": True}), flush=True)
    return run


def verify_scene(directory):
    from gear_sonic.research.cat_geometry import verify_scene_files
    return verify_scene_files(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--source-repo", type=Path, help="Use an existing matching checkout read-only instead of downloading")
    g = sub.add_parser("generate")
    g.add_argument("--scene", default="random")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--difficulty", type=float, default=.2)
    g.add_argument("--n-side", type=int, choices=range(10), default=9)
    g.add_argument("--n-floor", type=int, choices=range(4), default=3)
    g.add_argument("--n-ceiling", type=int, choices=range(4), default=3)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.source_repo)
    else:
        if not 0 <= args.difficulty <= 1 or args.seed < 0:
            parser.error("Difficulty must be in [0,1] and seed nonnegative")
        generate(args.scene, args.seed, args.difficulty, args.n_side, args.n_floor, args.n_ceiling)


if __name__ == "__main__":
    main()
