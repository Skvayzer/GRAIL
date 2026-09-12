#!/usr/bin/env python3
"""Generate disjoint multi-layout CAT banks with the unchanged pinned generator."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np

FAMILIES = ("lateral", "low", "overhead", "mixed")


def recipes(train_per_family=48, eval_per_family=12):
    rows = []
    for split, count, base in (("train", train_per_family, 310000), ("validation", eval_per_family, 910000)):
        for f, family in enumerate(FAMILIES):
            for i in range(count):
                rows.append(dict(split=split, family=family, seed=base+f*10000+i,
                    difficulty=(.2, .4, .6, .8)[i % 4],
                    n_rect_L=(1, 3, 6)[(i//4) % 3] if family in ("lateral", "mixed") else 0,
                    n_rect_R=(1, 3, 6)[(i//4) % 3] if family in ("lateral", "mixed") else 0,
                    n_rect_F=1+(i//4) % 3 if family in ("low", "mixed") else 0,
                    n_rect_C=1+(i//4) % 2 if family in ("overhead", "mixed") else 0))
    return rows


def build_one(job):
    root, row = job
    from cat_scenes import generator_modules
    import torch
    torch.set_num_threads(1)
    _, generator, _, pf = generator_modules()
    cfg = generator.Cfg(**{k: v for k, v in row.items() if k not in ("split", "family")})
    occ, xv, yv, zv = generator.generate_and_save(cfg, save=False)
    if not occ.any() or occ.all():
        return dict(**row, rejected="empty_or_full_after_native_morphology")
    sdf = pf.make_sdf(occ, cfg.voxel)
    bf = pf.grad3(sdf, cfg.voxel)
    travel, gf = pf.make_guidance_field_progressive(pf.PFConfig(),
        np.meshgrid(xv, yv, zv, indexing="ij"), occ, cfg.goal_w, bf, sdf)
    if not all(np.isfinite(a).all() for a in (sdf, bf, gf, travel)):
        raise ValueError(f"Nonfinite native potential fields {row}")
    name = f'{row["split"]}_{row["family"]}_{row["seed"]}'
    path = Path(root)/(name+".npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, obs=occ, sdf=np.asarray(sdf, np.float32),
            bf=np.asarray(bf, np.float32), gf=np.asarray(gf, np.float32))
    return dict(**row, file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        occupancy_sha256=hashlib.sha256(occ.tobytes()).hexdigest(), shape=list(occ.shape),
        resolution=cfg.voxel, origin=cfg.origin_w.tolist(), goal=cfg.goal_w.tolist(),
        occupied_cells=int(occ.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--train-per-family", type=int, default=48)
    p.add_argument("--eval-per-family", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 4 <= a.train_per_family <= 1000 or not 4 <= a.eval_per_family <= 1000 or not 1 <= a.workers <= 8:
        p.error("Explicit bounded counts required")
    from cat_scenes import verify_source
    source = verify_source()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="cat-generated-parallel-bank-v1", source=source,
        recipes=recipes(a.train_per_family, a.eval_per_family), scenes=[], rejected=[], complete=False,
        obstacle_semantics="original CAT forbidden SDF; flat physical floor",
        robot_actuation=False)
    out = a.output/"bank.json"
    out.write_text(json.dumps(manifest, indent=2)+"\n")
    # No seed replacement or retry-until-easy. Native morphology can erase a
    # low obstacle; record that recipe rather than count empty space as clutter.
    hashes = set()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(build_one, ((str(a.output), r) for r in manifest["recipes"])):
            if "rejected" in row:
                manifest["rejected"].append(row)
                continue
            if row["occupancy_sha256"] in hashes:
                manifest["rejected"].append(dict(**row, rejected="duplicate_geometry"))
                continue
            hashes.add(row["occupancy_sha256"])
            manifest["scenes"].append(row)
            out.write_text(json.dumps(manifest, indent=2)+"\n")
            print("GENERATED", len(manifest["scenes"]), row["file"], flush=True)
    for split in ("train", "validation"):
        for family in FAMILIES:
            if sum(r["split"] == split and r["family"] == family for r in manifest["scenes"]) < 4:
                raise ValueError(f"Insufficient generated diversity: {split}/{family}")
    manifest.update(complete=True, unique_geometries=len(hashes))
    out.write_text(json.dumps(manifest, indent=2)+"\n")
    print("COMPLETE", out, len(hashes), flush=True)


if __name__ == "__main__":
    main()
