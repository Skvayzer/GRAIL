#!/usr/bin/env python3
"""Reproducible CAT generator batches, not an automatic training launcher.

Generation is opt-in; every requested seed is recorded, including rejections.
Generated meshes are NOT automatically admitted on a GRAIL terrain. The same
source geometry must pass grounding, passage and physical-oracle checks there.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

from artifacts import ROOT
from cat_scenes import generate, sha256


SPLITS = {"development": range(1000, 1016), "validation": range(10000, 10004),
          "test": range(20000, 20008)}
STAGES = {
    "lateral": dict(difficulty=(.2, .4), n_side=(1, 3), n_floor=(0,), n_ceiling=(0,)),
    "low": dict(difficulty=(.2, .4), n_side=(0,), n_floor=(1, 2), n_ceiling=(0,)),
    "overhead": dict(difficulty=(.2, .4), n_side=(0,), n_floor=(0,), n_ceiling=(1, 2)),
    "mixed_easy": dict(difficulty=(.2, .4), n_side=(1,), n_floor=(1,), n_ceiling=(1,)),
    "mixed_hard": dict(difficulty=(.6, .7, .8, .9), n_side=(1, 3), n_floor=(1, 2, 3), n_ceiling=(1, 2)),
}


def recipes(stage, split):
    """Seed-major order gives a small prefix both geometry and seed diversity."""
    spec = STAGES[stage]
    combinations = list(itertools.product(*(spec[k] for k in spec)))
    # Round-robin parameter combinations within each seed cycle. In particular
    # the first N records are not N replicas of a single random seed.
    return [dict(stage=stage, split=split, seed=seed,
                 **dict(zip(spec, combinations[(cycle+i) % len(combinations)])))
            for cycle in range(len(combinations)) for i, seed in enumerate(SPLITS[split])]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=tuple(STAGES), default="lateral")
    p.add_argument("--split", choices=tuple(SPLITS), default="development")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    requested = recipes(args.stage, args.split)
    if not 1 <= args.count <= len(requested):
        p.error("count must fit the finite, nonrepeating stage/split recipe bank")
    requested = requested[:args.count]
    if not args.execute:
        print(json.dumps(dict(generation_started=False, training_started=False,
                              available=len(recipes(args.stage, args.split)), recipes=requested), indent=2))
        return
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_cat_curriculum")
    run.mkdir(exist_ok=False)
    report = dict(schema="grail-cat-generator-bank-v1", stage=args.stage, split=args.split,
        training_admitted=False, robot_actuation=False, terrain_composition_validated=False,
        rejection_policy="record failure; no retry-until-pass or seed replacement", requested=requested,
        results=[], complete=False)
    path = run/"bank.json"
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    save()
    print("CAT candidate bank: "+str(path), flush=True)
    for recipe in requested:
        row = dict(recipe=recipe)
        try:
            scene = generate(**{k: recipe[k] for k in ("seed", "difficulty", "n_side", "n_floor", "n_ceiling")},
                             role_trace="unique-physical-v2")
            meta = json.loads((scene/"scene.json").read_text())
            row.update(state="generated_candidate", scene=str(scene), sha256=sha256(scene/"scene.json"),
                       role_provenance=meta.get("role_provenance"))
        except ValueError as error:
            row.update(state="rejected_generation", reason=str(error))
        report["results"].append(row)
        save()
    report["complete"] = True
    save()
    print(json.dumps(dict(bank=str(path), generated=sum(r["state"] == "generated_candidate" for r in report["results"]),
                          requested=len(requested), training_admitted=False)), flush=True)


if __name__ == "__main__":
    main()
