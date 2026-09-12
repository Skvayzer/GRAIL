#!/usr/bin/env python3
"""Append checked training-state CAT labels; held-out data remains unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
from cat_distill_model import sha


def aggregate(parent, sources, output):
    manifest = json.loads((parent/"dataset.json").read_text())
    if sha(parent/"dataset.npz") != manifest["dataset_sha256"]:
        raise ValueError("Changed parent dataset")
    with np.load(parent/"dataset.npz", allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    extra, seen = [], set()
    for source in sources:
        meta = json.loads((source/"dagger.json").read_text())
        digest = sha(source/"dagger.npz")
        if (digest != meta["sha256"] or digest in seen or meta["robot_actuation"]
                or meta["source_dataset_sha256"] != manifest["dataset_sha256"]
                or any(s >= 6 or s < 0 for s in meta["training_seeds"])):
            raise ValueError("Invalid/duplicate/held-out DAgger source")
        seen.add(digest)
        with np.load(source/"dagger.npz", allow_pickle=False) as a:
            if set(a.files) != set(arrays) or a["validation"].any() or not (a["mode"] == 1).all():
                raise ValueError("DAgger may append flat training rows only")
            if any(not np.isfinite(a[k]).all() or a[k].shape[1:] != arrays[k].shape[1:] for k in a.files):
                raise ValueError("DAgger shape/nonfinite mismatch")
            for key in arrays:
                arrays[key] = np.concatenate((arrays[key], a[key]))
        extra.append(dict(path=str(source.resolve()), **meta))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/"dataset.npz", **arrays)
    manifest["parent_dataset_hashes"] = [manifest["dataset_sha256"]]
    manifest["dataset_sha256"] = sha(output/"dataset.npz")
    manifest["dagger_sources"] = manifest.get("dagger_sources", [])+extra
    (output/"dataset.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("AGGREGATED", len(arrays["obs"]), "heldout", int(arrays["validation"].sum()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    aggregate(args.parent, args.sources, args.output)
