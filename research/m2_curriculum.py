#!/usr/bin/env python3
"""Verify/export the exact reviewed pilot evidence without changing its hashes.

The compact bundle contains generated CAT arrays/mesh, the selected arm-pose
witness and its original reference sweep. The released backbone/motion assets
are separately fetched by artifacts.py. No credentials, logs, learner weights,
or unrelated run files are exported. No simulation or network is started.
"""
import argparse
import json
from pathlib import Path
import tarfile

from artifacts import ROOT
from cat_roles import role_masks
from gear_sonic.research.training_admission import ChallengeAdmission, artifact_path, digest

REPO = ROOT.parent
MANIFEST = ROOT/"config/m2_curriculum.json"


def evidence_files(manifest=MANIFEST):
    data = json.loads(manifest.read_text())
    if data.get("schema") != "grail-cat-m2-curriculum-v1" or not data.get("environment_review_approved"):
        raise ValueError("Reviewed M2 curriculum required")
    files = {manifest.resolve()}
    for spec in data["scenes"].values():
        report = artifact_path(spec["witness"])
        checked = ChallengeAdmission(report, spec["witness_sha256"], spec["case"])
        files.add(report)
        item = next(x for x in checked.report["witnesses"] if x["case"] == spec["case"])
        files.add(report.parent/item["file"])
        scene = artifact_path(checked.report["scene"])
        _, meta = role_masks(scene)
        if digest(scene/"scene.json") != checked.report["scene_sha256"]:
            raise ValueError("Reviewed scene metadata changed")
        files.add(scene/"scene.json")
        files.update(scene/name for name in meta["files"])
        sweep = artifact_path(checked.report["reference"]["reference_run"])/"reference_sweep.npz"
        if digest(sweep) != checked.report["reference"]["reference_sweep_sha256"]:
            raise ValueError("Original reference sweep changed")
        files.add(sweep)
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(REPO):
            raise ValueError("Export contains missing/linked/out-of-repository evidence")
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "pack"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    files = evidence_files()
    if args.command == "pack":
        if args.output is None:
            parser.error("pack requires --output (existing files are never overwritten)")
        with args.output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for path in files:
                archive.add(path, arcname=str(path.relative_to(REPO)), recursive=False)
        print(json.dumps(dict(bundle=str(args.output.resolve()), sha256=digest(args.output),
                              bytes=args.output.stat().st_size)))
    print(json.dumps(dict(verified=True, files=len(files), curriculum_sha256=digest(MANIFEST))))


if __name__ == "__main__":
    main()
