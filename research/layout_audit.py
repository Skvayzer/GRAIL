#!/usr/bin/env python3
"""Recheck cached physical terrain/reference with a candidate CAT placement.

No simulator or policy starts. Each attempt produces a separate diagnostic.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from artifacts import ROOT
from gear_sonic.research.cat_geometry import Placement
from gear_sonic.research.scene_audit import check, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--translation", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    args = parser.parse_args()
    report, arrays, _ = check(args.reference_run.resolve(), args.scene.resolve(),
        Placement(tuple(args.translation), args.yaw), args.device)
    run = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_layout_check")
    run.mkdir(parents=True, exist_ok=False)
    save(run / "layout_audit.json", report, arrays)
    print(f"{run}: geometry_screen_accepted={report['geometry_screen_accepted']}; physics not started")
    raise SystemExit(0 if report["geometry_screen_accepted"] else 2)


if __name__ == "__main__":
    main()
