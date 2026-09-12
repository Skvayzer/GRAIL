#!/usr/bin/env python3
"""Check saved live Isaac link/site frames against CAT's MuJoCo forward kinematics.

Use CAT's Python environment. CPU only, no physics stepping or robot connection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--cat-repo", type=Path, default=Path(__file__).resolve().parents[2]/"Click-and-Traverse")
    args = parser.parse_args()
    report = json.loads((args.run/"cat_teacher_shadow.json").read_text())
    packet = args.run/"cat_teacher_shadow.npz"
    if not report["complete"] or hashlib.sha256(packet.read_bytes()).hexdigest() != report["packet_sha256"]:
        raise ValueError("Incomplete/changed Isaac snapshot")
    source = args.cat_repo/"data/assets/unitree_g1/g1_mjx_feetonly_torque.xml"
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["contract"]["sources"][str(source.relative_to(args.cat_repo))]:
        raise ValueError("Native robot model changed")
    model = mujoco.MjModel.from_xml_path(str(args.cat_repo/"data/assets/unitree_g1/scene_mjx_feetonly_flat_terrain.xml"))
    data = mujoco.MjData(model)
    with np.load(packet, allow_pickle=False) as archive:
        saved = {k: archive[k] for k in ("body_pos", "body_quat", "joint_pos", "sites")}
    pelvis_id = report["body_names"].index("pelvis")
    addresses = [int(model.joint(n).qposadr[0]) for n in report["joint_names"]]
    sites = report["contract"]["sites"]
    site_ids = [model.site(s["site"]).id for s in sites]
    max_site = np.zeros(len(sites))
    samples = 0
    for t in range(0, len(saved["sites"]), max(1, len(saved["sites"])//32)):
        for b in range(saved["sites"].shape[1]):
            data.qpos[:3] = saved["body_pos"][t, b, pelvis_id]
            data.qpos[3:7] = saved["body_quat"][t, b, pelvis_id]
            data.qpos[addresses] = saved["joint_pos"][t, b]
            mujoco.mj_forward(model, data)
            errors = np.linalg.norm(data.site_xpos[site_ids]-saved["sites"][t, b], axis=-1)
            max_site = np.maximum(max_site, errors)
            samples += 1
    result = dict(schema="grail-cat-live-frame-parity-v1", samples=samples,
        passed=bool(np.max(max_site) < .005), tolerance_m=.005,
        max_site_errors_m={s["site"]: float(e) for s, e in zip(sites, max_site)},
        packet_sha256=report["packet_sha256"], method="Native CAT MJCF FK at saved Isaac pelvis pose and named joint angles",
        training_admitted=False, physics_steps_added=0)
    target = args.run/"cat_teacher_kinematics.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
