#!/usr/bin/env python3
"""Prepare or execute a bounded desktop-only terrain baseline evaluation.

Downloaded checkpoints and scenes stay immutable. Each run receives its own
checkpoint copy and derived USD files with *only* texture asset paths relocated.
No navigation, ROS, robot connection, deployment wrapper or motor API is used.
"""
import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from artifacts import ROOT, PROVENANCE, safe_path, verify_file

REPO_ROOT = ROOT.parent


def prepare_data(manifest, family, run):
    from pxr import Sdf, UsdUtils
    prefix = f"data/{family}/"
    for item in manifest["files"]:
        if not item["path"].startswith(prefix):
            continue
        source = ROOT / "artifacts" / safe_path(item["path"])
        target = run / safe_path(item["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if "asset_remaps" in item:
            layer = Sdf.Layer.OpenAsAnonymous(str(source))
            if not layer:
                raise ValueError(f"Unable to open USD {source}")
            remaps = item["asset_remaps"]
            UsdUtils.ModifyAssetPaths(layer, lambda p: remaps.get(p, p))
            if not layer.Export(str(target)):
                raise ValueError("Failed to export derived scene")
        else:
            shutil.copyfile(source, target)
    return run / "data" / family


def evaluation_command(run, data, stem, num_envs):
    overrides = {
        "headless": True, "num_envs": num_envs, "seed": 42,
        "eval_callbacks": "im_eval", "run_eval_loop": False,
        "eval_output_dir": str(run / "metrics"),
        "hydra.run.dir": str(run / "hydra"),
        "manager_env.config.terrain_motion_dir": str(data),
        "manager_env.config.flat_usd_path": str(ROOT / "assets/flat_placeholder.usda"),
        "manager_env.config.flat_motion_dir": None,
        "manager_env.config.flat_to_terrain_ratio": 0,
        "manager_env.config.render_results": False,
        "manager_env.config.debug_state_log": str(run / "trajectory.json"),
        "manager_env.commands.motion.flat_to_terrain_ratio": 0,
        "manager_env.commands.motion.motion_lib_cfg.motion_file": str(data / "robot"),
        "manager_env.commands.motion.motion_lib_cfg.object_motion_file": str(data / "objects"),
        "manager_env.commands.motion.motion_lib_cfg.filter_motion_keys": [stem],
        "manager_env.commands.motion.motion_lib_cfg.multi_thread": False,
        "manager_env.commands.motion.motion_lib_cfg.motion_shard_world_size": 1,
        "manager_env.commands.motion.motion_lib_cfg.motion_shard_rank": 0,
    }
    command = [str(REPO_ROOT / ".venv/bin/python"), "-m", "gear_sonic.eval_agent_trl",
               f"checkpoint={run / 'checkpoint/last.pt'}"]
    for key, value in overrides.items():
        value = json.dumps(value, separators=(",", ":"))
        command.append(f"++{key}={value}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["stair_p1", "curb", "slope", "sitting"], default="stair_p1")
    parser.add_argument("--num-envs", type=int, choices=range(1, 17), default=1)
    parser.add_argument("--execute", action="store_true", help="Run desktop physics, never robot control")
    parser.add_argument("--accept-isaac-eula", action="store_true",
                        help="Explicit user acceptance of the NVIDIA Omniverse license; never enabled by default")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.execute and not args.accept_isaac_eula:
        package = importlib.metadata.distribution("isaacsim")
        accepted = Path(package.locate_file("isaacsim/kit/EULA_ACCEPTED"))
        if not accepted.is_file() and os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() != "yes":
            raise SystemExit("Isaac Sim requires your license acceptance before simulation. Read "
                             "https://docs.omniverse.nvidia.com/platform/latest/common/"
                             "NVIDIA_Omniverse_License_Agreement.html and, only if you agree, "
                             "pass --accept-isaac-eula. No simulator was started.")
    manifest = json.loads((ROOT / "data_manifest.json").read_text())
    if manifest["revision"] != PROVENANCE["dataset_revision"]:
        raise ValueError("Unexpected dataset revision")
    for item in manifest["files"]:
        verify_file(ROOT / "artifacts" / safe_path(item["path"]), item)
    scene = next(x for x in manifest["scenes"] if x["family"] == args.family)
    run = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.family)
    run.mkdir(parents=True, exist_ok=False)
    data = prepare_data(manifest, args.family, run)
    (run / "checkpoint").mkdir()
    for name in ("last.pt", "config.yaml"):
        shutil.copyfile(ROOT / "artifacts/checkpoint/SONIC/models/terrain_release" / name,
                        run / "checkpoint" / name)
    command = evaluation_command(run, data, scene["stem"], args.num_envs)
    record = {"simulation_only": True, "family": args.family, "num_envs": args.num_envs,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)),
              "dataset_revision": manifest["revision"], "command": command,
              "status": "prepared_not_executed"}
    record_path = run / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Run directory: {run}", flush=True)
    if not args.execute:
        print("Prepared only; add --execute to run a new simulation evaluation")
        return
    env = os.environ.copy()
    env.update(WANDB_MODE="offline", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
    if args.accept_isaac_eula:
        env["OMNI_KIT_ACCEPT_EULA"] = "Yes"
    record["status"] = "running"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        with (run / "process.log").open("w") as log:
            process = subprocess.Popen(command, cwd=REPO_ROOT / "imports/SONIC", env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        record["exit_code"] = returncode
        record["status"] = "process_completed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        record.update(status="timed_out", exit_code=124)
    except KeyboardInterrupt:
        record.update(status="interrupted", exit_code=130)
    record["metrics_present"] = (run / "metrics/metrics_eval.json").is_file()
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "exit_code": record["exit_code"],
                      "metrics_present": record["metrics_present"], "log": str(run / "process.log")}))
    raise SystemExit(record["exit_code"] or (0 if record["metrics_present"] else 2))


if __name__ == "__main__":
    main()
