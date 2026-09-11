#!/usr/bin/env python3
"""Prepare or execute a bounded desktop-only terrain baseline or PPO smoke test.

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
import tempfile

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


def scene_overrides(run, data, stem, num_envs):
    return {
        "headless": True, "num_envs": num_envs, "seed": 42,
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


def evaluation_command(run, data, stem, num_envs, gui=False, clutter_audit=False,
                       cat_scene=None, cat_translation=(0., 0., 0.), cat_yaw=0., layout_audit=False):
    overrides = scene_overrides(run, data, stem, num_envs)
    overrides.update(eval_callbacks="im_eval", run_eval_loop=False,
                     eval_output_dir=str(run / "metrics"))
    if gui:
        overrides.update(headless=False, eval_callbacks=[], run_eval_loop=True,
                         realtime=True, run_once=False,
                         viewer_eye=[4.0, 4.0, 3.0], viewer_target=[0.0, 0.3, 1.0])
    if clutter_audit:
        overrides.update(eval_callbacks=[], run_eval_loop=True, run_once=True, max_render_steps=501,
                         research_clearance_output=str(run / "clearance_audit.json"))
        overrides["manager_env.config.research_clutter"] = "stair_side_v1"
    if cat_scene is not None:
        overrides.update(eval_callbacks=[], run_eval_loop=True, run_once=not gui, max_render_steps=0 if gui else 501,
                         research_cat_output=str(run/"cat_audit.json"))
        if gui:
            overrides["research_cat_gui_output"] = str(run)
        overrides.update({"manager_env.config.research_cat_scene": str(cat_scene),
                          "manager_env.config.research_cat_translation": list(cat_translation),
                          "manager_env.config.research_cat_yaw": cat_yaw})
    if layout_audit:
        if cat_scene is None or num_envs != 1:
            raise ValueError("Layout audit requires CAT and one environment")
        overrides["research_layout_output"] = str(run / "layout_audit.json")
        overrides["manager_env.config.research_layout_audit"] = True
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
    parser.add_argument("--gui", action="store_true", help="Interactive repeating policy demo, not a metrics run")
    parser.add_argument("--clutter-audit", action="store_true",
                        help="One bounded stair reference with physical side clutter; no training or avoidance claim")
    parser.add_argument("--cat-scene", type=Path, help="Opt-in CAT mesh/reference audit; no new learning")
    parser.add_argument("--layout-audit", action="store_true",
                        help="Check terrain support, passage and physical ray parity before CAT rollout; one environment")
    parser.add_argument("--cat-translation", type=float, nargs=3, default=(0., 0., 0.), metavar=("X", "Y", "Z"))
    parser.add_argument("--cat-yaw", type=float, default=0., help="CAT-to-terrain yaw in radians")
    parser.add_argument("--training-smoke", action="store_true",
                        help="Two PPO updates on a disposable checkpoint, not a full training run")
    parser.add_argument("--accept-isaac-eula", action="store_true",
                        help="Explicit user acceptance of the NVIDIA Omniverse license; never enabled by default")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.gui and args.training_smoke:
        parser.error("--gui is for the released-policy demo, not training")
    if args.layout_audit and (not args.cat_scene or args.num_envs != 1):
        parser.error("--layout-audit needs --cat-scene and --num-envs 1")
    if args.cat_scene:
        if args.training_smoke or args.clutter_audit or args.num_envs > 4:
            parser.error("CAT integration supports at most four environments; no training or primitive audit")
        from gear_sonic.research.cat_geometry import Placement, verify_scene_files
        Placement(tuple(args.cat_translation), args.cat_yaw)
        args.cat_scene = args.cat_scene.resolve()
        verify_scene_files(args.cat_scene)
    elif args.cat_translation != (0., 0., 0.) or args.cat_yaw != 0.:
        parser.error("CAT placement requires --cat-scene")
    if args.clutter_audit and (args.training_smoke or args.gui or args.family != "stair_p1"):
        parser.error("--clutter-audit is a separate headless stair-only diagnostic, not training/GUI")
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
    kind = "training_smoke" if args.training_smoke else ("gui_demo" if args.gui else "evaluation")
    if args.clutter_audit:
        kind = "clutter_audit"
    if args.cat_scene:
        kind = "cat_gui_demo" if args.gui else "cat_audit"
    run = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + args.family + "_" + kind)
    run.mkdir(parents=True, exist_ok=False)
    data = prepare_data(manifest, args.family, run)
    (run / "checkpoint").mkdir()
    for name in ("last.pt", "config.yaml"):
        shutil.copyfile(ROOT / "artifacts/checkpoint/SONIC/models/terrain_release" / name,
                        run / "checkpoint" / name)
    if args.training_smoke:
        from training_smoke import training_command
        command = training_command(run, scene_overrides(run, data, scene["stem"], args.num_envs))
    else:
        command = evaluation_command(run, data, scene["stem"], args.num_envs,
                                     gui=args.gui, clutter_audit=args.clutter_audit, cat_scene=args.cat_scene,
                                     cat_translation=args.cat_translation, cat_yaw=args.cat_yaw,
                                     layout_audit=args.layout_audit)
    record = {"simulation_only": True, "family": args.family, "num_envs": args.num_envs,
              "kind": kind,
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)),
              "dataset_revision": manifest["revision"], "command": command,
              "status": "prepared_not_executed"}
    record_path = run / "run.json"
    (run / "environment.json").write_text(json.dumps({
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    }, indent=2, sort_keys=True) + "\n")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Run directory: {run}", flush=True)
    if not args.execute:
        print("Prepared only; add --execute to start a new bounded simulation run")
        return
    env = os.environ.copy()
    # Isaac Lab otherwise uses shared /tmp/isaaclab, which may belong to a
    # different desktop user. Never chmod/delete another user's simulator data.
    # Keep the actual path short: multiprocessing's AF_UNIX sockets have a
    # ~108-byte path limit. The run-local symlink makes logs easy to locate.
    temporary_dir = Path(tempfile.mkdtemp(prefix="grail-", dir="/tmp"))
    (run / "tmp").symlink_to(temporary_dir, target_is_directory=True)
    env["TMPDIR"] = str(temporary_dir)
    record["temporary_directory"] = str(temporary_dir)
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
    outputs_valid = False
    if record["exit_code"] == 0:
        try:
            if args.training_smoke:
                from training_smoke import audit_training
                report = audit_training(run)
                record["training_audit"] = report
                outputs_valid = report["passed"]
            elif args.cat_scene:
                from evaluation_audit import finite_nested
                report = json.loads((run / "cat_audit.json").read_text())
                record["cat_audit_summary"] = {key: report[key] for key in (
                    "preflight_accepted", "reference", "min_clearance_by_link_m",
                    "peak_CAT_normal_force_by_link_N", "completed_envs", "failure_terminated_envs")}
                outputs_valid = (report["valid_diagnostic_run"] and report["preflight_accepted"]
                                 and report["rollout_completed_without_failure"] and finite_nested(report)
                                 and "Successfully loaded policy state dict" in (run / "process.log").read_text())
                record["note"] = "First-episode mesh/reference audit only; no terrain support or avoidance-training certification"
                if args.layout_audit:
                    layout = json.loads((run / "layout_audit.json").read_text())
                    record["layout_audit_passed"] = layout["accepted_for_reference_diagnostic"]
                    outputs_valid = outputs_valid and record["layout_audit_passed"] and finite_nested(layout)
                if args.gui:
                    record["viewer_ready"] = (run / "viewer_ready.json").is_file()
                    outputs_valid = outputs_valid and record["viewer_ready"]
            elif args.clutter_audit:
                from evaluation_audit import finite_nested
                report = json.loads((run / "clearance_audit.json").read_text())
                outputs_valid = (report["valid_diagnostic_run"] and finite_nested(report)
                                 and len(report["solids"]) == 6 and report["probe_count"] > 29
                                 and "Successfully loaded policy state dict" in (run / "process.log").read_text())
                record["note"] = "Geometry diagnostics only; not an avoidance/retention benchmark"
            elif args.gui:
                record["note"] = "Interactive demo closed; no benchmark metrics expected"
                outputs_valid = True
            else:
                from evaluation_audit import audit_evaluation
                report = audit_evaluation(run)
                record["evaluation_audit"] = report
                outputs_valid = report["valid_evidence"]
        except Exception as error:
            record["audit_error"] = f"{type(error).__name__}: {error}"
    record["outputs_valid"] = outputs_valid
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "exit_code": record["exit_code"],
                      "outputs_valid": outputs_valid, "log": str(run / "process.log")}))
    raise SystemExit(record["exit_code"] or (0 if outputs_valid else 2))


if __name__ == "__main__":
    main()
