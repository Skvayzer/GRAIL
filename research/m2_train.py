#!/usr/bin/env python3
"""Prepare/launch bounded M2 simulation, collection, training or evaluation.

Default: prepare only, no simulator or optimizer. Existing demos, released
checkpoints and robot software are never changed. User must explicitly choose
execution and, separately, optimizer updates. Validation split is evaluation
only. Each launch has its own run directory and logs.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

from artifacts import ROOT, PROVENANCE, safe_path, verify_file
from baseline import prepare_data, evaluation_command
from cat_roles import role_masks
from gear_sonic.research.residual_engine import UpdateConfig
from gear_sonic.research.training_admission import ChallengeAdmission, digest, artifact_path

REPO = ROOT.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curriculum", type=Path, default=ROOT/"config/m2_curriculum.json")
    p.add_argument("--scene", default="stair_development_0")
    p.add_argument("--retention-family", choices=("stair_p1", "curb", "slope", "sitting"),
                   help="Evaluation control: unchanged reference reset, CAT fixture 100m outside workspace")
    p.add_argument("--mode", choices=("collect", "train", "evaluate"), default="collect")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--approve-optimizer", action="store_true")
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--num-envs", type=int, choices=range(1, 5), default=4)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume", type=Path)
    p.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    p.add_argument("--wandb-entity", default="skvayzer")
    p.add_argument("--wandb-project", default="grail-cat")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--accept-isaac-eula", action="store_true")
    args = p.parse_args(argv)
    if (not 1 <= args.iterations <= 10000 or not 0 <= args.seed < 2**31
            or not 1 <= args.timeout <= 172800 or not 1 <= args.checkpoint_every <= 10000):
        p.error("Bounded iteration, seed, timeout and checkpoint values required")
    if args.mode == "train" and not args.approve_optimizer:
        p.error("Training requires --approve-optimizer; --execute is a separate simulation choice")
    if args.mode != "train" and args.approve_optimizer:
        p.error("Optimizer approval is only meaningful in train mode")
    if args.retention_family and args.mode != "evaluate":
        p.error("Retention controls are evaluation-only")
    config = UpdateConfig(horizon=args.horizon, epochs=args.epochs, minibatch_size=args.minibatch_size,
                          learning_rate=args.learning_rate)
    if args.minibatch_size > args.horizon*args.num_envs:
        p.error("Minibatch must fit the rollout")
    if args.resume and not args.resume.is_file():
        p.error("Resume checkpoint does not exist")
    return args, config


def main(argv=None):
    args, update = parse_args(argv)
    curriculum = json.loads(args.curriculum.read_text())
    if (curriculum.get("schema") != "grail-cat-m2-curriculum-v1"
            or curriculum.get("environment_review_approved") is not True):
        raise ValueError("Reviewed curriculum manifest required")
    spec = curriculum["scenes"][args.scene]
    if args.mode == "train" and spec["split"] != "development":
        raise ValueError("Never train on a validation/retention scene")
    witness = (REPO/spec["witness"]).resolve()
    challenge = ChallengeAdmission(witness, spec["witness_sha256"], spec["case"])
    scene = artifact_path(challenge.report["scene"])
    role_masks(scene)  # Replay source generation and verify role arrays before Kit.
    placement = challenge.report["placement"]
    if args.retention_family:
        spec = dict(spec, family=args.retention_family, split="retention")
        placement = dict(translation=[100., 100., 0.], yaw=0.)
    if args.execute and not args.accept_isaac_eula:
        package = importlib.metadata.distribution("isaacsim")
        if not Path(package.locate_file("isaacsim/kit/EULA_ACCEPTED")).is_file() and os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() != "yes":
            raise ValueError("Isaac requires the user's license acceptance (--accept-isaac-eula)")
    manifest = json.loads((ROOT/"data_manifest.json").read_text())
    if manifest["revision"] != PROVENANCE["dataset_revision"]:
        raise ValueError("Paired dataset revision changed")
    for item in manifest["files"]:
        if item["path"].startswith(("checkpoint/", "data/"+spec["family"]+"/")):
            verify_file(ROOT/"artifacts"/safe_path(item["path"]), item)
    family = next(item for item in manifest["scenes"] if item["family"] == spec["family"])
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_m2_"+args.mode)
    run.mkdir(parents=True, exist_ok=False)
    data = prepare_data(manifest, spec["family"], run)
    (run/"checkpoint").mkdir()
    for name in ("last.pt", "config.yaml"):
        shutil.copyfile(ROOT/"artifacts/checkpoint/SONIC/models/terrain_release"/name, run/"checkpoint"/name)
    command = evaluation_command(run, data, family["stem"], args.num_envs, cat_scene=scene,
        cat_translation=placement["translation"], cat_yaw=placement["yaw"], layout_audit=True,
        residual_preflight=True, avoidance_task=True)
    plan = dict(schema="grail-cat-training-plan-v1", simulation_only=True, execute=args.execute,
        approve_optimizer=args.approve_optimizer, environment_review_approved=True, mode=args.mode,
        iterations=args.iterations, scene=args.scene, split=spec["split"],
        retention_family=args.retention_family,
        witness=None if args.retention_family else str(witness),
        witness_sha256=spec["witness_sha256"], case=spec["case"], curriculum_sha256=digest(args.curriculum),
        num_envs=args.num_envs, seed=args.seed, checkpoint_every=args.checkpoint_every,
        resume=str(args.resume.resolve()) if args.resume else None,
        wandb_mode=args.wandb_mode, wandb_entity=args.wandb_entity, wandb_project=args.wandb_project,
        update={key: getattr(update, key) for key in ("horizon", "epochs", "minibatch_size", "learning_rate",
                                                    "max_gradient_norm", "target_kl", "gamma", "gae_lambda")},
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)))
    plan_path = run/"training_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2)+"\n")
    command.append("++research_training_plan="+json.dumps(str(plan_path)))
    record = dict(command=command, status="prepared", simulation_only=True)
    record_path = run/"run.json"
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print("Run directory: "+str(run), flush=True)
    if not args.execute:
        print("Prepared only. No simulation, W&B run or optimizer started.", flush=True)
        return
    environment = os.environ.copy()
    temporary = Path(tempfile.mkdtemp(prefix="grail-m2-", dir="/tmp"))
    (run/"tmp").symlink_to(temporary)
    environment.update(TMPDIR=str(temporary), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    environment.pop("WANDB_DISABLED", None)
    environment["WANDB_MODE"] = args.wandb_mode if args.wandb_mode != "disabled" else "disabled"
    if args.accept_isaac_eula:
        environment["OMNI_KIT_ACCEPT_EULA"] = "Yes"
    record["status"] = "running"
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    with (run/"process.log").open("w") as log:
        process = subprocess.Popen(command, cwd=REPO/"imports/SONIC", env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["pid"] = process.pid
        record_path.write_text(json.dumps(record, indent=2)+"\n")
        try:
            exit_code = process.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            exit_code = 124
    result_path = run/"training_result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    record.update(status="finished", exit_code=exit_code,
                  outputs_valid=exit_code == 0 and result.get("complete") is True and result.get("backbone_unchanged") is True)
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(dict(run=str(run), status=record["status"], exit_code=exit_code,
                          outputs_valid=record["outputs_valid"])), flush=True)
    if not record["outputs_valid"]:
        raise SystemExit(exit_code or 2)


if __name__ == "__main__":
    main()
