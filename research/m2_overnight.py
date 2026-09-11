#!/usr/bin/env python3
"""Bounded unattended Isaac pilot; no robot, automatic retries or host changes.

Run with m2_job.py for an SSH-independent owned job, or an approved service.
Checksums, passing setup suite and explicit execution/update approval required.
Checkpoints are run-local every 100 iterations. Stop on a stalled simulator,
less than 20 GiB free disk, launcher failure, or wall-clock budget expiration.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from artifacts import ROOT
from m2_curriculum import evidence_files
from m2_results import audit
from gear_sonic.research.training_admission import digest


def supervision_reason(free_gib, progress_age_s, elapsed_s, limit_s):
    if not all(math.isfinite(x) for x in (free_gib, progress_age_s, elapsed_s, limit_s)):
        return "Nonfinite supervisor measurements"
    if free_gib < 20:
        return "Less than 20 GiB free disk"
    if progress_age_s > 600:
        return "No training progress for ten minutes"
    if elapsed_s > limit_s:
        return "Wall-clock budget reached"
    return None


def stop_owned(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)  # m2_train cleans its owned Isaac group.
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            # Never look up or kill any other training/robot process by name.
            raise RuntimeError("Owned launcher failed to stop; inspect its recorded PID and run log")


def write_status(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-suite", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=8.)
    parser.add_argument("--iterations", type=int, default=8000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approve-optimizer", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours <= 12 or not 1 <= args.iterations <= 10000:
        parser.error("Bounded positive hours <=12 and iterations <=10000 required")
    if args.execute and not args.approve_optimizer:
        parser.error("Execution requires --approve-optimizer")
    suite = json.loads(args.preflight_suite.read_text())
    if (suite.get("schema") != "grail-cat-m2-evaluation-suite-v1" or not suite.get("pipeline_complete")
            or {x["name"] for x in suite["results"]} != {"validation", "stair_p1", "curb", "slope"}
            or len(suite["results"]) != 4 or not all(x["pipeline_passed"] for x in suite["results"])):
        raise ValueError("Validation and three locomotion setup evaluations must finish before overnight launch")
    for item in suite["results"]:
        checked = audit(item["run"])
        if checked["mode"] != "evaluate" or checked["optimizer_steps"] != 0:
            raise ValueError("Preflight suite must contain actual zero-update evaluation records")
    evidence_files()
    command = [sys.executable, str(ROOT/"m2_train.py"), "--mode", "train", "--approve-optimizer",
               "--scene", "stair_development_0", "--num-envs", "4", "--iterations", str(args.iterations),
               "--checkpoint-every", "100", "--wandb-mode", "online", "--accept-isaac-eula",
               "--timeout", str(int(args.hours*3600)), "--execute"]
    if not args.execute:
        import shlex
        print(shlex.join(command))
        print("Prepared only; no training started.")
        return
    run = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_overnight")
    run.mkdir(exist_ok=False)
    status = run/"status.json"
    report = dict(schema="grail-cat-overnight-v1", simulation_only=True, status="starting", command=command,
                  started_utc=datetime.now(timezone.utc).isoformat(), hours_limit=args.hours,
                  preflight_suite=str(args.preflight_suite.resolve()), preflight_sha256=digest(args.preflight_suite),
                  source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True).strip(),
                  no_automatic_restart=True, training_run=None)
    write_status(status, report)
    print("Overnight supervisor: "+str(run), flush=True)
    started = time.monotonic()
    log_path = run/"launcher.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        report["launcher_pid"] = process.pid
        try:
            while process.poll() is None:
                text = log_path.read_text()
                paths = [x.removeprefix("Run directory: ") for x in text.splitlines() if x.startswith("Run directory: ")]
                if paths:
                    report["training_run"] = paths[-1]
                    training = Path(paths[-1])
                    if not training.resolve().is_relative_to(ROOT/"runs"):
                        raise ValueError("Unexpected launcher output path")
                    metrics = training/"metrics.jsonl"
                    if metrics.exists():
                        with metrics.open("rb") as stream:
                            stream.seek(max(0, metrics.stat().st_size-65536))
                            lines = stream.read().decode().splitlines()
                        # A concurrent final line may not yet be complete.
                        for line in reversed(lines[-2:]):
                            try:
                                last = json.loads(line)
                                break
                            except json.JSONDecodeError:
                                continue
                        else:
                            last = {}
                        report["last_iteration"] = last.get("iteration")
                        report["environment_steps"] = last.get("environment_steps")
                        report["optimizer_steps"] = last.get("optimizer_steps")
                        report["seconds_since_progress"] = time.time()-metrics.stat().st_mtime
                    else:
                        report["seconds_since_progress"] = time.monotonic()-started
                    checkpoints = sorted(training.glob("learner_*.pt"))
                    report["latest_checkpoint"] = str(checkpoints[-1]) if checkpoints else None
                else:
                    report["seconds_since_progress"] = time.monotonic()-started
                free = shutil.disk_usage(ROOT).free/2**30
                report["free_disk_gib"] = free
                reason = supervision_reason(free, report["seconds_since_progress"],
                                            time.monotonic()-started, args.hours*3600)
                if reason:
                    raise RuntimeError(reason)
                report["status"] = "training"
                report["updated_utc"] = datetime.now(timezone.utc).isoformat()
                write_status(status, report)
                time.sleep(10)
            report["exit_code"] = process.returncode
            if process.returncode != 0 or not report["training_run"]:
                raise RuntimeError("Training launcher failed; no automatic retry")
            training = Path(report["training_run"])
            result = json.loads((training/"training_result.json").read_text())
            report["training_audit"] = audit(training)
            if not result["complete"] or not result["backbone_unchanged"] or not result["learner_changed"]:
                raise RuntimeError("Training result did not satisfy learner/backbone checks")
            report.update(status="completed", checkpoint=result["checkpoint"], wandb_url=result.get("wandb_url"),
                          total_optimizer_steps=result["total_optimizer_steps"],
                          optimizer_steps=result["total_optimizer_steps"], last_iteration=len(result["metrics"]),
                          latest_checkpoint=result["checkpoint"],
                          environment_steps=result["simulator_steps"]*result["plan"]["num_envs"])
            remaining = int(args.hours*3600-(time.monotonic()-started))
            if remaining > 900:
                evaluation = [sys.executable, str(ROOT/"m2_evaluate.py"), "--checkpoint", result["checkpoint"],
                              "--execute", "--wandb-mode", "online", "--accept-isaac-eula"]
                report["status"] = "post_training_evaluation"
                write_status(status, report)
                with (run/"evaluation.log").open("w") as eval_log:
                    process = subprocess.Popen(evaluation, cwd=ROOT.parent, stdout=eval_log,
                                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    process.wait(timeout=remaining)
                report["evaluation_exit_code"] = process.returncode
                report["status"] = "completed" if process.returncode == 0 else "trained_evaluation_failed"
            else:
                report["evaluation_skipped"] = "Less than 15 minutes remain in the wall-clock budget"
        except BaseException as error:
            report.update(status="stopped", error=type(error).__name__+": "+str(error))
            stop_owned(process)
            raise
        finally:
            report["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_status(status, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
