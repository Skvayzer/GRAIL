#!/usr/bin/env python3
"""Run deterministic pilot checks sequentially, never optimizer updates.

Defaults to printing commands. With --execute, evaluates the validation layout
and three locomotion controls, one simulator process at a time. Sitting is
explicitly not admitted by the standing/stepping support-graph contract. Scene
controls use a real CAT fixture outside the workspace, not masked distances.
Successful execution is distinct from successful task performance.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import signal
from pathlib import Path
import subprocess
import sys

from artifacts import ROOT
from gear_sonic.research.training_admission import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--accept-isaac-eula", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error("checkpoint must exist")
    checkpoint = args.checkpoint.resolve()
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["contract"]["algorithm"]
    commands = []
    for name in ("validation", "stair_p1", "curb", "slope"):
        cmd = [sys.executable, str(ROOT/"m2_train.py"), "--mode", "evaluate", "--resume", str(checkpoint),
               "--num-envs", "1", "--iterations", str(math.ceil(512/config["horizon"])), "--wandb-mode", args.wandb_mode,
               "--horizon", str(config["horizon"]), "--epochs", str(config["epochs"]),
               "--minibatch-size", str(config["minibatch_size"]),
               "--learning-rate", str(config["learning_rate"]),
               "--critic-mode", saved["model_contract"].get("critic_mode", "shared")]
        if not saved["contract"]["task"].get("guidance_termination"):
            cmd.append("--legacy-guidance-abort")
        cmd += (["--scene", "stair_validation_102"] if name == "validation" else ["--retention-family", name])
        if args.accept_isaac_eula:
            cmd.append("--accept-isaac-eula")
        commands.append((name, cmd))
    if not args.execute:
        import shlex
        for _, cmd in commands:
            print(shlex.join(cmd+["--execute"]))
        print("Prepared only. No simulation, W&B or optimizer started.")
        return
    output = ROOT/"runs"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")+"_m2_evaluation_suite")
    output.mkdir(exist_ok=False)
    report = dict(schema="grail-cat-m2-evaluation-suite-v1", checkpoint=str(checkpoint),
                  checkpoint_sha256=digest(checkpoint), task_performance_acceptance=False, results=[],
                  excluded_controls={"sitting": "Not admitted: chair seating contacts are outside the standing/stepping support graph; not a passed retention test"})
    print("Evaluation suite: "+str(output), flush=True)
    for name, command in commands:
        print("Evaluating "+name, flush=True)
        log_path = output/(name+".log")
        with log_path.open("w") as log:
            process = subprocess.Popen(command+["--execute"], cwd=ROOT.parent,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=45)
                raise
        records = [json.loads(line) for line in log_path.read_text().splitlines()
                   if line.startswith('{"run":') and '"outputs_valid"' in line]
        record = dict(name=name, exit_code=exit_code, pipeline_passed=False)
        if records:
            run = Path(records[-1]["run"])
            record["run"] = str(run)
            result_file = run/"training_result.json"
            if result_file.exists():
                result = json.loads(result_file.read_text())
                record.update(pipeline_passed=exit_code == 0 and result["complete"]
                    and result["backbone_unchanged"] and not result.get("learner_changed", True)
                    and result["optimizer_steps"] == 0,
                    failures=result.get("failures"), timeouts=result.get("timeouts"),
                    wandb_url=result.get("wandb_url"))
        report["results"].append(record)
        report["pipeline_complete"] = len(report["results"]) == len(commands) and all(r["pipeline_passed"] for r in report["results"])
        (output/"suite.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(record), flush=True)
        if exit_code == 124:
            break  # Don't start another scene after an operator interrupt/timeout.
    if not report["pipeline_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
