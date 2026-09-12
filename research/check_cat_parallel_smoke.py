#!/usr/bin/env python3
"""CPU audit of the completed four-phase generated-clutter optimizer smoke.

Checks update/checkpoint evidence, not navigation success. Starts no simulator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def finite_tree(value):
    if isinstance(value,torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor")
    elif isinstance(value,dict):
        for item in value.values(): finite_tree(item)
    elif isinstance(value,(tuple,list)):
        for item in value: finite_tree(item)
    elif isinstance(value,float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint scalar")


def audit(run):
    run=Path(run)
    if not (run/"complete.json").exists() or (run/"error.json").exists():
        raise ValueError("Smoke worker has not completed successfully")
    config=json.loads((run/"config.json").read_text())
    expected=[("lateral","transfer"),("lateral","ppo"),("generalist","dagger"),("generalist","ppo")]
    metrics=[json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    if config["args"]["smoke_updates"]!=1 or [(m["family"],m["kind"]) for m in metrics]!=expected:
        raise ValueError("Expected exactly one rollout/update for each of the four smoke phases")
    steps=0;updates=0;checkpoints=[]
    for metric in metrics:
        finite_tree(metric)
        if metric["total_steps"]<=steps or metric["optimizer_updates"]<=updates:
            raise ValueError("Learner counters did not advance")
        steps,updates=metric["total_steps"],metric["optimizer_updates"]
        path=run/f"checkpoint_{steps:012d}.pt"
        checkpoint=torch.load(path,map_location="cpu",weights_only=False)
        finite_tree(checkpoint)
        if (checkpoint["total_steps"]!=steps or checkpoint["total_updates"]!=updates
                or checkpoint["bank_sha256"]!=config["bank_sha256"]):
            raise ValueError("Checkpoint/config/counter mismatch")
        checkpoints.append(checkpoint)
    if any(c["frozen_hashes"]!=checkpoints[0]["frozen_hashes"] for c in checkpoints):
        raise ValueError("Frozen provenance changed")
    # Adapter last layer was initialized to zero, unlike its hidden layers.
    for checkpoint,family in ((checkpoints[0],"lateral"),(checkpoints[2],"generalist")):
        weights=checkpoint["models"][family]["adapter"]
        last=max(int(k.split(".")[0]) for k in weights if k.endswith(".weight"))
        if not weights[f"{last}.weight"].abs().sum()>0:
            raise ValueError("Transfer/distillation did not change adapter output weights")
    for before,after,family in ((checkpoints[0],checkpoints[1],"lateral"),
                                (checkpoints[2],checkpoints[3],"generalist")):
        if torch.equal(before["models"][family]["log_std"],after["models"][family]["log_std"]):
            raise ValueError("PPO did not update action distribution")
    latest=json.loads((run/"latest.json").read_text())
    path=run/latest["checkpoint"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=latest["sha256"]:
        raise ValueError("Latest checkpoint checksum mismatch")
    return dict(passed=True,num_envs=config["args"]["num_envs"],phases=expected,
        total_steps=steps,optimizer_updates=updates,
        mean_env_steps_per_second=sum(m["env_steps_per_second"] for m in metrics)/len(metrics),
        max_torch_reserved_gib=max(m["cuda_reserved_gb"] for m in metrics),
        min_device_free_gib=min(m["cuda_free_gb"] for m in metrics),
        finite_checkpoints=True,adapter_and_ppo_weights_changed=True,
        frozen_provenance_unchanged=True,latest_checkpoint_sha256=latest["sha256"],
        evidence="Four-phase optimizer integration check, NOT learned navigation success",robot_actuation=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    result=audit(args.run)
    if args.output:
        with args.output.open("x") as output: json.dump(result,output,indent=2)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
