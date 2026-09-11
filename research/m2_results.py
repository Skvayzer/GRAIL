#!/usr/bin/env python3
"""Read-only audit of completed M2 logs/checkpoints; no simulator or updates."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from gear_sonic.research.learning_checkpoint import contract_hash, finite_tree, validate_update_count
from gear_sonic.research.training_admission import digest


def audit(run):
    run = Path(run)
    result = json.loads((run/"training_result.json").read_text())
    plan = json.loads((run/"training_plan.json").read_text())
    execution = json.loads((run/"run.json").read_text())
    rows = [json.loads(x) for x in (run/"metrics.jsonl").read_text().splitlines()]
    if (execution["exit_code"] != 0 or not execution["outputs_valid"] or not result["complete"]
            or result["plan_sha256"] != digest(run/"training_plan.json") or result["plan"] != plan
            or len(rows) != plan["iterations"] or rows != result["metrics"]
            or not result["backbone_unchanged"] or not result["checkpoint_roundtrip_verified"]):
        raise ValueError("Incomplete/inconsistent M2 execution records")
    start = result["initial_optimizer_steps"]
    updates = start
    horizon, n = plan["update"]["horizon"], plan["num_envs"]
    for i, row in enumerate(rows, 1):
        finite_tree(row)
        expected_updates = row.get("update", {}).get("optimizer_steps", 0)
        if plan["mode"] != "train" and expected_updates != 0:
            raise ValueError("Non-training mode updated weights")
        updates += expected_updates
        if (row["iteration"] != i or row["environment_steps"] != i*horizon*n
                or row["optimizer_steps"] != updates):
            raise ValueError("Discontinuous iteration/sample/update counters")
    checkpoint = run/"learner_final.pt"
    if digest(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Saved checkpoint checksum differs from completed result")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    finite_tree(saved)
    validate_update_count(saved["optimizer"], saved["updates"])
    h = hashlib.sha256()
    for key, tensor in sorted(saved["model"].items()):
        h.update(json.dumps([key, list(tensor.shape), str(tensor.dtype)]).encode())
        h.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    changed = h.hexdigest() != result["initial_learner_sha256"]
    if (saved["updates"] != updates or result["total_optimizer_steps"] != updates
            or result["optimizer_steps"] != updates-start or h.hexdigest() != result["learner_sha256"]
            or changed != result["learner_changed"] or (plan["mode"] != "train" and changed)
            or saved["contract_sha256"] != contract_hash(result["contract"])
            or json.loads(json.dumps(saved["contract"])) != result["contract"]
            or result["simulator_steps"] != horizon*len(rows)):
        raise ValueError("Actual checkpoint contents differ from runtime contract/counters")
    return dict(run=str(run), passed=True, mode=plan["mode"], transitions=n*horizon*len(rows),
                optimizer_steps=updates-start, total_optimizer_steps=updates,
                failures=result["failures"], timeouts=result["timeouts"],
                minimum_cat_gap_m=min(x["minimum_cat_gap_m"] for x in rows),
                maximum_contact_n=max(x["maximum_contact_n"] for x in rows),
                maximum_foot_error_m=max(x["maximum_foot_error_m"] for x in rows),
                checkpoint_sha256=digest(checkpoint), learned_avoidance_certified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    for run in parser.parse_args().runs:
        print(json.dumps(audit(run)))
