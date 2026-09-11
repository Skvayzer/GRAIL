"""Check finite rollout evidence, distinct from task success on a sample."""
import json
import math


def finite_nested(value):
    if isinstance(value, dict):
        return all(finite_nested(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_nested(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def audit_evaluation(run):
    metrics = json.loads((run / "metrics/metrics_eval.json").read_text())
    trajectory = json.loads((run / "trajectory.json").read_text())
    log = (run / "process.log").read_text(errors="replace")
    all_metrics = metrics["eval/all_metrics_dict"]
    finite_all = finite_nested(all_metrics)
    # The upstream success-only aggregate may be NaN when every clip fails.
    # That is different from non-finite actual states or per-motion metrics.
    finite_states = bool(trajectory) and finite_nested(trajectory)
    shape_ok = all(len(s["actions"]) == 29 and len(s["joint_pos"]) == 29
                   and len(s["root_pos"]) == 3 and len(s["root_quat"]) == 4
                   for s in trajectory)
    restored = "Successfully loaded policy state dict" in log
    report = {
        "sample_only_not_a_benchmark": True,
        "actor_restored_strictly": restored,
        "trajectory_steps": len(trajectory),
        "trajectory_finite": finite_states, "action_state_shapes_valid": shape_ok,
        "per_motion_metrics_finite": finite_all,
        "motion_keys": all_metrics["motion_keys"],
        "terminated": all_metrics["terminated"],
        "progress": all_metrics["progress"],
        "global_mpjpe_mm": all_metrics["mpjpe_g"],
    }
    report["valid_evidence"] = finite_all and finite_states and shape_ok and restored
    report["all_sample_references_succeeded"] = bool(all_metrics["terminated"]) and not any(all_metrics["terminated"])
    (run / "evaluation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
