"""Check the actual no-update avoidance task capture, independently of Isaac."""
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_task(run):
    run = Path(run)
    report = json.loads((run/"avoidance_task.json").read_text())
    residual = json.loads((run/"residual_preflight.json").read_text())
    path = run/"avoidance_task.npz"
    if (report["schema"] != "grail-cat-posture-avoidance-task-v1" or report["data_file"] != path.name
            or hashlib.sha256(path.read_bytes()).hexdigest() != report["data_sha256"]
            or not residual["complete"] or not residual["original_rewards_changed"]
            or not residual["original_terminations_changed"]):
        raise ValueError("Missing/mutated explicit avoidance task capture")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    expected = {"clearance_penalty", "contact_penalty", "contact_failure", "root_progress", "goal_distance",
                "link_gaps", "min_gap", "contact_peak", "foot_world_error"}
    t, n, links = report["steps"], residual["num_envs"], len(report["collision_links"])
    if set(arrays) != expected or t != residual["steps"] or len(report["outcomes"]) != t:
        raise ValueError("Avoidance transitions are incomplete")
    for name, value in arrays.items():
        shape = (t, n, links) if name == "link_gaps" else (t, n)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("Malformed/nonfinite avoidance tensors")
    clearance = report["spec"]["clearance_m"]
    expected_penalty = np.minimum(np.maximum((clearance-arrays["link_gaps"])/clearance, 0)**2, 4).mean(-1)
    if (not np.allclose(arrays["clearance_penalty"], expected_penalty, atol=1e-6, rtol=0)
            or not np.array_equal(arrays["min_gap"], arrays["link_gaps"].min(-1))):
        raise ValueError("Per-link clearance penalty differs from geometry")
    if arrays["contact_failure"].dtype != bool or not np.array_equal(arrays["contact_failure"],
            arrays["contact_peak"] > report["spec"]["contact_failure_threshold_n"]):
        raise ValueError("CAT contact termination differs from physical threshold")
    if (arrays["contact_failure"].any() or (arrays["foot_world_error"] > report["spec"]["foot_world_failure_m"]).any()
            or any(any(row["terminated"]) for row in report["outcomes"])):
        raise ValueError("Frozen integration control failed the proposed task")
    return dict(passed=True, steps=t, environments=n, min_clutter_gap_m=float(arrays["min_gap"].min()),
        peak_cat_contact_n=float(arrays["contact_peak"].max()), max_foot_world_error_m=float(arrays["foot_world_error"].max()),
        reward_profile_loaded=True, new_policy_trained=False, avoidance_success_demonstrated=False)
