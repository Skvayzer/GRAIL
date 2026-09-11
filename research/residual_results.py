"""Independently check recorded zero-update learner transitions, without Isaac."""
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_runtime(run):
    run = Path(run)
    report = json.loads((run/"residual_preflight.json").read_text())
    path = run/"residual_preflight.npz"
    if (report["schema"] != "grail-cat-residual-runtime-preflight-v1" or not report["complete"]
            or report["data_file"] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != report["data_sha256"]
            or not report["base_backbone_unchanged"] or not report["learner_unchanged"]
            or report["optimizer_steps"] != 0 or report["exploration_enabled"] or report["training_data"]
            or not report["actor_action_parity_exact"]):
        raise ValueError("Incomplete/mutated runtime diagnostic or wrong execution contract")
    with np.load(path, allow_pickle=False) as data:
        a = {key: data[key] for key in data.files}
    required = {"state", "next_state", "post_step_state", "value", "next_value", "post_step_value", "reward",
                "terminated", "truncated", "dones", "reference_step", "advantages", "returns"}
    if set(a) != required:
        raise ValueError("Missing or unexpected runtime transition fields")
    t, n, d = report["steps"], report["num_envs"], report["state"]["state_dim"]
    if not 2 <= t <= 500 or not 1 <= n <= 4 or report["capture_calls"] != t:
        raise ValueError("Invalid runtime dimensions/capture count")
    for key, value in a.items():
        expected = (t, n, d) if key in ("state", "next_state", "post_step_state") else (t, n)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"Malformed/nonfinite transition tensor: {key}")
    term, trunc, done = a["terminated"], a["truncated"], a["dones"]
    if any(x.dtype != bool for x in (term, trunc, done)) or not np.array_equal(done, term | trunc):
        raise ValueError("Reset mask must be exact terminated OR truncated")
    if term.any() or not done.any(0).all() or not (trunc & ~term).any():
        raise ValueError("Integration control did not reach clean timeouts in every environment")
    if (not np.allclose(a["next_state"][~done], a["post_step_state"][~done], atol=3e-5, rtol=0)
            or not np.allclose(a["next_value"][~done], a["post_step_value"][~done], atol=3e-5, rtol=0)):
        raise ValueError("Non-reset next-state/value parity failed")
    if not np.array_equal(a["state"][1:], a["post_step_state"][:-1]):
        raise ValueError("Consecutive learner state/history continuity failed")
    h, p = report["state"]["history_length"], report["state"]["physical_dim"]
    history = a["post_step_state"][done, :h*p].reshape(-1, h, p)
    if not np.array_equal(history, np.broadcast_to(history[:, -1:], history.shape)):
        raise ValueError("Reset histories contain previous episode samples")
    # Independent NumPy implementation, not calling the tensor function under test.
    expected = np.zeros_like(a["value"])
    carry = np.zeros(n, dtype=np.float32)
    for i in range(t-1, -1, -1):
        delta = a["reward"][i]+.99*a["next_value"][i]*(~term[i])-a["value"][i]
        carry = delta+.99*.95*(~done[i])*carry
        expected[i] = carry
    if (not np.allclose(a["advantages"], expected, atol=2e-4, rtol=2e-5)
            or not np.allclose(a["returns"], expected+a["value"], atol=2e-4, rtol=2e-5)):
        raise ValueError("Recorded GAE differs from independent reset-aware calculation")
    timeout = trunc & ~term
    separation = float(np.abs(a["next_state"][timeout]-a["post_step_state"][timeout]).max())
    if separation < .01:
        raise ValueError("Recorded timeout did not demonstrate distinct terminal/reset states")
    return dict(passed=True, steps=t, environments=n, state_dim=d,
        timeouts=int(timeout.sum()), failures=int(term.sum()), timeout_state_reset_separation=separation,
        nonreset_state_parity=True, reset_history_cleared=True, independent_gae_verified=True,
        optimizer_steps=0, training_data=False)
