"""Independent read-only checks of saved shadow features and action parity."""
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_observation_shadow(run):
    run = Path(run)
    meta = json.loads((run/"observation_shadow.json").read_text())
    if (meta["schema"] != "grail-cat-observation-shadow-v1" or not meta["complete"] or meta["error"] is not None
            or meta["packet_file"] != "observation_shadow.npz" or meta["adapter_applied_to_environment"]
            or meta["original_actor_inputs_changed"] or meta["rewards_changed"] or meta["policy_training"]
            or meta["optimizer_steps"] != 0 or meta["physics_steps_added"] != 0
            or not meta["base_backbone_unchanged"] or not meta["adapter_unchanged"]):
        raise ValueError("Incomplete/unsafe shadow report")
    path = run/meta["packet_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["packet_sha256"]:
        raise ValueError("Shadow packet checksum mismatch")
    n, count, latent = meta["frames"], len(meta["probe_order"]), meta["latent_dim"]
    expected = dict(volume=(n, 4, *meta["packet_spec"]["shape"]), probes=(n, count, 8), guidance=(n, 9),
                    valid=(n,), root=(n, 3), quaternion=(n, 4), probe_centers=(n, count, 3), baseline_action=(n, 29),
                    zero_residual_action=(n, 29), live_action=(n, 29), residual=(n, latent))
    if not 1 <= n <= 500 or not 1 <= count <= 512 or not 1 <= latent <= 256:
        raise ValueError("Invalid shadow dimensions")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("Unexpected shadow packet fields")
        data = {k: archive[k] for k in archive.files}
    if any(data[k].shape != shape or not np.isfinite(data[k]).all() for k, shape in expected.items()):
        raise ValueError("Nonfinite or wrong-shaped shadow payload")
    if data["valid"].dtype != bool or (np.abs(np.linalg.norm(data["quaternion"], axis=1)-1) > 1e-4).any():
        raise ValueError("Invalid shadow pose/mask")
    v, p, g = data["volume"], data["probes"], data["guidance"]
    for mask in (v[:, 2:], p[..., 6:], g[:, 7:]):
        if not np.isin(mask, (0, 1)).all():
            raise ValueError("Nonbinary feature mask")
    if (np.abs(v[:, :2]) > 1).any() or (v[:, 1] < 0).any() or (np.abs(p[..., :6]) > 1).any():
        raise ValueError("Distance/geometry encoding range mismatch")
    if (p[..., 3] <= 0).any() or (p[..., 5] < 0).any():
        raise ValueError("Invalid radius or unsigned terrain channel")
    if (v[:, 0][v[:, 2] == 0] != 0).any() or (v[:, 1][v[:, 3] == 0] != 0).any():
        raise ValueError("Unknown volume lacks numeric placeholder")
    if (p[..., 4][p[..., 6] == 0] != 0).any() or (p[..., 5][p[..., 7] == 0] != 0).any():
        raise ValueError("Unknown probe lacks numeric placeholder")
    if (g[g[:, 7] == 0] != 0).any() or (np.abs(g[:, :7]) > 1+1e-6).any():
        raise ValueError("Invalid guidance encoding")
    valid = (v[:, 2:] == 1).reshape(n, -1).all(1) & (p[..., 6:] == 1).reshape(n, -1).all(1) & (g[:, 7] == 1)
    if not np.array_equal(valid, data["valid"]) or int(valid.sum()) != meta["valid_frames"]:
        raise ValueError("Validity flags do not match packet channels")
    if (data["residual"] != 0).any() or not np.array_equal(data["baseline_action"], data["zero_residual_action"]) or not np.array_equal(data["baseline_action"], data["live_action"]):
        raise ValueError("Saved action/residual parity failed")
    samples, outcomes = meta["samples"], meta["outcomes"]
    if (len(samples) != n or len(outcomes) != n or any(x["step"] != i for i, x in enumerate(samples))
            or any(x["step"] != i for i, x in enumerate(outcomes)) or not outcomes[-1]["terminal"]
            or sum(x["terminal"] for x in outcomes) != 1 or any(x["failure"] for x in outcomes)
            or any(s["parity_max_abs_error"] != 0 or s["valid"] != bool(valid[i]) for i, s in enumerate(samples))):
        raise ValueError("Unpaired, reset-contaminated or invalid shadow sequence")
    check = meta["gradient_check"]
    if (not check or not 0 <= check["step"] < n or not valid[check["step"]]
            or not np.isfinite(check["head_gradient_norm"]) or check["head_gradient_norm"] <= 0
            or check["frozen_actor_parameter_gradients"]):
        raise ValueError("Missing/invalid frozen-decoder gradient path check")
    return dict(passed=True, frames=n, valid_frames=int(valid.sum()), zero_residual_action_parity_exact=True,
                gradient_head_norm=check["head_gradient_norm"], adapter_applied=False, avoidance_training_ready=False)
