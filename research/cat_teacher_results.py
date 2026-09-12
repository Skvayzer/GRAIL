"""Independent saved-file teacher/action-history pairing audit; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gear_sonic.research.cat_bridge import CatObservationBridge, indices
from gear_sonic.research.cat_teacher import CatTeacher

ROOT = Path(__file__).resolve().parent


def audit_teacher_shadow(run, teacher_dir=ROOT/"artifacts/cat_teacher_v2"):
    run, teacher_dir = Path(run), Path(teacher_dir)
    meta = json.loads((run/"cat_teacher_shadow.json").read_text())
    if (meta["schema"] != "grail-cat-teacher-shadow-v1" or not meta["complete"]
            or not meta["base_backbone_unchanged"] or not meta["teacher_unchanged"]
            or meta["robot_actuation"] or meta["optimizer_steps"] or meta["training_admitted"]):
        raise ValueError("Incomplete or unexpected teacher shadow contract")
    path = run/"cat_teacher_shadow.npz"
    if meta["packet_file"] != path.name or hashlib.sha256(path.read_bytes()).hexdigest() != meta["packet_sha256"]:
        raise ValueError("Teacher packet checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    for key, value in data.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite saved field: {key}")
    t, b, features = data["cat_obs"].shape
    if t != meta["frames"] or features != 162 or b < 1:
        raise ValueError("Saved teacher dimensions mismatch")
    np.testing.assert_array_equal(data["cat_obs"][..., 52:64], data["last_action"])
    np.testing.assert_array_equal(data["cat_obs"][..., 64:76], data["previous_leg_targets"])
    names = meta["action_joint_names"]
    leg_ids = indices(names, meta["contract"]["action_joints"])
    applied = data["applied_targets"][..., leg_ids]
    for step in range(1, t):
        continuing = ~(data["terminated"][step-1] | data["truncated"][step-1])
        np.testing.assert_array_equal(data["history_ready"][step], continuing)
        np.testing.assert_allclose(data["previous_leg_targets"][step, continuing], applied[step-1, continuing], atol=1e-6)
        expected = (applied[step-1]-data["previous_leg_targets"][step-1])/meta["contract"]["action_scale"]
        np.testing.assert_allclose(data["last_action"][step, continuing], expected[continuing], atol=1e-6)
        np.testing.assert_array_equal(data["step"][step], np.where(continuing, data["step"][step-1]+1, 0))
    command = data["grail_actions"]
    limit = meta["wrapper_action_clip"]
    if limit is not None and limit > 0:
        command = np.clip(command, -limit, limit)
    np.testing.assert_allclose(command*data["action_scale"]+data["action_offset"], data["applied_targets"], atol=2e-6)
    weights = teacher_dir/"weights.npz"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != meta["teacher_sha256"]:
        raise ValueError("Different CAT teacher weights")
    teacher = CatTeacher(weights)
    bridge = CatObservationBridge(meta["contract"], meta["joint_names"])
    with torch.no_grad():
        obs = torch.from_numpy(data["cat_obs"].reshape(-1, 162))
        action = teacher(obs)
        labels = bridge.leg_targets(action, torch.from_numpy(data["previous_leg_targets"].reshape(-1, 12)))
    np.testing.assert_allclose(action.numpy().reshape(t, b, 12), data["teacher_action"], atol=1e-4, rtol=2e-5)
    np.testing.assert_allclose(labels.numpy().reshape(t, b, 12), data["teacher_leg_targets"], atol=6e-5, rtol=2e-5)
    obs_keys = [k for k in data if k.startswith("grail_obs__")]
    if not obs_keys or any(data[k].shape[:2] != (t, b) for k in obs_keys):
        raise ValueError("Missing/misaligned original GRAIL observations")
    return dict(passed=True, frames=t, samples=t*b,
        history_paired_samples=int(data["history_ready"].sum()),
        original_observation_keys=obs_keys, teacher_replay_max_error=float(np.max(
            np.abs(action.numpy().reshape(t, b, 12)-data["teacher_action"]))),
        all_sites_in_domain_samples=int(data["in_domain"].all(-1).sum()),
        kinematics_independently_verified=False, training_admitted=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit_teacher_shadow(args.run), indent=2))
