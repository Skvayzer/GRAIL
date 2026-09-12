#!/usr/bin/env python3
"""CPU export/parity check using CAT's own .venv; no simulation or updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=ROOT/"artifacts/cat_release")
    p.add_argument("--output", type=Path, default=ROOT/"artifacts/cat_teacher_v1")
    args = p.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    from brax.training.agents.ppo import checkpoint
    sys.path.insert(0, str(ROOT.parent/"imports/SONIC"))
    from gear_sonic.research.cat_teacher import CatTeacher, JOINTS
    manifest = json.loads((args.release/"manifest.json").read_text())
    for item in manifest["files"]:
        if hashlib.sha256((args.release/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Released checkpoint checksum changed")
    source = (args.release/"logs_v1/generalist_v1/checkpoints/005033164800").resolve()
    config = json.loads((source/"ppo_network_config.json").read_text())
    if config["normalize_observations"] or config["action_size"] != 12:
        raise ValueError("Unexpected CAT observation/action contract")
    params = checkpoint.load(source)
    actor = params[1]["params"]
    arrays = {f"actor_{i}_{key}": np.asarray(actor[f"hidden_{i}"][key])
              for i in range(5) for key in ("kernel", "bias")}
    args.output.mkdir(parents=True, exist_ok=False)
    weights = args.output/"weights.npz"
    with weights.open("xb") as stream:
        np.savez(stream, **arrays)
    teacher = CatTeacher(weights)
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    observations = np.concatenate((np.zeros((1, 162), np.float32),
                                  rng.uniform(-1, 1, (63, 162)).astype(np.float32)))
    native = checkpoint.load_policy(source, deterministic=True)
    expected, _ = native({"state": jnp.asarray(observations)}, jax.random.PRNGKey(0))
    expected = np.asarray(expected)
    with torch.no_grad():
        actual = teacher(torch.from_numpy(observations)).numpy()
    error = float(np.abs(expected-actual).max())
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-5)
    report = dict(schema="grail-cat-released-teacher-v1", checkpoint=str(source),
        revision=manifest["revision"], checkpoint_manifest_sha256=hashlib.sha256((args.release/"manifest.json").read_bytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        input="original CAT Bx162 observations; not GRAIL observations", action_joint_order=JOINTS,
        action="normalized 12 leg targets; native conversion is default_pos + 0.5 * action",
        native_framework="Brax/JAX", target_framework="PyTorch", device="cpu", samples=len(observations),
        maximum_action_error=error, arithmetic_parity_passed=True,
        robot_rollout_validated=False, grail_observation_bridge_implemented=False,
        distilled_into_grail=False, optimizer_steps=0, frozen=True)
    (args.output/"verification.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
