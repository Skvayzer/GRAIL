#!/usr/bin/env python3
"""CPU replay of saved GRAIL inputs and CAT leg-loss gradient through its decoder.

No optimizer, changed checkpoint, physics or hardware. This verifies the paired
data can drive whole-body distillation; it does NOT perform that training.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from cat_teacher_results import audit_teacher_shadow
from gear_sonic.research.cat_bridge import teacher_leg_loss
from gear_sonic.research.observation_shadow import state_hash


def load_policy_config(run, meta, config_run=None):
    """Allow explicit config recovery from another hash-identical actor run.

    The first live packet predates resolved-config export. Do not rewrite its
    evidence: record the config source separately and still replay its OWN
    checkpoint, observations and actions with strict parameter loading.
    """
    source = config_run or run
    source_meta = json.loads((source/"cat_teacher_shadow.json").read_text())
    for key in ("base_backbone_sha256", "action_joint_names", "wrapper_action_clip"):
        if source_meta[key] != meta[key]:
            raise ValueError(f"Policy config source differs: {key}")
    if not source_meta["complete"] or not source_meta["base_backbone_unchanged"]:
        raise ValueError("Policy config source did not complete with frozen actor")
    path = source/source_meta["policy_config_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source_meta["policy_config_sha256"]:
        raise ValueError("Resolved policy config changed")
    return json.loads(path.read_text()), str(path), digest


def main():
    from gear_sonic.trl.utils.common import custom_instantiate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--policy-config-run", type=Path,
                        help="Explicit recovery for older packets without resolved config; actor hash must match")
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit_teacher_shadow(args.run)
    meta = json.loads((args.run/"cat_teacher_shadow.json").read_text())
    cfg, config_path, config_hash = load_policy_config(args.run, meta, args.policy_config_run)
    env_cfg, algo_cfg = (OmegaConf.create(cfg[k]) for k in ("env_config", "algo_config"))
    policy = custom_instantiate(algo_cfg.actor, env_config=env_cfg, algo_config=algo_cfg,
                                module_dim_dict=algo_cfg.get("module_dim", {}), backbone_kwargs={}, _resolve=False)
    # Our previously hash-verified downloaded GRAIL checkpoint, copied per run.
    weights = torch.load(args.run/"checkpoint/last.pt", map_location="cpu", weights_only=False)
    state = weights.get("actor_model_state_dict", weights.get("policy_state_dict"))
    if state is None:
        raise ValueError("No actor in source checkpoint")
    policy.load_state_dict(state, strict=True)
    del weights, state
    policy.eval().requires_grad_(False)
    if state_hash(policy.actor_module) != meta["base_backbone_sha256"]:
        raise ValueError("CPU replay actor differs from live GRAIL")
    with np.load(args.run/"cat_teacher_shadow.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    # Only bounded, in-field, history-paired samples. Stair labels remain
    # unadmitted: this is an arithmetic/gradient check, not a training dataset.
    eligible = arrays["history_ready"] & arrays["in_domain"].all(-1)
    rows = np.argwhere(eligible)[::max(1, int(eligible.sum())//8)][:8]
    if not len(rows):
        raise ValueError("No in-domain, history-paired samples for replay")
    errors, gradients, losses = [], [], []
    for t, b in rows:
        obs = {k.removeprefix("grail_obs__"): torch.from_numpy(v[t, b:b+1]).clone()
               for k, v in arrays.items() if k.startswith("grail_obs__")}
        residual = torch.zeros(1, policy.actor_module.token_total_dim, requires_grad=True)
        result = policy.forward(obs, latent_residual=residual, latent_residual_mode="post_quantization")[:, -1]
        expected = torch.from_numpy(arrays["grail_actions"][t, b:b+1])
        error = float((result.detach()-expected).abs().max())
        # Cross-device FP32/TF32 replay is not bitwise; report actual error.
        torch.testing.assert_close(result.detach(), expected, atol=2e-3, rtol=2e-3)
        command = result
        limit = meta["wrapper_action_clip"]
        if limit is not None and limit > 0:
            command = command.clamp(-limit, limit)
        targets = command*torch.from_numpy(arrays["action_scale"][t, b:b+1]) + torch.from_numpy(arrays["action_offset"][t, b:b+1])
        loss = teacher_leg_loss(targets, meta["action_joint_names"], torch.from_numpy(arrays["teacher_leg_targets"][t, b:b+1]))
        gradient, = torch.autograd.grad(loss, residual)
        if not torch.isfinite(gradient).all() or gradient.norm() == 0:
            raise ValueError("CAT leg loss has no finite nonzero gradient through GRAIL decoder")
        errors.append(error)
        losses.append(float(loss.detach()))
        gradients.append(float(gradient.norm()))
    if (state_hash(policy.actor_module) != meta["base_backbone_sha256"]
            or any(p.grad is not None for p in policy.parameters())):
        raise ValueError("Frozen GRAIL checkpoint changed/received gradients")
    report = dict(schema="grail-cat-paired-gradient-replay-v1", passed=True,
        samples=rows.tolist(), maximum_action_replay_error=max(errors), leg_mse_rad2=losses,
        latent_gradient_norms=gradients, backbone_unchanged=True, optimizer_steps=0,
        physics_started=False, robot_actuation=False, training_admitted=False,
        policy_config_file=config_path, policy_config_sha256=config_hash,
        source_packet_sha256=meta["packet_sha256"])
    with (args.run/"cat_teacher_gradient.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
