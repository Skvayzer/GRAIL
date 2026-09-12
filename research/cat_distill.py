#!/usr/bin/env python3
"""Bounded flat CAT teaching frozen GRAIL decoder: prepare/train/evaluate.

First supervised motor-token pilot, not PPO, not full random-scene curriculum.
All 29 targets come from GRAIL's decoder. No ROS/SDK/robot connections.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from cat_direct_native import act, make_player, teacher
from cat_distill_model import (DistillStudent, ProprioHistory, load_grail,
                              objective, sha, state_hash, flat_field_admission)


def reset_player(player, constants, seed):
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(seed)
    constants.DEFAULT_QPOS = constants.DEFAULT_QPOS.copy()
    constants.DEFAULT_QPOS[1] = rng.uniform(-.12, .12)
    constants.DEFAULT_QPOS[3:7] = Rotation.from_euler("z", rng.uniform(-.08, .08)).as_quat(scalar_first=True)
    np.random.seed(seed)
    return player.reset()


def prepare(args):
    import mujoco
    from gear_sonic.research.cat_bridge import SITES
    actor, contract = load_grail(args.context)
    expert = teacher()
    records, episodes = [], []
    for seed in range(8):
        player, constants, _ = make_player("side1")
        names = [player.mj_model.actuator(i).name for i in range(29)]
        history = ProprioHistory(contract, names)
        state = reset_player(player, constants, seed)
        rows, outside, minimum, ground_extensions = [], 0, float("inf"), 0
        for step in range(250):
            obs = history.sample(player.mj_data.qpos.copy(), player.mj_data.qvel.copy(), state.info["motor_targets"].copy())
            cat = state.obs["state"].copy().astype(np.float32)
            action = act(expert, state)
            target = np.array(contract["offset"], np.float32)
            leg_targets = np.clip(state.info["motor_targets"][player.action_joint_ids]+.5*action,
                player._soft_lowers[player.action_joint_ids], player._soft_uppers[player.action_joint_ids])
            leg_ids = [contract["joints"].index(names[i]) for i in player.action_joint_ids]
            target[leg_ids] = leg_targets
            sites = np.stack([player.mj_data.site(n).xpos.copy() for n in SITES])
            index = (sites-player.pf_origin)/player.dx
            valid = bool(((index >= 0) & (index <= np.array(player.sdf.shape[:3])-1)).all())
            if args.support_boundary:
                pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in player.mj_data.contact}
                contacts = [frozenset((int(foot), int(player._floor_geom_id))) in pairs for foot in player._feet_geom_id]
                valid, extensions = flat_field_admission(sites, player.pf_origin, player.sdf.shape, player.dx, contacts)
                ground_extensions += extensions
            state = player.step(state, action)
            clearance = min(float(np.min(state.info[k+"df"])) for k in ("head", "feet", "hands"))
            minimum = min(minimum, clearance)
            if valid:
                rows.append(dict(obs=obs, cat=cat, base=np.zeros(64, np.float32), mode=np.float32(1),
                    target=target, episode=np.int64(seed), validation=np.bool_(seed >= 6)))
            else:
                outside += 1
            if player.mj_data.site("head").xpos[2] < .7 or player.mj_data.qpos[0] >= 1.9:
                break
        accepted = bool(minimum > -.04 and player.mj_data.site("head").xpos[2] > .7)
        if accepted:
            records.extend(rows)
        episodes.append(dict(seed=seed, accepted=accepted, retained=len(rows) if accepted else 0,
            outside=outside, verified_ground_boundary_queries=ground_extensions,
            min_sdf=minimum, steps=step+1, final_root=player.mj_data.qpos[:3].tolist()))
        print("CAT_EXPERT", json.dumps(episodes[-1]), flush=True)
    # Existing recorded GRAIL data is used ONLY for GRAIL action retention.
    # Its CAT labels were unadmitted on stairs and are deliberately not read.
    meta = json.loads((args.context/"cat_teacher_shadow.json").read_text())
    with np.load(args.context/meta["packet_file"], allow_pickle=False) as data:
        for t in range(len(data["grail_actions"])):
            obs = {k.removeprefix("grail_obs__"): torch.from_numpy(data[k][t].copy())
                   for k in data.files if k.startswith("grail_obs__")}
            with torch.no_grad():
                raw = actor(obs)[:, -1]
                base = actor.actor_module._last_full_latent_flat[:, -1].clone()
            expected = torch.from_numpy(data["grail_actions"][t])
            torch.testing.assert_close(raw, expected, atol=2e-3, rtol=2e-3)
            target = raw.clamp(-contract["clip"], contract["clip"])*torch.tensor(contract["scale"])+torch.tensor(contract["offset"])
            records.append(dict(obs=obs["actor_obs"][0, 0].numpy(), cat=np.zeros(162, np.float32),
                base=base[0].numpy(), mode=np.float32(0), target=target[0].numpy(),
                episode=np.int64(100), validation=np.bool_(t >= 48)))
    arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
    if not ((arrays["mode"] == 1) & arrays["validation"]).any():
        raise ValueError("No accepted held-out expert episode")
    np.savez_compressed(args.run/"dataset.npz", **arrays)
    manifest = dict(schema="cat-grail-flat-distill-v1", contract=contract, episodes=episodes,
        dataset_sha256=sha(args.run/"dataset.npz"), native_mujoco_version=mujoco.__version__,
        scene="side1", num_expert_episodes=8, split="seeds 0..5 train, 6..7 validation",
        retention="original GRAIL actions only; no CAT labels from stair diagnostic packet",
        upper_supervision="explicit GRAIL nominal posture prior, not CAT arm labels",
        flat_support_boundary=bool(args.support_boundary),
        robot_actuation=False, full_curriculum=False)
    (args.run/"dataset.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("DATASET_READY", len(records), flush=True)


def load(args):
    manifest = json.loads((args.run/"dataset.json").read_text())
    if sha(args.run/"dataset.npz") != manifest["dataset_sha256"]:
        raise ValueError("Dataset changed")
    actor, contract = load_grail(Path(manifest["contract"]["context"]))
    if contract != manifest["contract"]:
        raise ValueError("GRAIL contract changed")
    model = DistillStudent(actor, contract, teacher())
    del actor
    with np.load(args.run/"dataset.npz", allow_pickle=False) as a:
        data = {k: torch.from_numpy(a[k].copy()) for k in a.files}
    if any(not torch.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite dataset")
    obs = data["obs"][~data["validation"]]
    model.mean.copy_(obs.mean(0)); model.std.copy_(obs.std(0).clamp_min(.05))
    return model, data, manifest


def state(model, optimizer, generator, step, manifest):
    return dict(schema="cat-grail-flat-checkpoint-v1", adapter=model.adapter.state_dict(),
        mean=model.mean, std=model.std, optimizer=optimizer.state_dict(), rng=generator.get_state(),
        step=step, dataset_sha256=manifest["dataset_sha256"], contract=model.contract,
        frozen_hashes=model.frozen_hash(), robot_actuation=False)


def restore(model, optimizer, generator, checkpoint, manifest):
    if (checkpoint["dataset_sha256"] != manifest["dataset_sha256"]
            or checkpoint["contract"] != model.contract
            or tuple(checkpoint["frozen_hashes"]) != model.frozen_hash()):
        raise ValueError("Checkpoint source/contract/frozen model mismatch")
    model.adapter.load_state_dict(checkpoint["adapter"], strict=True)
    model.mean.copy_(checkpoint["mean"]); model.std.copy_(checkpoint["std"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    generator.set_state(checkpoint["rng"])
    return checkpoint["step"]


def minibatch(data, generator, size=32):
    ids = []
    for mode in (1, 0):
        pool = torch.where((data["mode"] == mode) & ~data["validation"])[0]
        ids.append(pool[torch.randint(len(pool), (size//2,), generator=generator)])
    ids = torch.cat(ids)
    return {k: data[k][ids] for k in ("obs", "cat", "base", "mode", "target")}


@torch.no_grad()
def measure(model, data):
    reports = {}
    for name, mask in (("train", ~data["validation"]), ("validation", data["validation"])):
        total, metrics = objective(model, {k: data[k][mask] for k in ("obs", "cat", "base", "mode", "target")})
        reports[name] = dict(loss=float(total), **metrics)
    return reports


def train(args):
    model, data, manifest = load(args)
    frozen = model.frozen_hash()
    opt = torch.optim.Adam(model.trainable(), lr=3e-4)
    rng = torch.Generator().manual_seed(1234)
    start = 0
    if args.resume:
        start = restore(model, opt, rng, torch.load(args.resume, weights_only=True), manifest)
    if args.warm_start:
        saved = torch.load(args.warm_start, weights_only=True)
        original_dataset = saved["dataset_sha256"]
        if original_dataset not in manifest.get("parent_dataset_hashes", []):
            raise ValueError("Warm start must be an explicitly recorded parent dataset")
        # Only this explicit aggregation path may change the dataset identity.
        # Architecture, decoder, action contract, optimizer and RNG remain checked.
        saved["dataset_sha256"] = manifest["dataset_sha256"]
        start = restore(model, opt, rng, saved, manifest)
    initial = measure(model, data)
    output = args.run/f"training_{start}_{start+args.updates}"
    output.mkdir(exist_ok=False)
    torch.save(state(model, opt, rng, start, manifest), output/"initial.pt")
    history, begun = [], time.monotonic()
    for step in range(start+1, start+args.updates+1):
        opt.zero_grad(set_to_none=True)
        loss, metrics = objective(model, minibatch(data, rng))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.trainable(), 1., error_if_nonfinite=True)
        opt.step()
        if step == start+1 or step % 50 == 0 or step == start+args.updates:
            entry = dict(step=step, loss=float(loss.detach()), grad_norm=float(norm), **measure(model, data))
            history.append(entry); print("DISTILL_UPDATE", json.dumps(entry), flush=True)
            torch.save(state(model, opt, rng, step, manifest), output/f"step_{step}.pt")
    if model.frozen_hash() != frozen or any(p.grad is not None for p in model.decoder.parameters()):
        raise ValueError("Frozen decoder/teacher changed")
    final = measure(model, data)
    # Test the next update twice, once in-memory and once restored from disk.
    # Restore again afterwards: diagnostic updates are not promoted as training.
    checkpoint = torch.load(output/f"step_{step}.pt", weights_only=True)
    predictions = []
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        next_loss, _ = objective(model, minibatch(data, rng))
        next_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1., error_if_nonfinite=True)
        opt.step()
        predictions.append(torch.cat([p.detach().flatten().clone() for p in model.trainable()]))
        restore(model, opt, rng, checkpoint, manifest)
    torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
    report = dict(initial=initial, final=measure(model, data), updates=args.updates, cumulative_updates=step,
        elapsed_s=time.monotonic()-begun, history=history, frozen_unchanged=True,
        robot_actuation=False, closed_loop_validated=False, physics_started=False,
        trainable_parameters=sum(p.numel() for p in model.trainable()),
        warm_start_sha256=sha(args.warm_start) if args.warm_start else None,
        resume_next_update_bit_exact=True)
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("DISTILL_TRAINED", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path, help="Continue from verified parent dataset after DAgger aggregation")
    parser.add_argument("--support-boundary", action="store_true", help="Admit only native contacting feet up to 2cm below known flat floor")
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error("Use resume OR explicit parent-dataset warm start")
    if not 1 <= args.updates <= 2000:
        parser.error("Bounded pilot requires 1..2000 updates")
    torch.set_num_threads(4); torch.manual_seed(42)
    if args.action == "prepare":
        if args.context is None:
            parser.error("--context is required for preparation")
        args.run.mkdir(parents=True, exist_ok=False)
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
