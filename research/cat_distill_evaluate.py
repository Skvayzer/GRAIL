#!/usr/bin/env python3
"""Closed-loop evaluation of all 29 GRAIL-decoded targets, simulation only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from cat_direct_native import make_player, save_episode, snapshot, act, teacher
from cat_distill import load, restore, reset_player
from cat_distill_model import ProprioHistory, sha, flat_field_admission


class StudentController:
    def __init__(self, dataset, checkpoint, native_names):
        args = argparse.Namespace(run=Path(dataset))
        self.model, _, manifest = load(args)
        opt = torch.optim.Adam(self.model.trainable(), lr=3e-4)
        rng = torch.Generator()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.step = restore(self.model, opt, rng, saved, manifest)
        self.model.eval()
        self.checkpoint_hash = sha(checkpoint)
        self.history = ProprioHistory(self.model.contract, native_names)
        self.inverse = [self.model.contract["joints"].index(n) for n in native_names]
        self.clip_count = self.target_count = 0
        self.latest = None

    @torch.no_grad()
    def targets(self, player, state):
        obs = self.history.sample(player.mj_data.qpos.copy(), player.mj_data.qvel.copy(), state.info["motor_targets"].copy())
        self.latest = dict(obs=obs.copy(), cat=state.obs["state"].copy().astype(np.float32))
        target = self.model(torch.from_numpy(obs)[None],
            torch.as_tensor(state.obs["state"], dtype=torch.float32)[None],
            torch.zeros(1, 64), torch.ones(1))[0].numpy()[self.inverse]
        clipped = np.clip(target, player._soft_lowers, player._soft_uppers)
        self.clip_count += int((target != clipped).sum()); self.target_count += 29
        if not np.isfinite(clipped).all():
            raise ValueError("Nonfinite student targets")
        return clipped

    def label(self, player, state, expert):
        from gear_sonic.research.cat_bridge import SITES
        action = act(expert, state)
        target = np.array(self.model.contract["offset"], np.float32)
        labels = np.clip(state.info["motor_targets"][player.action_joint_ids]+.5*action,
                         player._soft_lowers[player.action_joint_ids], player._soft_uppers[player.action_joint_ids])
        target[self.model.legs] = labels
        sites = np.stack([player.mj_data.site(n).xpos.copy() for n in SITES])
        pairs = {frozenset((int(c.geom1), int(c.geom2))) for c in player.mj_data.contact}
        contacts = [frozenset((int(f), int(player._floor_geom_id))) in pairs for f in player._feet_geom_id]
        valid, _ = flat_field_admission(sites, player.pf_origin, player.sdf.shape, player.dx, contacts)
        return dict(**self.latest, target=target, base=np.zeros(64, np.float32), mode=np.float32(1)), valid


def main():
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[6, 7])
    parser.add_argument("--collect-dagger", action="store_true")
    parser.add_argument("--teacher-fraction", type=float, default=0.)
    args = parser.parse_args()
    if not 0 <= args.teacher_fraction <= 1 or (args.teacher_fraction and not args.collect_dagger):
        parser.error("Teacher assistance is only for explicitly labelled DAgger collection")
    if args.collect_dagger and any(s >= 6 for s in args.seeds):
        parser.error("Held-out seeds may not enter DAgger training")
    if len(args.seeds) > 8 or any(s < 0 or s > 1000 for s in args.seeds):
        parser.error("Use at most eight bounded seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    results, labels = [], []
    for seed in args.seeds:
        player, constants, _ = make_player("side1")
        names = [player.mj_model.actuator(i).name for i in range(29)]
        student = StudentController(args.dataset, args.checkpoint, names)
        state = reset_player(player, constants, seed)
        expert = teacher()
        rng = np.random.default_rng(seed+1000)
        rows, start = [], time.monotonic()
        for step in range(250):
            target = student.targets(player, state)
            if args.collect_dagger:
                label, valid = student.label(player, state, expert)
                if valid:
                    labels.append(dict(**label, episode=np.int64(200+seed), validation=np.bool_(False)))
                if rng.random() < args.teacher_fraction:
                    # Teacher assistance is never counted as student success.
                    target[player.action_joint_ids] = label["target"][student.model.legs]
            action = (target[player.action_joint_ids]-state.info["motor_targets"][player.action_joint_ids])/.5
            state.info["motor_targets"] = target.copy()
            for _ in range(10):
                player.mj_data.ctrl[:] = constants.KPs*(target-player.mj_data.qpos[7:])-constants.KDs*player.mj_data.qvel[6:]
                mujoco.mj_step(player.mj_model, player.mj_data)
            state = player.observe_after_physics(state, action)
            rows.append(snapshot(player, state, action))
            if rows[-1]["head"][2] < .7 or player.mj_data.qpos[0] >= 1.9:
                break
        result = save_episode(args.output, f"side1_seed{seed}", "grail", rows, player, time.monotonic()-start,
            dict(policy="GRAIL frozen 29-joint decoder + CAT-trained motor-token adapter", grail_loaded=True))
        result.update(policy="GRAIL frozen 29-joint decoder + CAT-trained motor-token adapter", grail_loaded=True,
            checkpoint_sha256=student.checkpoint_hash, checkpoint_updates=student.step,
            applied_joint_count=29, soft_clipped_target_fraction=student.clip_count/student.target_count,
            native_mujoco_version=mujoco.__version__, scope="native CAT dynamics, not full GRAIL task environment")
        result.update(teacher_fraction=args.teacher_fraction, dagger_collection=args.collect_dagger)
        (args.output/f"side1_seed{seed}_grail.json").write_text(json.dumps(result, indent=2)+"\n")
        results.append(result)
    (args.output/"evaluation.json").write_text(json.dumps(results, indent=2)+"\n")
    if args.collect_dagger:
        if not labels:
            raise ValueError("No valid student-state teacher labels")
        np.savez_compressed(args.output/"dagger.npz", **{k: np.stack([r[k] for r in labels]) for k in labels[0]})
        (args.output/"dagger.json").write_text(json.dumps(dict(sha256=sha(args.output/"dagger.npz"),
            source_dataset_sha256=sha(args.dataset/"dataset.npz"), checkpoint_sha256=sha(args.checkpoint),
            training_seeds=args.seeds, frames=len(labels), teacher_fraction=args.teacher_fraction,
            robot_actuation=False), indent=2)+"\n")


if __name__ == "__main__":
    main()
