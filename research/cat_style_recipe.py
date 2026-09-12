#!/usr/bin/env python3
"""Copy the released CAT recipe and inventory its EXACT scene distribution.

No custom PPO implementation, simulator, policy update or actuation is launched.
This records what the whole-body integration must reproduce, not a claim that
37 different environments have already been connected to the Isaac trainer.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_CAT = ROOT.parent.parent/"Click-and-Traverse"


def recipe(config, cat_repo, num_envs):
    if num_envs not in (256, 512, 1024, 2048):
        raise ValueError("Choose a staged single-GPU batch benchmark: 256/512/1024/2048")
    cfg = copy.deepcopy(config)
    policy = cfg["policy_config"]
    original_envs, original_batch = policy["num_envs"], policy["batch_size"]
    batch = original_batch*num_envs
    if batch % original_envs:
        raise ValueError("Cannot preserve native rollout-to-environment ratio")
    policy.update(num_envs=num_envs, batch_size=batch//original_envs, max_devices_per_host=1)
    paths = cfg["env_config"]["pf_config"]["paths"]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Expected the distinct scene list from the release")
    inventory = []
    for name in paths:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path.parts[:3] != ("data", "assets", "RandObs"):
            raise ValueError("Unexpected CAT scene path")
        directory = Path(cat_repo)/path
        required = ("obs.npy", "sdf.npy", "bf.npy", "gf.npy")
        inventory.append(dict(scene=name, available=all((directory/f).is_file() for f in required)))
    return dict(schema="grail-cat-upstream-learning-recipe-v1", configuration=cfg,
        hardware_overrides=dict(original_num_envs=original_envs, num_envs=num_envs,
                                original_batch_size=original_batch, batch_size=policy["batch_size"],
                                original_max_devices=config["policy_config"]["max_devices_per_host"], max_devices=1),
        implementation="Reuse CAT PPO and specialist-to-generalist DAgger; no bespoke KL-abort trainer",
        stages=["released CAT teacher; GRAIL whole-body locomotion initialization",
                "flat-ground single-family specialists: lateral / low / overhead",
                "multi-seed mixed clutter specialists",
                "combine learned clutter skills with GRAIL slopes, curbs and stairs",
                "CAT-style DAgger specialist-to-generalist distillation, then PPO fine-tuning"],
        scene_count=len(paths), scene_inventory=inventory,
        training_started=False, grail_teacher_bridge_ready=False,
        note="Replicas increase sample throughput, not geometry diversity. Keep real obstacle failures as episode resets; do not label timeouts as successes.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cat-repo", type=Path, default=DEFAULT_CAT)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--output", type=Path, help="Write a new generated recipe; never replace existing files")
    args = p.parse_args()
    source = ROOT/"artifacts/cat_release/logs_v1/generalist_v1/checkpoints/config.json"
    result = recipe(json.loads(source.read_text()), args.cat_repo, args.num_envs)
    result["source_config_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(scene_count=result["scene_count"],
        missing_scenes=[r["scene"] for r in result["scene_inventory"] if not r["available"]],
        hardware_overrides=result["hardware_overrides"], training_started=False), indent=2))


if __name__ == "__main__":
    main()
