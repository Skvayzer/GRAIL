#!/usr/bin/env python3
"""Summarize verified pilot evidence; lower imitation loss is not navigation success."""
import argparse
import json
from pathlib import Path
import numpy as np
from cat_distill_model import sha


def full_horizon_success(meta, final_goal_distance):
    """An early exit or assisted rollout cannot establish student success."""
    return bool(meta.get("full_horizon") and not meta.get("dagger_collection")
        and meta.get("teacher_fraction", 0.) == 0
        and meta["simulated_s"] >= 5 and not meta["fell"]
        and meta["native_clearance_or_height_violation_steps"] == 0
        and np.isfinite(final_goal_distance) and final_goal_distance < .2)


def report(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=False)
    stages, episodes, validation = [], [], None
    for run in runs:
        manifest = json.loads((run/"dataset.json").read_text())
        if sha(run/"dataset.npz") != manifest["dataset_sha256"]:
            raise ValueError("Changed training dataset")
        with np.load(run/"dataset.npz", allow_pickle=False) as a:
            heldout = {k: a[k][a["validation"]].copy() for k in a.files}
        if validation is not None:
            for key in heldout:
                np.testing.assert_array_equal(heldout[key], validation[key])
        validation = heldout
        for path in sorted(run.glob("training_*/report.json")):
            result = json.loads(path.read_text())
            if not result["frozen_unchanged"] or not result["resume_next_update_bit_exact"] or result["robot_actuation"]:
                raise ValueError("Invalid training checkpoint evidence")
            stages.append(dict(path=str(path), sha256=sha(path), **result))
        for path in sorted(run.glob("**/*_isaac.json"))+sorted(run.glob("**/*_grail.json")):
            meta = json.loads(path.read_text())
            if not meta.get("grail_loaded") or meta.get("dagger_collection"):
                continue
            archive = path.parent/meta["archive"]
            if sha(archive) != meta["sha256"] or meta["robot_actuation"] or meta["optimizer_steps"]:
                raise ValueError("Invalid evaluation evidence")
            with np.load(archive, allow_pickle=False) as a:
                final_goal_distance = float(np.linalg.norm(a["qpos"][-1, :2]-[2., 0.]))
            episodes.append(dict(path=str(path), checkpoint_updates=meta["checkpoint_updates"],
                engine=meta["engine"], scene=meta["scene"], horizon_s=meta["simulated_s"],
                full_horizon=meta.get("full_horizon", False), fell=meta["fell"],
                min_sdf=meta["minimum_native_site_sdf_m"], violations=meta["native_clearance_or_height_violation_steps"],
                final_goal_distance=final_goal_distance, crossed_x_exit=meta["reached_exit"],
                success=full_horizon_success(meta, final_goal_distance)))
    stages.sort(key=lambda s: s["cumulative_updates"])
    summary = dict(stages=stages, evaluations=episodes, heldout_rows=len(validation["obs"]),
        unchanged_heldout_across_rounds=True, any_full_horizon_success=any(e["success"] for e in episodes),
        robot_actuation=False, ready_for_harder_curriculum=False)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for stage in stages:
        h = stage["history"]
        steps = [v["step"] for v in h]
        axes[0].semilogy(steps, [v["validation"]["leg_mse"] for v in h], label=Path(stage["path"]).parent.name)
        axes[1].semilogy(steps, [max(v["validation"]["retention_mse"], 1e-12) for v in h])
    axes[0].set(title="CAT leg imitation — same held-out starts", ylabel="MSE (rad²)", xlabel="Cumulative adapter updates")
    axes[1].set(title="GRAIL held-out action retention", ylabel="MSE (rad²); lower is better", xlabel="Cumulative adapter updates")
    for ax in axes:
        ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Real CAT → frozen GRAIL decoder training\nNo full-horizon navigation success yet; no real robot actuation")
    fig.savefig(output/"learning.png", dpi=150)
    plt.close(fig)
    print(json.dumps(dict(stages=len(stages), evaluations=len(episodes),
        cumulative_updates=stages[-1]["cumulative_updates"], full_horizon_success=summary["any_full_horizon_success"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    report(args.runs, args.output)
