"""Bounded M0 PPO test: four envs recommended, two updates, eight steps/update.

The source checkpoint is immutable. This is a training-plumbing test, NOT an
improved controller or a deployable model. No simulator or torch import at module
load; the launcher shares the baseline's artifact and consent checks.
"""
import json

from artifacts import ROOT

UPDATES = 2
STEPS_PER_ENV = 8


def smoke_overrides(run):
    return {
        "checkpoint": str(run / "checkpoint/last.pt"),
        "experiment_dir": str(run / "training"),
        "exp_base": "research_m0", "experiment_name": "terrain_training_smoke",
        "project_name": "GRAIL_CAT_M0", "base_dir": str(run),
        "use_wandb": False, "multi_gpu": False, "global_rank": 0,
        "resume": True, "warm_resume": True, "auto_load_latest": False,
        "algo.config.world_size": 1, "algo.config.global_rank": 0,
        "algo.config.num_learning_iterations": UPDATES,
        "algo.config.num_learning_epochs": 1,
        "algo.config.num_mini_batches": 1,
        "algo.config.num_steps_per_env": STEPS_PER_ENV,
        "algo.trl.num_total_batches": UPDATES,
        "algo.trl.num_ppo_epochs": 1, "algo.trl.num_mini_batches": 1,
        "algo.trl.report_to": "none",
        "callbacks": {
            "model_save": {
                "_target_": "gear_sonic.trl.callbacks.model_save_callback.ModelSaveCallback",
                "save_dir": str(run / "training"),
                "save_frequency": UPDATES, "save_last_frequency": UPDATES,
                "max_disk_usage": None,
            },
        },
        # Strict loading is opt-in; the research test may not silently ignore
        # missing/unexpected actor keys when restoring the terrain baseline.
        "trainer.strict_checkpoint_load": True,
    }


def training_command(run, overrides):
    from omegaconf import OmegaConf
    config = OmegaConf.load(run / "checkpoint/config.yaml")
    for key, value in {**overrides, **smoke_overrides(run)}.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    config_dir = run / "training_config"
    config_dir.mkdir()
    OmegaConf.save(config, config_dir / "smoke.yaml")
    return [str(ROOT.parent / ".venv/bin/python"), "-m", "gear_sonic.train_agent_trl",
            "--config-path", str(config_dir), "--config-name", "smoke"]


def audit_training(run):
    import torch
    # run was created by the checksum-verified launcher; only its own generated
    # checkpoint is deserialized here, never an arbitrary user-supplied pickle.
    before = torch.load(run / "checkpoint/last.pt", map_location="cpu", weights_only=False)
    after = torch.load(run / "training/last.pt", map_location="cpu", weights_only=False)
    initial, final = before["policy_state_dict"], after["policy_state_dict"]
    keys_match = initial.keys() == final.keys()
    shapes_match = keys_match and all(initial[k].shape == final[k].shape for k in initial)
    finite = all(bool(torch.isfinite(v).all()) for v in final.values())
    critic_finite = all(bool(torch.isfinite(v).all()) for v in after["value_state_dict"].values())

    def finite_optimizer(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_optimizer(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return all(finite_optimizer(v) for v in value)
        return True

    optimizer_finite = finite_optimizer(after["optimizer_state_dict"])
    changed = [k for k in initial if k in final and initial[k].shape == final[k].shape
               and not torch.equal(initial[k], final[k])]
    log = (run / "process.log").read_text(errors="replace")
    report = {
        "simulation_only": True, "not_a_deployment_checkpoint": True,
        "updates": after["state"].global_step,
        "actor_keys_match": keys_match, "actor_shapes_match": shapes_match,
        "actor_finite": finite, "changed_actor_tensors": changed,
        "critic_finite": critic_finite, "optimizer_tensors_finite": optimizer_finite,
        "skipped_nan_gradient": "NaN in gradient!" in log,
    }
    report["passed"] = (report["updates"] == UPDATES and shapes_match and finite
                        and critic_finite and optimizer_finite
                        and bool(changed) and not report["skipped_nan_gradient"])
    (run / "training_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
