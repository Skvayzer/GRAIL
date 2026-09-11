"""Bounded simulator-only M2 learner, called after Isaac and frozen actor load."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from .avoidance_task import AvoidanceTask
from .cat_audit import CatAudit
from .learner_state import LearnerStateSampler
from .learning_checkpoint import save_checkpoint, load_checkpoint
from .observation_shadow import state_hash
from .residual_engine import UpdateConfig, ResidualCollector, PPOUpdater
from .residual_learning import ResidualActorCritic
from .residual_simulator import SimulatorCollector
from .runtime_observation import RuntimeObservation
from .scene_audit import run_live
from .training_admission import ChallengeAdmission, ArmReset, digest


def identical_tree(left, right):
    """Exact checkpoint state comparison, including Adam moments and steps."""
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(identical_tree(a, b) for a, b in zip(left, right))
    return left == right


def tracking_metrics(row):
    values = {k: v for k, v in row.items() if k != "update"}
    update = row.get("update")
    if update:
        values["ppo/kl_early_stop"] = int(update["kl_early_stop"])
        values["ppo/optimizer_steps_this_iteration"] = update["optimizer_steps"]
        for name in ("loss", "policy_loss", "value_loss", "pre_tanh_entropy", "approx_kl",
                     "clip_fraction", "gradient_norm"):
            samples = [batch[name] for batch in update["batches"] if name in batch]
            if samples:
                values["ppo/"+name] = sum(samples)/len(samples)
    return values


class EvaluationCollector:
    """Deterministic evaluation; never produces PPO likelihood/target data."""
    def __init__(self, model, horizon):
        self.model, self.horizon, self.failed = model, horizon, False
        self.rows, self.pending = [], None

    @torch.no_grad()
    def sample(self, packet, state):
        if self.pending is not None or not packet["valid"].all() or len(self.rows) >= self.horizon:
            raise ValueError("Invalid deterministic evaluation sample")
        self.pending = self.model.deterministic(packet, state)[0].detach().clone()
        return self.pending.clone()

    def outcome(self, residual, reward, terminated, truncated, final_packet, final_state):
        if self.pending is None or not torch.equal(residual, self.pending) or not torch.isfinite(reward).all():
            raise ValueError("Evaluation action/outcome mismatch")
        self.rows.append(reward.detach().clone())
        self.pending = None

    def finish(self):
        if self.failed or self.pending is not None or len(self.rows) != self.horizon:
            raise ValueError("Incomplete deterministic evaluation")
        return {"reward": torch.stack(self.rows).flatten()}


def run_training(wrapper, frozen_policy, plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if (plan.get("schema") != "grail-cat-training-plan-v1" or plan.get("simulation_only") is not True
            or plan.get("execute") is not True or plan.get("mode") not in ("collect", "train", "evaluate")
            or plan.get("environment_review_approved") is not True
            or not 1 <= plan["iterations"] <= 10000 or not 1 <= wrapper.env.num_envs <= 4):
        raise ValueError("Bounded, reviewed, explicitly executed simulator plan required")
    if plan["mode"] == "train" and plan.get("approve_optimizer") is not True:
        raise ValueError("Optimizer updates require a separate explicit approval flag")
    if plan["mode"] == "train" and plan["split"] != "development":
        raise ValueError("Validation scenes must never receive optimizer updates")
    run = plan_path.parent
    frozen_policy.eval().requires_grad_(False)
    base_hash = state_hash(frozen_policy)
    challenge = (ChallengeAdmission(plan["witness"], plan["witness_sha256"], plan["case"])
                 if plan.get("witness") else None)
    cat = CatAudit(wrapper, run/"cat_audit.json", challenge=challenge)
    run_live(wrapper, cat, run/"layout_audit.json", allow_replicated=True, challenge=challenge)
    task = AvoidanceTask(wrapper, cat, run)
    if hasattr(wrapper.env, "research_avoidance"):
        raise ValueError("Another avoidance task is already attached")
    wrapper.env.research_avoidance = task
    reset_context = ArmReset(wrapper, challenge) if challenge else nullcontext()
    report = dict(schema="grail-cat-training-result-v1", plan_sha256=digest(plan_path), mode=plan["mode"],
        complete=False, simulation_only=True, trained_avoidance_claim=False, base_sha256=base_hash,
        plan=plan, metrics=[], optimizer_steps=0, error=None)
    tracking = None
    try:
        with reset_context as reset_hook:
            obs = wrapper.reset_all()
            sampler = LearnerStateSampler(wrapper)
            oracle = RuntimeObservation(wrapper, cat, run, challenge=challenge)
            config = UpdateConfig(**plan["update"])
            learner = ResidualActorCritic(len(cat.probes), sampler.manifest()["state_dim"],
                latent_dim=frozen_policy.actor_module.token_total_dim, seed=plan["seed"]).to(wrapper.env.device).eval()
            generator = torch.Generator(device=wrapper.env.device).manual_seed(plan["seed"])
            optimizer = torch.optim.Adam(learner.parameters(), lr=config.learning_rate)
            contract = dict(backbone_sha256=base_hash, observation=dict(state=sampler.manifest(),
                packet=oracle.observer.spec.manifest(), probes=oracle.contract["probe_order"]),
                algorithm=config.manifest(), curriculum_sha256=plan["curriculum_sha256"],
                task=dict(profile="m2-reference-conditioned-posture", spec=asdict(task.spec)))
            start_updates = (load_checkpoint(plan["resume"], learner, optimizer, generator, contract)
                             if plan.get("resume") else 0)
            initial_learner_hash = state_hash(learner)
            updater = PPOUpdater(learner, optimizer, generator, config)
            updater.optimizer_steps = start_updates
            report.update(initial_optimizer_steps=start_updates, initial_learner_sha256=initial_learner_hash,
                          contract=contract, learner=learner.manifest(), reset_hook=bool(reset_hook))
            if plan["wandb_mode"] != "disabled":
                import wandb
                tracking = wandb.init(entity=plan["wandb_entity"], project=plan["wandb_project"],
                    name=run.name, dir=str(run), mode=plan["wandb_mode"], job_type=plan["mode"],
                    config={"plan": plan, "contract": contract}, save_code=False)
                report["wandb_url"] = tracking.url
            def new_collector():
                return (EvaluationCollector(learner, config.horizon) if plan["mode"] == "evaluate" else
                        ResidualCollector(learner, generator, wrapper.env.num_envs, config))
            collector = new_collector()
            started = time.monotonic()
            with SimulatorCollector(wrapper, frozen_policy, oracle, sampler, collector,
                                    allow_simulation_exploration=True) as runtime:
                for iteration in range(plan["iterations"]):
                    if iteration:
                        runtime.collector = new_collector()
                    for _ in range(config.horizon):
                        obs, _, _, _ = runtime.step(obs)
                    data = runtime.collector.finish()
                    # Evaluation uses deterministic residuals through the same
                    # runtime, while collection/train use sampled behavior.
                    update = updater.run(data, optimize=plan["mode"] == "train") if plan["mode"] != "evaluate" else None
                    row = dict(iteration=iteration+1, environment_steps=runtime.steps*wrapper.env.num_envs,
                        optimizer_steps=updater.optimizer_steps, reward_mean=float(data["reward"].mean()),
                        failures=runtime.terminations, timeouts=runtime.timeouts,
                        minimum_cat_gap_m=min(float(r["min_gap"].min()) for r in task.rows),
                        maximum_contact_n=max(float(r["contact_peak"].max()) for r in task.rows),
                        maximum_foot_error_m=max(float(r["foot_world_error"].max()) for r in task.rows),
                        goal_distance_mean=float(task.rows[-1]["goal_distance"].mean()),
                        elapsed_s=time.monotonic()-started)
                    if update:
                        row["update"] = update
                    report["metrics"].append(row)
                    with (run/"metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    if tracking:
                        tracking.log(tracking_metrics(row), step=iteration+1)
                    print("M2_ITERATION "+json.dumps(row, allow_nan=False), flush=True)
                    # Keep per-step memory bounded. Per-horizon summaries are
                    # streamed durably; final horizon retains detailed signals.
                    if iteration+1 < plan["iterations"]:
                        task.rows.clear()
                        task.outcomes.clear()
                    if (iteration+1) % plan["checkpoint_every"] == 0:
                        save_checkpoint(run/f"learner_{iteration+1:06d}.pt", learner, optimizer, generator,
                                        contract, updater.optimizer_steps)
            checkpoint = run/"learner_final.pt"
            checkpoint_hash = save_checkpoint(checkpoint, learner, optimizer, generator, contract, updater.optimizer_steps)
            # Exercise a real populated checkpoint on independent model/Adam/RNG
            # instances. Simulation state is intentionally not restored.
            clone = ResidualActorCritic(len(cat.probes), sampler.manifest()["state_dim"],
                latent_dim=frozen_policy.actor_module.token_total_dim).to(wrapper.env.device).eval()
            clone_opt = torch.optim.Adam(clone.parameters(), lr=config.learning_rate)
            clone_rng = torch.Generator(device=wrapper.env.device)
            loaded = load_checkpoint(checkpoint, clone, clone_opt, clone_rng, contract)
            if loaded != updater.optimizer_steps or state_hash(clone) != state_hash(learner) or not torch.equal(
                    clone_rng.get_state(), generator.get_state()) or not identical_tree(
                    clone_opt.state_dict(), optimizer.state_dict()):
                raise ValueError("Actual learner checkpoint roundtrip mismatch")
            report.update(complete=True, optimizer_steps=updater.optimizer_steps-start_updates,
                total_optimizer_steps=updater.optimizer_steps, learner_sha256=state_hash(learner),
                learner_changed=initial_learner_hash != state_hash(learner), checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_hash, checkpoint_roundtrip_verified=True,
                optimizer_moments_roundtrip_verified=True,
                simulator_steps=runtime.steps, failures=runtime.terminations, timeouts=runtime.timeouts,
                reset_calls=reset_hook.calls if reset_hook else 0,
                reset_environment_ids=reset_hook.rows if reset_hook else [])
            task.report["diagnostic_scope"] = "last horizon only; all horizons summarized in metrics.jsonl"
            task.save()
    except Exception as error:
        report["error"] = type(error).__name__+": "+str(error)
        raise
    finally:
        report["backbone_unchanged"] = state_hash(frozen_policy) == base_hash
        report["complete"] &= report["backbone_unchanged"]
        (run/"training_result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if tracking:
            tracking.summary.update({k: report[k] for k in ("complete", "backbone_unchanged", "optimizer_steps")})
            tracking.finish(exit_code=0 if report["complete"] else 1)
        del wrapper.env.research_avoidance
    if not report["complete"]:
        raise ValueError("Training setup did not complete")
    return report
