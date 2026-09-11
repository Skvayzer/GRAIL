# M2 training and overnight operation

Current entry points: `m2_train.py`, `m2_evaluate.py`, `m2_results.py`,
`m2_job.py`. The user approved the environment review and, on 2026-09-12,
explicitly requested an overnight simulation training run. No real robot is
connected by these programs. The older preparation documents describe earlier
gates; this page supersedes their statements that no training launcher exists.

## What this first pilot trains

- Initialize the motor backbone from the pinned released GRAIL terrain model.
  Its encoder/decoder stay frozen, including the clone that executes actions.
- Train a new obstacle-aware actor/critic. The action is a 64D Gaussian
  pre-tanh latent, transformed by `0.1*tanh(z)` and added after quantization.
  GRAIL decodes it to **all 29 robot joints**, not only the legs.
- The privileged teacher observes a 2,221D reference/proprioception state,
  a pelvis/yaw-aligned 4×13×13×11 oracle volume, 104 collider-cover probes,
  and terrain-aware support-graph guidance. Terrain is unsigned and separate
  from signed, forbidden CAT clutter. Unknown geometry is not free space.
- Keep paired reference footholds, root/balance regularization, smoothness and
  joint limits. Relax upper-body reference tracking and add per-link clutter
  clearance, measured CAT-contact cost/failure, and root-to-goal progress.
- The reviewed reset changes only 14 arm joints. Root, waist and legs keep the
  normal GRAIL reset. **Future desired/reference arm motion is not replaced
  with the collision-free witness.** The witness is a geometric admission and
  initial-pose choice, not a solved avoidance trajectory fed to the decoder.

This is a **reference-conditioned stair/posture-avoidance pilot**, not a
goal-only navigator, manipulation adapter, deployable LiDAR policy, or complete
terrain benchmark. It uses one development layout (seed 0) and one separate
geometry-validation layout (seed 102). Both use the reviewed stair reference;
the validation layout is never optimized. Long optimization on this tiny set
can overfit. Broader randomized multi-terrain curricula and fresh test layouts
are required before a research performance/generalization claim.

The arm witness has checked nonlocal self-clearance at reset. This stage does
not yet add a FABRICS shield or a new full-body self-collision reward; the
existing simulator collision/contact model is not a hardware safety guarantee.

## Training contract

Defaults for the launcher: four parallel static-scene replicas, 32-step rollout,
four PPO epochs, minibatch 32, Adam **3e-5**, gradient norm cap 1, KL stop 0.02,
gamma 0.99, GAE lambda 0.95. The lower launch learning rate follows the initial
smoke test's excessive early KL stops at 3e-4; it is not a tuned optimum.

Collection stores the sampled pre-tanh action and its likelihood, verifies the
same latent is actually decoded, and captures final observations **before**
Isaac auto-reset. True terminations have zero bootstrap. Timeouts bootstrap
their valid final state, never a reset state. Each environment's history resets
independently. Invalid ongoing/timeout observations, nonfinite values or stale
behavior likelihoods abort the run rather than silently entering PPO.

The frozen evaluator enables TF32. The learner locally uses full float32 and
restores those original actor settings afterward. Otherwise GPU kernel changes
between collection and minibatch sizes can make a stored action's likelihood
appear stale after the mean head becomes nonzero. The strict consistency guard
is retained; a nonzero-head CUDA regression test covers this case.

Checkpoints contain learner weights, all Adam moments/step counters and the
dedicated learner RNG. Checkpoint load enforces backbone, model, observation,
curriculum and algorithm contracts. **Resume starts new simulator episodes**;
it does not restore PhysX state, contact history or exact environment RNG.
Existing checkpoint files and released artifacts are never overwritten.

## Commands

Run from `~/robotics/GRAIL-CAT`, using its own `.venv`:

```bash
cd ~/robotics/GRAIL-CAT

# Verify reviewed scene evidence; no simulator or network.
.venv/bin/python research/m2_curriculum.py verify

# Prepare only (the default); does not start Isaac, W&B or updates.
.venv/bin/python research/m2_train.py

# Real collection and gradient diagnostics, but zero optimizer updates.
.venv/bin/python research/m2_train.py --mode collect --iterations 16 \
  --num-envs 4 --wandb-mode online --execute --accept-isaac-eula

# Bounded training pilot. --execute and optimizer approval are separate.
.venv/bin/python research/m2_train.py --mode train --approve-optimizer \
  --iterations 500 --num-envs 4 --checkpoint-every 100 \
  --wandb-mode online --execute --accept-isaac-eula --timeout 3600

# Inspect / stop the owned overnight job, including after reconnecting by SSH.
.venv/bin/python research/m2_job.py status
.venv/bin/python research/m2_job.py stop
```

Every launch prints a unique run directory. To resume, add
`--resume /absolute/path/to/learner_000500.pt` to the training command and keep
the same horizon/epochs/minibatch/learning-rate/curriculum contract. A resume
gets a new directory; the source checkpoint stays intact.

For deterministic evaluation, without gradients or stochastic exploration:

```bash
.venv/bin/python research/m2_evaluate.py \
  --checkpoint /absolute/path/to/learner_final.pt \
  --execute --wandb-mode online --accept-isaac-eula

.venv/bin/python research/m2_results.py /absolute/path/to/completed/run
```

The suite runs the validation clutter scene and stair/curb/slope locomotion
controls sequentially. Sitting is explicitly excluded, **not passed**: chair
seating contacts do not satisfy the standing/stepping support-graph assumptions
(the attempted control had only 0.5685 supported-anchor fraction). Supporting
that mode requires a separate task/support contract, not a looser validity mask.

Controls leave the original arm reset untouched and
place the real CAT fixture at (20,20,0), outside the reference workspace but
inside the exact-mesh query radius. Distances are **not** forced to free space.
The learned adapter remains active. These use the same M2 task overlay, so
compare kinematics/contact/termination outcomes, not reward numbers against the
historical unmodified GRAIL reward configuration. The curb control uses a 0.32m
geometric graph height bound because its measured riser is 0.2921m; stairs and
slope retain 0.20m. These are explicit reference-control heuristics, not changed
collision geometry or certified robot step capabilities. A timeout is not automatically
a successful avoidance trial; failure causes are logged separately.

`pipeline_complete` means all evaluation programs finished with unchanged
weights, not that the trained controller succeeded in every environment.

## Unattended overnight run

`m2_job.py start` requires the path to a completed four-scene locomotion setup suite and
explicit execution/update approval. It starts a detached, lower-priority job,
with no sudo or changes to system services, login policy or other projects.
Before detaching it checks that logind does not kill all user processes on
logout. PID/start-time/boot-ID checks protect its stop operation against PID
reuse. Suspending/rebooting the PC still interrupts training.

```bash
.venv/bin/python research/m2_job.py start \
  --preflight-suite /absolute/path/to/passing/evaluation_suite/suite.json \
  --iterations 8000 --hours 8 --execute --approve-optimizer
```

The overnight run starts a **fresh** learner, not the short setup checkpoint.
It uses only the development split, keeps a checkpoint every 100 iterations
(a few minutes on this PC), and has an eight-hour wall-clock limit. It stops
on launcher errors, more than ten minutes without progress, or less than
20 GiB free disk. There are no automatic retries. If enough time remains after
training, it runs the deterministic evaluation suite on the final checkpoint.
All work is desktop Isaac simulation; nothing arms or publishes to the robot.

W&B account: `skvayzer`, project: `grail-cat`.
`https://wandb.ai/skvayzer/grail-cat`
Logged values include PPO losses/entropy/KL/gradient norms, actual Adam update
counts, rewards, CAT clearance/contact, foot error, and each termination cause.
No credentials or whole source tree are uploaded by this launcher.

Run-local files:

- `training_plan.json`, `run.json`, `process.log`: configuration and execution.
- `metrics.jsonl`: durable per-iteration metrics while the run is active.
- `learner_*.pt`: periodic/final learner checkpoints, never GRAIL replacements.
- `training_result.json`: completion, hashes and checkpoint roundtrip checks.
- `avoidance_task.npz`: detailed **last-horizon** signals, not the entire run.
- Supervisor `status.json`: heartbeat, latest checkpoint and stop reason.

`m2_results.py` independently checks saved files, sample/update counters, actual
checkpoint contents, optimizer step counters and contract/checkpoint hashes.
It does not certify learned avoidance from a successful process exit.

## Reproduce the reviewed scene inputs on another PC

Code/configuration are committed; generated evidence stays in ignored run and
artifact directories. `m2_curriculum.py pack --output NEW_FILE.tar.gz` exports
only the 22 required scene/witness/reference files (about 8 MB compressed),
without logs, credentials or released neural-network weights. The existing
bundle on this PC is `research/artifacts/m2-reviewed-curriculum-v1.tar.gz`:

`5c510896d034b7c950dbb5915ef59da68be0c4bec66a79d4a874c4ed3059da08`

Copy this bundle to the new repository, verify its SHA256, extract its
`research/runs/` files without overwriting existing files, and run `verify`.
Archived absolute paths remain provenance: the loader safely resolves their
`research/runs/` suffix against the new checkout, keeping original file hashes.
The backbone and paired reference assets are fetched separately using the
committed `data_manifest.json`; follow `README.md` and retain upstream licenses.

Source-generation recipes and all rejected/accepted geometry are documented in
`ENVIRONMENT_REVIEW.md`, `CAT_SCENES.md` and `PROGRESS.md`. A different generated
scene must pass the geometry/role/passage/witness checks before being admitted;
do not edit a checksum merely to bypass a failed check.
