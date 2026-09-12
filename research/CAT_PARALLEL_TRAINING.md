# Generated CAT clutter and parallel whole-body learning

## Current state — 12 September 2026

The single-scene pilot is superseded by an implemented generated-bank GPU task
and staged learner. **Sustained training has not started yet.** The 2048-env
physics benchmark passed, but the integrated optimizer smoke was stopped when
another user's potentially live G1 deployment was discovered on this GPU.
An operator must confirm that it is not controlling hardware, or stop that
deployment, before GPU training resumes. No other user's process was modified.

The new launch preflight refuses unreviewed robot-deployment GPU processes.
During training it checks again at update boundaries and checkpoints/stops if a
new one appears. This is a conservative coordination guard, not hard real-time
GPU isolation. Reserve the GPU appropriately when operating physical robots.

## Actual generated distribution

`cat_parallel_bank.py` calls the unchanged pinned CAT random obstacle generator
and progressive potential-field generator. It does not repeat `side1`.

| Family | Unique training layouts | Unique held-out layouts |
|---|---:|---:|
| Lateral | 48 | 12 |
| Low obstacles | 36 | 9 |
| Overhead | 48 | 12 |
| Mixed | 48 | 12 |
| **Total** | **180** | **45** |

240 recipes were requested with disjoint seeds; CAT's native morphology erased
15 tiny low-obstacle recipes. These are recorded as rejected, not retried with
easier seeds and not counted as distinct clutter. All 225 admitted occupancy
hashes are unique across both splits. No held-out geometry enters training.

Difficulty varies across 0.2/0.4/0.6/0.8, lateral rectangle counts 1/3/6 per side,
floor counts 1/2/3 and overhead counts 1/2 as appropriate to the family. Layouts
include the generator's curved passage, randomized masks, rotations and morphology.
Arrays are native XYZ, 4 cm resolution, nominal 3 x 2 x 1.5 m volume. Native
SDF/BF/GF are cached with hashes and sampled by scene ID for each environment.
The legacy CAT X/Z interpolation weights are preserved for checkpoint compatibility.

The physical floor supports each G1. **Clutter uses CAT's forbidden SDF training
semantics**, not solid full-body obstacle contacts. This is not yet a physical
contact/manipulation or stair training task. Full GRAIL terrain-task integration
remains separate from teaching its decoder in the native CAT robot dynamics.

## Many environments means actual parallel physics

`cat_parallel_env.py` creates independent GPU PhysX G1 articulations, separated
by 5 m with inter-environment collision filtering. Every reset resamples a
generated scene for that slot. Resets, target histories, GRAIL histories, gait,
delayed odometry and termination masks are per-environment.

Measured teacher-controlled physics throughput on the shared RTX 5090:

| Parallel environments | Measured environment-steps/s |
|---:|---:|
| 16 | 49 |
| 256 | 668 |
| 2048 | 4824 |

These are 64-step physics checks, **not learner-throughput or learned-policy
success claims**. The 2048-env check exercised all 48 lateral geometries and 72
completed/reset episodes. Another user's compute workload was already present;
no process was killed to improve these numbers. 2048 is the selected batch size.

## CAT learning style and explicit whole-body adaptations

1. Separate lateral, low, overhead and mixed whole-body specialists start with
   the released CAT generalist as a leg-action teacher. Teacher assistance
   decreases during transfer; labels always use the executed target history.
2. Each family specialist receives PPO updates in that generated distribution.
3. Whole-body specialists label generalist-visited states, routed by family;
   Gaussian KL DAgger distills their motor-token policies into one generalist.
4. Generalist PPO runs over the mixed family distribution.

The pretrained CAT hidden features are frozen. A CAT-sized 512/256/128/64 hidden
stack predicts 64 motor-token coordinates. The original frozen GRAIL `g1_dyn`
decoder outputs all 29 joint targets. CAT provides 12 leg labels only; nominal
upper posture is an explicitly separate transfer prior, not invented arm labels.
Native field rewards include head, hand, feet, knee and shoulder clearance.
Original recorded GRAIL actions/tokens provide a separate retention loss; this
does **not** prove retained stair capability.

Actor input contains native noisy/delayed CAT 162D observations plus GRAIL's
1029D state/history. The independent critic receives the native CAT 250D
noiseless privileged packet plus GRAIL proprioception. The motor-token action
space and privileged whole-body inputs necessarily differ from CAT's 12-action
network. This is a torch/PhysX adaptation, not a bit-identical JAX/MJX run.
The GRAIL flat reference object is virtual; no future motion clip or LiDAR is
used in this first navigation-teacher stage.

Native CAT settings retained:

- 500 Hz physics, 50 Hz control, 1000-step (20 s) episodes.
- Uniform initial XY offsets +/-1 m, yaw +/-pi/2, joint-position multiplier
  0.5..1.5 and initial root velocity +/-0.5, as in executable CAT reset code.
- Gain scales 0.75..1.25, torque perturbations, 1.3..1.5 Hz gait, native
  observation-noise scales, five-control-step odometry refresh and pushes.
- The 22 native reward equations/scales, including the native nonnegative
  total-reward clipping. Original legacy field clamping is preserved and
  out-of-domain queries are diagnostic, not silently dropped from the MDP.
- PPO learning rate 3e-4, entropy 0.003, discount 0.98, GAE 0.95, clip 0.2,
  rollout 32, four epochs, 64 minibatches and gradient limit 1.
- Adaptive scene failure sampling with EMA 0.95 and a nonzero coverage floor.

Explicit differences: a frozen 29-joint GRAIL decoder/64D Gaussian latent;
leg-target MSE for CAT-to-whole-body transfer; Gaussian KL between our whole-body
specialists; PhysX instead of MJX; hardware-scaled batch; nominal upper-body
and recorded GRAIL retention losses. Native self-contact shapes/filters are
enabled physically, but a separate pair-specific self-contact termination
query is not yet ported. Fall/upvector/SDF collision termination is implemented.

## Run budget, checkpoints and evaluation

Default transition budgets are independent of environment count:

- Per family: 4,194,304 teacher-transfer transitions + 16,777,216 PPO transitions.
- Generalist: 8,388,608 DAgger transitions + 134,217,728 PPO transitions.
- Total: **226,492,416 transitions**, subject to a checkpointed 24-hour wall limit.

At 2048 environments these are 64 transfer and 256 PPO rollout updates per
family, followed by 128 DAgger and 2048 generalist PPO rollout updates. Optimizer
update count is reported separately. Actual end-to-end learner throughput still
requires the pending integrated smoke; pure physics implies at least ~13 hours
on the measured shared-GPU load, before learning/evaluation overhead.

Held-out-geometry evaluations have no teacher assistance and run full 20 s
episodes. Success requires remaining upright, no post-grace collision, and
staying within 0.2 m of the goal for the last second. Exit-plane crossing alone
is never success. Completed episode records preserve geometry IDs/failure types.

Checkpoints contain all specialist/generalist learner parameters, optimizers,
RNGs, normalization, scene sampling/coverage and stage counters. Original frozen
weights remain hash-addressed. Resume starts fresh randomized PhysX episodes;
it is learner continuation, **not bit-exact simulator replay**. Source datasets
and earlier pilot checkpoints are not overwritten. Logs flush every rollout
update, checkpoints every 25 updates and at phase/stop boundaries. Low disk or
a new unreviewed GPU robot deployment causes a checkpointed stop.

W&B credentials were verified as `skvayzer`. The project is `skvayzer/grail-cat`;
a real full-run URL will be recorded only when that run actually starts.

## Reproduction and next launch

From `~/robotics/GRAIL-CAT`, after resolving GPU deployment coordination:

```bash
# Existing verified bank: research/runs/20260912_cat_generated_bank_v2
# To regenerate in a NEW directory:
.venv/bin/python research/cat_parallel_bank.py research/runs/new_cat_bank --workers 4

# Mandatory integration check before the expensive run: transfer, PPO, DAgger, PPO.
.venv/bin/python research/cat_parallel_train.py research/runs/new_cat_smoke \
  --num-envs 2048 --smoke-updates 1 --wandb-mode disabled \
  --accept-isaac-eula --detach

# Only after smoke completes without errors and its checkpoints are verified:
.venv/bin/python research/cat_parallel_train.py research/runs/new_cat_full \
  --num-envs 2048 --hours 24 --wandb-mode online \
  --accept-isaac-eula --detach

tail -f research/runs/new_cat_full/worker.log
# status.json / metrics.jsonl / latest.json / wandb.json live in the run directory.
# To stop, inspect process.json and send SIGTERM to that run's exact worker PID.
```

The deployment-review flag is not a generic safety bypass: use it only after an
operator confirms the exact named deployment PID is not controlling hardware.
Normally stop the conflicting deployment through its operator first.

## Verification completed so far

- 225 unique generated layouts with disjoint split/seed/geometry hashes.
- Field-bank sampling matches the independently tested per-scene sampler.
- Fast actor packing matches the native-verified CAT observation bridge.
- Batched GRAIL history matches single-environment history, including partial resets.
- All 22 reward terms compared to unchanged native JAX source on 32 synthetic
  states, within 2e-5 relative/absolute tolerance; source hashes recorded.
- Actual frozen GRAIL decoder CPU checks exercised transfer, DAgger and PPO
  gradients, all 29 outputs, frozen hashes and bit-exact learner next-update resume.
  These arithmetic checks are not physics learning or navigation success.
- 249 CPU tests pass; 8 GPU tests deliberately skipped while deployment is active.
- GPU physics benchmarks at 16/256/2048 environments pass. Integrated optimizer
  smoke and the sustained run are pending the operator clarification above.

Local evidence: `research/runs/20260912_cat_generated_bank_v2/`,
`20260912_cat_parallel_bench16_v2/`, `20260912_cat_parallel_bench256/`,
`20260912_cat_parallel_bench2048/`. Failed/aborted attempts were retained.
