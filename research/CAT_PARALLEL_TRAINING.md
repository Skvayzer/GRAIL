# Generated CAT clutter and parallel whole-body learning

## Current state — 13 September 2026

**Running and verified with 16,384 environments:**
`research/runs/20260913_cat_generated_full_16384_v1`, launch revision `05537ed`,
worker PID 920235 at launch. Resumes the verified 10,612,736-transition /
19,456-optimizer-step checkpoint below. The generated bank, stage budgets,
17 GiB Torch allocator cap, 1 GiB device-free guard and 24h wall cap are unchanged.

The user explicitly authorized sharing the GPU with existing deployment PID
604747 and stated its owner is not using it. Passed `--reviewed-deployment-pid`
for that exact PID only. Its Ethernet/all-output configuration was disclosed;
0% GPU usage is not proof that it is stale or disconnected from hardware.
New deployment PIDs still trigger stopping. No other process was signalled or
modified; our workload remains simulation-only. The run directory records the
approval scope in `deployment_review.json`. Preflight: roughly 30.4 GiB free,
existing deployment using 586 MiB.

[Current W&B run](https://wandb.ai/skvayzer/grail-cat/runs/d27kk9up) reports running.
After five resumed PPO rollouts: 13,234,176 cumulative transitions / 20,736
optimizer steps, about 48,000 environment-steps/s. Measured process VRAM is
19.855 GiB (21.32 decimal GB), with 10.492 GiB CUDA device-free. CPU audit of
`checkpoint_000012709888.pt` passes SHA-256, finite tensors, preserved progress,
updated PPO parameters and unchanged frozen/bank provenance. Twelve targeted
CPU tests pass. This is restart/checkpoint evidence, not learned-navigation
success: early PPO success is still zero. The detached process is left running
under the existing transition schedule and wall-time limit.

## Previous stop and restart attempts — 12 September 2026

**Historical: restart blocked by recurring GPU deployment jobs:** prepared 16,384
environments, a reduction of 384 (2.29%) from the previous batch. Latest attempt:
`research/runs/20260912_cat_generated_full_16384_v2`, launch revision `fe56e5c`,
worker PID 13018 at launch, sent SIGTERM during startup.

The first attempt (`20260912_cat_generated_full_16384_v1`, PID 12057) was stopped
during initialization when a new GPU G1 deployment/inference process appeared
after preflight. That process (PID 12452) exited without our intervention.
Verified the GPU was clear and retried in the new `v2` directory, preserving
the interrupted startup record and the original source checkpoint.
Another G1 deployment then appeared under the same user (PID 14243), so the
second startup was also stopped. These short-lived jobs can reappear after a
clean preflight; do not repeatedly restart or exempt them without coordination
with their operator. Their command line suggests loopback/ZMQ simulation, but
hardware isolation had not been confirmed by the operator. Those attempts were paused.
The PyTorch allocator cap was 17 GiB;
simulator allocations are additional. The 1 GiB device-free stop reserve and
runtime robot-deployment guard are unchanged, with no PID exemptions.

The previous 16,768-env run did not crash: its memory usage subsequently rose
to 20.068 GiB, leaving 0.959 GiB device-free. It saved and stopped via the
memory guard. Its final checkpoint is `20260912_cat_generated_full_16768_v1/`
`checkpoint_000010612736.pt`: **10,612,736 cumulative transitions / 19,456
optimizer steps**, in lateral PPO after completed lateral teacher transfer.
That checkpoint passed CPU SHA-256, finite-tensor and transition-counter checks
before the new launch. The restart continues those weights and optimizer state.
No earlier production progress was discarded.

At restart preflight the other compute job was no longer running and no GPU
compute processes were reported; approximately 31 GiB was free. We did not stop
any other user's process. Keep checking availability before future restarts;
this is shared GPU use, not guaranteed isolation or exclusive reservation.
All activity is desktop simulation only. Early success remained zero at the
previous stop; runtime evidence is **not successful learned navigation**.

Resize history: the user requested at least 20 GB of process GPU use.
The 2,048-env run was stopped checkpointed at 4,194,304 transitions (completed
lateral teacher transfer). Its checkpoint is retained; it is no longer running.
The 16,384-env capacity test passed all four optimizer phases at 17,176 env-steps/s,
with 12.79 GiB Torch-reserved and 1.27 GiB minimum device-free memory (about
19.8 GiB total process VRAM). The 16,640-env continuation resumed the original
learner/optimizers, not the capacity-test weights, then saved 6,856,704 transitions
after five PPO rollouts at 19.895 GiB process VRAM. The final 128-env increase
resumed that saved progress; neither production continuation was discarded.

Resizing preserves per-stage transition counters and budgets, rather than
interpreting the old 2,048-env rollout index at the new vector size. A last full
rollout can overshoot a stage's target by less than one vector batch. Counters,
checkpoint hash and parent run are recorded. Physics episodes restart fresh.
Checkpoint/evaluation cadence is also transition-based after resizing.

The following documents the original 2,048-env launch and its evidence:

The G1 deployment has stopped, and the user authorized sharing the remaining
GPU resources with the other compute job. The 2048-env four-phase optimizer
smoke passed: 262,144 transitions, 1,024 optimizer steps, finite checkpoints,
changed transfer/distillation/PPO weights and unchanged frozen provenance.
Measured end-to-end throughput was about 3,782 environment-steps/s; at least
15.56 GiB device memory remained free. This validates the learner plumbing,
**not successful learned navigation**. That original sustained run started with:

- Run: `research/runs/20260912_cat_generated_full_shared_gpu_v1`
- Launch revision: `1b79057`; worker PID recorded in `process.json` (2762897 at launch).
- [W&B run p3rsz237](https://wandb.ai/skvayzer/grail-cat/runs/p3rsz237)
- 2,048 environments, default 226,492,416-transition schedule, 24h wall limit.
- First training checkpoint saved at 65,536 transitions / 256 optimizer steps.

No other user's process was modified.

The new launch preflight refuses unreviewed robot-deployment GPU processes.
During training it checks again at update boundaries and checkpoints/stops if a
new one appears. This is a conservative coordination guard, not hard real-time
GPU isolation. Reserve the GPU appropriately when operating physical robots.
The default launch caps the PyTorch allocator at 14 GiB (Kit/PhysX allocate
separately) and stops checkpointed below 2 GiB device headroom. The current
explicitly sized run uses 17 GiB / 1 GiB respectively, as documented above.
Evaluations also check for deployments/headroom every 32 simulation steps.
Actual smoke usage was much smaller: at most 2.63 GiB Torch-reserved memory,
with roughly 5 GiB total process GPU use including simulator allocations.

## Actual generated distribution

Short CPU-rendered examples and reproduction/transfer commands are in
[CAT_TRAINING_LAYOUT_VIDEOS.md](CAT_TRAINING_LAYOUT_VIDEOS.md). They show actual
cached training layouts, not trained-policy rollouts.

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
no process was killed to improve these numbers. The original batch size was
2048; the current user-requested restart uses 16,384.

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
update count is reported separately. The shared-GPU optimizer smoke measured
about 3,782 environment-steps/s, implying roughly 16.6 hours for the transition
budget alone, plus held-out evaluation and initialization. Full-run throughput
will vary with other workloads and episode/reset behavior; the 24h cap remains.

Held-out-geometry evaluations have no teacher assistance and run full 20 s
episodes. Success requires remaining upright, no post-grace collision, and
staying within 0.2 m of the goal for the last second. Exit-plane crossing alone
is never success. Completed episode records preserve geometry IDs/failure types.

Checkpoints contain all specialist/generalist learner parameters, optimizers,
RNGs, normalization, scene sampling/coverage and stage counters. Original frozen
weights remain hash-addressed. Resume starts fresh randomized PhysX episodes;
it is learner continuation, **not bit-exact simulator replay**. Source datasets
and earlier pilot checkpoints are not overwritten. Logs flush every rollout
update; checkpoints are saved when crossing each 1,638,400-transition interval
(25 original 2048-env rollouts), and at first-resumed-update/phase/stop boundaries.
Periodic evaluation uses 16,384,000-transition intervals plus phase boundaries.
Low disk or
a new unreviewed GPU robot deployment causes a checkpointed stop.

W&B credentials were verified as `skvayzer`. The project is `skvayzer/grail-cat`;
the current run URL is recorded at the top of this page.

## Reproduction and next launch

From `~/robotics/GRAIL-CAT`, with no unreviewed GPU robot deployment running:

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
  --num-envs 2048 --hours 24 --wandb-mode online --torch-memory-limit-gib 14 \
  --accept-isaac-eula --detach

tail -f research/runs/new_cat_full/worker.log
# status.json / metrics.jsonl / latest.json / wandb.json live in the run directory.
# To stop, inspect process.json and send SIGTERM to that run's exact worker PID.
```

To monitor the current resized run:

```bash
tail -f research/runs/20260913_cat_generated_full_16384_v1/worker.log
cat research/runs/20260913_cat_generated_full_16384_v1/status.json
```

Exact current launch (record only; **do not start another copy** or reuse the
PID exemption without fresh review):

```bash
.venv/bin/python research/cat_parallel_train.py \
  research/runs/20260913_cat_generated_full_16384_v1 \
  --num-envs 16384 --hours 24 --wandb-mode online \
  --torch-memory-limit-gib 17 --min-free-gpu-gib 1 \
  --resume research/runs/20260912_cat_generated_full_16768_v1/checkpoint_000010612736.pt \
  --reviewed-deployment-pid 604747 \
  --accept-isaac-eula --detach
```

This large profile was measured on this shared RTX 5090, not a portable default.
Recheck other GPU users and available memory before any future launch; use a new
run directory and the latest verified production checkpoint for continuation.

The first checkpoint and W&B upload were verified. Startup compute was about
3,800 env-steps/s, process VRAM about 5.4 GiB with another user's job still
present. Check current logs rather than assuming this launch-time snapshot is
still current. Early transfer losses are not independent navigation evaluation.

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
  smoke `20260912_cat_parallel_smoke_shared_gpu_v1` completed all four phases.
  `check_cat_parallel_smoke.py` verifies finite checkpoints, increasing counters,
  changed adapter/PPO parameters and latest-checkpoint checksum on CPU.
  Its `audit.json` explicitly does not claim navigation success.

Local evidence: `research/runs/20260912_cat_generated_bank_v2/`,
`20260912_cat_parallel_bench16_v2/`, `20260912_cat_parallel_bench256/`,
`20260912_cat_parallel_bench2048/`. Failed/aborted attempts were retained.
