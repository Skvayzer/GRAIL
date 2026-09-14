# Project context and decisions

This document is a curated handoff of the relevant conversation, checked source,
and saved evidence. It is not a verbatim transcript or proof of untested claims.
The current task was to package context; the latest architectural decision is
the frozen-SONIC v2 proposal. No adapter-v2 implementation has been authorized
by this packaging operation.

## Objective and research scope

The user wants the best of GRAIL/SONIC whole-body terrain skills and CAT 3D
obstacle avoidance, eventually supporting manipulation-compatible balance.
The desired body interface has 29 joints: 12 legs, 3 waist and 7 per arm.
Fingers/grippers are separate. Moving wrists or having 29 outputs does not prove
grasping, hand-command tracking, stair retention or useful whole-body avoidance.

The earlier plan allowed progressively unfreezing terrain-controller weights.
That plan is now explicitly superseded. The current plan requires **no base
unfreezing**: train an adapter and training-only critic around an immutable
motor foundation. Preserve manipulation as a later requirement.

## Important user preferences and boundaries

- Keep current work and other projects intact; use a separate branch/worktree.
- Commit progress. The GitHub account is Skvayzer.
- Reuse CAT's generated clutter and specialist/generalist learning structure;
  do not substitute one fixed scene for a diverse training distribution.
- Use available GPU resources efficiently, but do not kill others' processes,
  infer a process is harmless from low utilization, or treat VRAM occupancy
  as evidence of learning. Previous PID-specific approvals are historical.
- User explicitly requested **no automatic policy evaluations** during
  training; they will evaluate later. This does not disable training metrics,
  checkpointing, or immediate numerical/contract checks.
- Show failure honestly. Successful imports, lower losses, rendered scenes,
  short exits and frozen hashes are not navigation success.
- The real robot is potentially dangerous. Only the user changes locomotion/
  FSM mode. No implicit motor actuation, concurrent control owners, remote
  service restarts or arming based on this handoff.
- Keep the handoff lightweight; no virtual environment, weights or giant
  scene arrays in the ZIP. Recover them separately if needed.

## Locations and repositories

Primary desktop: /home/constantinesmirnov/robotics/GRAIL-CAT.
GitHub fork: https://github.com/Skvayzer/GRAIL.git.
Research branch: research/grail-cat-terrain.
Upstream GRAIL base: aa31d8242ac79b11545b9e3635f73014a227bdfc.
Exact handoff HEAD is in the archive manifest, not a hardcoded moving branch.

Related local CAT: /home/constantinesmirnov/robotics/Click-and-Traverse,
branch setup/desktop-rtx5090. Original upstream:
https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/.
Generator pin: 866ba392f1c1e84b92ad75fa66550f26e8af8e48.

Related robot stack: /home/constantinesmirnov/robotics/humanoid_navigation,
branch feature/cat-perception-preview, GitHub Skvayzer/humanoid_navigation.
Robot historically accessed as unitree@192.168.50.179.
This handoff copies the local snapshot; it does **not** verify current robot
state or guarantee that local source matches every remote deployment.

Obsidian vault: /home/constantinesmirnov/research/research.
Desktop report: GRAIL_CAT_Report_20260914/.
The user previously used tl-2 as the desktop SSH/Tailscale name for SCP.
Recheck names/connectivity instead of assuming other-machine state.

## Robot-side background (separate from desktop research)

The earlier project ported tested Agibot X2 FAST-LIO/localization/Nav2 logic to
G1 using Foxy containers, Unitree-compatible CycloneDDS, and a host rosbridge
visualization path. It preserves vendor topic discovery and avoids modifying
other robot projects. The lab map was named teaching_lab.
Extrinsic/frame conventions, map-floor alignment, and ROS/Foxglove serialization
required fixes. Do not replace verified transforms with guessed identity values.

A CAT perception-preview branch was added for raw LiDAR/3D obstacle fields,
then non-actuating policy/kinematic previews. It is not the trained GRAIL-CAT
adapter or a proven low-level obstacle-avoidance deployment.
The source snapshots and CAT RESEARCH/README/VALIDATION notes explain its gates.
Navigation's killnav/velocity gateway is not automatically an adequate emergency
stop for a future joint-level controller. Never auto-run included deploy scripts.

Original source archives mentioned by the user include
robotics_slam_nav_backup_20260713_111143.zip (tested X2 baseline) and slam.zip
(reverse-engineering attempt). They are not included here; the current research
does not require copying every historical Downloads archive.

## What was implemented on the desktop

1. Restored a pinned GRAIL terrain checkpoint in a separate Isaac Lab environment.
   Existing stairs, curb, slope and sitting sample rollouts establish reproduction
   only. Mean global MPJPE: 35.67 / 52.63 / 26.32 / 16.88 mm respectively.
2. Ported CAT geometry/guidance and scene generation. Added support-vs-obstacle
   semantics, contact permissions, full-height fields and articulated coverage.
3. Fixed pelvis guidance over stair edges: a checked transit connection recovered
   validity from 334/499 to 499/499 reference frames. This is geometry, not balance.
4. Reviewed stairs/clutter layouts and a kinematic arm-clearance witness.
   User approved their appearance. These were not trained-policy successes.
5. An early reference-conditioned four-environment residual pilot failed.
   It is separate from the later large native-CAT-style training experiment.
6. Checked the original CAT controller in MuJoCo and Isaac on side1, hurdle1,
   crouch1. Geometry, state/action mapping and field parity were checked.
   The Isaac overhead episode violated the native -0.04 m SDF threshold once
   by about 0.35 mm: **partial pass**, not universal simulator validation.
7. Built a CAT-to-GRAIL latent learner, generated layout banks, batched physics,
   specialist transfer/PPO, specialist-to-generalist DAgger, generalist PPO,
   resumable checkpoints, W&B training logs and scoped GPU resource guards.
8. Increased batch sizes under user requests, preserving learner progress.
   The completed result is poor. A PDF reports this plainly.
9. Revised the proposal to a full frozen pipeline with a true nominal-motion
   residual. This revision is documentation only, not new code/training.

## Completed experiment: exact frozen boundary

See cat_distill_model.py, cat_parallel_policy.py, cat_parallel_env.py,
cat_parallel_train.py.

Frozen: GRAIL g1_dyn body decoder and pretrained CAT feature trunk.
Trainable: latent adapter, 64D Gaussian log_std, separate privileged critic.
Navigation bypasses the GRAIL motion encoder and FSQ; base is a zero vector.

    token = zero_base + 2 * tanh(sampled_adapter_output)
    q_target = frozen_decoder(token, proprioception_history)

A zero latent is not necessarily stand/walk. This network selected the whole
navigation token, not merely a correction to a valid encoded nominal motion.
Calling it residual-only was misleading in navigation mode. It still reused
pretrained features/decoder and CAT labels, so “everything learned from scratch”
would also be misleading.

Dimensions: native CAT input 162, frozen CAT features 64; GRAIL state packet
1029; base 64; mode 1; adapter input 1158. Large adapter:
1158 -> 512 -> 256 -> 128 -> 64 -> 64. Critic: 1279 -> 1024 -> 512 -> 256 -> 128 -> 1.
Adapter state includes virtual-object/reference-related contract limitations.

Losses differ by stage:
- Transfer: CAT leg-target MSE + 0.25 nominal upper-posture MSE, value and
  recorded GRAIL retention losses.
- DAgger: KL from the whole-body family specialist distribution, value/retention.
- PPO: clipped stochastic residual objective + entropy, value and retention.
  Logged leg MSE is diagnostic during PPO, not its imitation objective.

The task uses native CAT 50 Hz control / 500 Hz torque PD on physical flat
floor, with SDF-only clutter. It is not the earlier physical stairs/clutter task.
Training uses any of 11 sites at SDF < 0 as a violation, termination after a
50-step grace period, head-height/inversion fall checks, and a strict
1000-step/last-50-steps goal condition. This is different from the direct-CAT
-0.04 m monitored threshold/early exit-plane comparison. Do not equate results.

## Production lineage and outcome

Six parent-linked runs, not capacity tests or interrupted startups:

| Run suffix/date | Envs | Final cumulative transitions |
| --- | ---: | ---: |
| 20260912 ... shared_gpu_v1 | 2048 | 4,194,304 |
| 20260912 ... 16640_v1 | 16640 | 6,856,704 |
| 20260912 ... 16768_v1 | 16768 | 10,612,736 |
| 20260913 ... 16384_v1 | 16384 | 101,838,848 |
| 20260914 ... 24576_noeval_v1 | 24576 | 107,343,872 |
| 20260914 ... 25344_noeval_v1 | 25344 | 227,373,056 |

The final state is completed, W&B exit code 0, finished 14 September about
00:51 local Dubai time. 103,680 optimizer updates total.
Final checkpoint: research/runs/20260914_cat_generated_full_25344_noeval_v1/
checkpoint_000227373056.pt.
SHA256: 7f4240808cc9bd935e1d64e2799df997130eff3df251dbf51dbdc5a63189bdc8.

Final 20 logged updates: 285,754 completed training episodes; episode-weighted
success 0%, falls 88.43%, collision indicator 15.53%. Indicators may overlap.
No independent final deterministic evaluation. The failure cause is unresolved.

Bank: 180 training + 45 held-out layouts, not 25,344 unique geometries.
Families train/holdout: lateral 48/12, low 36/9, overhead 48/12, mixed 48/12.
240 requested recipes, 15 low recipes rejected after native morphology.
Grid 75x50x38 at 0.04 m, origin [-0.5,-1,0], nominal goal [2,0,0.75].
Generated-bank manifest is included; heavy NPZ fields are excluded.

Final run median throughput ~50.84k transitions/s, peak process VRAM
28.5547 GiB, minimum logged free memory 1.5865 GiB. Different sharing/stages
confound scaling. GPU status later showed no compute jobs, but that is stale
immediately; check live before scheduling.

## W&B clarification

Final run: https://wandb.ai/skvayzer/grail-cat/runs/rmx6awvl.
A read-only server query confirmed finished and 148 training-history records.

Keys include loss, value_loss, reward, retention_mse, leg_mse, fall_rate,
success_rate, collision_rate and total_steps. There is **no ppo/ namespace**.
Separate policy loss, entropy, approximate KL, clip fraction and explained
variance were not recorded. This is an instrumentation omission, not missing
PPO or disabled learning. Do not fabricate historical diagnostics.
Add them before another long run. No evaluation does not mean no logging.
The 45 validation-scenes-visited counter is inherited; no evaluation files
exist in the two no-eval continuations.

## Active v2 proposal

Freeze the complete terrain-trained SONIC encoder, terrain conditioning, FSQ,
decoder, base normalization and action mapping. Supply valid nominal motion
inputs. Train CAT/body/support/goal/task feature projections, a pre-FSQ residual
head/exploration and a separate critic.

    h_nominal = frozen_encoder(valid_motion, terrain)
    tokens = frozen_FSQ(h_nominal + small_scaled_residual)
    q_target = frozen_decoder(tokens, correct_state_history)

GRAIL's manipulation configuration uses pre_quantization and scale 0.1.
This is a starting reference, not a validated safe bound for our terrain model.
Preserve height-encoder ordering, masks, normalization and history. Zero
residual must reproduce the full original controller on identical inputs.

First learn reference-conditioned avoidance around usable motion references.
A later causal goal/route-to-motion-command provider is required for autonomous
navigation. It does not exist just because the diagram includes it. Do not
silently zero missing object/future-reference channels or revert to direct tokens.

Keep generated clutter, family specialists/generalist and later terrain/contact
and manipulation-compatible tasks. Preserve GRAIL's working low-level dynamics
first instead of silently swapping the robot/controller into another task.
No progressive base unfreezing. No claim freezing alone cures failure; the
completed decoder was already frozen.

## Next work, when the user requests implementation

1. Separate branch/worktree and distinct run lineage; do not overwrite old runs.
2. Full-controller action-transform wrapper with explicit freeze allowlist/hashes.
3. Zero-residual parity, including terrain, FSQ and observation/action contracts.
4. CAT residual observations and actor/critic using the GRAIL adapter workflow.
5. Generated-scene curriculum and complete PPO/residual/per-family logs.
6. Reference-conditioned flat-clutter training, no auto evaluation, regular saves.
7. Causal navigation commands, terrain/clutter and manipulation later.

The archived code is the old completed implementation plus the new proposal.
Do not infer that steps above are already implemented.
