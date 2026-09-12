# Progress

## 2026-09-12 — resized training running at 16,768 environments / 21.41 GB VRAM

- The temporary G1 GPU deployment exited naturally. Verified no GPU robot
  deployment before starting `20260912_cat_generated_full_16768_v1` at `5bb3265`.
  PID 2825440 at launch; W&B `https://wandb.ai/skvayzer/grail-cat/runs/eartbpzz`.
  Runtime deployment checks remain enabled, with no exemptions. No robot access
  or actuation; the other user's 9,876 MiB compute job was left running.
- Resumed the verified 6,856,704-transition checkpoint, retaining all earlier
  production progress. At audit: 9,539,584 transitions / 18,944 optimizer steps.
  Latest audited checkpoint `checkpoint_000009539584.pt` SHA-256:
  `2bce54b671bb2949213feea8c42f5d52e689816039115b296bdfb68c0950fe83`.
  CPU checks pass finite tensors, counters, updated PPO parameters and unchanged
  frozen provenance. W&B reports running; 12 targeted CPU tests pass.
- Measured steady process use 19.94 GiB = 21.41 decimal GB; approximately 1.08
  GiB CUDA device-free. This exceeds 20 decimal GB, but is slightly below 20 GiB.
  Kept the 1 GiB stop reserve rather than padding allocations or risking another
  batch increase. Torch allocator cap 17 GiB; simulator allocations are separate.
- Recent learner throughput about 11.2–12.1k environment-steps/s, first update
  17.2k. Generated bank and transition budgets unchanged. Lateral transfer is
  complete and lateral PPO continues; early success remains zero. Finite losses
  and checkpoint updates are not evidence of successful navigation yet.

## 2026-09-12 — 16,640-env continuation verified; new deployment blocks final resize

- Run `20260912_cat_generated_full_16640_v1` restored the original weights,
  optimizer and completed transfer budget, then performed five real PPO updates.
  Latest saved cumulative progress: 6,856,704 transitions / 17,664 optimizer
  steps. Peak measured NVIDIA process use: 19.895 GiB, about 1.2 GiB CUDA-free.
  Checkpoint: `checkpoint_000006856704.pt`; W&B run `k0db7rad`.
- Requested a graceful stop for one further 128-env sizing adjustment to
  16,768; all new progress was preserved. A separate G1 deployment/inference
  process then appeared on the GPU (PID 2823322 at inspection, about 1.46 GiB).
  Final restart has not been performed without renewed deployment coordination
  and sufficient memory headroom. No other user's process was stopped/changed.
- Added finer 128-env increments and preserved checkpoint/evaluation cadence
  in transitions when batch size changes. The 1 GiB free-memory stop threshold
  was not relaxed. Twelve targeted CPU tests pass.

## 2026-09-12 — resizing to the user's 20 GB VRAM target

- Stopped only our original worker gracefully at 4,194,304 transitions; lateral
  transfer is complete. Checkpoint `checkpoint_000004194304.pt` preserved and
  verified, along with its optimizer and normalization state. The in-progress
  held-out evaluation is marked partial; it is not treated as a full benchmark.
- Capacity run `20260912_cat_capacity_16384_v1`: 16,384 independent environments,
  four optimizer phases, 2,097,152 transitions and 1,024 optimizer updates.
  Checkpoint audit passes. Average 17,176 env-steps/s; peak Torch-reserved
  12.785 GiB; minimum CUDA device-free 1.272 GiB. Total process use approached
  19.8 GiB. Other user's 9.6 GiB GPU job was left running.
- Added transition-based resize/resume accounting and tests (including the
  real legacy checkpoint), environment-count choices up to 32,768, configurable
  free-memory stop threshold (minimum 1 GiB), and NVIDIA process-memory metrics.
  The measured next candidate is 16,640 environments; no dummy VRAM allocation.
- Eleven targeted CPU tests pass. These changes affect only our desktop
  simulation learner; no robot connection or other user's process changed.

## 2026-09-12 — full generated-clutter training launched

- Launched `20260912_cat_generated_full_shared_gpu_v1` at code revision `1b79057`,
  PID 2762897, detached from the SSH session. W&B:
  https://wandb.ai/skvayzer/grail-cat/runs/p3rsz237 (skvayzer/grail-cat).
- 2,048 independent simulated environments; generated 180-layout training bank,
  45 held-out geometries, full transfer/specialist-PPO/DAgger/generalist-PPO
  schedule, 226,492,416-transition budget and 24h wall cap. No real-robot actions.
- First update/checkpoint completed: 65,536 transitions / 256 optimizer steps,
  finite losses, about 3,817 env-steps/s. Total process VRAM about 5.4 GiB,
  approximately 16 GiB device-free; other compute workload still running.
  This is launch evidence, not a completed run or proof of navigation success.

## 2026-09-12 — shared-GPU optimizer smoke passed

- G1 deployment no longer running; user explicitly authorized using remaining
  GPU memory alongside the other compute job. Other users' jobs untouched.
- `20260912_cat_parallel_smoke_shared_gpu_v1`: 2,048 environments, four phases
  (teacher transfer/PPO/whole-body DAgger/PPO), 262,144 transitions and 1,024
  optimizer steps. CPU checkpoint audit passes finite tensors, updated adapter
  and stochastic-policy parameters, counters, frozen provenance and hashes.
- Mean learner throughput 3,782 env-steps/s; peak Torch-reserved 2.63 GiB,
  minimum device-free 15.56 GiB. These are plumbing/performance checks, not
  evidence of a trained navigation policy. Full scheduled training is next.
- Added 14 GiB Torch allocator cap, low-headroom checkpointed stopping,
  evaluation-time deployment checks, source revision/file hashes and terminal
  status. Seven targeted CPU tests pass. Simulator allocations are additional
  to the Torch cap; actual process VRAM was also inspected using NVIDIA tools.

## 2026-09-12 — generated training layout videos

- Created four 12-second 720p MP4s: lateral, low, overhead and mixed. Each uses
  three fixed cached training seeds (difficulty 0.4/0.6/0.8), with a rotating 3D
  mesh and exact occupied top/side projections. Includes twelve PNG stills and
  source/scene/video checksum manifest on Desktop, `CAT_training_layouts_20260912`.
- CPU rendering only; no simulator, GPU encoder, robot or training process
  started/changed. These are explicitly geometry previews, not policy rollouts
  or evidence of successful navigation. Native floating SDF volumes preserved.
- Every frame of all four movies decoded successfully (240 frames each), and
  selected-scene/checksum plus existing video/mesh tests pass (8 tests).
  Reproduction and laptop transfer: `CAT_TRAINING_LAYOUT_VIDEOS.md`.

## 2026-09-12 — generated-clutter parallel task and staged learner

- Replaced the single-scene pilot direction with 180 unique generated training
  layouts and 45 disjoint held-out layouts, using unchanged pinned CAT random
  occupancy/FMM generation. Fifteen empty low-obstacle recipes were rejected
  explicitly. Lateral, low, overhead and mixed families are all represented.
- Implemented independent GPU PhysX environments, per-episode field sampling,
  native CAT reset/PD/noise/gait/odometry mechanics, 162D actor and 250D privileged
  packets, and all 22 native reward terms. JAX-source arithmetic parity passes.
- Physics benchmarks: 16/256/2048 envs at 49/668/4824 env-steps/s on the shared
  GPU. The 2048-env check visited all 48 lateral layouts and reset 72 episodes.
  These are teacher-driven physics checks, not trained-student successes.
- Implemented family CAT-to-GRAIL transfer/PPO, whole-body specialist-to-generalist
  KL DAgger, generalist PPO, held-out evaluation, W&B, compact checkpoints and
  resume. Default budget is 226,492,416 transitions with 2048 envs and a 24h cap.
  CPU tests through the actual decoder pass all three gradient paths and exact
  learner resume; 249 CPU tests pass, 8 GPU tests were deliberately skipped.
- Another user's G1 deployment/inference process was discovered on the same
  GPU. Stopped only our pending integration smoke and requested operator
  confirmation before sustained GPU load. Added launch/runtime detection.
  **Full training has not started; GPU optimizer smoke remains pending.**
  W&B identity is confirmed as skvayzer. Other users' jobs/robot state untouched.
- Full scope, differences from native CAT, commands and evidence are in
  CAT_PARALLEL_TRAINING.md. This still uses native CAT SDF-only clutter and G1
  dynamics with the GRAIL decoder, not the original GRAIL terrain task or live
  sensor/manipulation integration. No hardware command entry point was added.

## 2026-09-12 — second DAgger round and full-horizon failure checks

- Added 1000 student-state labels including 500 from actual Isaac CPU PhysX.
  The aggregate has 3739 rows, with all 396 held-out rows unchanged. Continued
  the checked optimizer/RNG state to 2200 cumulative updates; final validation
  leg/retention MSE is 0.003189/0.001038 rad². Frozen decoder/teacher unchanged.
- Short unassisted tests now cross the x exit, but violate clearance and miss
  the goal radius. Added opt-in post-exit full-horizon checks: both MuJoCo
  validation starts and the Isaac seed-6 check fall after passing the exit.
  No successful-navigation claim or promotion of the latest checkpoint.
- Training is stopped after the bounded pilot. All checkpoints, failed tests
  and earlier slower/stabler comparisons remain intact. 252 tests pass; next
  work is stable full-horizon flat behavior, not harder stairs. No robot access,
  no actuation, no changes to the other user's GPU job.

## 2026-09-12 — actual flat CAT-to-GRAIL updates and first DAgger round

- Implemented a 337,856-parameter CAT-feature motor-token adapter through the
  real frozen GRAIL decoder, outputting all 29 joint targets. CAT supervises
  only legs; GRAIL nominal upper posture and recorded GRAIL action retention
  are separate losses. No claimed arm avoidance or manipulation training.
- Ran 200-update strict-boundary and 600-update contact-aware pilots, then
  800 further updates after 960 student-state DAgger labels (1400 cumulative).
  Actual Adam/checkpoint/RNG resume is bit-exact; decoder/teacher stay frozen.
- Confirmed live-recorded GRAIL joint/action/gravity history layout. Known flat
  foot-floor boundary handling is narrow and counted; no stair guard relaxed.
  Parent/split/hash checks prevent held-out starts entering DAgger aggregation.
- Initial unassisted MuJoCo tests fell; after DAgger both held-out starts stayed
  upright for 5 s but did not reach the exit. Assisted Isaac collection advanced
  but violated native clearance. Loss improvement is not called navigation
  success. Full details and reproduction commands are in CAT_DISTILLATION.md.
- No robot access or actuation. CPU training and bounded CPU-physics simulations
  leave the other user's GPU workload untouched. The subsequent Isaac-labelled
  DAgger and final unassisted checks are recorded in the entry above.

## 2026-09-12 — CAT directly controlling Isaac, paired with native MuJoCo

- Added a bounded simulation-only runner using unchanged released CAT weights,
  original native-player observations/history and explicit 500 Hz torque PD.
  CAT controls the 12 legs; remaining native joints retain nominal PD targets.
  No GRAIL actor, optimizer, robot commands or host robot changes.
- Recorded actual closed-loop `side1`, `hurdle1`, `crouch1` episodes in MuJoCo
  and Isaac. All six stay upright and finish within 0.2 m of the native XY goal.
  Native SDF/height acceptance passes lateral/low Isaac tests, but overhead has
  one sample at -0.040348 m against -0.04 m cutoff. A fresh overhead repeat is
  byte-identical. This remains a partial pass, not certified transfer.
- Original composed mesh vertices agree within 5.73e-8 m; sampled field values
  within 2.50e-6; imported initial FK within 7.19e-7 m. Body masses, COMs, joint
  limits and inertias agree numerically. Fixed only the inertia audit's initial
  mistaken double rotation (PhysX tensor values are already in link axes).
- ONNX parity on all 1587 saved observation vectors passes with maximum action
  error 6.26e-7. Added native post-step and inertia-frame regression tests.
- Restored native explicit contact pairs omitted by the MJCF importer, while
  preserving original SDF-only robot/clutter semantics. This does not certify
  physical full-body collision avoidance or the full random/stair distribution.
- All 241 automated tests pass (26.7 s). Saved hashed episodes, plot, and
  common-renderer trajectory comparison videos for all three scenes;
  reproducible commands and caveats are in CAT_DIRECT_EVALUATION.md. Isaac worker
  teardown is bounded to its own process group. Other GPU work was untouched.

## 2026-09-12 — live CAT teacher labels and real GRAIL decoder gradient replay

- Connected the verified native CAT observation/action bridge to live Isaac
  evaluation. Original GRAIL observations/actions/rewards/terminations stay
  unchanged; CAT only labels states. Actual processed joint targets are checked
  against Isaac actuator targets and captured before automatic reset. Offline
  pairing checks catch shifted history and post-reset target substitution.
- Ported CAT command projection, gait/stop update and five-step odometry delay;
  native-source CPU command/gait parity passes (maximum 5.96e-8 / zero error).
  Deterministic 1.4Hz, 0.07m foot-lift, noise-free teacher queries are not the
  full randomized CAT MDP. Added cached/vectorized legacy-field sampling with
  independent reference parity and explicit domain masks.
- Full diagnostic run `20260912T094136_943795Z_stair_p1_cat_audit`: 499 live
  frames / 498 executed-history pairs; frozen actor/teacher hashes unchanged,
  independent CPU CAT action replay error 1.58e-6. Native CAT MJCF forward
  kinematics at saved Isaac root/joints matches all eleven sites within
  9.36e-7m. This validates link-frame conventions, not learned avoidance.
- Short repeat `20260912T095135_246506Z_stair_p1_cat_audit`: 64 frames / 63
  history pairs, saved resolved GRAIL configuration, successful independent
  audit and native kinematic parity. An intermediate config serialization
  failure on PosixPath happened before rollout; fixed with a narrow serializer
  and regression test. Failed evidence retained, no checkpoint replaced.
- Reconstructed the actual frozen GRAIL policy on CPU and replayed eight
  in-domain states from the full run: maximum action error 1.08e-6. CAT absolute
  leg-target MSE yields finite, nonzero gradients through its frozen decoder to
  the 64D post-quantization input. No optimizer steps or parameter changes.
  The early full packet lacks resolved config, so replay explicitly borrows
  the short run's hash-checked config with matching actor hash/joint order/
  clipping; it loads the full run's own checkpoint and does not rewrite data.
- Both packets remain `training_admitted=false`: only 59/499 full-run frames
  have all sites in CAT's native field bounds, and none in the short prefix.
  Original clutter-only fields do not encode stair traversability. No guard
  was relaxed to call these a training dataset. Flat specialist distillation,
  GRAIL retention, generalist DAgger and PPO remain the next implementation.
- 239 automated tests pass; commands/limitations are in CAT_STYLE.md. Only
  small diagnostic simulations ran on the shared GPU; the other user's
  existing GPU job was not stopped or modified. No simulation/training job
  from this milestone remains running, and no robot was accessed or actuated.

## 2026-09-12 — native CAT observation/action bridge and same-state labels

- Added batched PyTorch bridge for CAT's original 162D observation, named G1
  joints/body-site offsets, heading-frame fields and incremental leg targets.
  Corrected the prior documentation: CAT integrates from the PREVIOUS target,
  not the default pose. The actor weights were correct and remain unchanged.
- Exported v2 teacher metadata with that corrected description; weight SHA256
  remains `4fd7022efb45478e9805b154ffb07599ba0ec365913b5921efb99a3e73159440`.
  Old artifacts retained. Explicit legacy CAT sampling is isolated from the
  corrected geometry queries, with out-of-domain samples counted separately.
- CPU MuJoCo parity on 300 moving states in lateral/low/overhead scenes passes.
  Maximum observation/action/target errors: 2.3842e-7 / 8.6427e-7 / 3.7253e-7.
  Native training observation source also checked with noise disabled and
  distinct delayed/current inputs. Refreshed MuJoCo kinematics before comparing
  both packets so stale post-integration sensor data does not mix timestamps.
- Result: `research/runs/20260912T093526_836048Z_cat_bridge/report.json`, with
  exported site/limit contract and hashed same-state label archive. The named
  articulation input uses Isaac-style tensors reconstructed from MuJoCo; this
  is **not a live Isaac transfer validation**. Zero native failure resets in
  these short samples is not evidence of complete obstacle traversal success.
- Added applied-student-target history with per-environment reset and leg-only
  imitation loss. Gradient tests update no arm/waist targets and never update
  the teacher. 230 automated tests pass. No GRAIL optimizer steps, full randomized
  MDP port, or whole-body distillation claimed. Live Isaac collector is next.
- No real robot access/actuation. No new heavy GPU run was started while the
  other user's training workload was active; no other job was stopped/changed.

## 2026-09-12 — failed-night diagnosis and CAT-style redirection

- Overnight `20260911T220756_837720Z_m2_train` failed at 135 iterations /
  17,280 transitions / 514 optimizer steps, not after a full night. Its
  invalid-final-observation guard aborted collection; 133/135 PPO updates
  stopped early at the KL bound. Reward remained about 0.119–0.120. Historical
  W&B GPU sampling averaged 40.45%, with about 4.3GiB allocated. The valid
  iteration-100 checkpoint contains 420 updates. No learned-avoidance claim.
- Added diagnostic snapshots, exception-path checkpoint/counter recovery,
  optional independent normalized actor/critic with separate gradient clipping,
  and support-route failure episode termination. Replica audit/storage limits
  support up to 16 for the experimental M2 path. These pass unit tests but do
  **not** have an extended live-training validation or confirmed fix of the
  precise original invalid component; the old log did not record that component.
- Diagnostic rerun `20260912T091212_927858Z_m2_train` was stopped intentionally
  after the user's redirection to CAT's learning style. GPU contention made
  iterations about 30 seconds versus 2.3 seconds previously. Only our launcher
  was signalled; the other user's workload was left untouched. No overnight
  training process is left running by this work.
- User requested CAT's recipe, without further bespoke residual-trainer work.
  Downloaded the 13-file / 5,549,990-byte released generalist parameter tree
  (actor/value/normalizer, not optimizer), pinned and hash-verified. Exported
  frozen PyTorch teacher; native Brax parity on 64 inputs has maximum action
  error 2.78279185295105e-6. Teacher integration/distillation into GRAIL remains
  explicitly unimplemented; GRAIL remains the whole-body controller goal.
- Actual pinned generalist config: 37 random scenes, 36 locally available,
  missing `D8G2L3O2S13`; source defaults include 65,536 environments over up to
  eight devices. Added exact-recipe inventory/scaled hardware profile utility.
  This is not 37 live Isaac environments or a completed training launch.
- Four deterministic new lateral CAT scene candidates generated in bank
  `20260912T091558_805884Z_cat_curriculum`; all source geometries exported,
  none automatically admitted on terrain. See CAT_STYLE.md for next integration.
- Final verification: 222 tests pass; git diff whitespace check passes. The
  PyTorch export and exact 37-scene / 512-environment hardware-profile recipe
  are present under `research/artifacts/cat_teacher_v1/`. No new policy updates
  were performed after the user's CAT-style redirection.

## 2026-09-12 — overnight pilot started; first periodic checkpoint verified

- Final locomotion setup suite `20260911T220207_819774Z_m2_evaluation_suite`
  passed all four execution audits: validation clutter, stair, curb and slope.
  Each locomotion control had zero failures and one timeout. Clutter validation
  had one real contact failure; its process/reset path passed, **not** avoidance
  performance. Sitting is explicitly excluded under the current support model.
- Detached lifecycle smoke `20260911T220702_602150Z_overnight_job` completed:
  128 transitions, 16 Adam steps, final checkpoint, independent saved-file audit,
  supervisor completion, and no lingering simulation job. No host services,
  login policy, sudo configuration or other projects were changed.
- Started the requested overnight run at **2026-09-11 22:07:56 UTC**
  (12 September 02:07:56 Dubai), within the user's two-hour setup/debug cap.
  Source: clean `b9b8d55d25363b723053fbd47a1c2d72d5df5631`.
  Job: `20260911T220754_075232Z_overnight_job`.
  Training: `20260911T220756_837720Z_m2_train`.
  W&B: https://wandb.ai/skvayzer/grail-cat/runs/46v3eekz
- Fresh learner; frozen GRAIL backbone; development seed 0 only; four
  environments, at most 8,000 PPO iterations or eight hours. Checkpoints every
  100 iterations; automatic final evaluation when sufficient time remains.
  No automatic retries. Stop on invalid data/process failure, ten minutes of
  stalled progress, low disk (<20GiB), or wall-clock budget exhaustion.
- **Handoff observation at 22:12 UTC:** still training at iteration 111,
  14,208 transitions, 451 Adam steps, one failure and 27 timeouts. Expected
  early learning failures are counted rather than called successes. GPU memory
  about 4.3GB; free disk about 183GiB; supervisor heartbeat healthy.
- First periodic `learner_000100.pt` independently loaded and checked: 420 Adam
  steps, finite tensors, consistent optimizer counters/contract, 4,776,468 bytes.
  SHA256: `50b32e83ffa4211c682be1f1a033fd8040ba33b1e6e131ec7540743b22ada96a`.
- 215 automated tests passed. Use `research/m2_job.py status` for **current**
  state; the figures above are historical handoff observations, not a claim of
  completed overnight training. Commands also saved on the desktop as
  `GRAIL_CAT_training_commands.txt`. No real robot actuation occurred.

## 2026-09-12 — integrated M2 updates/resume and supervised overnight preparation

- User approved the reviewed environments and explicitly requested overnight
  simulation training, with a two-hour maximum for unresolved setup/debugging.
  W&B identity verified as `skvayzer`; actual runs sync to `skvayzer/grail-cat`.
  No robot connection, mode change, ROS or SDK actuation was performed.
- Added a prepare-only-by-default launcher with separate execution/optimizer
  flags, exact reviewed challenge/placement/geometry admission, arm-only reset
  integration, stochastic on-policy collection, deterministic evaluation,
  PPO updates, W&B metrics, checkpoint resume and read-only result auditing.
  Original reference arm tables are not replaced by the geometric witness.
- Collection run `20260911T213906_169796Z_m2_collect`: 512 transitions, one
  timeout, zero failures/updates, unchanged backbone/learner, saved checkpoint.
- The first optimizer attempt correctly stopped on inconsistent likelihoods
  after its mean head became nonzero. Fixed learner TF32/batch-size numerical
  drift locally while restoring the frozen actor's precision settings; kept
  the strict guard and added a CUDA nonzero-head regression test. Pilot launch
  rate reduced to 3e-5 after excessive early KL stopping at 3e-4.
- Training smoke `20260911T214502_102941Z_m2_train`: 24 horizons ×32 steps ×4
  environments = 3,072 transitions; **177 real Adam steps**, four timeouts,
  zero failures. Independent saved-file audit passes. Learner changed; frozen
  backbone unchanged. Actual weights, Adam moments/counters and RNG roundtrip.
  W&B: https://wandb.ai/skvayzer/grail-cat/runs/h8h6iorn
- Fresh-process resume `20260911T214652_229891Z_m2_train`: 256 transitions,
  eight additional Adam steps (177→185), loaded state verified, unchanged
  backbone. This resumes learner/optimizer/RNG, not PhysX episode state.
- Stochastic validation collection `20260911T220033_204025Z_m2_collect`:
  2,560 transitions, zero updates, four actual CAT-contact terminations.
  Reset groups were `[0,1,3]` then `[2]`, independently of the initial batch
  reset. Other environments kept collecting through each reset. Minimum cover
  gap -0.00595m, peak CAT contact 69.75N; this is a reset-path test, not success.
- Added explicit retention controls with learned adapter active and a real
  CAT fixture outside the workspace. Fixed exact-surface extraction for planar
  convex quad terrain faces; nonplanar/concave faces still reject and physical
  ray parity remains mandatory. The released curb has a 0.2921m riser; its
  geometric control graph uses 0.32m, with unchanged 0.20m for stair/slope.
- Sitting is **excluded, not passed**. Its chair contact geometry violates the
  standing/stepping support-graph model (supported-anchor fraction 0.5685).
  It needs a separate seating/task support contract. No checks were relaxed
  to call this a successful four-terrain whole-body retention benchmark.
- 215 tests pass (including real CUDA math but no unit-test optimizer steps).
  Added PID/start-time/boot-ID-safe detached job control, disk/stall/time
  supervision, durable metrics, no-retry behavior, and automatic post-training
  evaluation. No sudo/login/service settings changed. Verified logind's
  existing `KillUserProcesses=false` policy for SSH-independent job lifetime.
- Exported the exact reviewed pilot evidence as an 8.16MB, 22-file bundle,
  with source hashes unchanged and safe path rebinding on a new checkout.
  Generated evidence stays outside Git; commands and scope are in
  `M2_TRAINING.md`. Initial integration commit: `8545417` (pushed).
- Final locomotion suite and detached-job start are recorded in the next
  handoff entry once verified. This is a small reference-conditioned pilot,
  not demonstrated general obstacle avoidance or manipulation.

## 2026-09-12 — collision-screened challenge examples for human review

- Added explicit v2 CAT role provenance: padding is a boundary condition, not
  a second physical obstacle class. A unique physical causal role plus padding
  is classified; padding-only/mixed physical roles remain unresolved. Legacy
  v1 replay and default behavior stay unchanged. Upstream occupancy is exact.
- Traced the failed arm witnesses to actual reference hand/hip capsule overlap
  at spawn (left minimum -0.02985 m). Added offline constant-arm examples, with
  all 14 arm joints within limits and root/waist/legs bitwise unchanged. This
  changes initial arm configuration; it is NOT yet a validated simulator reset.
- Fixed six-seed, 10,098-placement voxel-grid screen found 55 arm-conflict
  candidates with protected-body clearance and role-valid terrain placement:
  development seeds 0/2/3/42: 7/19/11/9; geometry-validation 101/102: 1/8.
  All rejections are retained. These validation seeds are not a final unseen
  policy benchmark. Run: `20260911T211346_576408Z_training_layout_screen`.
- Follow-up 18-posture sweeps each found four geometric examples: development
  seed 0 at (-0.4,-0.08,0), validation seed 102 at (0.5,0.4,0), both yaw pi/2.
  Reports: `20260911T211743_796471Z_posture_witness` and
  `20260911T211311_525513Z_posture_witness`. Case 17 retains exactly the same
  feet, has minimum CAT cover gaps 0.07043/0.03652 m and nonlocal capsule-pair
  clearance 0.06144 m. Route existence is geometric, not a dynamic certificate.
- Added a read-only video renderer with source/witness hashes, exact CAT and
  terrain meshes, display-tessellated imported capsules and independently
  recomputed cover gaps. Renderings are labeled kinematic, NOT policy output.
  The first rendering was interrupted (exit 143); it is not a delivered demo.
- The 198-test suite passes. No new physics/controller execution, optimizer
  step or robot access in this milestone. Training readiness remains false.
- Stop for user environment review. Still required afterward: witness-backed
  challenge admission and reset integration, live stochastic collector checks,
  approval-bound pilot launcher, and approved update/resume validation.

## 2026-09-12 — bounded learner engine and simulator-bridge preparation

- Added on-policy residual collection, sampled/executed latent pairing, separate
  terminal/timeout bootstrap, clipped PPO iteration, gradient limits, KL stop,
  finite checks and stale-behavior rejection. Default update mode computes and
  clears gradients without optimizer steps; no automatic learning entry point.
- Added an opt-in simulator bridge with a frozen decoder clone, pre-reset
  capture, per-environment history clearing and wrapper-recovery mismatch
  rejection. It is tested with a fake simulator, not live stochastic Isaac.
  Existing frozen evaluation commands are not rerouted through it.
- Extended oracle/state queries to select surviving environments before
  physical quaternion checks. Invalid terminal observations never enter value
  evaluation; invalid ongoing/timeout inputs still reject collection.
- Hardened checkpoint schema v2 with optimizer type/name ordering, fixed
  hyperparameters, moment/counter checks and validation before live mutation.
  Nonempty moment fixtures roundtrip; real post-update resume remains untested.
- All 193 tests pass, including full teacher-dimension CUDA gradients and
  asynchronous fake-simulator resets. Optimizer calls in tests are either
  prohibited or no-op spies; **zero real optimizer steps** for this milestone.
- Frozen four-environment Isaac run `20260911T205918_208015Z_stair_p1_cat_audit`
  exited 0 with valid outputs: 499 steps, four timeouts, zero failures, unchanged
  weights, exact actor action parity, selected-batch observation error 0.0,
  next-state error 2.9802322e-7. No new stochastic actions were applied.
- Remaining: admissible challenge/retention curriculum, approval-bound launcher,
  live collector validation, bounded approved update/resume and evaluation.
  Details and reproducible test commands: `RESIDUAL_ENGINE.md`. No robot access.

## 2026-09-12 — opt-in avoidance task and rejected posture challenge screen

- Implemented a distinct M2 posture-avoidance reward/termination overlay, leaving
  original retention configuration unchanged. Upper-body tracking is relaxed;
  foot placement, root/balance, action smoothness and joint limits remain. CAT
  contact failure includes feet; no heuristic terrain-contact permissions added.
- Counterfactual tests cover per-link clearance, physical contact thresholds,
  root-only telescoping progress, true world-foot error and failure event units.
  Actual one-/four-environment frozen-action runs each completed 499 steps with
  clean timeouts and unchanged weights; independent report checks pass. Exact
  run IDs and bounds are in `TRAINING_PREPARATION.md`. This is not training or
  evidence of learned avoidance; altered-task runs are not claimed numerically
  identical to every older frozen rollout.
- Reconstructed the original reference using pinned upstream motion/MJCF with
  exact agreement to all 499 imported probe frames. Added deterministic posture
  blending, named scalar DOF mapping (excluding the upstream free root), exact
  capsule segment-distance checks and a bounded candidate placement screen.
- The fixed 108-placement sweep yielded one actual arm-intersection candidate.
  All 18 follow-up posture trials remained rejected: shifted geometry failed
  role/terrain placement, and nonlocal capsule self-pair interpretation needs
  additional work. Reports preserve rejection, not an invented feasible solution.
- All 174 tests passed with CUDA available. No optimizer steps or robot access.


## 2026-09-12 — zero-update Isaac learner state and terminal/reset integration

- Added a privileged 2,221D teacher state with all 29 joints, explicit ten-step
  physical history and ten future frames of 14 named bodies. Exact scales,
  frame conventions, reference offsets and body/joint orders are versioned.
  This is not a deployable student input and does not replace GRAIL's inputs.
- Added read-only capture after original reward computation and before Isaac
  auto-reset, with no extra physics, command advancement, observation-manager
  call or global RNG consumption. Final-state +1 reference queries and physical
  history agree with actual non-reset next states; resets clear only their own
  histories. Invalid oracle inputs fail closed, not a training recovery claim.
- Added opt-in `--residual-preflight`: original frozen actions only, exact
  zero-residual clone parity, no exploration and no optimizer updates. Static
  support-grid reuse verifies physical meshes/poses for every replica first.
- One environment `20260911T201349_917151Z_stair_p1_cat_audit`: exit 0, outputs
  valid, 499 steps, one clean timeout, zero failures. Maximum state discrepancy
  2.3841858e-7. CAT clearance samples/minima/contact peaks match the previous
  clean frozen headless run exactly. An earlier configuration lookup error
  exited before rollout and is preserved in its failed run directory.
- Four environments `20260911T201532_835046Z_stair_p1_cat_audit`: exit 0,
  outputs valid, 499 steps, four clean timeouts, zero failures. Verified physical
  replicas 1/2/3; maximum state discrepancy 2.9802322e-7. Actor and learner hashes
  unchanged. Independent recorded-data checks verify continuity, reset history,
  timeout/final separation and GAE via NumPy rather than the tensor code.
- All 156 tests pass with CUDA enabled (14 new tests). Sandbox-only run passes
  with six CUDA tests skipped; the full GPU rerun has no skips. Existing
  nonfatal imageio pipe ResourceWarnings remain in synthetic video tests.
- Reports, scope limits, data hashes and commands are in `RESIDUAL_RUNTIME.md`.
  These deterministic transitions are not PPO behavior data. Avoidance rewards,
  demanding/held-out curriculum, stochastic runtime/update/checkpoint loop,
  invalid-state handling and final training-approval gate remain unfinished.
  No real-robot access or task training occurred.

## 2026-09-11 — review videos and residual learning components (no training)

- Added `baseline.py --record-video`: single-environment headless frozen-policy
  recording, separate from GUI/training. Pins 1280x720/25 fps, readable opt-in
  captions and hidden reference debug markers. Recorder output is run-local;
  HDF5 export is disabled rather than using Isaac's shared `/tmp/isaaclab` file.
  An initial CLI syntax failure and a shared-default-path failure were retained
  in their own runs; the failed owned simulator was terminated. Other users'
  temporary data was not modified.
- Camera run `20260911T193059_716652Z_stair_p1_cat_audit` exited 0, outputs valid:
  249 decoded video frames, 499/499 valid obstacle packets, exact zero-residual
  action parity and unchanged actor/adapter weights. No failure termination;
  minimum sampled CAT cover gap 0.404890 m. This is the original controller
  descending stairs beside sparse CAT lateral fixtures, not learned avoidance.
- Added checksummed offline MP4 visualizations of the corrected guidance and
  3D/per-body obstacle inputs. Before/after uses the identical saved pelvis
  trajectory and validates source hashes; 334/499 -> 499/499 valid frames.
  Both diagnostic videos have 300 decoded frames / 12 s, with one-second holds.
  Camera video is a separate evaluation, not frame-synchronized with them.
- Three labeled videos, previews, descriptions and provenance are packaged at
  `~/Desktop/GRAIL_CAT_review_20260911_v2.zip` (~7.2 MiB). Archive integrity and
  full MP4 decoding checked; sampled camera frames visually inspected. No
  Viser/web listener or firewall change. See `REVIEW_DEMOS.md`.
- Added tensor-only residual actor/obstacle-aware critic, pre-tanh stochastic
  latent contract, pure clipped-PPO loss, reset-aware GAE, detached bounded
  rollout storage and contract-bound learner/RNG checkpoint helpers. The
  existing shadow adapter retains its state keys and zero-residual behavior.
  Parameters receive test gradients but **no optimizer steps** were applied.
- Tests verify exploration RNG isolation/replay, zero deterministic mean,
  invalid-input rejection, timeout-versus-terminal bootstrap, target detachment,
  clipping signs, time/env ordering, no-overwrite saves and zero-update checkpoint
  roundtrip. Checkpoint resume resets simulator episodes; no exact PhysX-resume
  claim or post-update optimizer-resume validation yet. Core commit `2d20ae2`.
- All 142 tests pass, including 19 new tests. A nonfatal imageio/ffmpeg pipe
  ResourceWarning appears during synthetic video validation; decoded outputs
  pass. The owned simulator processes exited; no real robot was accessed.
- Remaining before the first approved learning pilot: integrate the runtime
  update loop and pre-reset final observations, avoidance-specific rewards and
  terminations, genuinely challenging/held-out layouts, approval-bound launcher,
  and 1-/4-environment no-update preflight. **Training has not started.** Full
  preparation remains in progress; tracked in `TRAINING_PREPARATION.md`.

## 2026-09-11 — checked pelvis-to-support guidance connections

- Fixed the new guidance sampler's support/transit mismatch. The moving pelvis
  is a temporary transit root with bounded, fully cell-checked connections to
  existing goal-reachable support nodes. No changes to support masks, graph
  edges, geometry, actor inputs, rewards, policy weights or control source.
- Closed segment/cell intersection includes corner touches and boundaries;
  preserves 28 cm stride and 20 cm height-excursion bounds. Explicit unknown,
  blocked, out-of-grid and unsupported-target rejection. Shortest total cost
  with deterministic lookahead ties avoids steering back into the current cell.
- Versioned packet/connection semantics and recorded connection witnesses;
  old packets are preserved, not relabeled. New offline comparison verifies
  source checksums, CPU/CUDA and GPU batches 1/16, and every accepted connector
  with a separate scalar intersection implementation.
- Development `20260911T190949_101780Z_pelvis_guidance_audit`: recovered all
  165 formerly invalid stair frames, 499/499 now valid, no lost valid frames;
  all 499 independent connector checks passed. Maximum XY connection 0.278362 m,
  maximum CPU/GPU packed-feature error 2.03e-6. All 123 tests pass, including
  ten new connector tests and an updated blocked-source regression.
- Clean `8ab40fa` comparison `20260911T191130_206279Z_pelvis_guidance_audit`
  reproduced all 165 recoveries and 499 independently checked connectors.
  Clean Isaac run `20260911T191127_746866Z_stair_p1_cat_audit` exited 0 with
  valid outputs and 499/499 valid packets. All 499 action-parity checks passed;
  backbone/adapter hashes unchanged, head gradient norm 0.11701956. Clearance
  samples/minima, contact counts/classifications and the complete contact
  payload hash exactly match the earlier no-shadow reference. Recorded target
  cells and crossed-cell counts match the independent comparison exactly.
- Full packet replay at the same clean revision passed: CUDA
  `20260911T191305_276925Z_observation_replay` has exact feature equality; CPU
  `20260911T191308_344564Z_observation_replay` maximum errors are 1.78814e-7 /
  1.49012e-7 / 2.02656e-6 for volume/probes/guidance, below unchanged 2e-5.
  Validity masks match. Shadow sampling averaged 9.85 ms, maximum 32.64 ms;
  not an end-to-end real-time benchmark.
- The owned simulator exited; no robot, ROS or real-motion changes. This fixes
  the recorded guidance validity issue, not avoidance learning or full-body
  feasibility. Scope, limitations and commands: `PELVIS_GUIDANCE.md`.

## 2026-09-11 — obstacle observation interface and zero-residual shadow adapter

- Added yaw-only, pelvis-centred 3D oracle samples: 13 x 13 x 11 at 16 cm,
  signed CAT and unsigned terrain distances with separate masks. Additional
  per-collider-cover features follow all 104 imported probes through articulation.
  Goal/support-height delta, support-graph direction and cost have explicit
  validity/goal flags. No new signed terrain or sensor-free-space claim.
- Added a separate small 3D/probe encoder with zero-initialized 64D latent head,
  bounded at 0.1 per coordinate. Reuses the existing post-quantization residual
  argument on a frozen actor clone only. The original actor remains the sole
  source for `env.step`; no trained adapter loader, optimizer, rewards, ROS,
  robot connection or real motion. Bound is not a hardware safety guarantee.
- New headless `--observation-shadow` requires the existing single-environment
  physical layout/reference preflight. It captures before each actual policy
  step and pairs outcomes through terminal reset. Tests require exact cloned /
  original / zero-residual action parity, unmodified observations and RNG,
  unchanged backbone/adapter weights and a nonzero finite gradient to the new
  head through the frozen decoder. No gradient update is applied.
- Development run `20260911T184716_604447Z_stair_p1_cat_audit`: exit 0, outputs
  valid, 499 exact action-parity samples; head gradient norm 0.116968. Backbone
  and adapter hashes unchanged. Geometry masks all known; 334 valid guidance
  frames and 165 explicitly invalid ones. All 165 correspond to unsupported
  small-patch cells below the pelvis, not missing terrain or blocked transit.
  Do not use this hard gate as a final continuous-control design.
- This run reproduced the prior no-shadow random-scene run's sampled clearance
  sequence and minima exactly, plus 499 / 1,996 policy/physics steps and 217,710
  contact records with identical classification counts. Mean shadow processing
  was 9.32 ms, maximum 77.60 ms on this GPU; not an end-to-end real-time claim.
- 112 tests pass (15 new), including CPU/CUDA/frame parity, invalid masks,
  support-vs-clutter channels, zero/RNG/gradient/bounds, launcher restrictions,
  saved action tamper detection and terminal pairing. No new runtime conflicts.
- Clean `762ce22` run `20260911T185232_282806Z_stair_p1_cat_audit` confirmed
  all 499 action checks and raw probe-centre capture. Its clearance samples,
  minima, contact classification counts and contact payload hash exactly match
  the earlier no-shadow regression. Offline CPU/CUDA geometry replay then
  correctly rejected a maximum normalized volume error of 0.0002823174.
- Diagnosed the replay discrepancy: the upstream evaluator enables TF32, which
  affected our new small `einsum` coordinate transforms; probe transforms also
  changed with batch size. Replaced only observation-coordinate transforms
  with explicit float32 products, leaving all actor precision settings intact.
  A new non-axis-yaw regression failed before the fix and exercises CPU/CUDA,
  TF32 on/off and batches 1/16. The 2e-5 replay tolerance was not relaxed.
  Prior recordings are retained, not regenerated in place.
- Clean `200144e` confirmation:
  `20260911T185640_513446Z_stair_p1_cat_audit` exited 0 with valid outputs,
  499 exact action-parity frames, unchanged weights and head-gradient norm
  0.11696785. Its complete contact payload hash, classification counts and
  sampled clearance sequence/minima exactly match the no-shadow reference.
  Shadow sampling averaged 9.01 ms, maximum 34.24 ms; not an end-to-end timing
  or avoidance-performance benchmark.
- Independent replay of all 499 frames passed at clean `200144e`:
  CUDA `20260911T185752_319799Z_observation_replay` reproduced every feature
  exactly; CPU `20260911T185800_161438Z_observation_replay` had maximum errors
  1.78814e-7 / 1.49012e-7 / 5.96046e-8 in volume/probe/guidance normalized
  channels. Validity masks match exactly on both devices. 113 tests pass.
- Next: bounded transit-to-support guidance for the moving pelvis (334/499
  frames valid remains unchanged), genuinely avoidance-demanding placements,
  and reviewed contact semantics before a bounded residual-learning trial.
  The owned simulator exited. No policy training or robot changes in this
  milestone. Commands and limitations: `OBSERVATION_SHADOW.md`.

## 2026-09-11 — random role provenance and terrain-aware diagnostic guidance

- Added original-output-exact random CAT tracing without editing pinned source,
  changing random draws or modifying geometry. Preserve source memberships;
  morphology additions get a role only with unambiguous single-role dependency.
  Mixed/padding dependencies remain unresolved. Traces are checksummed and
  fully replayed before role-based placement. Old untraced scenes stay readable
  as geometry but still cannot pass role-based placement.
- Added shared bounded support graph and offline `terrain_guidance.py`: goal
  graph distance, next node and world-XYZ direction on known support patches.
  Swept transit, trunk clearance, riser/stride bounds and no corner cutting;
  explicit NaN/false/-1 for unreachable cells, zero direction at the goal.
  No policy input/reward/command change or full-3D/whole-body feasibility claim.
- Development batch `20260911T183023_271089Z_random_clutter_audit` tested fixed
  seeds 0/1/42 at difficulty 0.2 across three recipes, without resampling.
  Original-count scenes all reject ambiguous roles (465/917/395 voxels).
  Sparse lateral seeds 0/42 resolve, seed 1 rejects (224 voxels). Floor-only
  outputs are empty after original CAT morphology and reject. Four of six
  sparse placements pass cached geometry/role/reference screens; two fail
  rigid grounding at the stair footprint. These are controls, not evidence
  of a learned avoidance response or physical mounting validity.
- Guidance on the retained fixed-scene control and all four passing sparse
  placements gives 12-node routes, 2.610522 m graph distance, support heights
  0..1.244169 m. Occupancy affects reachable area but these trials do not require
  a route change from the reference corridor. Original CAT fields are untouched.
- 97 tests pass (13 new): independent Python/sparse graph distance parity,
  holes/cliffs/walls, invalid goals, resource bounds, world-grid/hash checks,
  source parity, morphology replay, immutable upstream globals and trace tamper
  rejection. All 18 baseline artifacts verify; no unexpected runtime conflicts.
- Clean `60c86e3` confirmation batch
  `20260911T183236_696188Z_random_clutter_audit` reproduced all nine case outcomes
  and four passing placements with `complete=true`, `dirty=false`. Independently
  checked all trace hashes and the four guidance exports' hashes, descending
  costs, goal behavior and invalid masks.
- Clean headless random-scene regression
  `20260911T183246_963741Z_stair_p1_cat_audit` used sparse seed 0 at (0,0.2,0),
  yaw pi/2. Exit 0, outputs valid, completed without failure; all 96 physical
  support rays passed. Recorded 499 policy / 1,996 physics steps including
  four terminal samples, minimum sampled CAT cover clearance 0.353909 m and
  zero CAT contact force across measured links. 217,710 contact records;
  provisional classifications still include 340 riser/side and 17 swing-support
  mismatches, not cleared or converted into permissions. This is a reference
  regression with nonblocking clutter, not a newly learned avoidance result.
- Commands, assumptions and remaining gates: `RANDOM_CLUTTER_GUIDANCE.md`.
  Owned simulator exited; no robot, ROS or host-controller changes.

## 2026-09-11 — controlled articulated contact fixtures and review queue

- Added nine headless physical fixtures with the imported 29-joint G1, measuring
  all collision-bearing links against every fixture. Tread/rotated tread, injected
  swing/unknown labels, forefoot/riser, shin/riser, underside, CAT-role box and
  no-contact control are checked. Zero-gravity/imposed-root diagnostic only;
  no policy, robot connection, actuation, observation/reward or driver changes.
- Shared actual capsule-derived sole bounds with the rollout audit. No clearance
  cover radius is substituted for the real collider. The former rollout sole
  derivation was moved unchanged, with added finite/dimension rejection.
- Initial `20260911T154903_971568Z_contact_fixtures` rejected two poor fixture
  placements. Corrected the pitched-shin face alignment and isolated the
  forefoot underside patch; no filters/forces were suppressed. The next run
  `20260911T175139_905885Z_contact_fixtures` exposed arbitrary SDF-gradient
  selection at a box edge. Exact outward normal-cone checking resolves it;
  inward/tangent/off-surface normals still reject. Open stair normals unchanged.
- `20260911T175353_451267Z_contact_fixtures`: all nine passed; 742 approach
  physics steps, 407 force-bearing records, no unintended pairs. Peak force
  reconstruction error 0.0000229 N. Independent saved geometry/label/count/hash
  validation passed. Test phases are injected scenarios, not reference truth.
- New offline `review_contacts.py` verifies the prior capture before producing
  immutable-source review JSON/Markdown. Stair contact records group into four
  riser/side intervals, one 20 ms swing/support mismatch, one initial unverified
  normal interval and 22 phase-unknown intervals. None is auto-labeled for RL.
- 84 tests pass (12 new, including adversarial saved-data checks), with CPU/CUDA
  parity exercised. Environment check has no new conflicts. See
  `CONTACT_FIXTURES.md` for exact commands, scope and remaining gates.
- Clean `8b870fa` confirmation: `20260911T175656_637177Z_contact_fixtures`
  reproduced all nine passing cases / 407 records with `dirty=false` and saved
  artifact verification. Review run `20260911T175658_037134Z_contact_review`
  reproduced interval counts. Stair regression
  `20260911T175754_038972Z_stair_p1_cat_audit` exited 0 with valid outputs and
  `dirty=false`: 1,996 physics samples, identical sole regions/classifications
  and identical contact NPZ checksum to the prior clean replay. All test
  simulators exited; no new training or robot action.

## 2026-09-11 — articulated contact capture before reset

- Added headless `--contact-audit`, requiring the existing one-environment
  layout/reference preflight. Instance-local context-managed `scene.update`
  observer reads each physics substep before RL automatic reset, without extra
  physics or observations. No global/dependency patch or controller change.
- Added independent raw PhysX view for all 14 imported collision-bearing links
  versus terrain, ground and CAT. Six ragged buffers are validated against their
  counts/offsets and independent aggregate-force matrix. Saturation, overlap,
  nonfinite data, ambiguous names and changed filter ordering fail explicitly.
- Initial run `20260911T153503_445016Z_stair_p1_cat_audit` rejected before the
  policy loop because the runtime returns leaf names, not full prim paths.
  Fixed explicit identity binding and added regression coverage; no normal
  flipping, contact dropping or physics changes to make diagnostics pass.
- Successful development run `20260911T153625_910820Z_stair_p1_cat_audit`:
  strict actor load, completed without failure, minimum CAT cover gap 0.09351 m.
  Captured 499 policy steps / 1,996 physics steps, including four terminal steps
  with 43 force-bearing point records formerly missed by post-reset observation.
  217,451 total contact records, 203,515 at/below 0.1 N; reconstruction error
  <=0.0000916 N. No CAT or non-foot terrain/ground force measured.
- Conservative geometry/phase classification found 8,541 stance-support
  candidates, 5,045 phase-unknown records, 327 foot-side/riser records, 20
  swing-support candidate mismatches and three initially unverified normals.
  Point records are not independent events. Candidate phase comes from the
  reference, not actual forces; do not call it verified contact truth or use it
  blindly as a reward. Foot-edge/phase review is now a concrete next gate.
- Sole bounds use the actual imported capsule radii/endpoints and articulated
  WXYZ body transforms. Terrain/ground/CAT remain separate paired surfaces.
  No collision exemption or contact permission is granted by this diagnostic.
- Added independent saved artifact validation: hashes, finite payloads, index
  ranges, classifications, per-step counts, contiguous counters and complete
  terminal/decimation alignment. Parent launcher requires this for valid output.
- 72 tests pass (11 new). No new packages/unexpected environment conflicts,
  GUI, robot connections, ROS edits, training or actuation. Detailed contract,
  measured findings and remaining steps: `ARTICULATED_CONTACTS.md`.
- Clean-commit confirmation at `8be37fc`:
  `20260911T154050_843634Z_stair_p1_cat_audit` recorded `dirty=false`, exit 0
  and valid outputs. It reproduced 1,996 physics steps, four terminal steps,
  217,451 contact records, all classification counts above and the 0.09351 m
  minimum CAT cover gap. All owned simulation processes exited. This confirms
  capture integrity, not verified foot phases or collision-free certification.

## 2026-09-11 — feature-preserving placement and full-height oracle fields

- Added explicit floor/lateral/overhead replay for 20 pinned CAT fixed-scene
  recipes, requiring their union to match cached occupancy exactly. Connected
  hurdles/posts retain separate roles. Unknown/random role provenance rejects
  explicitly; the original generator and scene files remain unchanged.
- Added optional rigid grounding from all footprint cell centres/corners.
  Uneven/missing/downward support rejects rather than warping or burying clutter.
  Separate role-retention checks preserve datum and exposed feature height;
  overhead fixtures are explicit, not silently converted to support.
- Added exact **unsigned** distance to open terrain/ground and full-height
  simulation-oracle feature volumes, alongside exact signed CAT distances.
  Treads/risers stay present in the surface band. There is no fabricated terrain
  inside/outside, free-space certificate, sensor-observation claim or extended
  flat CAT guidance. Exact mesh distances remain the reference-clearance gate.
- New offline tool `research/compose_scene.py`; source/output hashes, placement,
  source revision/dirty state, runtime versions, interpolation errors and
  rejection reasons are retained. Contract/results: `SCENE_COMPOSITION.md`.
- Original displayed scene: full reference field coverage increased from
  74.4605% to 100% (51,896 samples; 56 x 74 x 72 cells). Maximum discrepancies
  from exact queries: <1.95 cm CAT and <1.71 cm terrain, at 4 cm resolution.
  New retention gate rejects the buried hurdle despite the prior path passing.
- Grounded landing placement (0,0.2,0), yaw pi/2: 354 footprint queries, zero
  support spread, roles retained, but 70 reference frames violate the 3 cm
  cover margin (minimum gap -0.0430 m). Correctly exported as rejected, exit 2.
  This separates valid scene geometry from an unchanged tracker failing to
  negotiate the challenge. Clear control beyond the endpoint (0,1.2,0) passes
  the cached screen, exit 0; it is not a hurdle-crossing result.
- 61 tests pass (nine new), including CPU/CUDA parity, unsigned open-surface
  normals, missing support, role union/tamper rejection, grounding, full sphere
  extent coverage, and unknown/XYZ interpolation. One initial test used an
  overly strict decimal assertion for float32 ray height; it now uses 1e-6 m.
  No new dependencies, environment conflicts, GUI/simulation launches, policy
  training, ROS changes or robot connections. `avoidance_training_ready=false`.
- Clean-commit confirmation at `a606d3c`: all three offline compositions
  reproduced the expected reject/reject/accept outcomes and 100% reference
  coverage with `adapter_dirty=false`. Runs:
  `20260911T152752_420279Z_composed_scene` (buried hurdle),
  `20260911T152753_322907Z_composed_scene` (grounded/reference conflict),
  `20260911T152754_494837Z_composed_scene` (clear control beyond endpoint).

## 2026-09-11 — combined terrain/support/passage validation

- Stopped the user's repeating stair/CAT viewer; no GUI was left running.
- Added exact enabled-collider extraction at the live terrain rigid-body pose,
  preserving uniform scale and rejecting units, mirroring, unsupported geometry
  and moving terrain/reference mismatches. The released stair mesh has 40,000
  triangles and is open; support uses rays, not a fabricated signed volume.
- Added upward support/patch screening, bounded step/stride graph edges with
  intermediate support/trunk checks, endpoint projection bounds, existing
  sampled whole-body reference clearance, and buried/floating component review.
  New CLI: `baseline.py --cat-scene ... --layout-audit`; offline cached
  candidate checker: `research/layout_audit.py`. See `SCENE_VALIDATION.md`.
- The initial adjacent-cell graph incorrectly disconnected tread patches across
  riser-edge bands. Corrected it to bound stride and check intermediate transit
  separately from stance-patch validity. Tests reject holes, cliffs, diagonal
  corner cutting, sealed corridors and a visible-but-too-narrow 0.30 m opening.
- PhysX callback tests exposed a typed `RaycastHit` API (not the dictionary used
  by `raycast_closest`). More importantly, world-space CPU queries missed the
  GPU-positioned stair: CPU pose remained (2,0,0)/identity while the live GPU
  pose was (0,0,0.6291) with its paired rotation. A physics-only warmup/reset
  did not fix this and was removed. NVIDIA documents this Direct GPU API
  limitation. Reconcile the two ray frames against the same cooked collider;
  static ground is queried separately. No pose, policy input or extra step is
  changed in the final implementation. Intermediate rejected runs are retained.
- Development run `20260911T150945_632435Z_stair_p1_cat_audit` passed: all 96
  PhysX/cooked-geometry rays (30 hitting terrain above ground) agree within
  0.0000117 m. Support graph finds a 2.6105 m route with 12 nodes; all reference
  pelvis samples have support within the declared height bounds. These are
  geometric screens, not foot-contact/balance or continuous-path certification.
- Its unchanged-policy rollout finished: 498 sampled batches, no failure
  termination, minimum CAT cover gap 0.0935 m, measured CAT normal force 0 N.
  Exit 0 and output audit passed. Offline conflicting overhead placement
  `20260911T150536_192407Z_layout_check` rejects 426/499 reference frames.
- The approved passage has 13.84% of occupied voxel centres wholly below the
  terrain support envelope under the diagnostic tolerance, and no floating base
  columns. This is flagged for placement review; no hurdle difficulty credit or
  silent geometry correction is applied. `avoidance_training_ready` stays false.
- 52 unit tests pass, including CPU/CUDA ray parity, live/authored/CPU query
  rotations, scale retention, inverted normals and gap rejection. Environment
  audit reports no unexpected dependency conflicts. No new dependencies,
  robot connections, ROS changes, policy training or real actuation.
- Clean-commit confirmation at `1f2f375`:
  `20260911T151123_875346Z_stair_p1_cat_audit` exited 0 with validated outputs,
  accepted layout, 498 sampled batches and no failure termination. All launched
  simulators have exited; the GUI remains stopped.

## 2026-09-11 — live stairs with CAT clutter

- Enabled `baseline.py --gui --cat-scene ...` for a repeating desktop demo.
  Headless audits retain their one-episode/500-step bound. Both modes retain
  the reference-clearance rejection gate; training and primitive-audit mixing
  remain prohibited. The released actor and physical scene are unchanged.
- Added a GUI legend, camera-reset button, readiness record and actual viewport
  capture. The first episode's diagnostic is persisted immediately; further
  replays are explicitly not audited and do not accumulate diagnostic samples.
- Development run `20260911T145325_647000Z_stair_p1_cat_gui_demo` loaded the
  original stairs with `side-hurdle2`, placement (0,-1,0), yaw pi/2, one G1.
  Inspected `overview.png`: the robot, stairs and CAT side obstacles are visible
  alongside upstream yellow height-scan and reference-axis debug markers.
- The first episode completed without failure: 498 sampled batches, minimum
  sampled cover gap 0.0935 m and no measured CAT normal contact force. This is
  unchanged-reference tracking, not learned obstacle avoidance. Terminal
  contact/coverage limitations from the integration audit still apply.
- 38 unit tests and syntax/diff checks passed. Viewer intentionally remains open
  for the user, bounded to one hour; close Isaac Sim to stop sooner. No real
  robot connection, ROS changes, actuation or training was performed.

## 2026-09-11 — CAT/GRAIL reference and physical-mesh integration

- Added explicit CAT-to-terrain translation/yaw, per-environment physical mesh
  spawning, rotated voxel-field queries, and strict unknown masks. No actor,
  observation layout or reward changes. No robot access or new training.
- Added Warp signed-distance queries to the exact closed CAT triangle mesh,
  conservative imported-G1 cover checking at every stored reference frame, and
  a 3 cm minimum-gap rejection gate before the policy rollout. Stored reference
  sweeps are checksummed and reusable for offline candidate-placement checks.
- 37 unit tests pass, including CPU/CUDA mesh-distance parity, vector rotation,
  unknown handling, limb-specific rejection, source tamper rejection, and opt-in
  launch configuration. Equal-distance surface normals can differ at medial
  axes; tests retain distance parity and test normals only where unique.
- Development run `20260911T144115_853042Z_stair_p1_cat_audit`: original stair
  reference plus random CAT clutter translated 3 m sideways, two environments,
  498 sampled batches, 104 probes, no measured CAT contact. Minimum actual gap
  2.801 m. The CAT field is correctly invalid there (0% coverage), not free.
- Deliberately conflicting combined clutter at translation (-1,0,1): runtime
  `20260911T144242_845662Z_stair_p1_cat_audit` refused before any policy-loop
  steps; 426/499 reference frames flagged, minimum sphere gap -0.167 m. Offline
  `20260911T144246_233341Z_cat_reference_check` reproduced the rejection.
- A wider original CAT `side-hurdle2` passage (no geometry edits), yaw pi/2 and
  translation (0,-1,0), passed the reference screen: minimum gap 0.0974 m. The
  two-environment physical run `20260911T144407_165784Z_stair_p1_cat_audit`
  completed 498 sampled batches, minimum actual gap 0.0948 m, zero measured CAT
  contact. Field coverage of reference probes was 74.46%, not silently extended
  above CAT's fixed-height volume. This is not learned obstacle avoidance.
- Added explicit failure-versus-timeout outcome reporting. Clean `937ffbb`
  confirmation `20260911T144627_656824Z_stair_p1_cat_audit` exited 0: both
  environments reached reference timeout without failure termination, 498
  sampled batches, minimum actual gap 0.0948 m and zero measured CAT contact.
  Metadata audit reports no unexpected conflicts. All launched runs exited.
  Documentation and remaining support,
  reference/task-direction and terminal-contact gates: `CAT_GRAIL_INTEGRATION.md`.

## 2026-09-11 — interactive CAT clutter gallery

- Generated fresh scenes with unchanged pinned CAT code: random seed 42,
  difficulty 0.2 (`20260911T142935_132257Z_cat_scene_random`, 16,966 occupied
  cells), and the typical combined scene
  (`20260911T142936_467194Z_cat_scene_side-hurdle-crouch2`, 5,628 cells).
- Added `research/cat_viewer.py`: a bounded GUI supervisor, checksummed source
  scenes, translation-only gallery layout, static physical meshes, lighting,
  camera reset, source provenance, and automatic viewport screenshot. Start/goal
  markers are visual-only projections, not obstacles or policy inputs.
- First GUI attempt depended on an optional, unloaded window-title extension
  and failed. Removed that unnecessary dependency, caught worker exceptions,
  and require `viewer_ready.json` before accepting a successful GUI exit.
  The earlier run's zero exit code alone was not valid visualization evidence.
- `20260911T143131_626282Z_cat_gallery` loaded both meshes and wrote
  `viewer_ready.json` and `overview.png`. Visually inspected that actual Isaac
  viewport capture: random clutter and the typical passage/overhead geometry
  are visible. Development run provenance correctly records a dirty worktree.
- All 31 unit tests passed; viewer dry-run verifies scenes without starting Kit.
  The GUI is intentionally left open for inspection, with a one-hour lifetime
  limit. No robot, policy inference, avoidance training or ROS was started.

## 2026-09-11 — actual CAT generator reuse and Isaac export

- Exporter implementation committed/pushed as `568233f`. Clean confirmation
  scene `20260911T142124_665013Z_cat_scene_side-hurdle-crouch2` and physics run
  `20260911T142150_928880Z_contact_fixtures` passed 384 rays plus sphere contact.
  All 29 unit tests passed at that commit, with no generator tests skipped.
- Final unchanged-policy stair regression at clean `568233f`:
  `20260911T142229_031745Z_stair_p1_evaluation` exited 0 with strict actor
  restoration, 499 finite steps, no failure termination, and progress 1.0.
  Global MPJPE was 35.66776480923695 mm, matching the original sample baseline.
  All launched simulation processes have exited; the GUI demo remains stopped.
- Per the user's clarification, reused the original CAT generator rather than
  extending the hand-authored primitive fixtures as a training distribution.
- Pinned six unchanged source/license files to CAT commit `866ba39`; all hashes
  were verified against that Git revision. The user's CAT checkout was read-only.
- Added isolated generator fetch, original random/typical occupancy + FMM field
  generation, and static non-convex USD mesh export. Corrected exporter world/
  half-cell coordinates and closed boundary surfaces; preserved source arrays.
- Generated random seed 42/difficulty 0.2 (16,966 occupied cells), `side0`
  (4,560), and `side-hurdle-crouch2` (5,628). All report their source and hashes.
- `20260911T141639_670710Z_contact_fixtures`: side0 passed 384 PhysX ray tests and
  a dynamic sphere contact. `20260911T141822_060048Z_contact_fixtures`: random
  scene passed the same checks. Failed development runs remain in runs/.
- Corrected validation plumbing: explicitly enable headless PhysX scene queries,
  serialize native scalars, and select genuinely planar sphere-contact patches
  rather than assuming a point ray implies a planar sphere contact.
- Generator dependencies added only to this research environment. Restored/pinned
  ImageIO to Isaac Sim's 2.37.0 requirement; no new metadata exceptions remain.
- 29 unit tests passed. Comparing unchanged generator code in the original CAT
  environment (NumPy 2.1.3 / SciPy 1.16.3) with GRAIL (1.26.4 / 1.15.3) found
  exactly two differing occupancy cells in seed 42, out of 142,500. Source parity
  is verified, not cross-runtime bitwise parity. Cache/hash generated geometry
  for matched experiments; do not disturb GRAIL's pinned numerical runtime.
- Clean `7532d6c` two-environment diagnostic run
  `20260911T141245_169292Z_stair_p1_clutter_audit`: exit 0, 498 sample batches,
  104 imported-shape probes, consistent physical clutter poses. Self sensors
  measured right-wrist/right-hip contact up to 17.94 N; not a collision-free claim.
- M1 is still partial: scene composition with terrain, complete articulated
  support/contact semantics, oracle field parity and learned avoidance remain
  future gates. No policy training, robot connection or actuation in this work.

## 2026-09-11 — M1 diagnostic geometry (partial)

- Stopped the requested stair GUI demo and its owned simulator process. No robot
  connection or actuation occurred.
- Added opt-in ideal primitive fields, support/contact permission logic, and
  conservative whole-body collider probes. Policy weights/inputs/rewards unchanged.
- Confirmed a CAT interpolation X/Z weight-order mismatch using the original
  methods in CPU JAX; pinned the numerical legacy fixture and tested explicit v1
  corrections. Original CAT checkout remains untouched.
- 25 unit tests passed. Seven sphere/PhysX contact fixtures passed in
  `20260911T140916_411053Z_contact_fixtures` (exit 0). The earlier successful
  fixture run stalled during Kit teardown; stopped only that owned process and
  made the fixture runner exit after its report is flushed.
- `20260911T140707_169344Z_stair_p1_clutter_audit`: exit 0, strict actor restore,
  498 finite samples, 104 probes over actual imported capsules/sphere, 14 links,
  six physical side-clutter fixtures. Minimum added-clutter clearance ~0.439 m;
  no measured clutter contact. This is not an avoidance result.
- Added independent self-contact sensors and physical scene-pose consistency
  checks after that run; confirmation pending. A -0.052 m sphere-cover self-gap
  in the initial run is a broad-phase warning, not evidence of actual collision.
- User clarified that training clutter must come from CAT's actual generator.
  Our primitives remain validation fixtures; next implementation adapts CAT's
  simulator-independent occupancy generation into physical USD/PhysX geometry.

## 2026-09-11 — M0 started

- Forked NVlabs/GRAIL into Skvayzer/GRAIL.
- Created `research/grail-cat-terrain`, retaining upstream main unchanged.
- Recorded source/model/data pins and the simulation-only boundary.
- Found a separate existing desktop Isaac Sim 5.1 / Isaac Lab 2.3.2 installation;
  it will be inspected read-only and will not be upgraded or modified.
- Checkpoint restoration, physical rollouts, and training are **not yet verified**.

### Pinned assets and tooling verified

- Downloaded/verified 18 official artifacts: terrain checkpoint/config and one
  paired robot/object/scene sample each for stairs, curb, slope, and sitting.
- Checkpoint CPU audit passed: 43 finite actor tensors, 20,624,570 tensor elements,
  29-action standard-deviation head, 29 movable URDF joints, history length 10,
  simulation timestep 0.005 s and decimation 4 (50 Hz nominal policy).
- Six tooling unit tests passed; shell syntax and diff whitespace checks passed.
- Successfully prepared an isolated stair run, including texture-only USD fixes.
- The public stair USD refers to absent metallic/roughness textures. The manifest
  explicitly records their omission in derived copies; source hashes are intact.
- Independent environment installation is in progress. The CPU audit used the
  existing reference Python read-only; it does not validate the new environment.
- Physical policy restoration and training are still pending. No task-success
  or terrain-retention result has been established.

### Isolated installation finished; user license decision required

- Installed the separate `.venv` and a separately cloned Isaac Lab v2.3.2.
- Verified imports resolve to this fork/its dependency checkout, not the existing
  desktop SONIC checkout. PyTorch CUDA arithmetic passed on the RTX 5090.
- First physics attempt stopped before simulator startup at the Omniverse EULA
  prompt (`EOF when reading a line`). License was **not accepted** by the agent.
  The launcher now checks this gate explicitly before staging/starting a run.
- Added process-group cleanup for timed-out/interrupted simulation evaluations.
- Remaining installation audit: the inherited package snapshot has seven metadata
  conflicts (packaging, typing-extensions, NumPy, click, psutil, Starlette).
  These must be reconciled/documented before declaring the runtime validated.
  In particular GRAIL/SONIC uses NumPy 1.26.4 while Isaac Sim metadata pins 1.26.0;
  do not blindly upgrade NumPy or modify the existing environment to resolve it.
- **Next:** obtain user EULA decision, reconcile research-environment dependency
  pins, rerun the bounded stair evaluation, then a tiny PPO smoke test. No
  simulator rollout, training result, or autonomous navigation is claimed yet.

### Explicit simulator consent and runtime verification resumed

- User explicitly accepted the NVIDIA Omniverse EULA in conversation. Simulation
  may now be launched with `--accept-isaac-eula`; the default consent gate remains.
- Added an isolated runtime pin overlay for packaging, click, psutil and
  typing-extensions. This does not change the original desktop environment.
- Upstream metadata cannot all be satisfied simultaneously: SONIC requires NumPy
  1.26.4 versus Isaac Sim's 1.26.0; Isaac Lab requires Starlette 0.49.1 versus
  Isaac Sim's FastAPI 0.115.7 requiring Starlette <0.46; ONNX requires newer
  typing-extensions than Isaac Sim pins. Retain the research stack requirements,
  document these exceptions, and validate simulation separately. No FastAPI or
  remote simulator streaming service is enabled by our launcher.
- The installed metadata audit now reports exactly those three documented
  conflicts and no unexpected conflicts. Support package versions are pinned.
- Isaac Sim starts on the RTX 5090. Fixed our temp-directory isolation to use
  short, unique paths (multiprocessing AF_UNIX sockets cannot use long run paths).
- First complete stair physics rollout restored the actor strictly, finished
  the single reference with no termination, and reported 35.668 mm global MPJPE.
  However, the upstream debug-trajectory exporter then crashed because `Path`
  was not imported. Added the missing import; a clean rerun is still required.
  This is one reference clip, not a general terrain success-rate estimate.

### Baseline reproduced; bounded training tooling added

- Clean stair rerun at commit `2d4bfff`: exit 0, 499 debug steps, strict actor
  restoration, finite states/actions, no failure termination, 35.668 mm global
  MPJPE. Run: `20260911T112807_060354Z_stair_p1`.
- Curb sample also completed: no failure termination, 52.631 mm global MPJPE.
  Slope and sitting samples are being checked separately.
- Added a two-update/four-environment recommended PPO smoke command, with a new
  output directory and opt-in strict checkpoint loading. No CAT reward, obstacle
  representation, task adapter, or robot integration is being trained yet.
- Added finite-rollout evidence audit and bounded-training configuration tests.
  Nine tooling unit tests pass. Actual PPO backward/update verification pending.

### PPO update and save verified

- All four pinned sample references completed without failure termination:
  stairs 35.668 mm, curb 52.631 mm, slope 26.317 mm, sitting 16.879 mm global
  MPJPE. Finite action/state recordings and strict checkpoint restoration passed.
  These are four sample checks, not a representative benchmark or new result.
- Training initially segfaulted inside `libomni.kit.app.plugin.so`. Moving TRL
  imports did not help and was reverted. The effective fix is removing already
  consumed Hydra CLI flags from Kit's `sys.argv`, while preserving AppLauncher
  options in its parsed namespace. Added a regression test and shared this fix
  with evaluation, which otherwise has the same custom-config CLI risk.
- Run `20260911T134423_383394Z_stair_p1_training_smoke` completed two PPO updates
  over four environments / eight steps each (64 transitions). It restored the
  released actor strictly plus critic/optimizer, reset counters, changed actor
  weights, and saved a new step-2 checkpoint. Actor, critic and optimizer tensors
  were finite, with no skipped NaN-gradient update detected.
- M0's initial install/restore/rollout/backward/save gates are now demonstrated.
  A clean-commit confirmation run is next. M1 physical clutter, support/obstacle
  semantics, full-body clearance, and CAT-conditioned learning remain future work.

### Clean-commit confirmation completed

- At clean revision `68382c4`, both the two-update PPO run
  `20260911T134531_031926Z_stair_p1_training_smoke` and the released-checkpoint
  stair run `20260911T134556_310014Z_stair_p1_evaluation` exited 0 and passed
  their audits. Training changed all 43 actor state tensors; actor, critic and
  optimizer tensors remained finite. The baseline stair metric was unchanged.
- Ten tooling tests pass; all 18 input artifacts still verify; no unexpected
  dependency conflicts. Detailed scope, metrics and run IDs: `M0_RESULTS.md`.
- No training/simulator process is intentionally left running. No real-robot
  connection, ROS configuration change, actuation, or full avoidance training
  was performed in this milestone.

### Interactive stair demonstration

- Added `baseline.py --gui`: a repeating, paced live simulation using the
  released terrain checkpoint, without the one-shot evaluation exit callback.
  GUI mode is separate from benchmark metrics and from training smoke mode.
- Launched and visually verified the G1 and stair geometry in Isaac Sim 5.1.0.
  First GUI renderer startup took roughly 90 seconds; it then loaded the actor
  and began the stair sequence. The window is intentionally left open for the
  user, with a one-hour wall-clock limit and close-window exit handling.
- Eleven tooling tests pass. This remains desktop-only, without robot access.
