# Progress

## 2026-09-11 — actual CAT generator reuse and Isaac export

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
