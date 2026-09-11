# Progress

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
