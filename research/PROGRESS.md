# Progress

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
