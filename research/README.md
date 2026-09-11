# GRAIL–CAT research branch

Desktop simulation only. No ROS, robot SDK, motor-command publishers, or robot
launchers are part of these research entry points. Existing CAT, SONIC, and robot
environments are not modified.

## Objective

Use the GRAIL terrain-aware SONIC model as the motor initialization, introduce
CAT-style 3D obstacle fields with explicit support/contact semantics, then learn
goal-conditioned whole-body control and manipulation-compatible commands.

The original tracker needs future motion/object references. Loading its checkpoint
is not autonomous navigation, and a successful stair rollout is not proof that
obstacle avoidance or manipulation works.

## Milestones

1. **M0 (initial gates passed):** isolated installation, pinned terrain checkpoint and small
   paired scene/motion dataset, observation/action audit, baseline rollout and
   tiny training smoke test.
2. **M1 (in progress):** physical clutter, support geometry, full-body clearance and contact tests.
3. M2: reference-conditioned avoidance with per-terrain retention evaluation.
4. M3: deployable goal-conditioned student without oracle inputs.
5. M4/M5: masked hand commands, payload/contact geometry, frozen-base task adapters.
6. M6: sensor realism, export parity, then separately reviewed shadow integration.

Upstream baseline: `aa31d8242ac79b11545b9e3635f73014a227bdfc`.
Isaac Lab baseline: `v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0`.
Source and model licenses are retained. GRAIL's root license restricts use to
non-commercial research/evaluation; do not relabel this fork as unrestricted.
Large checkpoints/data are fetched by hash, never committed.

Progress and actual test results are recorded in `PROGRESS.md`. Do not infer a
completed milestone from the existence of a launcher or configuration.
The verified four-sample baseline and two-update training smoke results are in
[`M0_RESULTS.md`](M0_RESULTS.md). CAT obstacle-avoidance training is not implemented yet.
Geometry contracts and outstanding gates are in [`M1_GEOMETRY.md`](M1_GEOMETRY.md).
The actual CAT clutter-generator reuse and Isaac USD export are documented in
[`CAT_SCENES.md`](CAT_SCENES.md).
The explicit mesh placement, sampled-reference rejection and GRAIL diagnostic
rollout are documented in [`CAT_GRAIL_INTEGRATION.md`](CAT_GRAIL_INTEGRATION.md).
Combined terrain/support/passage screening for new layouts is documented in
[`SCENE_VALIDATION.md`](SCENE_VALIDATION.md).

## Desktop commands (from repository root)

```bash
bash research/bootstrap.sh
uv run --no-project --python 3.11.16 --with usd-core==26.3 python research/artifacts.py fetch
git lfs pull --include='imports/SONIC/gear_sonic/data/assets/robot_description/**' --exclude=''
.venv/bin/python research/artifacts.py verify
.venv/bin/python -m unittest discover -s research/tests -v
.venv/bin/python research/checkpoint_audit.py
.venv/bin/python research/baseline.py --family stair_p1
.venv/bin/python research/baseline.py --family stair_p1 --execute
# Training plumbing only: 2 updates x 8 steps x 4 environments.
.venv/bin/python research/baseline.py --family stair_p1 --num-envs 4 --training-smoke --execute
```

`fetch` uses the committed manifest. Only maintainers preparing a *new* manifest
use `artifacts.py prepare`. The default baseline command stages a run but does
not launch physics; `--execute` starts a bounded desktop evaluation. Results and
logs go to `research/runs/`. A process exit code alone is not a passed task.

First launch requires the user's acceptance of the [NVIDIA Omniverse
license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html).
Read it and, only if you agree, add `--accept-isaac-eula` to the `--execute`
command. Acceptance is never enabled automatically by these scripts.

The environment snapshot is from an existing compatible desktop installation,
not a claim of matching every version in NVIDIA's installer. Core versions and
package differences must be recorded in evaluation reports. The bootstrap uses
`--no-deps` deliberately to reproduce that package snapshot; runtime validation
is still required. Build-time setuptools is constrained for flatdict.

`env/runtime-pins.txt` overlays compatible support-package versions on the
snapshot. `check_environment.py` rejects unexpected active dependency conflicts;
three exact upstream metadata conflicts are documented in `PROGRESS.md` (NumPy,
typing-extensions, FastAPI/Starlette). No web/streaming service is enabled. A
passing metadata audit is not a physics or training validation.

Each run copies the model and derives its own scene. Some released USDs refer to
the authors' absolute texture paths; the manifest records relocations to the
packaged per-scene textures. Missing metallic/roughness maps are explicitly
cleared in the derived scene, never silently replaced with different physics.
Original downloaded files retain their upstream checksums.

## Evidence and the training smoke test

For a repeating live Isaac Lab window using the released stair policy:

```bash
.venv/bin/python research/baseline.py --family stair_p1 --gui --execute --timeout 3600
```

First use still requires the license acceptance described above. `--gui` disables
the one-shot metrics callback, repeats the reference after episode reset, and
paces execution to no faster than real time. Close the Isaac Sim window to exit;
the launcher also enforces the supplied wall-clock timeout. GUI demos do not
produce benchmark metrics and cannot be combined with `--training-smoke`.

Each run retains its configuration, package versions, process log and audit JSON.
Evaluation validates finite trajectories and strict actor restoration separately
from the reference clip's success/failure. `trajectory.json` records environment
zero; the metrics contain all evaluated references.

`--training-smoke` warm-resumes actor, critic and optimizer into a **new** run
directory, resets training counters, and runs exactly two PPO updates. It opts in
to strict actor-key loading. It uses the original tracking task, not CAT rewards.
The audit requires matching actor shapes, finite actor tensors, changed weights,
and step 2 in the saved checkpoint. Nothing overwrites the downloaded baseline.
The resulting tiny-run model is disposable and **must not be deployed**.

Short private temporary directories are retained at `/tmp/grail-*`, with a link
from each run's `tmp` directory. They avoid both cross-user log collisions and
UNIX socket path-length failures; do not delete another run's temporary data.
