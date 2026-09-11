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

1. **M0 (current):** isolated installation, pinned terrain checkpoint and small
   paired scene/motion dataset, observation/action audit, baseline rollout and
   tiny training smoke test.
2. M1: physical clutter, support geometry, full-body clearance and contact tests.
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
```

`fetch` uses the committed manifest. Only maintainers preparing a *new* manifest
use `artifacts.py prepare`. The default baseline command stages a run but does
not launch physics; `--execute` starts a bounded desktop evaluation. Results and
logs go to `research/runs/`. A process exit code alone is not a passed task.

The environment snapshot is from an existing compatible desktop installation,
not a claim of matching every version in NVIDIA's installer. Core versions and
package differences must be recorded in evaluation reports. The bootstrap uses
`--no-deps` deliberately to reproduce that package snapshot; runtime validation
is still required. Build-time setuptools is constrained for flatdict.

Each run copies the model and derives its own scene. Some released USDs refer to
the authors' absolute texture paths; the manifest records relocations to the
packaged per-scene textures. Missing metallic/roughness maps are explicitly
cleared in the derived scene, never silently replaced with different physics.
Original downloaded files retain their upstream checksums.
