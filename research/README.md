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
