# Environment review handoff — 2026-09-12

Training has NOT started. Stop for the user's environment review before
continuing challenge integration. Visual approval is separate from approval
to run optimizer updates; remaining engineering gates are listed below.

## What the new videos show

Desktop directory: `~/Desktop/GRAIL_CAT_environment_review_20260912_v2/`.

- `01_stairs_and_clutter.mp4`: same stair descent reference, original CAT lateral
  clutter, original arms versus a collision-screened arm posture side by side.
- `02_arm_clearance_closeup.mp4`: the final four seconds near the obstacle.
- `*_conflict.png`: stills at the original reference's closest approach.
- `provenance.json`: source hashes, pose case, placement and decoded video info.

These are **offline kinematic visualizations**, not Isaac physics rollouts and
not learned avoidance. The robot is drawn using imported collision primitives,
not its visual mesh. The underlying terrain/CAT meshes are the checked exports.
Five frames per reference second are rendered at 5 fps; motion duration remains
approximately real-time. The lower graph uses conservative body-cover gaps.

The original reference intersects clutter; case 17 changes arm references from
spawn, keeps root/waist/legs unchanged and passes geometric clearance checks.
There is no assertion that the frozen controller can execute that arm posture
while retaining balance. Existing actual frozen-policy Isaac video is in the
older `GRAIL_CAT_review_20260911_v2.zip`; it shows a different nonblocking control.

## Please evaluate

1. Do terrain orientation, obstacle height and traversal direction look right?
2. Is this a useful first *arm-clearance* challenge while retaining the stair
   foot trajectory? This is not yet a detour/crouching/goal-only navigation task.
3. Are CAT's synthetic lateral fixtures acceptable, including floating blocks,
   or should the initial curriculum require supported furniture-like objects?

The example obstacles are on/near the bottom landing rather than scattered on
every tread. A sampled geometric passage exists. This does not certify dynamic
passability or establish a broad curriculum. Visual review cannot certify it.

## Evidence and limits

The fixed 4 cm placement grid tests 1,683 poses per seed, six seeds total. It
found 55 arm-conflict candidates; candidates are not automatically admitted.
Two fixed placements received 18-posture searches, four successes each:

| Split / seed | Translation (m), yaw | Example CAT gap | Nonlocal self gap |
| --- | --- | --- | --- |
| Development / 0 | (-0.4, -0.08, 0), pi/2 | 0.07043 m | 0.06144 m |
| Geometry validation / 102 | (0.5, 0.4, 0), pi/2 | 0.03652 m | 0.06144 m |

Metrics use case 17; feet change by exactly 0 m. Self checks use imported
capsules with nonlocal pair exclusions, not claimed PhysX-filter equivalence.
The rendered example is validation seed 102. Validation seeds are for geometry
development, not a final unseen-policy test set. No CAT occupancy is modified.

Source report paths:

- `research/runs/20260911T211346_576408Z_training_layout_screen/screen.json`
- `research/runs/20260911T211743_796471Z_posture_witness/witness.json`
- `research/runs/20260911T211311_525513Z_posture_witness/witness.json`

Re-render without simulation or training:

```bash
.venv/bin/python research/record_posture_review.py \
  --witness-run research/runs/20260911T211311_525513Z_posture_witness \
  --case 17 --output /PATH/TO/NEW/REVIEW_DIRECTORY
```

## Still required before training

- A witness-backed challenge loader and valid initial arm/reset integration;
  do not bypass the original reference-clear gate for these new scenes.
- Live, bounded stochastic collector/reset verification in Isaac.
- An explicit approval-bound training launcher and pilot/evaluation commands.
- After training approval: tiny update, checkpoint/resume and retention checks.

The learner engine exists and 198 tests pass. These remaining items mean the
project is ready for *environment review*, not yet for an end-to-end training run.
