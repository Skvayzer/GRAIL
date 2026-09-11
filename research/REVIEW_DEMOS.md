# Laptop review videos — 2026-09-11

Three H.264 MP4s, 1280 x 720 at 25 fps. No browser server, Viser installation,
firewall change, robot connection, optimizer update or new trained policy.

Desktop bundle: `~/Desktop/GRAIL_CAT_review_20260911_v2/`.

1. `01_isaac_frozen_stairs_cat.mp4` (~10 s): actual headless Isaac Lab physics,
   released GRAIL controller descending the stair reference, original CAT sparse
   lateral clutter. Not kinematic reference playback. Adapter is shadow-only.
   Debug markers are hidden and captions identify the control source.
2. `02_guidance_before_after.mp4` (12 s): an earlier recorded, checksummed
   frozen rollout with old versus corrected pelvis guidance. Green support
   nodes, amber transit cells, current pelvis and checked white connector.
   Recovers 334/499 to 499/499 valid guidance frames. This is an offline
   diagnostic visualization, not an avoidance-policy rollout.
3. `03_whole_body_observations.mp4` (12 s): recorded oracle distance slices,
   articulated collider-cover centres and per-link clutter clearance. CAT is
   signed; terrain is unsigned. Numerical observation distances are clipped at
   their configured 2 m scale. This is not simulated LiDAR or a policy result.

Videos 2/3 use the previous clean headless recording. Video 1 is a separate
camera-enabled evaluation; do not compare frame positions as synchronized
replays. Both individually passed action-parity and geometry validity checks.
Rendering enabled changes timing/recorder observation calls; a video is not an
exact numerical repeat of the headless regression benchmark.

## What to inspect

- Terrain orientation, robot stance and the overall traversal corridor.
- Whether the generated obstacle shapes are a useful starting distribution.
  Original CAT lateral blocks can float: they are static synthetic fixtures,
  not physically grounded furniture. No invented supports are added.
- Stair-edge motion: the pelvis can pass through an amber transit region
  without that region becoming a valid foot-support patch.
- Whole-body coverage: legs, torso and arms contribute obstacle information;
  terrain/support is not silently classified as forbidden CAT clutter.

This sparse scene leaves the original reference clear. It is an integration
control, not a scene proving learned detours, crouching or collision avoidance.
Human visual review is useful but does not replace geometry/physics tests.

## Copy to a laptop

Run on the **laptop**, replacing `PC_IP` with the address used to SSH into this
workstation (not the Unitree robot's address):

```bash
scp constantinesmirnov@PC_IP:/home/constantinesmirnov/Desktop/GRAIL_CAT_review_20260911_v2.zip .
```

Unzip and open the MP4s in the laptop's video player. Use its loop/repeat
function if desired. The archive contains descriptions and provenance as well.
Nothing needs to remain running on the workstation for video playback.

## Reproduce (no training)

From the repository root, after preparing the source scene/reference runs:

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/20260911T183237_023782Z_cat_scene_random \
  --cat-translation 0 .2 0 --cat-yaw 1.5707963267948966 \
  --layout-audit --observation-shadow --record-video \
  --execute --accept-isaac-eula --timeout 600

.venv/bin/python research/record_review_demos.py \
  --run research/runs/20260911T191127_746866Z_stair_p1_cat_audit \
  --guidance-comparison research/runs/20260911T191130_206279Z_pelvis_guidance_audit \
  --output /PATH/TO/A/NEW/OUTPUT_DIRECTORY
```

Isaac license acceptance is the user's choice, as documented in `README.md`.
`--record-video` is single-environment evaluation only, never training. Video
and recorder outputs stay inside the new run; shared `/tmp/isaaclab` datasets
are not changed. All videos are decoded and checked for moving frames. The
validator does not claim that a moving image proves policy correctness.

Final actual camera run: `20260911T193059_716652Z_stair_p1_cat_audit`, exit 0,
499 valid packets, exact zero-residual versus actor action parity, unchanged
backbone and adapter, no failure termination. Captured 249 video frames. Minimum
sampled CAT cover clearance ~0.405 m. Source reports retain revision and dirty
state: the video feature was being developed during this run.

See `TRAINING_PREPARATION.md` for the remaining runtime/curriculum/reward gates.
Training has not started and must wait for explicit approval after review.
