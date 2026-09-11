# Terrain-aware CAT placement and full-height oracle fields

M1 diagnostic implementation, **not avoidance training or robot control**.
The original CAT generator, source scenes, GRAIL checkpoint, actor observations,
rewards and live launcher behaviour are unchanged. All output is a new ignored
run directory. This tool reads cached physical terrain/reference snapshots;
it does not start Isaac, a policy, ROS or a robot connection.

## Feature-preserving placement

`cat_roles.py` reconstructs explicit floor/lateral/overhead masks using the
unchanged pinned CAT **fixed-scene** functions. Their union must match the
checksummed cached occupancy exactly. Roles may overlap: a hurdle joining two
posts must not disappear into one unlabelled connected component. Unknown
recipes fail explicitly. New random exports retain replay-verified role traces;
old random exports without traces and ambiguous morphology additions still
reject placement. See [RANDOM_CLUTTER_GUIDANCE.md](RANDOM_CLUTTER_GUIDANCE.md).

`--ground-on-terrain` proposes a **rigid Z translation only** from terrain
support beneath all occupied footprint-cell centres and corners. Translation
in XY and yaw are explicit inputs; pitch/roll are never inferred. The source
floor datum must fit a nearly level footprint (height spread <= 4 cm), with
known upward support. Stair edges, missing surfaces or inverted support reject
the proposal. It does not warp columns, erase obstacles, shorten posts, stretch
a hurdle across different tread heights or modify the source USD/FMM fields.
Only the proposed placement is recorded; no running scene is modified.

Feature retention is then checked independently:

- Grounded floor/lateral roles must preserve their source terrain-relative
  datum within 4 cm and retain at least 95% of their occupied-column height
  above the terrain top envelope, at every sampled column.
- Explicit overhead roles retain their source bottom-clearance datum. They are
  declared suspended static CAT fixtures, not falsely labelled floating floor
  blocks. Their mounting/construction validity is **not** simulated.
- No role grants foot, shin, hand or body contact permission.

These tolerances are a conservative geometry screen, not hardware limits.
Sampling does not certify sub-voxel cracks, structural support, exact solid
intersection, balance or articulated passage. Open/multi-layer terrain still
requires more than a top-down support envelope.

## Full-height representation

`VolumeSpec.fit` fits all terrain/CAT vertices and every reference collider-cover
sphere's full bounding box, including head/arms at the top of stairs. It uses
4 cm XYZ cells, at least one interpolation-cell margin, and an 8-million-cell
resource guard. The sample origin is explicitly the centre of cell (0,0,0).

Saved `volume.npz` fields:

| Field | Meaning |
| --- | --- |
| `clutter_sdf`, `clutter_outward_normal` | Exact closed CAT mesh signed distance and outward normal at grid centres |
| `clutter_inside` | Negative/zero CAT distance at sample centres; not terrain occupancy |
| `terrain_unsigned_distance`, `terrain_authored_normal` | Nearest actual terrain triangle/explicit ground; no invented inside/outside sign |
| `terrain_surface_band` | Distance <= half voxel diagonal; **support surfaces stay present** |
| `*_distance_valid` | Geometric query validity, not sensor visibility or known free space |
| `support_height`, `support_authored_normal`, `support_known`, `support_candidate` | Separate top-down terrain/ground candidates; not a foothold planner/contact allowance |

An unsigned distance below a stair is still positive. Being outside the surface
band **does not mean free space**. There is deliberately no `terrain_inside` or
combined signed occupancy/free-space mask. Closest-face authored normals are
not gradients of unsigned distance; normals at edges/ties can be ambiguous.
The band includes both treads and risers and is never erased for support.

The diagnostic sampler exposes distances only. It uses XYZ trilinear sampling,
explicit masks, and NaN/false outside the volume. Binary support/contact labels
and normal vectors are not silently interpolated. Exact mesh queries remain
the clearance gate; interpolated distances have voxel-scale approximation error.
All reference samples are checked against exact queries with a conservative
sqrt(3)*voxel 1-Lipschitz interpolation bound.

This is **simulation-oracle geometry**, not the G1 LiDAR observation interface.
CAT's original guidance/FMM remains restricted to its original domain. It is
neither extended by padding nor presented as terrain-aware guidance.
The separate `terrain_guidance.py` diagnostic now exports bounded 2.5D support
graph guidance from a layout snapshot; it does not replace the original CAT
fields or add policy observations. Composition acceptance alone does not connect
that guidance or authorize training.

## Commands

First obtain a cached `baseline.py --layout-audit` run as described in
[SCENE_VALIDATION.md](SCENE_VALIDATION.md), and an original `side-hurdle2` scene.

```bash
.venv/bin/python research/compose_scene.py \
  --reference-run research/runs/YOUR_COMBINED_AUDIT \
  --scene research/runs/YOUR_SIDE_HURDLE2_SCENE \
  --translation 0 0.2 0 --yaw 1.5707963267948966 \
  --ground-on-terrain --device cuda:0
```

Use Z=0 with automatic grounding; combining an explicit nonzero Z with automatic
Z selection is rejected. Omit `--ground-on-terrain` to audit an explicit XYZ pose.
The original displayed placement was `(0,-1,0)`, yaw pi/2.

Outputs: `composition.json`, `volume.npz`, `volume_slice.png`, and the existing
`layout_audit.json/npz/png`. The report records input/output hashes, source revision,
dirty status, runtime versions, exact placement, role retention and field error.
The slice plot is a geometry diagnostic, not a simulation rollout.

Exit 0 means **role retention AND the existing cached reference/route screen**
passed. Exit 2 means the exported diagnostic is rejected; consult both reports.
Invalid input/provenance or impossible rigid grounding fails before volume export.
Neither exit 0 nor a complete field implies obstacle-crossing skill. A clear
control beyond the goal can pass without the robot encountering that obstacle.
Physics must be rechecked for a new pose; this offline checker never claims it.
The live baseline's prior gates are unchanged; this is an additional offline
composition/curriculum gate, not a new implicit bypass or training launcher.

## Verified results (2026-09-11)

Using clean physical/reference snapshot
`20260911T151123_875346Z_stair_p1_cat_audit` and original scene
`20260911T144243_747872Z_cat_scene_side-hurdle2`, yaw pi/2:

| Placement XYZ | Feature retention | Cached reference screen | Interpretation |
| --- | --- | --- | --- |
| (0,-1,0) | Rejected: buried hurdle | Passed | Original demo is not a retained hurdle challenge |
| (0,0.2,0), automatic grounding | Passed | Rejected: 70 flagged frames; min cover gap -0.0430 m | Visible grounded hurdle, but unchanged tracker does not clear it |
| (0,1.2,0), automatic grounding | Passed | Passed | Clear control beyond the reference endpoint; **not** hurdle-crossing success |

Development reports: `20260911T152242_205946Z_composed_scene`,
`20260911T152514_502606Z_composed_scene`, and
`20260911T152537_952668Z_composed_scene`, respectively.

Clean `a606d3c` confirmations reproduced all three expected outcomes with
`adapter_dirty=false`: `20260911T152752_420279Z_composed_scene`,
`20260911T152753_322907Z_composed_scene`, and
`20260911T152754_494837Z_composed_scene`, respectively.

Original-demo field coverage rises from **74.4605% to 100%** of 51,896 stored
reference-probe samples (499 frames x 104 probes). Grid: 56 x 74 x 72, height
2.88 m. Maximum interpolation discrepancies: 0.01948 m CAT distance and
0.01706 m terrain unsigned distance. The grounded landing uses 354 footprint
queries with zero support-height spread. The YZ slice was visually inspected.

61 unit tests pass, including nine new unsigned-surface, role replay/retention,
grounding rejection, height/extent, XYZ interpolation/unknown and CPU/CUDA
parity tests. No new packages or unexpected environment conflicts. No simulator,
training process or real robot was started for these offline checks.

## Remaining M1 gates

1. Traceable roles for the **random** CAT generator; terrain/placement curriculum
   that preserves intended challenges without silently changing the distribution.
2. Articulated terrain/swing-foot contact permissions; no terrain-wide collision
   exemptions. [ARTICULATED_CONTACTS.md](ARTICULATED_CONTACTS.md) now implements
   complete pre-reset/terminal capture; independent phase labels and permission
   validation remain outstanding.
3. Terrain-aware route/guidance with support semantics and bounded physical
   validation for each scene family; no straight-line/flat-FMM shortcut.
4. Then avoidance observation/reward wiring, bounded training smoke, terrain
   retention comparisons, and eventually goal-conditioned student training.

`avoidance_training_ready` remains false. Rejected references should motivate
motion adaptation/learning, not weakened collision gates or buried obstacles.
