# Pelvis transit guidance fix

Simulation-only `bounded-transit-root-v2`. This changes the new observation's
guidance calculation, not the support graph, terrain, CAT geometry, released
GRAIL actor or its control inputs. The adapter remains zero initialized and
disconnected from `env.step`. No robot/ROS changes or avoidance training.

## Cause and correction

The old query accepted only the nearest support-graph node. At tread edges, an
8 cm-wide sampled patch spans different step heights and is not a valid stance
patch. A moving pelvis can nevertheless pass over that edge while feet are
supported elsewhere. All 165 invalid frames in the prior 499-frame stair
recording were in clear transit cells but not supported-patch cells.

Keep the support nodes and graph edges unchanged. Introduce the current pelvis
XY position as a temporary **transit root**, connected to goal-reachable support
nodes within the existing 28 cm horizontal stride limit. Unlike a nearest-node
snap, each proposed connection must pass these checks:

- Root remains inside the sampled grid and has a known live terrain height.
- Target is an existing, goal-reachable supported node.
- Every grid cell intersected by the closed root-to-target segment is in bounds,
  has known support height and passes the existing trunk-clear transit mask.
  Corner touches and lines along cell boundaries include both neighboring cells.
- The maximum height excursion over those cells **and the live root's support
  height** is at most the existing 20 cm bound.
- No out-of-bounds clamp, unknown-ground fallback or cached last-good direction.

Segment/cell intersections use slab clipping, not sparse samples along a line.
The local search has at most 289 targets/cells per root and 16 roots per batch;
larger search radii reject explicitly. Float32 elementwise calculations leave
the actor's TF32 setting untouched.

Select the lowest connector endpoint-distance plus existing goal-graph cost.
Within a 0.01 mm cost tie, prefer the longest lookahead, then a deterministic
cell order. Do not steer back into the current cell centre. Direction is from
the actual root toward the selected node, with support-height difference as Z.
Arrival retains the existing goal-cell convention: zero direction and cost.
This avoids cell-centre stop/backtracking on a straight route without stateful
hysteresis; it is not a promise of continuous direction at genuine route branches.

This is still a 2.5D **grid-level geometric screen**, not a continuous geometry,
balance, stance-contact or full-body feasibility certificate. Subcell obstacle
coverage and upper/lower-level ambiguities are not newly solved here. The
independent articulated reference and physical-layout preflights remain in place.

## Recording and validation

Observation channel dimensions remain unchanged, but the packet manifest now
names the guidance version. The report records its bounds and selection contract,
plus each selected target, connector length and crossed-cell count. Replay rejects
an incompatible manifest instead of silently reinterpreting old recordings.

```bash
# Compare new guidance against saved pre-fix poses; no Isaac or actor started.
.venv/bin/python research/audit_pelvis_guidance.py \
  --run research/runs/20260911T185640_513446Z_stair_p1_cat_audit \
  --require-all-valid

.venv/bin/python -m unittest discover -s research/tests -v
```

`--require-all-valid` is for this known traversable reference, not arbitrary
scenes. Normal invalid guidance must remain invalid. The audit uses a separate
scalar connector witness checker, compares CPU/CUDA and GPU batches 1/16, and
saves a new report without changing source files. Original recordings retain
their old behavior; use their recorded commit for exact old-version replay.

Tests cover ascending/descending tread transitions, straight-line progress,
arrival, blocked source cells, walls, holes, excessive height changes, unknown
root support, outside-grid positions, diagonal corner cutting, parallel boundary
segments, randomized independent cell covers, witness corruption, resource
bounds and CPU/CUDA/TF32/batch parity.

Development replay `20260911T190949_101780Z_pelvis_guidance_audit` recovered all
165 rejected frames: 334/499 -> 499/499, with no lost valid frames. All 499
selected connectors passed independent witness checks. Maximum connection
length was 0.278362 m (limit 0.28); CPU/GPU feature difference <=2.03e-6.
Clean-commit physics and replay confirmation are recorded in `PROGRESS.md`.

Clean revision `8ab40fa` confirmed the result:

- Comparison `20260911T191130_206279Z_pelvis_guidance_audit`: 499 independent
  witnesses pass, 165 recovered and zero lost valid frames.
- Isaac `20260911T191127_746866Z_stair_p1_cat_audit`: 499/499 valid packets,
  499 exact action-parity checks, unchanged weights, no failure termination.
  Clearance samples and the complete contact payload hash match the earlier
  no-shadow reference exactly. Recorded witness cells/counts match the separate
  checker. The owned simulator exited.
- CUDA replay `20260911T191305_276925Z_observation_replay`: all features exact.
  CPU replay `20260911T191308_344564Z_observation_replay`: maximum normalized
  error 2.02656e-6, below unchanged 2e-5 tolerance; masks match exactly.
- All 123 tests pass. No policy or adapter training was performed.

Validity recovery is not smooth-control certification. On this reference the
world horizontal heading changes by median 0.60 degrees per frame, 95th percentile
8.63 degrees, maximum 25.80 degrees; no >90-degree reversal. Discrete target and
support-height transitions still exist. The feature is not a velocity command,
and continuous-control behavior must be evaluated before a learned adapter is
allowed to drive the simulator. No goal-arrival success is claimed by this
reference-replay check.

Next work remains challenging obstacle layouts and reviewed contact/phase
semantics before a bounded residual-learning trial. This fix is not evidence
that the controller has learned obstacle avoidance.
