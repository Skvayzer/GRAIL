# Random CAT provenance and terrain-aware guidance

M1 simulation-only tooling. No robot connection, ROS, SDK, policy action,
observation/reward change or new avoidance training is introduced here.

## Original random geometry, traceable roles

`random_cat_roles.py` observes two stages in a private function-global copy of
the pinned CAT generator: rotated role extrusion and morphology. Original module
functions and files are not patched. Auxiliary role reconstructions use copied
RNG state, and a separate untouched invocation must match final occupancy and
XYZ axes exactly. The original source hashes remain the contract.

Floor, lateral and overhead masks are captured after start/goal carving. Their
union must reproduce pre-morphology occupancy. Closing/opening runs exactly as
upstream, including occupied border padding; conservative dependency bits are
propagated through all dilation/erosion stages. Final widening may only remove
voxels. This is a provenance record, not a replacement morphology algorithm.

- Surviving source cells retain their original roles, including overlaps.
- New cells receive a role only when all traced dependencies are that single
  role, without padding dependencies.
- Mixed-role or padding-dependent new cells stay explicitly unresolved.
- Role-based placement rejects any unresolved cells. It does not delete them,
  infer roles from connected components, or silently relabel everything lateral.

The conservative dependency trace is **not** an exclusive semantic ground truth
or a minimal causal explanation. In particular, erosion can depend on several
nearby roles even when a human would assign one role to the final shape.
Resolving such ambiguity requires an explicit reviewed placement contract.

Random `scene.json` now includes optional `role_provenance`, and `role_trace.npz`
joins the existing six checksummed assets. The trace stores source masks,
pre/post-morphology occupancy, final masks, added-cell dependency bits and the
unresolved mask. Before placement, both the unmodified generator and the whole
trace are replayed; matching only the final union is insufficient. Old random
scenes are still readable as geometry, but cannot pass this placement gate.

Lateral random blocks can be elevated in the original CAT scene. Preserving
their floor-relative datum does not make them physically grounded. They remain
static synthetic fixtures; real mounting/construction validity is not checked.
No semantic role grants a foot/hand/body contact exemption.

## Shared support graph and guidance

`SupportGraph` is now shared by the layout route check and the goal-field export.
It represents one support height per world XY cell, **not** a full 3D CAT FMM.
Defaults inherited from `LayoutLimits`:

| Quantity | Value |
| --- | --- |
| XY resolution | 0.04 m |
| Maximum swept support-height range | 0.20 m |
| Maximum XY edge/stride length | 0.28 m |
| Support patch half-width / height variation | 0.04 / 0.06 m |
| Trunk radius / extra clearance | 0.20 / 0.03 m |
| Trunk sphere-centre height range | 0.45..1.35 m above support |

Nodes must be supported trunk-clear patches. Every edge's conservative line-cell
cover must have known, trunk-clear transit support with bounded height range.
This permits crossing a tread edge without claiming that a foot can stand on
that edge. It rejects holes, walls, cliffs and diagonal corner cutting. Equal
height endpoints cannot jump over an intervening large height excursion.

The sparse symmetric graph is capped at 1,048,576 cells, eight million potential
directed edges, and a 16-cell reach. Dijkstra from the selected goal produces:

- `goal_cost`: shortest sum of 3D **endpoint** edge lengths in metres.
- `next_index`: row-major flat index of the next support node, -1 at the goal or
  for unreachable cells.
- `next_delta`, `direction`: world-XYZ node displacement and its unit vector.
  Both are zero at the goal, NaN if unreachable.
- `reachable`: explicit validity mask; invalid cost is NaN, never zero.

Directions descend cost strictly. There is no continuous interpolation across
invalid cells, fallback to flat terrain, goal clamping or policy command output.
The node route is not a foot swing trajectory: straight endpoint segments can
intersect a stair riser, and these vectors are not executable velocities.
This is an upright-trunk geometric screen, not balance, articulation, crouching,
support-contact permission, underpass or stacked-floor certification.

`terrain_guidance.py` verifies the saved layout hash and world-grid coordinates.
By default it uses the layout's endpoints; for legacy snapshots it recovers
them only from verified on-grid route endpoints. An explicit goal cell must be
walkable. A valid goal may yield an unreachable start, which is reported without
inventing a route. Original CAT `gf.npy` and `travel.npy` remain unchanged.

Every export records source hashes, graph limits, git revision/dirty state,
reachability, field checksum and a PNG. It does **not** revalidate role retention
or promote a rejected layout to an accepted training scene. `compose_scene.py`
and physical/reference gates remain separate requirements.

## Reproduce the bounded diagnostics

From the repository root, using a previously captured stair terrain/reference
run from [ARTICULATED_CONTACTS.md](ARTICULATED_CONTACTS.md):

```bash
.venv/bin/python research/audit_random_clutter.py \
  --reference-run research/runs/YOUR_STAIR_CAT_AUDIT --device cuda:0

.venv/bin/python research/terrain_guidance.py \
  --layout research/runs/YOUR_LAYOUT_OR_COMPOSITION/layout_audit.json

.venv/bin/python research/cat_scenes.py generate \
  --scene random --seed 0 --difficulty .2 --n-side 1 --n-floor 0 --n-ceiling 0

.venv/bin/python -m unittest discover -s research/tests -v
```

No Isaac process starts for these commands. The batch is deliberately fixed:
seeds 0/1/42; original-count, sparse-lateral and floor-only recipes; difficulty
0.2; XY placements (0,-1), (0,0.2), (0,1.2) at yaw pi/2, with rigid Z grounding.
There is no retry-until-pass loop or seed replacement. Every trace, empty output,
role rejection and placement outcome is retained; `complete=false` identifies
an unfinished batch. Source data is read-only; outputs use new ignored run dirs.

Development results (`20260911T183023_271089Z_random_clutter_audit`):

| Recipe | Seed 0 | Seed 1 | Seed 42 |
| --- | --- | --- | --- |
| Original counts: 9 per side, 3 floor, 3 ceiling | 465 unresolved cells | 917 unresolved | 395 unresolved |
| Sparse lateral: 1 per side, no floor/ceiling | roles resolved | 224 unresolved | roles resolved |
| Floor-only: 1 floor rectangle | empty after morphology | empty | empty |

For each resolved sparse scene, the first placement fails the footprint support
gate; the other two pass role, support-route and cached whole-body-reference
screens. All four yield the same 12-node / 2.610522 m start-to-goal distance,
with support height 0..1.244169 m. They change reachable areas but do not require
deviating from the original corridor. Do not report these as avoidance success.
The fixed retained control yields the same route distance.

Clean revision `60c86e3` reproduced the batch at
`20260911T183236_696188Z_random_clutter_audit` (`complete=true`, `dirty=false`).
All nine trace hashes and four guidance hashes/invalid masks/descending costs
were independently checked. The development batch predates the explicit
`complete` flag; use the clean run for that check.

A clean headless Isaac run with sparse seed 0 at translation (0,0.2,0), yaw pi/2,
completed at `20260911T183246_963741Z_stair_p1_cat_audit`: exit 0, outputs valid,
no failure termination, 499 policy / 1,996 physics steps, all 96 physical
support-ray checks passed. Minimum sampled CAT cover clearance was 0.353909 m;
measured CAT normal contact force was zero across all covered links. The policy,
observations and rewards stayed unchanged. Contact capture retained four
terminal samples and 217,710 records. Provisional phase/geometry classifications
still include 340 riser/side and 17 swing-support mismatches; these are not
independently verified labels or approved contact permissions.

97 unit tests pass, including CPU/CUDA geometry parity and 13 new trace/guidance
tests. All 18 pinned baseline artifacts verify and no unexpected package
conflicts were introduced. The owned simulator exited; no robot was accessed.

## Next gates

1. Review a placement rule for ambiguous morphology additions without silently
   deleting/warping original clutter. Keep rejection rates and recipe distribution.
2. Add role-valid clutter configurations that actually challenge the intended
   traversal corridor, with physical placement checks and a body-feasible route.
3. Observation-only geometry/guidance conditioning and zero-residual checkpoint
   parity are now implemented and verified; see `OBSERVATION_SHADOW.md`.
   Before residual learning, fix bounded guidance attachment at tread transitions
   without inventing support or crossing obstacles. The released tracker still
   needs future motion/object references; it is not goal-conditioned.
4. Complete reference foot-phase/riser-contact review before using the contact
   candidate labels as RL permissions or reward ground truth.

No full training, manipulation adaptation, deployable student or robot trial is
claimed by this milestone.
