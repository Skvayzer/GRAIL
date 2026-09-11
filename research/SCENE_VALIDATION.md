# Valid terrain + CAT composition: explicit screening gates

Desktop-only M1 tooling. This does not modify the original CAT generator,
GRAIL policy, observations, rewards, or physical scene. It is not avoidance
training, a motion planner for deployment, or a whole-body safety certificate.

The subsequent [SCENE_COMPOSITION.md](SCENE_COMPOSITION.md) adds an **offline**
explicit-role/rigid-grounding gate and full-height oracle features. It does not
change the live diagnostic gates below or turn the open terrain into a signed
solid. The original displayed placement now explicitly fails hurdle retention.

## What must agree

1. **Coordinates and physical geometry.** Metres, Z-up, WXYZ rotations. Extract
   only enabled triangle-mesh colliders with `none` approximation. Apply the
   terrain's live rigid-body pose exactly once, preserving uniform scale; reject
   mirrored/sheared roots, unsupported colliders and moving terrain references.
   Authored USD axes are not assumed to equal the final stair orientation.
2. **Support.** Cast downward against the actual terrain mesh and the explicitly
   verified horizontal ground plane. CAT geometry is never promoted to support.
   Missing support stays unknown. Winding-derived normals identify upward-facing
   candidates; the ray-facing normal must not make an inverted surface walkable.
3. **Passage.** Build a 4 cm support graph with small stance patches, a bounded
   step/stride and a conservative upright trunk envelope checked against the
   exact CAT mesh. Intermediate transit cells must remain known and clear. A
   stride can cross a riser-edge band unsuitable for a stance patch, but not a
   hole, an excessive height excursion, a blocked corridor or a diagonal corner.
4. **Whole-body nominal motion.** Independently check all 104 conservative
   collider-cover spheres throughout the sampled GRAIL reference against CAT,
   with the existing 3 cm margin. A graph route does not replace this check.
5. **Physics parity.** Compare terrain/ground heights against independent PhysX
   scene rays, filtered by exact collider paths. Mismatches reject the diagnostic
   before the policy loop. The scene-query flag is opt-in for these audits.
   Direct-GPU mode leaves CPU scene-query actor poses stale. Rays are therefore
   rigidly mapped from the live GPU terrain pose into its CPU query pose, against
   the **same cooked collider**, preserving metric distance. Ground is queried
   separately in world coordinates. Both poses and this method are recorded;
   this is cooked-geometry parity, not a claim that CPU pose readback is current.
   No extra simulation step, scene-pose change or policy-input change is needed.
6. **Obstacle embedding review.** Report occupied voxels below the terrain's
   support envelope and columns whose bases float above support. Overhead
   obstacles need an explicit role/attachment review, not automatic rejection
   or automatic support labels. A buried hurdle must not be counted as a visible
   hurdle in the curriculum. No automatic lifting, rotation or mesh edits.

The **open** stair mesh supports surface rays, not a reliable signed inside/outside
volume. "Below support envelope" is therefore a placement-review flag, not proof
of a volumetric mesh intersection. Underpasses and stacked floors require a
multi-layer representation; this top-down screen does not validate them.

## Current numerical screen (not hardware-certified limits)

| Quantity | Value |
| --- | --- |
| Grid spacing | 0.04 m |
| Maximum support height excursion per edge | 0.20 m |
| Maximum horizontal stride | 0.28 m |
| Candidate support normal | upward Z >= 0.7071 |
| Small support patch | 0.08 x 0.08 m, sampled at nine positions |
| Maximum patch height variation | 0.06 m |
| Trunk screen | radius 0.20 m; sphere centres 0.45–1.35 m above support |
| Clutter clearance margin | 0.03 m |
| Maximum endpoint projection | 0.12 m in XY, bounded support-height difference |
| PhysX height parity tolerance | 0.005 m |

The sphere chain's radius includes the inter-sphere cover allowance. The trunk
screen is deliberately separate from foot/arm geometry. A candidate patch is
not a full sole, double support, friction, balance, swing-foot or reachability
check. Pelvis projections are checked for nearby support height, not required
to be stance footholds at every instant. The route connects the first reference
anchor to its furthest XY excursion, so a returning clip cannot pass with a
zero-length start-to-start route.

## Run the combined check

After generating `side-hurdle2`, use its printed directory:

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/YOUR_SIDE_HURDLE2_SCENE \
  --cat-translation 0 -1 0 --cat-yaw 1.5707963267948966 \
  --num-envs 1 --layout-audit --execute --accept-isaac-eula --timeout 240
```

One environment is currently required for this extended check. This is the
recommended entry point when checking a **new terrain/clutter combination**;
the older `--cat-scene` diagnostic alone checks CAT/reference clearance only.
`--gui` may be added for subsequent visual review, subject to the same gates.
Use the EULA flag only after acceptance as documented in `README.md`.

Artifacts are written into a new run directory:

- `terrain_snapshot.npz/json`: physical mesh at the live pose, collider paths,
  explicit ground, coordinates and checksum.
- `reference_sweep.npz` / `cat_audit.json`: nominal whole-body cover and separate
  policy-rollout diagnostic.
- `layout_audit.json/npz/png`: limits, route/support grids, reference alignment,
  component review flags, PhysX parity and a top-down review plot. Black in the
  plot means no valid stance patch/trunk placement; a valid bounded stride may
  cross a black riser-edge strip if its intermediate transit checks all pass.

Recheck an alternative CAT placement without starting Isaac or the policy:

```bash
.venv/bin/python research/layout_audit.py \
  --reference-run research/runs/YOUR_COMBINED_AUDIT \
  --scene research/runs/YOUR_CAT_SCENE \
  --translation 0 -1 0 --yaw 1.5707963267948966 --device cuda:0
```

The offline checker verifies input hashes and creates a **new** report. It never
claims a fresh physics parity check; that requires a new live audit. Exit 2
means geometric screening rejected the candidate; a live refusal records the
failed checks and runs no policy-loop steps.

## Before avoidance training

`geometry_screen_accepted` and `accepted_for_reference_diagnostic` are not
`avoidance_training_ready`. The latter remains false. Required next gates:

- Extend explicit roles/rigid placement beyond the supported fixed CAT recipes
  to traceable random clutter. The new offline composition gate rejects buried
  challenges and unsuitable rigid grounding; it never silently deforms them.
- Full-height mesh-distance fields now cover the entire reference, with terrain
  unsigned distance/support separate. Complete terrain solid/contact semantics
  and terrain-aware guidance without erasing stairs or inventing free space.
- Complete articulated terrain/swing-foot and terminal-contact accounting.
- Verify visual appearance and bounded dynamic traversal for each curriculum
  family, not just one geometric graph or one released reference.
- Only then add avoidance observations/rewards and run retention comparisons.

PhysX query API details are checked against the installed 5.1 bindings and
[NVIDIA's scene-query documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.1/extensions/runtime/source/omni.physx/docs/dev_guide/scene_queries.html).
The [Direct GPU API limitations](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/DirectGPUAPI.html)
explain why an ordinary CPU world-space ray cannot validate a moved GPU actor.

## Verified development results

- Clean `1f2f375` confirmation `20260911T151123_875346Z_stair_p1_cat_audit`
  exited 0 with accepted layout, validated outputs and no failure termination.
  No GUI or headless simulator was left running.
- `20260911T150945_632435Z_stair_p1_cat_audit`: approved wider CAT passage,
  unchanged GRAIL stair tracker. All 96 cooked-geometry parity rays passed
  (30 terrain hits), maximum height difference 0.0000117 m. The graph found a
  2.6105 m route, with a support-height candidate below every sampled pelvis.
- The subsequent first episode completed without failure: 498 diagnostic
  batches, minimum CAT cover gap 0.0935 m and no measured CAT normal force.
  Terminal-contact and continuous-path limitations still apply.
- The passage retains a placement warning: 13.84% of CAT occupied voxel centres
  are wholly below the terrain support envelope, with no floating base columns.
  The source obstacle remains unchanged; do not count the buried low hurdle as
  a retained challenge in a training curriculum.
- The conflicting overhead scene in `20260911T150536_192407Z_layout_check`
  was rejected: 426/499 nominal reference frames violate CAT cover clearance.
- 52 tests pass, including inverted normals, incorrect units/scale handling,
  CPU/GPU ray parity and query-frame rotations, blocked passages, narrow gaps,
  stride-over-riser handling, holes, excessive steps and diagonal corners.
