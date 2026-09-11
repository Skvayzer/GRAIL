# Articulated contact capture before reset

M1 simulation diagnostic. This closes the terminal-step **capture** gap for a
new opt-in audit, not all contact-permission/avoidance-training gates.
No robot connection, new training, policy/action/reward/observation changes,
additional physics steps or modified Isaac Lab dependency files.

## Why the old contact report missed the last step

Isaac Lab's `ManagerBasedRLEnv.step` runs each physics step and updates scene
buffers, computes termination/reward, resets terminated environments, and only
then returns to the evaluator. Reading body poses/contact buffers after that
return cannot recover the pre-reset terminal state. The earlier CAT audit
correctly skipped such samples, but therefore did not cover terminal contacts.

The new `PhysicsStepTap` observes **after the original `scene.update`** on each
physics substep, before the reset can occur. It is a context-managed wrapper
on one scene instance, not a patched global class. The original call/return is
preserved; it is restored on normal completion or exception. There is no extra
observation computation (which could consume noise/history), reset, simulation
step or call that changes actions. Duplicate tap owners fail explicitly.

Every returned policy step must have exactly the configured number of captured
physics steps, with contiguous simulator counters. Termination flags are attached
after the environment returns; the contact samples were already copied. The
last step and its forces cannot be replaced by the reset state. No GUI replay,
multi-environment or training mode is supported by this extended audit yet.

## Contact data and geometry

An independent PhysX tensor view reads each of the 14 imported collision-bearing
G1 links against three explicit partners: the terrain rigid body, the ground
collision plane and the CAT mesh. The robot's existing contact reporting is
used; no additional collider or force is introduced. This audit excludes
self-contact, friction, payloads and future finger geometry.

[PhysX's detailed contact API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data)
provides six buffers: normal force magnitude, point, normal, separation, count
and start offset. Read only referenced ranges; unused slots can contain NaN.
Use physics dt to obtain forces. Reject malformed, overlapping, saturated or
nonfinite data. Reconstructed sums of point forces times their normals must
agree with the independent pair-force matrix; never flip normals to fit a label.

The pinned runtime reports **leaf names**, not full prim paths. Requested exact
paths, returned names and explicit sensor/filter counts are checked. Leaf-name
ambiguity or changed filter ordering rejects the run. The initial development
attempt `20260911T153503_445016Z_stair_p1_cat_audit` assumed full paths and
refused before policy-loop stepping. The corrected binding has a regression test.

Terrain contacts query the actual paired terrain triangles, not a nearby ground
surface. Ground contacts use the separately verified plane. CAT contacts query
the exact closed clutter mesh. Static terrain pose is checked every substep.
World-to-foot coordinates use the actual articulated body pose and WXYZ rotation.
Sole bounds come from the imported G1 foot **capsules' real radii/endpoints**,
not the inflated clearance-cover sphere radii. Current diagnostic sole bounds:
X approximately [-0.061, 0.139], Y [-0.033, 0.033], Z [-0.047, -0.023] m.
These are conservative diagnostic regions, not calibrated hardware permissions.

## Classification is not a granted contact permission

Point-force threshold: 0.1 N. Above that, record separate categories for:

- CAT contact (no automatic foot/hand exemption).
- Non-foot terrain/ground contact.
- Foot riser/side contact (normal Z < 0.7071).
- Excess separation penetration (>1 cm) or point force (>2,000 N).
- Contact outside the derived sole region.
- Unverified surface distance/normal agreement (>1.5 cm or dot <0.9).
- Reference stance-support candidate, swing-support mismatch, or phase unknown.

All limits are recorded. These are diagnostic thresholds, not hardware safety
limits, a stability criterion, or current training rewards. A flagged foot-edge
contact needs interpretation; a completed stair rollout is not collision-free
certification.

The dataset has no verified foot-contact annotations. Phase comes independently
from **reference** sole-to-support clearance and reference ankle speed, not actual
measured contact force. Candidate stance requires gap between -1 and +4 cm and
speed <=0.25 m/s; gap >7 cm or speed >0.50 m/s gives candidate swing. Unknown
support, intermediate values and transitions stay unknown. Both ends of a
policy interval must agree. These heuristic labels can disagree with legitimate
touchdown timing, retargeting or heel/toe rolling. They are explicitly **not
ground truth** and do not authorize a contact, add a penalty or erase terrain.

## Run and audit

Use an existing original CAT `side-hurdle2` scene:

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/YOUR_SIDE_HURDLE2_SCENE \
  --cat-translation 0 -1 0 --cat-yaw 1.5707963267948966 \
  --num-envs 1 --layout-audit --contact-audit \
  --execute --accept-isaac-eula --timeout 300
```

The existing strict checkpoint/reference-clearance/terrain-layout gates still
run first. The scene above retains the documented buried-hurdle warning and
is an unchanged-reference diagnostic, **not a hurdle curriculum example**.
See [SCENE_COMPOSITION.md](SCENE_COMPOSITION.md) for the separate retention gate.

Additional outputs:

- `contact_audit.json`: source geometry, exact partner binding, limits, every
  physics counter, policy/terminal alignment, classifications, peaks and labels.
- `contact_audit.npz`: packed per-contact records with point/normal/separation,
  force, local coordinates, reference frame and classification; schema/column
  order and hash in the JSON.
- `run.json`: parent source revision/dirty state, contact artifact verification
  and the separate original-policy completion result.

`contact_results.audit_contact_capture(run)` independently verifies the checksum,
packed counts/index ranges/classifications, finite payloads, complete decimation
and terminal coverage. Parent output validation requires it. `capture_complete`
means capture integrity, not collision-free behaviour. The older `cat_audit.json`
retains its older skipped-reset scope; do not misread it as this new report.

## Development result (2026-09-11)

Run `20260911T153625_910820Z_stair_p1_cat_audit` completed with exit 0 and valid
outputs. Original actor loaded strictly, no failure termination, same observed
minimum CAT cover gap 0.09351 m. New capture:

- 499 policy steps / **1,996 physics steps**, including all four terminal steps.
- 217,451 detailed contact records; 203,515 at/below 0.1 N. Records are contact
  points across time, **not independent collision events**.
- 43 point contacts above threshold in the terminal policy interval, now retained.
- Force reconstruction maximum discrepancy 0.0000916 N.
- No CAT-clutter or non-foot terrain/ground force recorded.
- 8,541 stance-support candidates; 5,045 phase-unknown contacts.
- 327 foot riser/side records over 318 physics frames, about 4.41–6.045 s by
  zero-based sample time. The strongest point had normal Z 0.402, surface
  normal agreement 1.0 and force about 850 N. Review these edge contacts; do not
  silently waive them or assume all are unsafe motion.
- 20 reference swing-support mismatches over four physics steps near 6.12 s;
  these are **label/timing candidates**, not established swing-foot violations.
- Three initially unverified support-normal records, during settling at the
  first two physics steps. Peak aggregate left-foot terrain normal force was
  about 1,369 N. No startup steps were suppressed from the audit.

72 tests pass (11 new), covering pre-reset capture, exception restoration,
leaf-name identity, ragged buffers/aggregate parity, terminal/hash checks and
contact classifications. No new dependencies or unexpected environment conflicts.

## Next work, in order

1. Review/annotate reference foot phases and the flagged edge/touchdown events;
   validate articulated stance/swing/sole rules against controlled negative
   fixtures. Keep uncertain labels out of automatic training penalties.
2. Retain roles through random CAT generation/morphology and add terrain-aware
   guidance that respects actual support and preserves the obstacle challenge.
3. Wire the validated obstacle observations/rewards into a bounded training smoke
   test, then evaluate terrain-skill retention against the frozen baseline.
4. Goal-conditioned distillation and manipulation commands follow; real-robot
   shadow deployment remains a separate, later review.

`phase_truth_verified=false`, `contact_permissions_granted=false`,
`collision_free_certified=false`, `avoidance_training_ready=false` remain explicit.
