# Controlled articulated G1 contact fixtures

M1 desktop-only geometry/label regression tests. These use the **actual imported
29-joint G1**, not a sphere substituted for a foot or shin. They do not train a
policy, annotate the released reference, certify stability or grant collision
permissions. The existing terrain/actor/reward configuration remains unchanged.

## Physical setup and coverage

A fresh headless Isaac scene imports the pinned `main_nodex.urdf` into the new
run directory. Cylinder-to-capsule conversion, actuator configuration and enabled
self-collisions come from `G1_CYLINDER_CFG`. Imported collision primitives and the
URDF checksum are recorded. Sole bounds share the same implementation as the
rollout audit, using real capsule radii, not inflated clearance-cover radii.

This **diagnostic** scene uses zero gravity, the original joint-position targets,
and an imposed root approach of 0.10 m/s. Poses/joints reset only between cases;
the articulated joints respond dynamically during contact. Root velocity is
imposed at each substep, so this is not a natural locomotion/stability experiment.
No checkpoint, ROS, SDK, robot connection or hardware command is involved.

Nine cases:

| Case | Required diagnostic result |
|---|---|
| Left foot onto a tread | Stance-support candidate |
| Right foot onto a tread, robot and box yawed 90 degrees | Stance-support candidate |
| Same support with an explicitly injected swing label | Swing/support mismatch |
| Same support with unknown phase | Unknown phase; no automatic permission |
| Forefoot into a riser face | Riser/side contact |
| Shin into a riser face | Non-foot contact |
| Forefoot upwards into an overhead patch | Downward-normal contact, not support |
| Foot onto a box with CAT semantic role | CAT contact, no foot exemption |
| Clear approach stopped short of the same tread | No contact |

The CAT-role fixture uses an analytic box, **not the procedural CAT mesh**. It
checks role handling; generated-mesh physical validation remains the separate
`physics_contact_check.py --cat-scene ...` suite. Phase labels above are explicit
scenario inputs, independent of forces; they do not validate a phase estimator.

The paired contact view covers all 14 imported collision-bearing robot links
against every fixture box. Unintended pairs reject the test. Six raw PhysX
buffers are verified against their force matrix. Each contact case starts with
a 4 cm geometric gap and retains ten physics samples from contact onset. The
clear control takes 30 steps and stops about 2.5 cm short. Physics dt is 5 ms.
The 20 settling steps before each approach have the boxes parked away; they are
not part of the contact measurement window. Self-contact/friction are not audited.

## Findings during test development

The initial run `20260911T154903_971568Z_contact_fixtures` passed seven cases and
rejected two fixture placements. The pitched shin's forward capsule endpoint
hit a box edge instead of the intended face. The broad underside plate also
intersected the leg. The face is now centred at the actual forward capsule
extremum, and the underside patch is restricted to the forefoot. No collision
filter or measured contact was suppressed to make these cases pass.

Run `20260911T175139_905885Z_contact_fixtures` corrected both placements but found
a separate normal-check issue. At a box edge, a single arbitrary SDF gradient
can select the side face even when a downward contact normal is valid. The
fixture now checks membership in the convex box's **outward normal cone** at the
contact point. Only faces within 10 micrometres of the boundary participate.
Inward normals, tangent directions and off-surface points are rejected. Measured
normals are not flipped or replaced; single-SDF-normal agreement is still saved.
The independent expected approach-normal check (>0.98 dot product), surface
error (<5 mm) and penetration (<5 mm) limits are unchanged.

This normal-cone check applies only to these analytic boxes. No automatic change
to open stair-mesh normals or rollout contact labels is implied.

Successful development run `20260911T175353_451267Z_contact_fixtures`:

- Nine cases passed; 742 approach physics steps, plus 180 settling steps.
- 407 force-bearing point records (>0.1 N); no unintended link/box contacts.
- All eight contact cases first touched at zero-based step 79, about 0.400 s.
- Maximum force reconstruction discrepancy: 0.0000229 N.
- Maximum recorded surface error: 0.0000000373 m; penetration: 0.00000479 m.
- Minimum outward-cone agreement >0.9999999999; peak point force about 83.27 N.
- 24 underside records demonstrated the single-SDF-normal ambiguity.
- The independent saved-artifact checker passed on all nine cases / 407 records.

The sandbox could not enumerate the GPU in one attempt; that process exited
before physics. The subsequent authorized headless run used the existing GPU
and packages. No driver/dependency change was made.

## Reproduce and inspect

From the repository root, using the already accepted Isaac licence:

```bash
timeout --signal=TERM --kill-after=15s 300s \
  .venv/bin/python research/physics_contact_check.py \
  --articulated --execute --accept-isaac-eula
.venv/bin/python -m unittest discover -s research/tests -v
```

The fresh `research/runs/*_contact_fixtures/` directory contains the imported USD,
`articulated_fixture_report.json` and checksummed
`articulated_fixture_contacts.json`. The checker independently replays the saved
point/body transforms, box geometry, classifications, identities, counts and
onset/window checks. A process exit code alone is not sufficient evidence.

For the previous stair rollout, export a **read-only review queue**:

```bash
.venv/bin/python research/review_contacts.py \
  research/runs/20260911T154050_843634Z_stair_p1_cat_audit
```

This verifies the original capture first and writes a fresh `*_contact_review`
directory containing `review.json` and `REVIEW.md`. It groups consecutive physics
samples separately by link, partner and label; it never bridges gaps or silently
assigns reward labels. End-of-step timestamps start at dt, not zero.

The captured stair run contains four riser/side intervals (327 point records),
one 20 ms swing/support mismatch (20 records, reference frame 306), one initial
unverified-normal interval and 22 phase-unknown intervals. These remain a review
queue, not verified violations. In particular, the two longer heel/edge intervals
need interpretation before any terrain-contact penalty is trained.

## Remaining work

Controlled contact classification is now physically exercised. The reference
foot-phase heuristic still needs annotation/validation, and exact stair edge
semantics remain unresolved. Do not convert these uncertain labels to rewards.

Next implementation track: preserve floor/lateral/overhead roles through random
CAT generation and morphology, then build terrain-aware guidance with actual
support and passage checks. After those gates, wire the obstacle observations
and rewards and run bounded fine-tuning with terrain-skill retention evaluation.
No claim of completed obstacle-avoidance training or deployability is made.
