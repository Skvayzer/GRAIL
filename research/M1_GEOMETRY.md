# M1 geometry integration — in progress

The newer CAT mesh/field placement and full-reference diagnostic integration
are in [CAT_GRAIL_INTEGRATION.md](CAT_GRAIL_INTEGRATION.md). These additions do
not remove the outstanding terrain/support/terminal-contact gates below.
The later [SCENE_VALIDATION.md](SCENE_VALIDATION.md) adds live triangle-surface
support extraction and conservative passage screening; it does not complete
articulated terrain contact or avoidance-training validation.
The newer [SCENE_COMPOSITION.md](SCENE_COMPOSITION.md) adds explicit fixed-CAT
roles, rigid grounding and full-height signed-clutter/unsigned-terrain oracle
features. Open terrain is still not given an invented signed volume; contact
permissions and random-clutter role provenance remain separate work.
The newest [ARTICULATED_CONTACTS.md](ARTICULATED_CONTACTS.md) adds opt-in
pre-reset capture of articulated terrain/ground/CAT contacts at every physics
step. This closes capture coverage for that diagnostic, not phase-label truth
or the full contact-permission/avoidance-training gate.

Desktop simulation only. No new policy training, deployed observation interface,
robot SDK, ROS changes or actuation. The original terrain checkpoint stays
immutable. These diagnostics do not make the tracker obstacle-aware.

## Implemented diagnostic contract

- `gear_sonic.research.geometry`: signed distances/outward normals for yawed
  boxes and upright finite cylinders, sphere-radius/margin clearance, explicit
  unknown masks, designated support-top queries and link/phase/region/normal/
  penetration/force contact permissions. Metres, Z-up, XYZ grid axes.
- Support surfaces stay in 3D occupancy. A stance foot on a tread is a permitted
  contact; a shin or swing foot hitting its riser is not. Support height alone
  is not a traversability/reachability decision.
- `body_envelope` audits the pinned URDF collision primitives. `usd_envelope`
  covers the **actual imported** sphere/capsule geometry, with exact name-based
  body binding and WXYZ rotations. Isaac converts this model's cylinders into
  capsules; using only the original cylinders could miss their end caps.
- Covers are conservative for these collider primitives, not certified for
  visual meshes, real hardware, hands with fingers, clothing or carried objects.
  Unrecognized collider geometry/scaling fails explicitly.
- A separate diagnostic stream records per-link clutter clearances, pair-filtered
  PhysX normal forces, and broad-phase self-cover gaps. It is not inserted into
  actor/critic inputs or rewards. Negative cover gaps alone do not prove contact.
- Older contact histories skip samples after automatic resets. The new opt-in
  articulated audit reads raw contact buffers before reset and verifies complete
  terminal coverage; independent stance/swing label validation remains outstanding.

## CAT sampler parity and intentional v1 differences

Pinned numerical fixture: `tests/fixtures/cat_sampler.json`, produced by CAT's
unaltered `world_to_grid`/`sample_field` methods on CPU JAX. The source file hash
is recorded and checked by `cat_sampler_reference.py`.

CAT's corner list varies X fastest but its flattened weight array varies Z
fastest. On index coordinates `(0.2,0.4,0.8)`, an affine field `x+10y+100z`
returns about **24.8**, versus the mathematically interpolated **84.2**. V1
corrects that ordering, includes the final grid interval and marks out-of-bounds
or contributing unknown samples invalid. `legacy_cat=True` exists only for
parity tests and is not the new policy contract. Matching these tests does not
claim parity of the complete CAT FMM/guidance pipeline.

Reproduce the legacy fixture with CAT's existing Python (read-only checkout):

```bash
/path/to/Click-and-Traverse/.venv/bin/python research/cat_sampler_reference.py \
  --cat-repo /path/to/Click-and-Traverse
```

## Diagnostic commands

```bash
.venv/bin/python -m unittest discover -s research/tests -v
timeout --signal=TERM --kill-after=10s 180s .venv/bin/python \
  research/physics_contact_check.py --execute --accept-isaac-eula
.venv/bin/python research/baseline.py --family stair_p1 --clutter-audit \
  --execute --accept-isaac-eula --timeout 300
```

Use the EULA flag only after reading/accepting the license as documented in the
README. Tests save new reports under `research/runs`; no baseline data overwritten.
The primitive fixture runner exits its process after closing/flushing its report,
as upstream's evaluator does, avoiding Kit extension teardown stalls.

The six `stair_side_v1` solids are **test fixtures**, deliberately outside the
pinned motion path, not the intended training distribution. The actual CAT
generator is now reused with an Isaac exporter: see [CAT_SCENES.md](CAT_SCENES.md).

## Verified so far / remaining gates

- 25 unit tests passed, including legacy JAX fixture parity, corrected sampling,
  normals, radii, unknowns, contact permissions and imported capsule coverage.
- Seven PhysX sphere-contact fixtures passed, including tread, riser, swing-foot,
  edge, rail, overhead beam and pole. Measured contact positions/normals agree
  with analytic geometry. These are semantic probe fixtures, not articulated
  foot/hand contact-skill validation.
- Initial G1 side-clutter diagnostic: 498 finite samples, 104 probes, 29 imported
  colliders (28 capsules + 1 sphere), 14 collision-bearing links. No clutter
  contact was expected or measured. A clean two-environment run at `7532d6c`
  confirmed the self-contact instrumentation and physical pose-consistency gates.
  It measured up to 17.94 N between the right wrist/hip collision links. Do not
  mistake a passing diagnostic run for a collision-free walking benchmark.
- Not implemented yet: arbitrary signed terrain-volume fields and multi-layer
  support, full articulated/payload/contact-permission validation,
  calibrated self-pair exclusions and whole-mesh coverage, sensor realism, or
  the full guidance-field parity/goal-conditioned policy.
- **Do not enable new avoidance rewards or begin M2 based on these partial gates.**
