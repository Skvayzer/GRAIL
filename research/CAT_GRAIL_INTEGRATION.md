# CAT clutter in the GRAIL environment — diagnostic stage

Simulation only. No robot or ROS access. This is **not** avoidance training or
goal-conditioned navigation. The released GRAIL actor, observation layout and
reward configuration remain unchanged; it still follows its paired reference.

## What is connected

- The same checksummed static, non-convex CAT mesh used in the gallery can now
  be spawned in each GRAIL environment, using an explicit XYZ translation and
  Z-axis yaw. No scale changes, source modification or implicit floor alignment.
- Voxel SDF, boundary and guidance fields share that placement. Points transform
  into CAT coordinates; vectors rotate back into terrain coordinates. Sampling
  uses corrected XYZ trilinear weights and a strict validity mask. Out-of-domain
  samples remain **unknown**, not zero/free.
- A separate Warp query computes signed distance to the **exact closed CAT
  triangle mesh**, using winding-number sign classification. It validates
  nondegenerate, consistently outward-wound, watertight components. Open terrain
  surfaces are deliberately not accepted by this signed-clutter query.
- The actual imported G1 collision primitives are covered by 104 conservative
  spheres. Every stored reference frame is checked before the policy loop. A
  sphere-to-mesh gap below 3 cm rejects this unchanged-reference diagnostic.
  This catches conflicting reference targets without running PPO into them.
- Per-link CAT contact-force histories and mesh clearances are recorded during
  an accepted rollout. Field coverage is reported independently from clearance.
  The physical mesh transform is checked against the field transform in every
  environment, including yaw and the environment spacing offsets.

This uses the complete exported CAT object as known geometry. Being outside its
voxel field does not invalidate the separate distance to that finite object,
but also **does not make unknown terrain or sensor space free**.

## Commands

First generate a scene and keep its printed directory:

```bash
.venv/bin/python research/cat_scenes.py generate --scene side-hurdle2
```

Check the original stair reference with this wider CAT passage, rotated so its
local +X direction lies along terrain +Y (yaw in **radians**):

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/YOUR_GENERATED_SCENE_DIRECTORY \
  --cat-translation 0 -1 0 --cat-yaw 1.5707963267948966 \
  --num-envs 2 --execute --accept-isaac-eula --timeout 240
```

License flag requires the user's prior acceptance, as in the main research
README. Omit `--execute` to stage/inspect without starting physics. CAT audit
cannot be combined with GUI, primitive diagnostics or training smoke, and is
capped at four environments and one episode / at most 500 policy steps.

Outputs include `cat_audit.json`, `reference_sweep.npz`, provenance and logs.
The sweep records imported-collider sphere centres/radii at all reference
frames, not the policy's trajectory. Its checksum allows candidate placements
to be checked without restarting Isaac:

```bash
.venv/bin/python research/cat_reference_check.py \
  --reference-run research/runs/YOUR_GRAIL_CAT_AUDIT_DIRECTORY \
  --scene research/runs/YOUR_GENERATED_SCENE_DIRECTORY \
  --translation 0 -1 0 --yaw 1.5707963267948966 --device cuda:0
```

CPU is the default. The offline checker exits 2 for a flagged reference; the
runtime audit refuses to enter its policy loop for a flagged reference. A
negative conservative sphere gap is a reason to inspect/reject this reference,
**not proof of an actual robot-mesh collision or an impossible task**. A future
avoidance learner may succeed using a different motion.

## Limits and next gate

- This reference screen checks discrete poses, not continuous-time swept
  articulation or dynamic reachability. Mesh normals at equal-distance ties
  need not be unique; CPU/CUDA signed distances agree while a tie normal may
  choose a different equally close face.
- These are collider covers, not certified visual meshes, physical G1 geometry,
  clothing or carried objects. No new clearance rewards are enabled.
- Contact histories are sampled after policy steps; auto-reset samples are
  excluded. Terminal contact accounting is still incomplete, and no contact
  safety certification is inferred from a diagnostic pass.
- CAT's original field does **not** yet include the stair geometry. The wider
  passage case can have low obstacle geometry buried in the original stairs.
  This is a controlled integration scene, not a vetted training environment.
- Roughly 1.52 m vertical coverage is insufficient for an entire stair course;
  full-body field validity is reported rather than extrapolated silently.
- CAT start/goal and guidance are not driving GRAIL yet. A reference can descend
  and ascend while the CAT field points only one way. Before M2, define feasible
  reference segments/tasks, compose terrain/support geometry with clutter,
  complete terminal contact checks and use larger/appropriate field coverage.

No run from this milestone produces a deployable obstacle-avoiding checkpoint.
