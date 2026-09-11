# CAT clutter generation in Isaac Lab

Reuse CAT's generator, **not a replacement random-box distribution**. Only the
physical scene exporter is new. The previously added simple solids remain
small validation fixtures; they are not our training-scene generator.

## Preserved upstream components

Source: [Click-and-Traverse](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse/tree/866ba392f1c1e84b92ad75fa66550f26e8af8e48/procedural_obstacle_generation),
revision `866ba392f1c1e84b92ad75fa66550f26e8af8e48`.
`cat_generator_manifest.json` pins the six source/license files by SHA-256.
All six were checked against that Git revision. Downloads remain unmodified.

- `random_obstacle.py`: the original seeded lateral, floor and ceiling masks,
  corridor/gap curves, rotations, start/goal clearance, morphology and local
  widening. The same difficulty and obstacle-count controls are exposed.
- `typical_obstacle.py`: named hurdles, narrow passages, crouching obstacles
  and combinations, including `side-hurdle-crouch2`.
- `grid_config.py` and `pf_grid_config.yaml`: original axes/grid conventions.
- `pf_modular.py`: unchanged fast-marching SDF, boundary gradients, travel-time
  and progressive guidance fields. We do not use the top-level `main.py` wrapper,
  which has fixed output paths and visualization side effects.

This generation is NumPy/SciPy/scikit-fmm code; it does **not** require MuJoCo.
MuJoCo was part of CAT's simulation/training runtime. Our training runtime remains
the isolated GRAIL Isaac Lab environment, with no JAX/MuJoCo trainer imported.
Dependencies are pinned in `env/runtime-pins.txt`; ImageIO stays at Isaac Sim's
required 2.37.0. The metadata audit still has only the three documented M0 exceptions.

Source parity is not cross-version bitwise parity: the existing CAT environment
uses NumPy 2.1.3 / SciPy 1.16.3, while GRAIL uses 1.26.4 / 1.15.3. For the
seed-42 fixture, exactly two of 142,500 occupancy cells differ (16,968 versus
16,966 occupied cells). Do not upgrade GRAIL's NumPy to hide this. Generate once
and reuse the cached, checksummed occupancy/meshes for matched comparisons;
record both the source and numerical dependency versions.

## What the Isaac exporter changes

`cat_scenes.py` builds a closed marching-cubes surface of the generated occupied
cells, then authors a static USD triangle collider. It does **not** convexify the
whole scene: that would fill passages or remove the meaning of overhangs.
Occupancy, SDF and guidance arrays remain unchanged and are saved together with
the mesh, source pins, configuration, dependency versions and output checksums.

The exporter explicitly applies XYZ world origin and the half-voxel sample-center
offset. A free padding layer closes occupied volumes touching grid boundaries.
Face winding points out of occupied space. These are versioned mesh-export
differences from CAT's visualization wrapper, not changes to its generator.

The stock grid has 4 cm voxels and **75 × 50 × 38** cells. Rounding the nominal
1.5 m height gives an actual 1.52 m vertical extent. `origin_corner` and
`sample_origin` are both recorded; the latter is the location of array index zero.
Out-of-grid samples are unknown/invalid in the new field contract, not free space.
The corrected sampler is documented in [M1_GEOMETRY.md](M1_GEOMETRY.md).

CAT's `constants.py` labels its feet-only flat scene for training and its mesh
scene for visualization/testing. Full-body physical clutter contacts in GRAIL
are an **intentional extension**, not a claim of identical MuJoCo dynamics.
All CAT clutter is initially forbidden geometry; the exporter does not label
floor obstacles as walkable stair treads or authorize handrail contact.

## Commands

From `~/robotics/GRAIL-CAT` after the isolated bootstrap:

```bash
# Fetch just the pinned generator, not a second trainer or robot deployment stack.
.venv/bin/python research/cat_scenes.py fetch

# Or reuse a matching existing checkout read-only:
.venv/bin/python research/cat_scenes.py fetch --source-repo ../Click-and-Traverse

.venv/bin/python research/cat_scenes.py generate --scene random --seed 42 --difficulty 0.2
.venv/bin/python research/cat_scenes.py generate --scene side0
.venv/bin/python research/cat_scenes.py generate --scene side-hurdle-crouch2
.venv/bin/python -m unittest discover -s research/tests -v
```

Each generation prints a new directory under `research/runs`. It contains
`scene.usda`, `scene.json`, `obs.npy`, `sdf.npy`, `bf.npy`, `gf.npy`, `travel.npy`.
It does not launch a policy or alter a prior scene. Scene files are not committed.

After the user's Isaac license acceptance, replace the example directory below
with the generated path:

```bash
timeout --signal=TERM --kill-after=10s 180s .venv/bin/python \
  research/physics_contact_check.py --execute --accept-isaac-eula \
  --cat-scene research/runs/YOUR_GENERATED_CAT_SCENE_DIRECTORY
```

This starts headless Isaac Lab, checks occupied **and empty** columns with rays
in both directions along all three axes, then drives a simulation-only sphere
into a planar patch and checks measured contact. Headless scene-query support
is explicitly enabled. It is not a robot movement command or a walking test.

## Evidence and next gate

- 29 unit tests passed with fetched upstream source (none skipped).
- Seed 42, difficulty 0.2, default counts: **16,966 occupied cells**, 14,796
  triangles. Occupancy byte hash:
  `237c442b738826bc734757b746df65327b847a96f9d840f29d95a14478a7e242`.
- `side0`: 4,560 occupied cells and 7,464 triangles. All 384 PhysX ray checks
  passed, including 192 empty-column checks, plus a stopping physical contact.
  Maximum ray surface error was about `1.34e-7 m` on that fixture.
- Random seed-42 scene: all 384 ray checks and its planar-patch contact passed.
  A first contact fixture incorrectly assumed a randomly selected point-ray hit
  was planar for a finite-radius sphere; it hit a bevel correctly. The fixture
  now selects a planar patch from occupancy, without relaxing the normal test.

These establish scene-export/physics agreement for sample scenes, **not learned
obstacle avoidance**. Next: compose this actual CAT clutter with GRAIL terrain
and reachable motion/task corridors, validate full-body terrain/contact fields,
then introduce learning branches. The original flat, fixed-height CAT volume
cannot simply be pasted over multi-level stairs without support-aware placement
and adequate vertical coverage. Do not force a reference motion through clutter
and then reward both exact tracking and avoiding that same clutter.
