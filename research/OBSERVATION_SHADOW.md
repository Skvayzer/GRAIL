# Obstacle observations and frozen-checkpoint shadow adapter

Simulation-only M1 interface/gradient plumbing. This is **not a trained avoidance
controller**, does not change actor observations/rewards, and cannot send shadow
actions to the environment through this launcher. No robot/ROS/SDK integration.

## What the interface contains

`obstacle_observation.py` samples exact CAT and terrain meshes using the live
robot pose. Origin is the named pelvis/anchor, X follows its horizontal forward
axis, and Z remains world up. Only yaw is used for the local sampling frame;
full WXYZ body rotations are still used to place the articulated collision
probes. Invalid/non-normalized or vertical-forward poses fail explicitly.

| Input | Default shape (without batch) | Meaning |
| --- | --- | --- |
| `volume` | 4 x 13 x 13 x 11 | Sparse 3D distance samples plus two validity channels |
| `probes` | 104 x 8 for this imported G1 | Articulated collider-cover positions, radii and geometry distances/masks |
| `guidance` | 9 | Local goal delta, next support-node direction, graph cost, validity and goal-cell flag |
| `valid` | boolean | All queried geometry known AND root guidance valid |

Grid samples are 16 cm apart, spanning local X/Y -0.96..0.96 m and Z
-0.80..0.80 m relative to the pelvis. These are query points, not a revoxelized
occupancy grid: the CAT source geometry remains at its original resolution.
Thin obstacles may fall between query samples; the full-body probes add local
information but do not make this a complete perception or collision certificate.

Volume channel order is signed closed-CAT distance / 2 m, unsigned terrain
distance / 2 m, CAT validity, terrain validity. Distances are clipped to [-1,1]
or [0,1] respectively. A point below a tread still has positive terrain distance;
terrain has no invented solid sign, occupancy or known-free mask.

Each probe has local XYZ / 2 m, radius / 2 m, CAT distance minus cover radius /
2 m, unsigned terrain **centre** distance / 2 m, and two validity flags. Probe
order/names/offsets/radii come from the actual imported collider cover and are
recorded. Feet, legs, torso and arms remain present. Terrain distances are not
converted to clearance/contact permissions because stance and riser semantics
are not yet validated for RL.

Guidance contains local goal XYZ delta / 5 m (Z is the difference in support
height, not goal-ground minus pelvis height), unit next-node XYZ direction,
graph cost / 5 m, valid, at-goal-cell. Nearest grid cell is used only within the
grid and when that cell is reachable/supported with compatible support height.
There is no search for a more convenient walkable cell, no clamp-to-edge
fallback, and no interpolation across invalid cells. Invalid numeric payloads
are zero only alongside validity zero. Invalid guidance gates the residual off.

This is an omniscient **simulation oracle** against static geometry, not a
LiDAR/depth field. Geometry beyond a finite mesh-field cache can still be queried
exactly here; this must not be confused with observed free space. Sensor masking,
occlusion, latency, coverage and deployment export remain later work.

## Adapter and preservation of GRAIL

`obstacle_adapter.py` contains a 4→8→16 channel 3D CNN with average pooling, a
flattened probe MLP to 64 features, and a 128-wide fusion MLP including guidance.
The last linear layer is zero initialized. Its output is `0.1*tanh(head)`, with
the packet-valid gate applied afterward. For the released checkpoint this is a
64D residual (two 32D tokens), not 29 direct joint commands.

The bound is an experimental latent magnitude, **not** a speed/torque/joint
limit or hardware-safety proof. Zero initialization preserves the released
terrain-conditioned encoding/FSQ/decoder path. Input features and the 29-joint
action order are not replaced. Unknown inputs cannot be marked valid simply by
giving the network zero placeholders: channel masks are checked independently.

The actor already supports `latent_residual_mode="post_quantization"`.
`observation_shadow.py` uses that interface on a deep-copied, frozen actor.
It never modifies the real actor, adds a rollout/history step, samples an action
distribution again, or returns its predictions to the simulator. Its sample
method returns `None`; the evaluator still passes only the original
`policy_model.action_mean` into `env.step`.

For every executed step, including the action before terminal reset:

1. Capture live pose/probes and pack the independent obstacle inputs.
2. Use cloned buffered observations for the frozen actor's baseline and
   zero-residual forward passes. Require exact equality to the real action mean.
3. Require zero residual, finite features/actions, unchanged actor observations
   and unchanged CPU/CUDA RNG state through the shadow forwards.
4. Once a packet is valid, verify a finite nonzero gradient from action loss to
   the new head through the frozen decoder. Use `autograd.grad`, no optimizer,
   no actor parameter gradients and no weight updates.
5. Pair with exactly one environment-step outcome. Save complete/incomplete
   status and verify backbone/adapter hashes on exit.

Initialization uses a forked CPU RNG stream; moving the adapter to CUDA does not
seed the live rollout generator. The actor clone changes internal diagnostic
token caches only. The original actor remains the action source, including
its existing noise/std settings (the shadow checks deterministic action means).

## Commands

Use an explicitly reviewed, role-valid CAT placement. The existing physical
layout and sampled-reference rejection still run before any policy step.

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/YOUR_SPARSE_RANDOM_SEED_0_SCENE \
  --cat-translation 0 0.2 0 --cat-yaw 1.5707963267948966 \
  --num-envs 1 --layout-audit --contact-audit --observation-shadow \
  --execute --accept-isaac-eula --timeout 300

.venv/bin/python research/replay_observations.py \
  --run research/runs/YOUR_SHADOW_RUN --device cpu
.venv/bin/python research/replay_observations.py \
  --run research/runs/YOUR_SHADOW_RUN --device cuda:0
```

EULA acceptance remains explicit; use it only if already accepted. The flag
rejects GUI, multiple environments, missing layout preflight, and training.
No trained adapter load/apply switch is exposed. Existing default launch modes
do not instantiate the adapter or change their inputs.

`observation_shadow.json/.npz` retain channel contracts, probe order, source
hashes, root/quaternion/raw world probe centres, packed features and all three
action means, zero residuals, per-frame parity/timing/outcomes and gradient
evidence. The parent requires independent saved payload validation; an exit
code of zero alone is insufficient.

`replay_observations.py` verifies all relevant source/packet hashes and rebuilds
features from recorded poses and exact meshes on CPU or CUDA. Feature errors
must be <=2e-5 in normalized units and validity masks must match. It loads no
actor and starts no Isaac process. Each replay report is a new ignored run dir;
source data is immutable. This reproduces coordinate/geometry sampling, not
the correctness of hypothetical sensor inputs or learned behavior.

## Findings and remaining gates

Development run `20260911T184716_604447Z_stair_p1_cat_audit` completed with 499
exact action-parity checks, unchanged weights, and adapter-head gradient norm
0.116968. The original rollout's clearance sequence matched the earlier
no-shadow run exactly. 112 unit tests pass, with CPU/CUDA parity and adversarial
saved-payload checks. The final raw-probe/replay extension was added afterward;
clean-run evidence is recorded separately in `PROGRESS.md`.

All geometry queries were valid, but guidance was valid for only 334/499 frames.
Every invalid frame projected the pelvis into a cell that fails the small
support-patch test even though transit is clear. This is expected at tread
edges; a pelvis is not a stance foot. We record the masks instead of erasing
the support test or snapping through obstacles. The hard on/off gate is **not**
ready for continuous control with a trained nonzero residual.

Next work is to design and validate bounded transit-to-support guidance for
the pelvis, build challenge placements that genuinely require avoidance, and
complete phase/contact semantics before RL reward integration. Then introduce
a tiny, separately gated residual-learning experiment with terrain-retention
evaluation. Goal-only control, manipulation adaptation and real-robot use are
not enabled here.
