# Frozen CAT controlling Isaac: direct comparison

Date: 12 September 2026. **Partial pass, not complete sim-to-sim validation.**

The original released CAT generalist now actually controls the imported native
G1 in Isaac/PhysX. This is not the earlier GRAIL rollout with CAT shadow labels.
No GRAIL policy, training updates, ROS, robot SDK or hardware commands were used.

## Recorded results

Three deterministic original `TypiObs` fixtures were run in both engines, using
the same unchanged policy and original native-player observation/control logic.

| Native scene | MuJoCo / Isaac steps | Minimum monitored-site SDF, MuJoCo / Isaac | Violation steps, MuJoCo / Isaac |
| --- | --- | --- | --- |
| `side1`: lateral obstacle | 1000 / 108 | -0.008828 / -0.013986 m | 0 / 0 |
| `hurdle1`: low obstacle | 179 / 107 | 0.113090 / 0.117935 m | 0 / 0 |
| `crouch1`: overhead obstacle | 97 / 96 | -0.014089 / **-0.040348 m** | 0 / **1** |

All six episodes stay upright and finish within 0.2 m XY of the native goal
at `[2, 0]`. That radius is a diagnostic metric, not the native player's
success flag. The harness stops at root `x >= 1.9`, a head-height failure, or
1000 policy steps (20 s). MuJoCo `side1` reaches the goal vicinity but remains
at x=1.878 and runs to the time cap; Isaac crosses the exit plane. Do not infer
equal traversal times or identical trajectories from these results.

CAT's native acceptance requires monitored head/feet/hand SDF values above
-0.04 m and head height above 0.7 m. The overhead Isaac run fails that test at
one sampled step, by approximately 0.35 mm. This is **not rounded into a pass**.
A fresh repeat produced the exact same trajectory archive SHA256:
`904e557486b291e15c83e8eb14faa10953f7cc491d448b0dc9fcb80f85d8b830`.

## What was verified

- Original OBJ triangles, original scene placement `[-0.5, -1, 0]` and the
  composed Isaac stage agree: maximum world-vertex error 5.73e-8 m, face indices
  exactly equal. No reconstructed or half-cell-shifted mesh was substituted.
- Ported PyTorch guidance/boundary/SDF sampling agrees with the native player
  at sampled body sites along both engines' saved trajectories: maximum error
  2.50e-6. Out-of-domain queries remain explicitly counted; native legacy
  clamping is not interpreted as known free space.
- Imported robot initial body forward kinematics agree within 7.19e-7 m.
  Maximum mass, COM, joint-limit and inertia-tensor errors are respectively
  6.38e-7 kg, 7.25e-9 m, 3.48e-7 rad and 1.09e-8 kg m².
- PhysX tensor `get_inertias()` is COM-centered but expressed in **link axes**.
  An initial diagnostic mistakenly rotated it a second time; the audit was
  corrected and regression-tested. This did not change the simulated robot.
- Frozen PyTorch checkpoint export matches the original ONNX actor on all
  1587 saved observation vectors across these episodes: maximum action error
  6.26e-7. Actor weights were not fine-tuned to pass the test.
- Native-player post-physics extraction is regression-tested against the
  original complete MuJoCo step, including observations and target history.
  The full automated suite passes: 241 tests in 26.7 s.

The evidence does not indicate an axis flip, misplaced native mesh, wrong joint
mapping or changed checkpoint. It does show different closed-loop trajectories
between engines. The precise cause of the overhead clearance difference has
not been isolated; contact/solver and integration differences remain candidates.

## Fidelity choices and limits

1. Native CAT controls 12 leg joints with incremental position targets:
   `clip(previous_target + 0.5 * action, soft_limits)`. The other 17 joints
   retain native nominal targets through PD, not learned whole-body actions.
2. Policy period is 0.02 s; explicit torque PD is evaluated every 0.002 s.
   Native gains, limits, armature and passive joint settings are carried over.
   The native player uses 1.5 Hz gait and 0.07 m foot lift. This is different
   from the earlier 1.4 Hz deterministic **training-query** shadow bridge.
3. Isaac advances the robot using CPU PhysX. MuJoCo inside the Isaac process
   is only a kinematic/sensor mirror, with no `mj_step` for Isaac dynamics.
   The original player's 2 ms sensor-cache convention is preserved.
4. The MJCF importer omitted native explicit contact pairs because ordinary
   robot collision masks are zero. The runner reconstructs the named foot-floor
   and selected self-contact primitives/pair filters. It does not enable every
   visual geometry as a collider. PhysX and MuJoCo contact solvers, softness and
   pair-specific friction/`condim` behavior are not numerically identical.
5. **Native CAT clutter avoidance is SDF-based:** ordinary robot-clutter
   physical contacts are disabled by the original model. This comparison keeps
   those semantics. No physical collision response is not evidence of no
   intersection. Monitored sites also do not certify full-body clearance.
6. Original `side1` and `hurdle1` meshes are not watertight; `crouch1` is. Native
   stored fields remain the policy's source, not a closed-mesh sign assumption.
7. This verifies three native flat fixtures, **not** the full random training
   distribution, our combined stairs/clutter scenes, real LiDAR representation,
   learned whole-body behavior or safe robot deployment.

## Artifacts and reproduction

Local evidence, intentionally outside git:

- `research/runs/20260912_cat_direct_comparison/`: six hashed episode archives,
  summaries, `comparison.json`, `onnx_parity.json`, imported-model audits,
  `comparison.png`, and `side1_comparison.mp4`, `hurdle1_comparison.mp4`,
  `crouch1_comparison.mp4`.
- `research/runs/20260912_cat_direct_repeat/`: independent overhead repeat.

Videos replay saved trajectories in a **common Isaac renderer**; left is the
MuJoCo physics trajectory, right is Isaac physics. They do not run new physics
or represent a native MuJoCo renderer capture. The common-clock comparison ends
with the shorter recording; single-engine replay is capped at 8 s.

From `~/robotics/GRAIL-CAT`, with the existing pinned CAT checkout, release
artifacts, teacher export and desktop Isaac environment installed:

```bash
CAT_RUN=research/runs/my_cat_direct_check
../Click-and-Traverse/.venv/bin/python research/cat_direct_native.py "$CAT_RUN"
for scene in side1 hurdle1 crouch1; do
  .venv/bin/python research/cat_direct_isaac.py "$CAT_RUN" \
    --scene "$scene" --accept-isaac-eula
done
.venv/bin/python research/cat_direct_results.py "$CAT_RUN"
../Click-and-Traverse/.venv/bin/python research/cat_direct_native.py "$CAT_RUN" --verify-onnx
.venv/bin/python research/cat_direct_isaac.py "$CAT_RUN" \
  --scene crouch1 --render-replay --accept-isaac-eula
.venv/bin/python -m unittest discover -s research/tests -v
```

Use a new run directory: episode archives, comparison reports and videos refuse
overwrite. Native reference physics used MuJoCo 3.3.1 in CAT's Python 3.12
environment; Isaac Sim 5.1 / Isaac Lab 2.3.2 uses Python 3.11 and MuJoCo 3.12.0
for kinematics only. Checkpoint and native-player source hashes are checked
before running. No dependency upgrades are performed by these commands.

Isaac's CPU teardown sometimes hangs after outputs are saved. The launcher
bounds and cleans up only its own child process group, using an explicit
completed/error outcome rather than assuming a zero Kit exit means success.
Another user's running GPU workload was left untouched; these were diagnostic
single-robot CPU-physics runs, not throughput or maximum-GPU-utilization tests.

## Next gate

Expand the native-flat evaluation to multiple starts and more original scenes,
and isolate overhead sensitivity to physics/contact settings without retuning
the teacher merely to pass one fixture. Separately test physical whole-body
clearance/contact semantics needed for GRAIL. Then perform an actual small
distillation update/checkpoint/retention test before broad curriculum training.
This check alone does not make overnight whole-body training ready.
