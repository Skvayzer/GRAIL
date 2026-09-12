# CAT teaching the GRAIL whole-body decoder: first flat pilot

12 September 2026. Real optimizer updates and student-controlled physics now
exist. This is a bounded first-stage distillation experiment, **not yet a
successful autonomous whole-body obstacle-avoidance policy**.

## Model and learning signal

- Frozen released CAT hidden layers extract a 64D feature from the native 162D
  observation. No randomly initialized replacement avoidance teacher.
- A 337,856-parameter adapter consumes this feature, normalized 1029D GRAIL
  proprioceptive/object-state history, 64 reference tokens, and a task-mode bit.
  Its correction is `2*tanh(adapter(...))` in continuous post-quantization space.
- Flat CAT mode uses zero base tokens and learns motor tokens from current
  CAT/GRAIL observations. It does **not** require a future demonstration at
  inference, but uses oracle CAT navigation fields, not live LiDAR.
- GRAIL-retention mode uses the original encoder's tokens from recorded GRAIL
  observations. The pretrained GRAIL decoder is frozen in both modes and
  produces **all 29 joint targets**, mapped by name and original action scales.
- Loss: CAT absolute-leg-target MSE + 0.25 * nominal-upper-posture MSE + GRAIL
  all-joint retention MSE. CAT supervises only 12 legs. The other 17 nominal
  targets are an explicit GRAIL-default posture prior, **not CAT arm labels**.
  No claim of learned arm collision avoidance, hand commands or manipulation.
- Adam, learning rate 3e-4, batch 32 (16 flat + 16 retention), gradient norm
  clipped to 1. Adapter statistics are fit on training rows only. Frozen model
  hashes and absence of decoder gradients are checked. These are initial
  supervised-transfer hyperparameters, not a claim to reproduce CAT PPO.

## Data and simulator scope

Eight native `side1` lateral-scene demonstrations vary initial lateral position
within +/-0.12 m and yaw within +/-0.08 rad. Seeds 0..5 train, 6..7 validate.
This is **one geometry with varied starts**, not the full random distribution.
The recorded GRAIL retention packet supplies original GRAIL actions/tokens only;
its previously unadmitted CAT-on-stairs labels are never used.

The direct simulation uses native CAT's G1 MJCF, torque PD and SDF-only clutter
semantics. MuJoCo collection uses the desktop environment's 3.12.0; Isaac uses
CPU PhysX with the previously checked native robot import. This does not yet
validate the full GRAIL terrain task/actuator setup or physical clutter contacts.

Proprioception follows the actual pinned `PolicyCfg` term order: 10-frame
angular velocity, joint position, joint velocity, previous action, gravity;
then object future deltas and current relative pose. Joint/gravity/action
history agrees with the saved live GRAIL packet (joint/action error zero,
gravity error below 1e-7). Flat mode uses a static virtual object at world
origin with identity relative-future rotation, not a zero/invalid rotation.
Full observation equivalence to an actual flat GRAIL environment remains a
separate gate; this runner reconstructs the flat inputs explicitly.

The first strict dataset excluded ordinary contacting foot sites slightly
below z=0. The optional `--support-boundary` rule admits native lower-Z
clamping **only** for a foot with a native floor-contact check, within 2 cm of
the known flat floor. XY, upper-Z, other sites and noncontacting feet stay
strict. Stair-recording guards are unchanged. Isaac collection uses the native
kinematic mirror's ground-contact check, not a measured PhysX contact force.
These boundary exceptions are counted in the expert manifest.

## Actual initial results

- `20260912_cat_distill_flat_v3`: strict-boundary pilot, 468 total rows,
  200 updates. Held-out leg MSE fell to 0.01084 rad², and exact next-update
  save/resume passed. Student fell on both held-out MuJoCo starts; Isaac stayed
  upright for 5 s but did not traverse. This is a failed navigation result.
- `20260912_cat_distill_flat_v4`: contact-aware dataset, 1779 total rows,
  including 1715 flat and 64 GRAIL-retention rows. 600 updates; held-out leg MSE
  0.004733 rad². Both held-out MuJoCo starts still fell.
- `20260912_cat_distill_dagger_v1`: appended 960 labels from student/teacher
  mixed rollouts (50% teacher assistance), preserving the 396 held-out rows.
  Continued to 1400 cumulative updates. Held-out leg MSE 0.003526 rad²;
  held-out retention MSE 0.000881 rad² (retention is not perfect).
  Both unassisted held-out MuJoCo runs stayed upright for 5 s and made limited
  progress, but neither reached the exit. A 25%-assisted Isaac training rollout
  advanced to x=1.334 m but had 40 native clearance violations: not a success.
- `20260912_cat_distill_dagger_v2`: appended 1000 more labels (500 MuJoCo,
  500 Isaac, with 25% or no teacher assistance). Dataset now has 3739 rows;
  the same 396 validation rows remain byte-identical. Continued to **2200
  cumulative updates**, held-out leg MSE 0.003189 rad² and retention MSE
  0.001038 rad². All three tested unassisted short rollouts (MuJoCo seeds 6/7,
  Isaac seed 6) cross x=1.9, but miss the 0.2 m goal radius and violate clearance.
  Extending beyond the exit reveals falls: MuJoCo at 1.88/2.36 s and Isaac at
  2.40 s. **Not a navigation success; the newer checkpoint is not promoted.**

The 1400-step checkpoint remains useful as a slower stability comparison; it
also did not solve the task. None of these checkpoints is ready for deployment.
Full-horizon checks use `--full-horizon` (MuJoCo) or
`--distill-full-horizon` (Isaac) so an exit-plane crossing cannot hide an
immediate fall. Native direct-CAT benchmark defaults remain unchanged.

The training was bounded and is now stopped, with checkpoints and optimizer
states retained. No unattended overnight run was launched. CPU training was
used for this small learner; this is not a GPU-throughput benchmark or the
planned 256/512/1024-environment parallel trainer. The other user's GPU workload
was not stopped or changed.

The first two preparation attempts failed before a dataset was saved (checkpoint
key handling, then NumPy boolean JSON serialization); both were fixed and their
empty run directories retained. All completed run artifacts remain separate.

## DAgger, checkpoint and split integrity

`cat_distill_evaluate.py --collect-dagger` queries the frozen teacher on the
student's **actual current state and executed target history**. Teacher actions
may assist a configurable fraction of collection steps. Assisted rollouts are
explicitly labelled, never reported as independent student success. Validation
seeds cannot be collected into training. Isaac has the same opt-in path via
`--distill-collect` on `cat_direct_isaac.py`.

`cat_distill_aggregate.py` checks source hashes and parent dataset identity,
rejects held-out/duplicate/nonfinite sources, and creates a new dataset without
modifying its parent. `--warm-start` requires that verified parent relationship;
ordinary `--resume` requires the identical dataset. Optimizer, RNG, normalization
and step counts are saved. The next update is tested both in-memory and restored
from disk, bit-exact on CPU, then diagnostic updates are rolled back in memory.

## Reproduction

From `~/robotics/GRAIL-CAT` (existing pinned local installation):

```bash
CAT_RUN=research/runs/my_flat_distillation
CONTEXT=research/runs/20260912T095135_246506Z_stair_p1_cat_audit
.venv/bin/python research/cat_distill.py prepare "$CAT_RUN" \
  --context "$CONTEXT" --support-boundary
.venv/bin/python research/cat_distill.py train "$CAT_RUN" --updates 600
.venv/bin/python research/cat_distill_evaluate.py "$CAT_RUN" \
  "$CAT_RUN/training_0_600/step_600.pt" "$CAT_RUN/eval_600" --full-horizon

.venv/bin/python research/cat_distill_evaluate.py "$CAT_RUN" \
  "$CAT_RUN/training_0_600/step_600.pt" "$CAT_RUN/dagger" \
  --seeds 0 1 2 3 --collect-dagger --teacher-fraction 0.5
.venv/bin/python research/cat_distill_aggregate.py "$CAT_RUN" \
  research/runs/my_dagger_round1 "$CAT_RUN/dagger"
.venv/bin/python research/cat_distill.py train research/runs/my_dagger_round1 \
  --warm-start "$CAT_RUN/training_0_600/step_600.pt" --updates 800

.venv/bin/python research/cat_direct_isaac.py "$CAT_RUN/isaac_eval" \
  --scene side1 --steps 250 --distill-run "$CAT_RUN" \
  --distill-checkpoint "$CAT_RUN/training_0_600/step_600.pt" \
  --distill-seed 6 --distill-full-horizon --accept-isaac-eula
```

Use new output paths. Source checkpoints and datasets are immutable. Evaluation
reports contain the exact student checkpoint hash and cumulative update count.
No ROS/SDK/hardware entry point exists in these scripts.

`cat_distill_report.py` verifies that validation data stayed unchanged and
produces a paired leg-imitation/retention plot plus explicit full-horizon
success checks. Current evidence: `research/runs/20260912_cat_distill_summary/`.
The automated suite passes 252 tests, including actual saved GRAIL history
layout, gradient/frozen-weight checks, resume, held-out leakage rejection,
and rejection of early-exit, assisted, fallen or clearance-violating successes.

## Remaining gates

Unassisted flat traversal and clearance must improve before low/overhead tasks
or stairs. We need broader flat scenes, batched Isaac collection, original
GRAIL task/actuator integration and live terrain-retention rollouts. Recorded
action retention does not prove retained stair capability. Physical whole-body
clearance/contact semantics, PPO, hand commands and sensor inputs remain later
stages. Do not launch the historical failed M2 overnight recipe instead.
