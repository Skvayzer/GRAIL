# CAT-style learning: current direction

The user redirected implementation on 12 September: reproduce CAT's learning
style, not extend the bespoke four-environment M2 residual pilot. **GRAIL stays
the whole-body/terrain controller.** CAT's 12-leg-action policy is a pretrained
avoidance teacher, not a replacement whole-body controller.

**Direct checkpoint check:** CAT has now controlled Isaac in three original
flat fixtures, compared against MuJoCo. Geometry/field/model checks pass and all
runs remain upright, but overhead clearance fails one native threshold sample.
See [CAT_DIRECT_EVALUATION.md](CAT_DIRECT_EVALUATION.md) for actual results,
videos, reproduction commands and the limits of this partial pass.

## Implemented in this change

- Download the authors' original trainable generalist actor/value/normalizer
  parameter tree, pinned to model revision
  `46ce4b57ba0639168d51741b661ff62f7ce6f045`, with per-file remote hash checks.
  This is a weight warm-start, not restoration of the authors' optimizer.
- Export a frozen PyTorch CAT teacher usable in the Isaac/PyTorch process.
  Its 162D input and 12D action contract are unchanged. A CPU comparison against
  the original Brax policy on 64 synthetic inputs passes: maximum action error
  `2.78279185295105e-6`. This is arithmetic verification, not a robot rollout.
- Copy the **actual released training configuration and scene list**, without
  manually inventing replacement PPO hyperparameters. A hardware profile scales
  environment count and batch size together; it does not change scene sampling,
  reward, entropy, discount, clipping, rollout length or epoch settings.
- Add a reproducible CAT-generator bank utility with recorded seeds/rejections.
  Four new lateral-clutter candidates were generated. They are not automatically
  admitted on stairs, and are not substitutes for the release's full scene set.

## Learning recipe to reproduce

1. Use pretrained CAT avoidance knowledge and GRAIL whole-body locomotion.
2. Start on flat terrain with separate lateral, low and overhead specialist tasks.
3. Train over varied seeds/geometries, then mixed clutter; do not replicate one
   stair fixture and describe it as distributional training.
4. Introduce slopes, curbs and stairs with clutter after the easy skills work,
   retaining GRAIL terrain-only episodes to check for forgetting.
5. Use CAT's specialist-to-generalist DAgger procedure, followed by PPO fine-tuning.

Use CAT's actual HumanoidPF observations, collision/clearance rewards and episode
reset behavior for the avoidance teacher. Do not silently feed our unrelated
oracle packet to its 162D input. Whole-body adaptation must map teacher outputs
by joint name and account for the native **incremental** convention:
`clip(previous_motor_target + 0.5 * action, soft_joint_limits)`.
The earlier description as offsets from default pose was incorrect; defaults
initialize/reset target history, they are not added at every step.
The teacher supplies no wrist/arm targets. Those remain a GRAIL/whole-body task;
zero-padding twelve actions is not a whole-body pretrained policy.

## What remains (not claimed implemented)

Actual flat CAT-to-GRAIL decoder updates, student-controlled simulation,
teacher-assisted DAgger aggregation and exact save/resume now run. See
[CAT_DISTILLATION.md](CAT_DISTILLATION.md) for the initial results and limitations.
The first student has not passed unassisted obstacle traversal. Multi-scene
parallel curriculum, full GRAIL terrain-task integration, retention rollouts
and PPO are not connected. No renewed overnight run should be described as
ready merely because supervised losses and checkpoint tests pass.
The old M2 pilot is not this pipeline and its overnight command must not be used
as a substitute. No custom-model training is left running after this change.

The pinned generalist config lists **37** random scenes; 36 are installed in
the desktop CAT checkout. The missing scene is `D8G2L3O2S13`. The current upstream
README says 42 scenes; the executable configuration is the reproduction source
of truth for this revision, not the README count. Native held-out generation
spec is `eval_scene_specs/randobs_eval_v1.yaml` (480 recipes); these should not
be silently added to the training split.

## Commands / files

From `~/robotics/GRAIL-CAT`:

```bash
.venv/bin/python research/cat_release.py --fetch
# Requires the already-installed CAT Python environment; uses CPU only.
../Click-and-Traverse/.venv/bin/python research/export_cat_teacher.py
.venv/bin/python research/cat_style_recipe.py --num-envs 512

# Optional candidate generation, not a training launch:
.venv/bin/python research/cat_curriculum.py --stage lateral --count 4 --execute
```

The export refuses to overwrite an existing output; the verified export is in
`research/artifacts/cat_teacher_v2/` (same weights as v1, corrected action
description). Large artifacts remain outside git.
Try 256, 512, 1024 environments in measured single-GPU benchmarks, **not** the
authors' multi-GPU 65,536-environment generalist configuration blindly. At this
change another user's workload occupies the GPU; it was not stopped or modified.

## Teacher bridge milestone — 12 September

`cat_bridge.py` implements the original 162D packing, 23 named observed joints,
11 body sites, yaw-only field coordinates, previous-action/target history, and
soft-limit-clipped incremental leg targets. Site offsets/defaults/limits are
exported from CAT's loaded MJCF rather than guessing them from our URDF.

The released teacher uses CAT's legacy X/Z interpolation-weight ordering and
edge clamping. Its compatibility sampler is explicit and separate from our
corrected geometry/safety queries; out-of-grid queries are counted, not treated
as observed free space. The actor gets group-major GF/BF/distance features,
not a per-point-interleaved 77D vector.

Verification command (CPU, no hardware commands):

```bash
../Click-and-Traverse/.venv/bin/python research/check_cat_bridge.py --steps 100
```

Passed on 300 moving MuJoCo states across `side1`, `hurdle1`, `crouch1`:
maximum observation error `2.39e-7`, actor error `8.65e-7`, target error
`3.73e-7` radians. Names were reordered to test MuJoCo/PhysX indexing independence.
The original native training `_get_obs` also passes with noise disabled and
deliberately different current/delayed fields and commands. Only the native
ONNX actor drives the CPU simulation; the port labels exactly those states.
This checks arithmetic/packing, not successful complete obstacle traversal.

Report, model contract and labelled samples:
`research/runs/20260912T093526_836048Z_cat_bridge/`.
230 automated tests pass, including batch-512 packing and leg-only imitation
gradients. The latter is a unit test, **not GRAIL training**.

The live Isaac teacher-label collector is now implemented (next section): it
verifies equivalent named link frames, provides CAT command/gait/delay state,
records the **actually applied GRAIL targets**, and pairs teacher leg labels
with the original GRAIL observations.
`AppliedTargetHistory` never advances from unexecuted teacher predictions.
`teacher_leg_loss` supervises decoded **absolute leg targets in radians** only;
arms/waist receive no fabricated CAT labels. Then connect this to whole-body
distillation with GRAIL retention, before flat specialist/generalist curriculum
and throughput benchmarks. No new overnight training was launched in this step.

## Live teacher collection and GRAIL replay — 12 September

`CatTeacherShadow` is an opt-in recorder in the existing evaluation loop. Only
the unchanged released GRAIL actor drives simulated joints. It saves pre-action
GRAIL observations, CAT observations/labels and the actually applied joint
targets captured before automatic reset. Executed-target history is verified
independently; the initial/reset hold has `history_ready=false`. The teacher
never advances its history from an action that was not executed.

The CAT command projection and stop/gait update are compared directly against
the original native training functions: maximum command error `5.96e-8`, gait
update error zero. The collector uses deterministic 1.4Hz gait, 0.07m foot lift,
five-step odometry delay and no observation noise. This is teacher-query state,
**not a claim to have ported the complete randomized training MDP**. A cached,
vectorized field sampler matches the separate legacy reference sampler.

Evidence:

- Full unchanged-controller rollout:
  `research/runs/20260912T094136_943795Z_stair_p1_cat_audit/`.
  499 frames, 498 with executed history; independent CAT CPU replay error
  `1.58e-6`. Native CAT MuJoCo forward kinematics at the saved Isaac root pose
  and named joint positions matches all 11 sites within `9.36e-7m`.
- Bounded 64-frame repeat with vectorized sampling and resolved GRAIL config:
  `research/runs/20260912T095135_246506Z_stair_p1_cat_audit/`.
  Pairing/replay passed, 63 history-paired frames; native site error below
  `9.27e-7m`. About 56 seconds of rollout on a shared, heavily occupied GPU;
  this is not a training-throughput benchmark.
- CPU reconstruction of the actual GRAIL policy/checkpoint, replaying eight
  in-domain states from the full recording, reproduces original actions within
  `1.08e-6`. CAT absolute-leg-target MSE backpropagates through the frozen decoder
  into its 64D post-quantization input, with finite nonzero gradients. Backbone
  hashes are unchanged, and **zero optimizer steps** were taken.

Only 59/499 frames in the stair run have all sites inside CAT's native field
volume; the 64-frame prefix has none. Original CAT clutter-only fields do not
describe stair traversability, and legacy out-of-bounds clamping does not make
these reliable labels. Both packets therefore remain `training_admitted=false`.
The offline gradient check uses in-domain frames for arithmetic verification,
not as an admitted obstacle-avoidance dataset. This reinforces flat specialists
first, followed by terrain-aware integration; do not train blindly on these
diagnostic stair labels.

Reproduce the collector on the previously reviewed fixture:

```bash
.venv/bin/python research/baseline.py --family stair_p1 \
  --cat-scene research/runs/20260911T144243_747872Z_cat_scene_side-hurdle2 \
  --cat-translation 0 -1 0 --cat-yaw 1.5707963267948966 --layout-audit \
  --cat-teacher-contract research/runs/20260912T093526_836048Z_cat_bridge/contract.json \
  --cat-teacher-steps 499 --execute --accept-isaac-eula --timeout 900

# Replace RUN with the new output directory; no simulation in these checks.
.venv/bin/python research/cat_teacher_results.py RUN
../Click-and-Traverse/.venv/bin/python research/check_cat_shadow_kinematics.py RUN
.venv/bin/python research/replay_cat_teacher_grail.py RUN
```

The first full recording predates resolved-config export. Its gradient check
used `--policy-config-run research/runs/20260912T095135_246506Z_stair_p1_cat_audit`
explicitly. Recovery requires matching frozen-actor hash/joint order/action
clipping, then strict loading of the original recording's own checkpoint and
action replay. The original evidence was not rewritten. New recordings export
their own hashed resolved configuration. An intermediate config-export failure
on a `PosixPath` was fixed and regression-tested before the successful repeat.

Next: connect admitted flat-clutter collection to real whole-body distillation
updates/checkpoints with GRAIL retention; then CAT specialist/generalist DAgger
and PPO over varied scenes, followed by measured 256/512/1024-env scaling. The
one-env diagnostic recorder is not the parallel training implementation.

Sources: the pinned local CAT source/checkpoint, and the authors' repository:
https://github.com/GalaxyGeneralRobotics/Click-and-Traverse
