# CAT-style learning: current direction

The user redirected implementation on 12 September: reproduce CAT's learning
style, not extend the bespoke four-environment M2 residual pilot. **GRAIL stays
the whole-body/terrain controller.** CAT's 12-leg-action policy is a pretrained
avoidance teacher, not a replacement whole-body controller.

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

The named-articulation observation/action bridge and same-state label primitives
are implemented and CPU-verified (below). They are **not yet wired into live
Isaac rollouts**. Actual whole-body distillation and the multi-scene curriculum
trainer are not connected. No renewed overnight run should be described as ready until that path
has a real rollout/update/checkpoint test and comparison with its teachers.
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

Next implementation is the live Isaac teacher-label collector: verify equivalent
named link frames, provide CAT command/gait/delay state, record the **actually
applied GRAIL targets**, and pair teacher leg labels with the GRAIL observations.
`AppliedTargetHistory` never advances from unexecuted teacher predictions.
`teacher_leg_loss` supervises decoded **absolute leg targets in radians** only;
arms/waist receive no fabricated CAT labels. Then connect this to whole-body
distillation with GRAIL retention, before flat specialist/generalist curriculum
and throughput benchmarks. No new overnight training was launched in this step.

Sources: the pinned local CAT source/checkpoint, and the authors' repository:
https://github.com/GalaxyGeneralRobotics/Click-and-Traverse
