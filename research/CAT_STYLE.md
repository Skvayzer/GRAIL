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
by joint name and account for the native `default_pos + 0.5 * action` convention.
The teacher supplies no wrist/arm targets. Those remain a GRAIL/whole-body task;
zero-padding twelve actions is not a whole-body pretrained policy.

## What remains (not claimed implemented)

The CAT-to-Isaac observation/action bridge, same-state teacher labels, actual
whole-body distillation and multi-scene curriculum trainer are **not connected
yet**. No renewed overnight run should be described as ready until that path
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
`research/artifacts/cat_teacher_v1/`. Large artifacts remain outside git.
Try 256, 512, 1024 environments in measured single-GPU benchmarks, **not** the
authors' multi-GPU 65,536-environment generalist configuration blindly. At this
change another user's workload occupies the GPU; it was not stopped or modified.

Sources: the pinned local CAT source/checkpoint, and the authors' repository:
https://github.com/GalaxyGeneralRobotics/Click-and-Traverse
