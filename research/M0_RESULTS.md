# M0 desktop smoke results — 2026-09-11

These validate installation and training plumbing, **not** a CAT-augmented policy,
terrain generalization, manipulation, or real-robot safety. No robot was accessed
or actuated. The original CAT and SONIC installations were not modified.

## Runtime

RTX 5090, driver 580.178.04; Python 3.11.16; PyTorch 2.7.0+cu128;
Isaac Sim 5.1.0.0; Isaac Lab v2.3.2. Source/model/data pins are in
`provenance.json` and `data_manifest.json`. Package pins and the three explicit
metadata exceptions are described in `README.md` and `PROGRESS.md`.

## Released terrain checkpoint, no fine-tuning

One alphabetically selected paired scene/motion from each family, seed 42,
one environment, original 50 Hz control / 200 Hz physics. These are released
dataset examples, not a held-out evaluation set. A clip ending normally is
different from a failure termination. The audit checks the latter separately.

| Reference family | Debug steps | Failure termination | Global MPJPE (mm) |
| --- | ---: | --- | ---: |
| Stairs | 499 | No | 35.668 |
| Curb | 499 | No | 52.631 |
| Slope | 499 | No | 26.317 |
| Sitting | 248 | No | 16.879 |

All four restored actor weights strictly and recorded finite states and actions.
Raw run directories (ignored by Git, kept locally):

- `20260911T112807_060354Z_stair_p1`
- `20260911T133959_984177Z_curb`
- `20260911T134106_554169Z_slope_evaluation`
- `20260911T134150_217188Z_sitting_evaluation`

Each contains `run.json`, `process.log`, `metrics/metrics_eval.json`,
`trajectory.json`, and `evaluation_audit.json`. Source revisions/dirty flags
are recorded per run; some checks ran while training-only tooling was developed.
The checkpoint and physics inputs did not change between these checks.

## PPO smoke test

Clean code revision: `68382c4`.
Run: `20260911T134531_031926Z_stair_p1_training_smoke`.

- Four environments, eight control steps per update, two PPO updates:
  64 transitions total, using the original reference-tracking objective.
- Strict actor restoration; critic and optimizer restored; fresh counters and
  output directory (the released step-20000 checkpoint remains immutable).
- All 43 actor state tensors changed, with matching keys/shapes and finite values.
- Critic and optimizer tensors finite; no skipped NaN-gradient update detected.
- Step-2 `training/last.pt` saved; exit 0 and `training_audit.json` passed.

This tiny trained checkpoint is disposable, not a candidate for deployment.
It establishes a backward/update/save path, not improved task performance.

The released checkpoint's stair evaluation was also repeated at clean revision
`68382c4`: `20260911T134556_310014Z_stair_p1_evaluation`, exit 0, valid finite
evidence, no failure termination, and the same 35.668 mm global MPJPE.

Ten tooling unit tests pass. All 18 original artifacts still match their hashes
after training, and the dependency audit reports no unexpected conflicts.

## Next gate

M1: physical clutter with separate support/contact and forbidden-obstacle
semantics, full-body clearance probes, and scene/field consistency tests. Then
reference-conditioned avoidance and terrain-retention evaluation can begin.
Goal-conditioned autonomy and manipulation adapters remain later milestones.
