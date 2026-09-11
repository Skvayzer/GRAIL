# Residual learner runtime preflight (no training)

`baseline.py --residual-preflight` checks the new learner's inputs and episode
boundaries inside Isaac. The released GRAIL policy alone supplies actions;
the learner's zero residual is evaluated only through a frozen clone. No
exploration or optimizer updates occur. These deterministic recordings are
**not PPO behavior data** and must not be used as a training rollout.

## State and reset contract

The privileged M2 state is 2,221 floats for the released configuration:

- Ten oldest-to-newest physical samples of 96 floats: all 29 joint positions
  relative to defaults, 29 joint velocities, named-pelvis linear/angular
  velocities, gravity direction, and 29 last-executed actions. Scales and
  exact joint order are stored in the report.
- Ten future frames for 14 named bodies: position relative to the live pelvis
  in its yaw frame and two rotation-matrix columns, plus normalized clip phase.
  Motion-table offsets are 0, 5, ... 45 steps; indices clamp at the clip end.

This is an oracle/reference-conditioned teacher, not a deployable goal-only
student. The original GRAIL actor retains its own observations and history.

An instance-local `PreResetCapture` calls the original reward computation
unchanged, then reads simulator state **before Isaac auto-reset**. It queries
reference tables at +1 without advancing the command, calling the observation
manager or consuming RNG. The corresponding hypothetical physical history is
used for the next value. For non-reset transitions this is compared with the
actual post-step state/value. For timeouts the captured final value is retained;
the reset state's value is not substituted. Reset physical histories repeat the
new initial sample independently for each environment. True termination masks
bootstrap in the tensor GAE utility (tested separately).

Unknown/invalid current, final or reset geometry fails this preflight. That is
intentional for the unobstructed integration control. Recovery from invalid
observations, falling states and changed/dynamic layouts remains work for the
training runtime; this preflight is not evidence that those cases are solved.

The oracle uses each environment's local coordinates. Reusing a support graph
requires verifying every replica's physical triangle mesh, terrain pose,
environment height and paired reference. Ordinary layout/shadow diagnostics
still default to one environment; replication is opt-in and bounded to four.

## Verified runs

Both runs exited 0 with `outputs_valid=true`, 499 captured transitions per
environment, unchanged actor/learner hashes and exact zero-residual action
parity at every step. Neither used a learned avoidance action.

| Environments | Run (UTC directory name) | Clean timeouts | Failures | Maximum non-reset state difference |
| --- | --- | --- | --- | --- |
| 1 | `20260911T201349_917151Z_stair_p1_cat_audit` | 1 | 0 | 2.3841858e-7 |
| 4 | `20260911T201532_835046Z_stair_p1_cat_audit` | 4 | 0 | 2.9802322e-7 |

The independent `residual_results.audit_runtime` checks checksums, tensor
dimensions/finiteness, consecutive state continuity, per-environment reset
history, next-state/value parity and GAE with a separate NumPy calculation.
Final/reset states differ by ~0.998 in the largest normalized coordinate,
demonstrating that the captures are not simply the new episode's observation.
Asynchronous resets and failure-versus-timeout behavior also have synthetic
unit tests; the four live control episodes time out simultaneously.

The one-environment CAT clearance sample sequence, per-link minima and contact
peaks exactly match the earlier frozen headless run
`20260911T191127_746866Z_stair_p1_cat_audit`. This is a regression check, not a
general proof of identical behavior in all environments.

Recorded transition SHA256:

- One environment: `f331526ad6a4615e1c2157c0ad1615cc9d38be763bc1992ac122f9eaaef8eb35`
- Four environments: `5e8a6d22bd068068abd7f0c8d6e0a62d62da4a561385dfb38f12c57d34f5ca3a`

Actor state SHA256: `4a80b9ba80cf3785999d161867c5d725cc56b76032ce59b723e812a2e96a3ced`.
Learner initial state SHA256: `7ca32b4d57e57eaa77c051e328be3ee29b631ae482c1c9f55653560aff7fb555`.
These were development runs with recorded dirty state. An earlier configuration
lookup failure is retained at `20260911T195732_251509Z_stair_p1_cat_audit`; it
exited before the rollout. No failed record is relabeled as a pass.

## Reproduce

From the repository root, with the previously generated and verified source
scene present (see `REVIEW_DEMOS.md` and `CAT_SCENES.md`):

```bash
.venv/bin/python research/baseline.py --family stair_p1 --num-envs 4 \
  --cat-scene research/runs/20260911T183237_023782Z_cat_scene_random \
  --cat-translation 0 .2 0 --cat-yaw 1.5707963267948966 \
  --layout-audit --residual-preflight --execute --timeout 600
```

Use `--num-envs 1` for the single-environment check. Isaac license acceptance
must already be recorded or explicitly supplied after the user agrees. No
robot network connection is used. `--residual-preflight` rejects training,
interactive GUI, camera recording and other research observation/contact
recorders; an ordinary baseline without the flag is unchanged.

Remaining training gates are tracked in `TRAINING_PREPARATION.md`.
