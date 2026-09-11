# Training preparation and review gate

Actual avoidance training is **not authorized yet**. The user will first review
recorded environment/setup demos, then decide whether to start. Desktop-only
frozen-policy rollouts, offline tests and mathematical checks are preparation;
they are not a trained obstacle-avoidance result. Do not run the old M0 PPO smoke
command as a substitute for this approval.

## Target of the first training stage

Reference-conditioned, whole-body **residual obstacle avoidance** on the frozen
GRAIL terrain backbone. Retain the original terrain encoder/decoder and their
weights; learn a bounded post-quantization latent adapter and an obstacle-aware
value function. Reference motion remains an input at this stage. Goal-only
student distillation, manipulation adapters and deployment are later stages,
not capabilities to claim from the first training run.

## Completion checklist

- [x] Reproducible frozen terrain checkpoint, four paired reference/scene samples.
- [x] Physical CAT clutter port with source hashes, role and placement checks.
- [x] Oracle 3D/body-cover observations and checked pelvis-to-support guidance.
- [x] Recorded zero-residual action parity and unchanged frozen rollout.
- [ ] Deterministic avoidance-demanding curriculum plus no-clutter retention
  controls and held-out layouts; a side obstacle that never blocks the reference
  is only an integration control, not an avoidance challenge.
- [x] Pure-tensor residual actor/obstacle-aware critic, stochastic action
  contract, clipped PPO objective, reset-aware GAE and detached rollout storage.
- [x] Learner/RNG checkpoint roundtrip at zero updates, no-overwrite and frozen
  backbone/observation contract checks. No PhysX state-resume claim.
- [ ] Runtime optimizer/update loop, post-update checkpoint/resume validation
  and training logs. No real task optimization has been run at this stage.
- [ ] Isaac runtime integration, including pre-reset timeout observations,
  explicit handling of invalid geometry, and 1-/4-environment dry-run checks.
- [ ] Avoidance reward/termination configuration. Exact world-frame foot or
  anchor tracking must not forbid intended detours. Do not treat provisional
  stair contact-phase labels as validated reward targets.
- [ ] Explicit training approval/configuration gate, defaulting to no updates.
- [x] Three actual MP4 review demos, provenance and laptop-copy instructions
  (`REVIEW_DEMOS.md`). These show the integration control, not a trained avoider.
- [ ] Final preflight report and exact pilot/evaluation/resume commands.
- [ ] User has reviewed the demos and approved the first training run.

## Training/evaluation design constraints

Keep CAT signed distance separate from unsigned terrain distance. The open
terrain surface is not a signed solid, and legitimate stance contacts must not
be punished as clutter. The first avoidance reward should use verified clutter
geometry and goal/route progress, plus balance/style regularization; preserve
original terrain retention evaluation separately. Label contact heuristics as
diagnostics until their semantics are validated.

Store the sampled **pre-tanh latent action**, its log probability, observation,
reward, terminal/timeout flags and the correct next-state value. Reset states
must never bootstrap the previous episode. A zero-initialized residual mean
reproduces GRAIL only in deterministic evaluation: Gaussian exploration is
nonzero and must be opt-in, simulator-only and bounded before decoding.

Training readiness means the pipeline and checks are ready to attempt a bounded
pilot, not that it is guaranteed to learn. Four upstream samples are not a
representative terrain benchmark. Full training needs a larger, explicitly
partitioned dataset and retention/avoidance/whole-body clearance evaluations.

No robot SDK, ROS publisher, real-robot connection or actuation is part of these
entry points. Latent limits are not hardware safety guarantees.

## Current tensor implementation

`residual_learning.py` adds a shared obstacle/state trunk, a zero-mean 64D
Gaussian latent policy and an obstacle-aware scalar critic. The caller must
version the concatenated proprioception/reference state. Per-coordinate latent
output is `0.1*tanh(z)`; bounded log standard deviation is -4..-0.5. A dedicated
learner RNG must be provided explicitly. Invalid packets can produce a gated
zero deterministic residual for diagnostics, but cannot silently enter
exploration/likelihood calculations or rollout storage.

`residual_rollout.py` keeps detached, copied, time-major transitions. True
termination removes bootstrap; timeouts bootstrap a separately provided final
state value and stop GAE propagation across the reset. A trajectory horizon
still bootstraps. Advantages use population variance, normalized once over the
whole rollout. No synthetic timeout value is invented by these utilities.

`learning_checkpoint.py` saves only the residual learner, optimizer and its RNG
under explicit model/backbone/observation/algorithm contracts. It never
overwrites existing files and loads tensor dictionaries with `weights_only`.
Resume intentionally starts new simulator episodes. The original actor must
remain outside this optimizer; exact optimizer ownership is checked.

These modules do not launch Isaac, load a released actor, apply an optimizer
step or claim that the end-to-end CAT trainer exists. Their mathematical and
initial-state checkpoint tests are preparation for runtime integration.
