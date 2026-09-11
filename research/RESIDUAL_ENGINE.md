# M2 learner engine — preparation, not training approval

Historical engine handoff. The separate approved runtime/overnight launcher is
now implemented; see [M2_TRAINING.md](M2_TRAINING.md) for current operation.
The launcher's pilot Adam rate is 3e-5; the tensor engine's default is unchanged.

The collection/update and simulator-bridge modules are implemented. Existing
Isaac entry points still use frozen actions only; **the new bridge is not yet
exposed through an approved training launcher**. Do not treat this document as
permission to start task optimization or to use the robot.

## Implemented behavior

- `residual_engine.ResidualCollector`: a bounded on-policy horizon with a
  dedicated exploration RNG. Samples a pre-tanh Gaussian latent and stores its
  likelihood/value with a detached copy of the current observation. The latent
  reported as executed must exactly match the sample. Changed learner weights,
  invalid current observations or unpaired steps reject the horizon.
- True terminations have zero bootstrap and never evaluate their invalid final
  observations. Timeouts **must** provide a valid pre-reset final state; invalid
  timeout/ongoing observations abort collection. They are not replaced with
  zeros or silently skipped. Timeouts and terminations stop GAE across resets.
- `PPOUpdater`: clipped policy/value objectives, full-rollout advantage
  normalization, bounded gradient norm, KL early stop and finite checks before
  an optimizer call. It owns exactly the residual learner, never GRAIL. Stale
  likelihood/value data is rejected before gradients. Defaults to gradients
  only: no optimizer calls, no parameter changes, shuffle RNG restored afterward.
  `optimize=True` exists for a future separately approved launcher.
- Defaults: horizon 32, four epochs, minibatch 32, Adam learning rate 0.0003,
  gradient norm 1.0, KL stop 0.02, gamma 0.99, lambda 0.95. These are initial
  experiment settings, not experimentally validated learning hyperparameters.
- `residual_simulator.SimulatorCollector`: opt-in bridge around the actual
  wrapper/decoder API, using a frozen policy clone and the sampled residual at
  the existing post-quantization interface. Retains the pinned evaluator's
  one-step decoder buffering; upstream observation history remains intact.
  Rewards/final states are tapped before Isaac resets. Post-reset learner
  physical history is cleared per environment. Wrapper recovery that changes
  captured rewards or flags invalidates the rollout rather than rewriting it.
- `LearnerStateSampler` and `RuntimeObservation` can query selected environments
  before quaternion/geometry validation. A true-terminal row need not be scored
  for another environment's valid timeout bootstrap. Nonfinite physics/rewards
  that occur before our capture still abort; no general numerical recovery claim.
- Checkpoint schema v2 validates Adam/AdamW type, exact ordered parameter names,
  fixed hyperparameters, moment shapes/dtypes/signs and matching step counters
  before changing live state. Load first exercises optimizer loading on a copy.
  No overwrite. Resume restores learner/optimizer/RNG, **not PhysX state**; it
  starts fresh simulator episodes. V1 checkpoints are intentionally rejected.

## What was actually tested

```bash
cd ~/robotics/GRAIL-CAT
.venv/bin/python -m unittest discover -s research/tests -q
```

193 tests passed, with CUDA available. New tests exercise real tensor sampling,
likelihoods and gradients, including the full 2,221D state / 104-probe /
13x13x11-volume CUDA shapes. Optimizer calls are prohibited or replaced with
no-op spies. No real optimizer step is performed by these tests.

The bridge tests use an **in-memory fake simulator**, including asynchronous
true termination, timeout, invalid terminal rows, wrapper recovery mismatch and
invalid timeout handling. They are not evidence of live stochastic Isaac
rollouts. Nonempty checkpoint moments are directly constructed fixtures, not
post-training checkpoints; real post-update resume remains to be verified.

Actual frozen four-environment Isaac regression:
`20260911T205918_208015Z_stair_p1_cat_audit`, exit 0, outputs valid. All 499 steps
completed; four timeouts, zero failures, exact clone-action parity and unchanged
GRAIL/learner hashes. Selected-environment current/final queries matched the
full batch exactly; maximum non-reset next-state error 2.9802322e-7. No
exploration or optimizer updates. This validates shared observation changes,
not the unexecuted stochastic bridge.

## Remaining integration before an approved pilot

1. Feasible avoidance-demanding scenes, retention controls and held-out splits.
   The current runtime oracle admits validated reference-clear controls only;
   do not bypass that gate to insert a rejected, collision-conflicting layout.
2. Approval-bound launcher/coordinator with bounded total iterations, run-local
   logs/checkpoints, restoration counters, scene/task contracts and evaluation.
   It must check readiness and explicit user approval before enabling exploration
   or optimization. No environment-variable shortcut around these gates.
3. Live Isaac validation of the new collector and asynchronous reset behavior,
   then an explicitly approved tiny update/checkpoint/resume test. The full
   sampler/collector/updater path has not yet been run as task training.
4. Broader terrain-retention and avoidance evaluation before any long run.

The three review videos remain in `REVIEW_DEMOS.md`. They show the frozen
integration control and diagnostics, not learned avoidance.
