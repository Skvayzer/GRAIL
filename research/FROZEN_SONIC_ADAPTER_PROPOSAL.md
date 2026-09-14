---
title: GRAIL-CAT - Frozen SONIC Adapter Proposal v2
aliases:
  - Frozen SONIC CAT adapter
  - GRAIL CAT proposal v2
tags:
  - robotics
  - humanoid
  - unitree-g1
  - reinforcement-learning
  - grail
  - click-and-traverse
  - research-proposal
created: 2026-09-14
updated: 2026-09-14
status: revised-proposal-not-implemented
---

# GRAIL × CAT — Frozen SONIC Adapter Proposal v2

Supersedes [[GRAIL-CAT - Terrain-Aware Whole-Body Control Proposal]]. Related: [[Click-and-Traverse - Training and Policy Architecture]] and [[Obstacle-Aware Loco-Manipulation - Research Contributions]].

> [!important] Decision
> Preserve the complete pretrained SONIC whole-body motor foundation and train a CAT-conditioned obstacle-avoidance adapter around its valid motion commands. **Do not fine-tune the controller, progressively unfreeze it, or replace its nominal motion tokens with zeros.** Train only newly added adapter components, their exploration parameters, and a separate critic.

> [!warning] Scope of this revision
> This is a documentation change, not a new implementation or experiment. No training, simulation, evaluation, robot commands, or deployment changes are authorized by this page. Existing checkpoints and the previous implementation remain intact. Automatic policy evaluation remains disabled under the user's standing request.

## 1. Correction: what the completed experiment actually trained

The completed 227.37M-transition native-CAT-style experiment **did not fine-tune the whole-body action decoder**.

| Component in the completed experiment | Actual treatment |
|---|---|
| GRAIL `g1_dyn` whole-body decoder | Frozen parameters, excluded from the optimizer |
| CAT hidden-feature trunk | Frozen copy of the pretrained teacher trunk |
| CAT-conditioned latent adapter | Trainable |
| 64-dimensional exploration log standard deviation | Trainable |
| Separate privileged value network | Trainable |
| Original GRAIL motion encoder and quantizer | Bypassed during flat navigation; not fine-tuned |
| Nominal navigation token | A zero 64D vector, not an encoded valid motion |

Source audit on 14 September 2026:

- `research/cat_distill_model.py`: `eval().requires_grad_(False)` on the restored controller, copied decoder, and CAT feature trunk.
- `research/cat_parallel_policy.py`: optimizer parameters filtered by `requires_grad`; decoded token is `base + 2*tanh(z)`.
- `research/cat_parallel_env.py`: flat navigation supplies a zero `base`.
- `research/cat_parallel_train.py`: Adam over the trainable list and frozen-module hash checks before checkpointing.

Thus the new experiment is **not “unfrozen decoder versus frozen decoder.”** It is **direct token generation through a frozen decoder versus a GRAIL-style residual around a functioning, fully preserved motion-control pipeline**.

The final 20 training updates recorded 0% success and 88.43% falls. That demonstrates poor training behavior, not the cause. Loss of nominal motion conditioning, command/observation mismatch, exploration, reset distribution, rewards, and contact/termination semantics are hypotheses to distinguish. Freezing alone cannot be claimed as the fix because the decoder was already frozen. The final checkpoint has not been evaluated.

## 2. What “GRAIL-style” means here

GRAIL's object-aware approach keeps SONIC's encoder, quantizer, and decoder frozen and learns a latent residual adapter. Its terrain branch instead fine-tunes the controller with height conditioning. These are different recipes. We propose borrowing the **frozen-base adapter recipe**, conditioning it on CAT geometry rather than assuming its manipulation weights already solve avoidance. [GRAIL §3.3](https://arxiv.org/html/2606.05160v1#S3.SS3)

The pinned manipulation configuration uses `pre_quantization` residuals with scale `0.1`; the local implementation adds the scaled residual to the encoder latent before applying FSQ. Its finger primitives are separate from body control and are not needed for the first avoidance experiment. [Pinned adapter configuration](https://github.com/NVlabs/GRAIL/blob/aa31d8242ac79b11545b9e3635f73014a227bdfc/imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml)

**Proposed foundation:** the already restored `terrain_release/last.pt`, treated as an immutable, terrain-trained SONIC model. This preserves the best available starting point for our stairs objective. It was trained upstream; **we will not train its weights**. Original flat-ground SONIC can be a separate labeled baseline, not a silent replacement or an assumed stair-capable model.

## 3. Architecture: preserve nominal behavior, learn the correction

    Valid nominal motion / causal motion-command generator
                            |
                   Frozen motion encoder
                 + frozen terrain conditioning
                            |
                 Continuous pre-FSQ latent h
                            |
                            + <--- scaled residual Δh
                            |          ^
                       Frozen FSQ      |
                            |     Trainable CAT adapter
                   Frozen body decoder |
                            ^          |
                            |     CAT body fields, guidance,
    Correct proprioception--+     support, goal, nominal intent,
    and action history            task masks and own history
                            |
                     29 joint targets
                            |
                   Existing PD / dynamics

First proposed body interface:

    h_nominal = frozen_encoder(valid_motion_inputs, terrain_inputs)
    r ~ adapter_distribution(observations)
    h_adapted = h_nominal + 0.1 * residual_transform(r)
    tokens = frozen_quantizer(h_adapted)
    q_target = frozen_decoder(tokens, correct_proprioception_history)

The exact insertion order, tensor shape, clipping and scaling must follow the selected checkpoint. `0.1` is an upstream starting point, not a verified safe bound. A bounded residual transform is our proposed extension; its effect on FSQ token changes must be logged. Do not transplant the previous post-FSQ `2*tanh(z)` path and call it the same design.

Two requirements:

1. **Zero residual reproduces the original full controller** on the same state, terrain, reference and history. A generic residual flag can take a different code path; explicitly preserve height conditioning, masks, normalizers and quantizer behavior.
2. **The adapter modifies coordination before body decoding.** It does not overwrite individual arms or legs after the controller. All 29 joints remain jointly controlled: 12 legs, 3 waist, 7 per arm. Fingers/grippers remain a separate later interface.

Frozen weights preserve parameters, not guaranteed balance under arbitrary inputs. Residuals can still produce bad behavior; an out-of-distribution reference or invalid sensor observation can also break the frozen controller.

## 4. Exact frozen/trainable boundary

| Frozen for every training stage | Trainable outside the base |
|---|---|
| Motion/token encoder and any recurrent base state parameters | CAT/support/task feature encoder |
| Terrain height encoder/projector | Residual policy head |
| FSQ settings and any learned quantizer parameters | Adapter action-distribution parameters |
| Whole-body decoder and original action mapping | Separate critic, discarded at deployment |
| Base normalization statistics and fixed gains/scales | Later explicit command/task adapter, if introduced |
| Immutable baseline checkpoint | Adapter-only normalization fitted and versioned separately |

Implementation requirements for the next approved change:

- Use an explicit optimizer allowlist, not just an informal “frozen” label.
- Assert no base parameters are in any optimizer; assert no base gradients or mutable-normalization updates.
- Hash all base weights and persistent buffers before/after updates and save the hashes in every checkpoint.
- Save adapter, critic, normalization, RNG, curriculum and optimizer states separately from the base artifact.
- PPO acts on the sampled residual variable and its exact old log probability. Quantization is part of the action transformation/environment; PPO does not require differentiating through physics or FSQ.
- Any optional differentiable auxiliary loss may propagate to adapter inputs through frozen modules, but must never update base parameters. Do not confuse gradient flow through a module with parameter training.

No automatic “unfreeze if progress is poor” fallback. If the frozen interface cannot express a needed skill, report that limitation and revise the adapter/commands within this boundary.

## 5. Nominal commands: the missing prerequisite

The released terrain checkpoint is a motion tracker, not an already available velocity/goal-to-joint navigation policy. Reinstating its encoder requires real motion inputs; simply placing an encoder box in a diagram is insufficient.

### First experiment: reference-conditioned avoidance

Use compatible locomotion and terrain clips with correct reference state initialization and history. Begin with flat walking/turning/standing references; add clutter that permits a route or posture adjustment. Keep the original robot model, gains, step rate, offsets and observation definitions.

The adapter sees nominal intent and is rewarded for progress, balance and clearance. Do not require exact upper-body tracking through obstacles. Do not place a wall across a compulsory reference and punish both deviation and collision. For larger detours, change the nominal route/reference or allow stopping; small residuals are not a global planner.

A future reference generated by a command system is different from oracle knowledge of future robot state. The former may be a legitimate command, but its generator must exist and be included in the deployed system.

### Later experiment: autonomous goal-conditioned avoidance

Before claiming navigation, provide a **causal motion-command interface** that maps a goal/route, current state and measured support into valid commands for this frozen encoder. Start with a tested reference library/selector for flat locomotion if feasible; stair commands must respect actual support geometry.

This command provider is not already implemented. If it needs learning, train it as a separately identified adapter outside the frozen base. Do not silently revert to zero tokens or a new direct-latent controller.

Audit every existing object/future-reference channel. In early reproduction it can be explicitly reference-conditioned. At deployment it must come from an actual command, measured state, or a documented valid transformation. Zero-filling missing channels is only a diagnostic comparison, not removal of oracle dependence.

## 6. CAT's role: representation, generated scenes and learning structure

Reuse the pinned CAT clutter generation and potential-field/guidance code, with recorded intentional changes for complete-body coverage and terrain contact semantics. Preserve native CAT as a comparison, not as a stair or arm-control expert. [Official CAT repository](https://github.com/GalaxyGeneralRobotics/Click-and-Traverse)

- Train across generated lateral, low, overhead and mixed layouts, with geometry/placement/difficulty distributions recorded.
- Many parallel robots must sample many geometries, not one fixed obstacle. Environment count and unique-layout count are different quantities.
- Reuse the existing scene-generation tools, but expand and version the generated banks as stages grow; do not silently treat the old 180 training layouts as the full CAT distribution.
- Pre-generate banks asynchronously/outside rollout collection where useful; log root seeds, hashes, family coverage, rejection reasons and sampling distribution. Reserve unseen seeds/combinations independently.
- Retain the specialist → generalist recipe: family-specific **residual adapters on the same frozen base**, then adapter-distribution distillation/DAgger and generalist PPO.
- CAT teacher guidance or matched task-level demonstrations can help warm-start flat traversal. Its 12-leg action head is not a 29-joint teacher or a set of compatible residual-token labels. Avoid making full CAT joint-action imitation the main objective for SONIC.

Use the native GRAIL/Isaac Lab adapter workflow as the starting trainer and action-transform path. Add CAT observations/tasks to it rather than first replacing its working low-level dynamics with the previous custom native-CAT model.

## 7. Environment semantics and curriculum

Separate support, forbidden obstacles and allowed contact pairs. Treads may support feet; risers and railings remain obstacles for other links. Include elbows, forearms, hands, torso, pelvis and legs in clearance; use articulated self-collision rules.

Two explicit task tracks prevent an unreported benchmark change:

- **Native CAT comparison:** retain its SDF-only clutter/contact conventions and label results accordingly.
- **Target research task:** use physical clutter plus support-aware fields in GRAIL's simulator. This is a deliberate extension, not an exact CAT reproduction. Fix field/contact thresholds and allowed pairs before comparing runs.

Proposed training order:

| Stage | Learned behavior | Foundation treatment |
|---|---|---|
| A | Neutral adapter with intact standing/walking/turning reference interface | Fully frozen |
| B | Generated easy flat side clutter; small avoidance corrections | Fully frozen |
| C | Separate low/overhead/mixed specialist adapters; progressive gaps and density | Fully frozen |
| D | Generalist adapter distilled from specialists, then mixed-layout PPO | Fully frozen |
| E | Uncluttered terrain rehearsal plus progressively cluttered curbs/slopes/stairs | Fully frozen |
| F | Causal goal-to-motion commands with the same residual interface | Fully frozen |
| G | Masked hand tasks, loads and manipulation-compatible avoidance | Fully frozen |

Track progress per family using training episodes, not reward alone. Avoid a first curriculum dominated by falls: initialize from valid reference states, introduce state/dynamics randomization gradually, and widen geometric difficulty without silently filtering away every hard scene. Keep blocked-route stopping distinct from successful traversal.

A curriculum does not by itself repair an incorrect action transform, frozen-controller input mismatch, or impossible tracking objective.

## 8. Training objectives and useful instrumentation

Start from GRAIL's working tracking/regularization framework, adding CAT clearance/progress objectives with explicit conflict handling:

- Preserve balance, support timing, foot clearance and useful locomotion intent.
- Relax nonessential arm/waist pose tracking when it conflicts with avoiding clutter.
- Reward route progress and stopping at the destination, not merely survival.
- Penalize prohibited contact, self-collision, missed support and joint/torque limits.
- Regularize residual magnitude and rate, plus unnecessary token switching.
- Rehearse uncluttered original behavior under a neutral/small adapter.
- Track manipulation constraints separately when enabled.

Do not blindly carry over reward clipping, strict zero-SDF termination, or nominal upper-body imitation from the failed experiment. Match geometry, margins and reference feasibility first; report each changed semantic.

Before another long run, add training diagnostics:

- `ppo/policy_loss`, `ppo/value_loss`, `ppo/entropy`, `ppo/approx_kl`, `ppo/clip_fraction`, explained variance and gradient norm.
- Residual magnitude/rate, action standard deviation, clipping/saturation, FSQ token-change rate and frozen-base hash status.
- Per-family training completion, falls, prohibited contacts, clearance, duration, route error and reward components.
- Active stage/family, unique generated scenes, training transitions, optimizer steps, throughput and VRAM.

These are **training logs, not evaluation rollouts**. Keep automatic evaluation disabled as requested. Later performance evaluation is user-directed; until then, no held-out-success or deployment-readiness claim.

## 9. Preserve the manipulation objective without adding another uncontrolled policy

Reserve hand-pose masks, reference frames, torso intent, payload context and permitted contacts in the adapter schema. First use free hands; later train feasible hand commands while balancing and traversing clutter.

For manipulation, prefer a shared/fused task-and-geometry adapter or a trained composition with explicit permissions—not two independently trained latent residuals added together without evidence. The frozen base remains the same. GRAIL's original manipulation adapter weights cannot be assumed compatible with the terrain-trained checkpoint.

Learn stand/reach, walk with one constrained hand, and bimanual carrying in progressively harder conditions. Only then add physical grasp/place tasks and embodiment-matched finger/gripper actions. Hand pose tracking and attached-payload tests do not establish grasping.

FABRICS is optional later reference shaping. It is not needed to test the frozen-base hypothesis and does not replace balance/contact validation.

## 10. What is different from the completed run

| Previous completed route | Revised proposal |
|---|---|
| Frozen decoder used alone for flat navigation | Full frozen encoder + terrain conditioning + FSQ + decoder |
| Zero nominal token, direct post-FSQ correction | Valid encoded nominal motion plus small pre-FSQ residual |
| Frozen CAT features + supervised CAT leg-target transfer | CAT fields/guidance plus residual-adapter PPO; task-compatible warm-start only |
| Custom CAT-native physics/reset/control task | Preserve GRAIL's working low-level environment first |
| Flat SDF-only clutter | Separate comparison track from physical terrain/clutter research task |
| No autonomous nominal-motion generator | Explicit reference-conditioned stage and later causal command-provider milestone |
| Generic loss logging | Complete PPO, residual, code-usage and per-family training logs |

The old result remains an auditable baseline. Start a new experiment lineage for the changed action interface; do not resume its optimizer or relabel its checkpoint as the new adapter.

## 11. Research hypothesis and limits

**Hypothesis:** a CAT-conditioned residual that modifies valid SONIC motion latents can acquire obstacle avoidance with less disruption than learning navigation tokens around an empty base.

Candidate contributions, conditional on experiments:

1. Contact-aware 3D avoidance adapters over an immutable terrain-capable motor foundation.
2. Preservation of nominal motion conditioning while connecting CAT guidance to whole-body coordination.
3. Manipulation-compatible geometry/task conditioning over the same frozen base.
4. Controlled comparison of direct-latent generation and reference-centered adaptation, including terrain retention and held-out terrain/clutter/task combinations.

These are not novelty claims merely from combining GRAIL and CAT. Later user-directed tests must separate reference-conditioned tracking from autonomous navigation, adapter-off parameter preservation from adapter-on behavioral retention, and geometry-only diagnostics from physical success.

## 12. Next implementation sequence — not executed by this revision

1. Preserve the completed experiment and create a distinct adapter-v2 config/run lineage.
2. Restore the full terrain controller through the action-transform wrapper, with explicit frozen-parameter/buffer checks and unchanged low-level dynamics.
3. Establish zero-residual parity, including terrain height conditioning, FSQ, normalization and history.
4. Add CAT/support/body-clearance observations and the pre-FSQ residual actor with training-only critic.
5. Add generated-scene banks, curriculum and the missing PPO diagnostics.
6. Start reference-conditioned flat-clutter adapter training only after implementation is requested; keep automatic evaluations off and checkpoint regularly.
7. Add autonomous command generation, terrain/clutter composition and manipulation stages as separate deliverables, not assumed capabilities.

No motor publishers, SDK actuation, locomotion-mode changes or robot resource changes belong to this revision.

## Source and artifact record

- [GRAIL paper §3.3](https://arxiv.org/html/2606.05160v1#S3.SS3), checked 14 September 2026.
- [Official GRAIL tracking documentation](https://nvlabs.github.io/GRAIL/tracking.html).
- [Pinned pre-quantization adapter configuration](https://github.com/NVlabs/GRAIL/blob/aa31d8242ac79b11545b9e3635f73014a227bdfc/imports/SONIC/gear_sonic/config/exp/manager/universal_token/hoi/pnp_table.yaml).
- Local `imports/SONIC/gear_sonic/trl/modules/universal_token_modules.py` and `envs/wrapper/manager_env_wrapper.py`: actual residual/quantizer/action-transform order.
- Local `research/cat_distill_model.py`, `cat_parallel_policy.py`, `cat_parallel_env.py`, `cat_parallel_train.py`: completed-run frozen/learned boundary.
- Immutable terrain-checkpoint SHA-256: `699b17677fa7e6f98ed4cf3dfb99d2fedb56276116a56f2be593aa756fc00120`.
- Completed-run lineage: `20260914_cat_generated_full_25344_noeval_v1`; not the proposed adapter-v2 experiment.
- Version-controlled copy: `research/FROZEN_SONIC_ADAPTER_PROPOSAL.md` in the `research/grail-cat-terrain` branch.
