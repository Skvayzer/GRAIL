# Suggested opening prompt for a new session

Read START_HERE.md, PROJECT_CONTEXT.md, RESTORE_AND_BRANCH.md and
notes/GRAIL-CAT - Frozen SONIC Adapter Proposal v2.md in this handoff archive.
Inspect the actual source and latest Git state before editing.

We are developing a GRAIL/SONIC + Click-and-Traverse research controller for
Unitree G1. The previous 227.37M-transition native-CAT-style run finished but
failed to learn robust traversal. Its decoder was already frozen; navigation
bypassed the encoder and generated tokens around zero. Do not mistake it for
full-body parameter fine-tuning or successful obstacle avoidance.

The active proposal is a full frozen terrain-trained SONIC pipeline with valid
nominal motion inputs and a CAT-conditioned pre-quantization residual adapter.
It is proposed, not implemented. No progressive unfreezing. Autonomous
goal-to-motion generation and manipulation compatibility are separate later
milestones, not inherited capabilities.

I want to work on a separate branch/worktree without disturbing the original
project, robot stack or others' GPU work. Preserve and commit progress.
Do not automatically start training, evaluation or robot actuation merely from
this handoff. Automatic policy evaluations remain disabled unless I ask for
them; training metrics/checkpoints should remain enabled.
Use diverse generated CAT clutter, and add proper PPO diagnostics before
another long training run.

First summarize what is implemented versus proposed and confirm the exact
branch/worktree and next task with my current instructions.
