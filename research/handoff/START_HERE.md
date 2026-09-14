# GRAIL-CAT: start here in the next session

Snapshot date: 14 September 2026. This is a **lightweight context handoff**, not
a complete simulator installation or a raw chat export. Source revisions and
archive checksums are in `MANIFEST.json` and `git/*/repository.json`.

## Current truth

- Active direction: **full frozen SONIC pipeline + CAT-conditioned adapter**.
  Read `notes/GRAIL-CAT - Frozen SONIC Adapter Proposal v2.md`.
- The revised adapter-v2 proposal is **not implemented or trained**.
- The previous native-CAT-style whole-body run **finished**, not crashed, at
  227,373,056 transitions. Its decoder was already frozen, but navigation
  bypassed the encoder/quantizer and generated tokens around a zero base.
- Its final 20 training updates show 0% success and 88.43% falls. Final policy
  evaluation was deliberately not run. Do not call it a successful controller.
- Historical “running” statements/PIDs in older run notes are snapshots, not
  current process status. Check live state before using shared GPU resources.
- No training, evaluation or hardware action should be inferred from opening
  this archive. The next user request determines what to implement.

## Read in this order

1. This file and `PROJECT_CONTEXT.md`.
2. The frozen-SONIC v2 proposal in `notes/`.
3. `source/GRAIL-CAT/research/PROGRESS.md` (newest entries first).
4. `reports/GRAIL_CAT_Implementation_Report_20260914.pdf`.
5. `source/GRAIL-CAT/research/CAT_PARALLEL_TRAINING.md`,
   `CAT_DIRECT_EVALUATION.md`, and `CAT_DISTILLATION.md`.
6. `RESTORE_AND_BRANCH.md` before creating a separate worktree/environment.

The PDF predates the frozen-SONIC v2 proposal. It describes the completed
implementation and results, not the newly proposed adapter.

## Archive layout

- `source/GRAIL-CAT/`: tracked text source, configs, tests and documentation.
- `source/Click-and-Traverse/`: related desktop CAT text source snapshot.
- `source/humanoid_navigation/`: related robot stack/CAT-preview text snapshot.
- `git/`: exact revisions, branches, commit logs, main-project change patch
  and replayable research commit series; no .git database or credentials.
- `notes/`: the six project-related Obsidian pages.
- `evidence/runs/`: saved lightweight logs/configs/metrics; large JSON
  records may have a clearly named summary instead of the original file.
- `evidence/artifacts/`: small artifact metadata, not model weights.
- `reports/`, `figures/`: PDF, plotted report data and selected existing images.
- `ASSET_RECOVERY.json`: locations, hashes/pins and recovery descriptions for
  excluded weights and data. `EXCLUSIONS.json` makes packaging omissions explicit.

Intentionally absent: .venv, model checkpoints, large NumPy scene/trajectory
arrays, meshes/textures, videos, caches, .git object stores, SSH/GitHub/W&B
credentials, and unrelated personal files. No full 105 GB copy was made.

This archive is for private project handoff. Upstream code/assets retain their
licenses; it is not a newly licensed redistribution or a robot deployment bundle.
