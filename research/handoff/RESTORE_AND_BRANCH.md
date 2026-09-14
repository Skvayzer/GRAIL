# Restore context and create an independent branch

Nothing in this file runs automatically. Reading the archive does not require
Isaac Sim, a GPU, model loading, internet authentication or robot access.

## On the original desktop (recommended)

The full source/assets already exist at
/home/constantinesmirnov/robotics/GRAIL-CAT. First inspect:

    cd /home/constantinesmirnov/robotics/GRAIL-CAT
    git status --short
    git branch -vv

After deciding on an unused branch/directory name with the user:

    git worktree add -b research/cat-frozen-adapter-v2 ../GRAIL-CAT-adapter-v2 SNAPSHOT_SHA

Replace SNAPSHOT_SHA with git/GRAIL-CAT/repository.json -> head. This creates
a new worktree without switching or resetting the original branch. If the name
already exists, choose another; do not delete/reset an existing worktree.

A separate worktree does not isolate Python editable installs. The existing
.venv points to source in the original worktree. Do not repoint that shared
environment or globally upgrade packages behind another session. Set up a
separate environment for the new worktree when implementation requires it.

## On another machine

Obtain the full repository at the recorded revision:

    git clone --branch research/grail-cat-terrain https://github.com/Skvayzer/GRAIL.git GRAIL-CAT-adapter-v2
    cd GRAIL-CAT-adapter-v2
    git switch -c research/cat-frozen-adapter-v2 SNAPSHOT_SHA

Authenticate independently if necessary. Credentials are not in this archive.
The GitHub repository is named GRAIL; GRAIL-CAT was the local directory name.

If the fork becomes unavailable but the upstream base is accessible, the
archive contains git/GRAIL-CAT/research-series.mbox and implementation.patch.
The series can be applied with git am to a **fresh clean branch** based on
aa31d8242ac79b11545b9e3635f73014a227bdfc. Do not apply both the series and the
aggregate patch. Replayed commit IDs can differ; compare the resulting tree.

The source/ trees are lightweight text snapshots for inspection. They are not
complete offline clones: binary assets, symlinks and Git databases are omitted.
They are not executable recovery substitutes without the missing dependencies.

## Recover runtime dependencies only when needed

- Consult source/GRAIL-CAT/research/bootstrap.sh and research/env/*.txt.
  The original environment was Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2
  at 37ddf626871758333d6ed89cf64ad702aef127d0, Torch 2.7/CUDA 12.8.
  Hardware/driver/platform support must be checked on the new host.
- Keep Click-and-Traverse as a sibling directory where scripts expect it.
  Its snapshot, commit and upstream URL are included.
- See ASSET_RECOVERY.json for excluded checkpoints, generated NPZ banks,
  pretrained teacher assets, terrain examples and original locations.
- Artifact download scripts and manifests are in research/artifacts.py,
  research/cat_release.py, research/export_cat_teacher.py and the CAT desktop/
  tools. Read their help/implementation before executing. New downloads may
  require license acceptance/access; do not auto-accept licenses.
- Our final learned checkpoint is only known to be on the original desktop;
  no external checkpoint upload is claimed. Copy it separately if desired.
  The new v2 proposal starts a new adapter lineage, not that failed optimizer.
- Recreate heavy generated fields from the included bank/recipe manifests and
  pinned generator, or selectively copy them from the original machine.
- Historical configs include absolute paths. Relocate explicit required inputs
  in new configs. Do not bulk-rewrite the evidence or original run records.

No restore step should contact/reconfigure the robot, arm motion, start policy
evaluation, kill other users' GPU jobs, or launch an overnight job automatically.

## Verify the archive

Check the companion .sha256 before extraction. MANIFEST.json records the
SHA-256, byte count and category of every archived member except itself.
The archive was checked for CRC integrity, duplicate/unsafe member paths,
and common credential-file/token patterns at creation.

The manifest, EXCLUSIONS.json and ASSET_RECOVERY.json are the authoritative
inclusion/exclusion record. Large JSON evidence may appear as .summary.json,
explicitly labeled as derived and with its original checksum; smaller raw
metrics/configs are retained unchanged.

This handoff has curated conversation context, not hidden model memory or a
full raw chat transcript. Read the latest user request before taking action.
