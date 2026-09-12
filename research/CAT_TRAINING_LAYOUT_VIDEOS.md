# Generated training layout previews

CPU-only inspection movies of the accepted generated bank, not learned-policy
rollouts or recordings from Isaac. No simulator, GPU encoder, robot connection,
or training process starts. Training files are verified and read without changes.

The September 12 bundle contains four 12-second H.264 / 720p / 20 fps movies:
lateral, low, overhead, and mixed clutter. Each shows three fixed training seeds
at difficulties 0.4, 0.6, and 0.8. Seed offsets 1, 18, 47 are selected from each
family's requested recipes, independent of visual appearance or success.

The left panel rotates around the original 4 cm occupancy surface. Right panels
show height-colored top and side projections; these are **not planned paths or
traversability maps**. Green is nominal start, red is goal, coordinates are
meters, and Z points up. Native floating forbidden volumes remain floating;
no furniture supports, robot animation, or feasible route is invented. This
training stage uses CAT's SDF obstacle fields with a physical flat floor, so the
solid visualization does not imply physical contact with the obstacle surfaces.

Reproduce from the repository root, using a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python research/render_cat_training_layouts.py \
  --bank research/runs/20260912_cat_generated_bank_v2/bank.json \
  --output /home/constantinesmirnov/Desktop/CAT_training_layouts_20260912 \
  --workers 2
```

The script refuses an existing output directory, verifies each selected NPZ and
occupancy checksum, renders without downsampling the source grid, and decodes
every completed movie to check dimensions, frame count, and changing frames.
The bundle includes twelve PNG stills and a provenance manifest. Its source bank
has 180 unique training layouts and 45 held-out layouts; this is just a sample.

Copy the prepared archive from the laptop:

```bash
scp constantinesmirnov@tl-2:/home/constantinesmirnov/Desktop/CAT_training_layouts_20260912.zip .
```

No viewer server or running simulator is needed; extract and open the MP4s in a
normal video player.
