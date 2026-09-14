# Implementation report — 14 September 2026

An 18-page PDF documents the terrain-aware whole-body proposal, actual code
interfaces, generated clutter, prior simulator comparisons, training schedule,
production learning curves, GPU scaling, and remaining research gaps.

The report deliberately separates the reference-conditioned GRAIL terrain and
physical-clutter diagnostic stack from the later native-CAT-style flat-floor,
SDF-clutter training task. Completion of training is not task success.

## Generate from the preserved local evidence

From the GRAIL-CAT repository:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python research/build_implementation_report.py \
  --output /home/constantinesmirnov/Desktop/GRAIL_CAT_Report_20260914
```

Use `--overwrite` only to regenerate this report's own output files. The builder
requires the six named production-run directories, baseline and direct-CAT
records, original proposal, and existing Desktop review images referenced in
the source. These local run/media artifacts are not bundled in Git. No new
simulation, inference, policy evaluation, training, GPU computation, or robot
connection is performed. Matplotlib/NumPy/Pillow in the existing venv suffice;
no extra document-rendering dependency is required.

Output files:

- `GRAIL_CAT_Implementation_Report_20260914.pdf`: selectable text/vector plots,
  architecture diagrams, existing rendered environment images, and references.
- `report_data.json`: joined log records and derived summaries (strict JSON).
- `manifest.json`: exact source revision and input/PDF SHA-256 fingerprints.
- `report_text.md`: textual report content.
- `page_01.png` through `page_18.png`: document-layout previews.

## Recorded training outcome

The final `20260914_cat_generated_full_25344_noeval_v1` status is **completed**:
227,373,056 cumulative environment transitions, 103,680 optimizer updates,
25,344 final parallel environments. Final checkpoint:

```text
research/runs/20260914_cat_generated_full_25344_noeval_v1/checkpoint_000227373056.pt
SHA-256: 7f4240808cc9bd935e1d64e2799df997130eff3df251dbf51dbdc5a63189bdc8
```

The final 20 logged updates contain 285,754 completed training episodes, with
episode-weighted success 0%, fall 88.43%, and collision indicator 15.53%.
Indicators can overlap. These are training metrics, not a deterministic final
evaluation; the final checkpoint remains unevaluated at the user's request.
Neither robust traversal nor terrain retention or manipulation is established.

The existing historical evaluations are reported, not rerun. No evaluation
files exist in either no-evaluation continuation. The inherited 45 visited
validation-scene counter is not evidence of evaluations after `--no-eval`.

Maximum final-run logged process VRAM: 28.5547 GiB (~30.66 decimal GB); median
training throughput ~50.84k environment-steps/s. Different sharing conditions
and stages preclude causal scaling claims. Large VRAM use does not establish
learning quality.

Plot construction verifies parent transition counters and final checkpoint
hash. Layout inspection renders only document pages, not policy trajectories.
