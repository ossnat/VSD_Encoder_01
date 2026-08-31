# Leave-one-out encoding experiment

ROI review is **done**. Frozen boxes live in `rois/` (window-independent).

## Status checklist

| Item | Status |
|------|--------|
| 1. ROI freeze (`rois/`) | **done** |
| 2. Window `[35, 43)` → `win_0035_0043` | **done** (config + averaged + encoding pairs). Full non-LOO ridge optional. **201118b excluded** (`src/stimuli/exclusions.py`; Control-attention). **201118a** is in catalog/pairs for **train only** (`train_only_sessions` / `--train-only-dates`); letter tests are **201118c/d**. Z-score variant: `configs/windows/evoked_35_43_zscore.yaml` → `win_0035_0043_zscore`. |
| 3. Stimulus taxonomy | **done** (`stimulus_taxonomy.yaml` / `.csv`) |
| 4. ROI-mask + dual disk/ROI metrics | **done** (`src/evaluation/roi_mask.py`, `dual_metrics.py`, stage-04 `--dual-roi`, LOO runner) |
| 5. LOO scaffolding (protocols A, B & C) | **done** (code + fold manifests + smoke folds) |
| 6. Overview PDF | **done** (`sanity_and_roi_overview.pdf`: sanity, 2×3 ROI, disk vs ROI, taxonomy, LOO smoke) |
| 7. Full protocol A/B sweep | **not started** (smoke only; commands below) |

## Smoke results (protocol B · ResNet18/layer3 · win_0035_0043)

| Fold | n_test | pixel-r disk | pixel-r ROI | spatial-r disk | spatial-r ROI |
|------|-------:|-------------:|------------:|---------------:|--------------:|
| `B__white_point_0.1` | 22 | — (NaN*) | — | 0.449 | 0.477 |
| `B__letter_A_white_1` | 38 | 0.127 | 0.196 | 0.260 | 0.335 |

\*Pixel-r across trials is undefined when reconstructions are constant (identical stimulus features within `stimulus_id`), as for `white_point_0.1`. Prefer **spatial-r**. `letter_A` varies by position across sessions, so pixel-r is defined.

## Key paths

| Path | Role |
|------|------|
| `rois/` | Frozen accepted ROI YAML + masks |
| `heldout_list.yaml` | Shared held-out stimulus IDs for protocols A, B & C |
| `stimulus_taxonomy.yaml` | Taxonomy + heldout/ROI flags |
| `runs/YYYY-MM-DD_35-46_resnet18_l3_{zscore\|raw}/` | **New (default) flat run root** — date + frames + model + layer + norm |
| `runs/.../protocol_A_zscore_NChull_clean/` | Flat leaf — protocol + norm + ROI + cleanliness |
| `runs/<window>/<model>/<layer>/` | **Legacy deep** fold trees (historical runs; still readable) |
| `configs/windows/evoked_35_43.yaml` | Frames `[35, 43)` exclusive end |
| `sanity_and_roi_overview.pdf` | Overview PDF |

## Flat results layout (default for new runs)

New LOO runs write under a short root tagged with normalization
(``_zscore`` / ``_raw``). Raw vs zscore can still share one untagged
root as sibling leaves via ``--run-root`` (Protocol A SLURM pipeline):

```
experiments/loo_encoding/runs/
  2026-08-06_35-46_resnet18_l3_zscore/   # default run root (one window)
    protocol_A_zscore_NChull_clean/      # leaf
      params.yaml                        # full run parameters
      folds_index.yaml
      loo_summary.csv
      overview/                          # optional (triplet overview tool)
      <fold_id>/
        model.joblib                     # saved by default
        metrics.json
        sanity_orig_recon_residual.png   # main VSD-colormap plot only
        alphas_per_target.npy
        ...
  2026-08-06_35-46_resnet18_l3_raw/
    protocol_A_raw_NChull_clean/
    protocol_B_raw_NChull_all/
```

| Piece | Rule |
|-------|------|
| Run root | `YYYY-MM-DD_{start}-{end}_{model}_{layer}_{zscore\|raw}` — e.g. `resnet18_imagenet`→`resnet18`, `layer3`→`l3`, `baseline_zscore`→`zscore`. Shared untagged roots still work via `--run-root` |
| Leaf | `protocol_{A\|B\|C}_{zscore\|raw}_{NChull\|disk\|full\|…}_{clean\|all}` |
| Collision | Default **resumes** into an existing leaf. `--fresh` creates `_HHMM` / `_v2` sibling |
| Models | Saved by default; opt out with `--no-save-model` |
| Legacy | `--layout deep` keeps `runs/<window_id>/<model>/<layer>/protocol_*/` |

Overview tools (`make_loo_triplet_overview.py`, pooled pixel-r plots, replot)
take an explicit `--protocol-dir` and work for **both** flat leaves and old
deep dirs — pass the leaf/protocol directory path.

### After encode: inclusive all-triplets + final per-pixel r

A full (non-array) `run_loo_encoding.py` now writes these at the end of each
leaf (skip with `--skip-post-encode-figures`). SLURM array workers still
need the post stage. Artifacts live under the leaf `overview/`:

| File | What |
|------|------|
| `overview/all_folds_triplets.png` | **Inclusive all-triplets** — every fold's orig \| recon \| residual on one page (`_all` = all trials, not QC-clean-only) |
| `overview/triplet_overview__batch01_of_01.png` | Same collage (batch name) |
| `overview/pooled_fold_pixel_r__{raw\|zscore}.png` | **Final per-pixel r** — Pearson r across fold-mean orig vs recon, NC hull |
| `overview/pooled_fold_pixel_r2__{raw\|zscore}.png` | Same pooling, R² |

Protocol B uses **test-split fold means** (all test trials of the held-out
`stimulus_id`). Protocol C uses stimulus-level mean orig vs a single held-out
ŷ. Both reuse cached `fold_mean_{orig,recon}.npy` — **no ridge refit**.

If encode already finished without these figures:

```bash
# Inclusive all-triplets (one page, every fold)
scripts/py experiments/loo_encoding/finalize_loo_leaf.py \
  --protocol-dir experiments/loo_encoding/runs/<run-root>/protocol_B_zscore_NChull_all \
  --make-overview --overview-inclusive

# Final per-pixel r / R² from cached fold means
scripts/py experiments/loo_encoding/assemble_protocol_B_pooled_maps.py \
  --run-root experiments/loo_encoding/runs/<run-root>

# Protocol A / C assemblers (same layout):
#   assemble_protocol_A_pooled_maps.py
#   assemble_protocol_C_pooled_maps.py
```

### Full Protocol A on SLURM (cluster)

Massive all-fold Protocol A (zscore/raw × clean/all + odd/even noise + PDF):

See **`experiments/loo_encoding/slurm/README.md`**.

```bash
# Path dry-run (safe on laptop)
scripts/py experiments/loo_encoding/prepare_protocol_A_pipeline.py \
  --config experiments/loo_encoding/slurm/protocol_A_full.yaml --paths-only

# On cluster:
PARTITION=generic RUN_DATE=2026-08-07 \
  bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
```

Array workers use `--array-worker` so concurrent fold jobs do not race on
leaf-level `folds_index.yaml` / `loo_summary.csv`.

## Decisions (locked)

- Train/val inside remainder; LOO only for **test**
- Protocol **A** (condition LOO), **B** (stimulus LOO), and **C** (condition
  LOO without same-stimulus train) share the same held-out list
- Baseline model: **ResNet18 / layer3**
- Stimulus CNN features are **window-independent** (do not re-extract for new windows)
- Dual metrics: circular eval disk **and** stimulus ROI mean pixel-r (+ mean trial spatial-r)

## Prepare window 35–43 (already run)

```bash
scripts/py scripts/01_build_averaged_trials.py --window configs/windows/evoked_35_43.yaml
scripts/py scripts/01c_build_encoding_pairs.py --window configs/windows/evoked_35_43.yaml --require-nc
# optional baseline ridge (not required for LOO folds; LOO trains its own models):
scripts/py scripts/03_train_ridge_encoder.py --window configs/windows/evoked_35_43.yaml \
  --model configs/models/resnet18.yaml --feature-layer layer3
```

## Build taxonomy + PDF

```bash
scripts/py experiments/loo_encoding/build_stimulus_taxonomy.py
scripts/py experiments/loo_encoding/make_sanity_and_roi_overview_pdf.py
```

## Run LOO

```bash
# Fold manifests only (flat layout by default)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol both --dry-run

# Smoke one protocol-B fold
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol B \
  --fold-id 'B__white_point_0.1' --smoke

# Full protocol B (all present held-outs; ~10 folds)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol B

# Protocol A (many folds: one per date/condition of each held-out stim; ~69)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol A

# Protocol C (~20 folds: one per stimulus_id; train excludes same stimulus_id)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol C

# Legacy deep tree (optional)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_43.yaml --protocol B --layout deep
```

Flat CLI extras: `--run-date YYYY-MM-DD` (continue a prior day's root),
`--run-root NAME` (explicit root under `runs/`), `--fresh` (new leaf if name
exists). Models are saved by default (`--no-save-model` to skip).

### Training target / loss ROI (`--target-mask` / `--loss-roi`)

Ridge Y targets (MSE) can be restricted to a spatial mask. Both flag names are
aliases; default is **`none`** (full-frame MSE). Resolution lives in
`src/evaluation/loss_roi.py`.

| Flag | Effect | Flat leaf ROI token | Deep dir (legacy) |
|------|--------|---------------------|-------------------|
| `--loss-roi none` (default) | Full FOV multi-output Ridge | `full` | `protocol_{A,B}/` |
| `--loss-roi disk` (or `circular`) | MSE only inside centered circle (radius from `configs/ridge/default.yaml` → `evaluation.mask_radius`, usually 50) | `disk` | `protocol_{A,B}_disk/` |
| `--loss-roi box_union` | MSE inside `experiments/loo_encoding/roi_compare/union_of_boxes__mask.npy` | `boxunion` | `protocol_{A,B}_box_union/` |
| `--loss-roi noise_ceiling_hull` | MSE inside official global naive hull (across-condition thr=0.90 magenta; see path below); **errors clearly if file missing** | `NChull` | `protocol_{A,B}_noise_ceiling_hull/` |
| `--loss-roi roi` | Fit only pixels in the held-out stimulus **box** from `--roi-dir` (default `rois/`) | `boxroi` | `protocol_{A,B}_box_roi/` |
| `--loss-roi path/to/mask.npy` (or `.yaml`) | Custom mask (polygon/ellipse/union) | mask stem | `protocol_{A,B}_<mask_stem>/` (or `protocol_{A,B}_<run-tag>/`) |

**Official path** for `noise_ceiling_hull` (naive / magenta hull; currently
built on `win_0035_0046` raw thr=0.90, frames 35–45 — see
`experiments/noise_ceiling_roi/`). Switch is **not** inside ridge: CLI
`--loss-roi noise_ceiling_hull` loads the installed alias; rebuild with NC
ROI `--window`. The mask file is independent of the LOO `--window`: analysis
uses its config; ROI creation uses NC ROI `--window`; LOO just loads the
installed `.npy`:

`experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`

Predictions are scattered back to the full FOV for plotting (out-of-mask = NaN).
Eval still reports dual **disk / ROI / full** spatial-r metrics (NaN pixels ignored),
plus **train-mask** spatial-r when a target mask is set. Optional `--roi-dir` and `--run-tag`.

```bash
# Full-frame MSE (default; same as omitting the flag)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --loss-roi none \
  --stimuli black_triangle_contour_0.4 --force --no-save-model

# Disk / circular (r=50 from ridge eval config)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --loss-roi disk \
  --stimuli black_triangle_contour_0.4 --force --no-save-model

# Named union-of-boxes (no --run-tag needed)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --loss-roi box_union \
  --stimuli black_triangle_contour_0.4 black_bar_vertical_0.3 letter_D_white_1 \
  --force --no-save-model
# → runs/.../protocol_B_box_union/

# Protocol B: train + score inside per-fold box ROI
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --target-mask roi \
  --stimuli black_triangle_contour_0.4 black_bar_vertical_0.3 letter_D_white_1 \
  --force --no-save-model

# Clean-only (join QC CSV; does not mutate FoundationData indexes)
# Flat leaf: protocol_B_zscore_NChull_clean/  (deep: protocol_B_noise_ceiling_hull__clean_good/)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --loss-roi noise_ceiling_hull \
  --stimuli black_triangle_contour_0.4 black_bar_vertical_0.3 letter_D_white_1 \
  --trial-cleanliness-csv Data/VSD_Encoder_01/qc/trial_cleanliness_gandalf__win_0035_0046_zscore.csv \
  --trial-cleanliness-keep good \
  --force
# Same cleanliness flags work on scripts/03_train_ridge_encoder.py.

# Equivalent custom path + run-tag (legacy style for box_union)
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46_zscore.yaml --protocol B \
  --target-mask experiments/loo_encoding/roi_compare/union_of_boxes__mask.npy \
  --run-tag box_union \
  --stimuli black_triangle_contour_0.4 black_bar_vertical_0.3 letter_D_white_1 \
  --force --no-save-model
```

**Flat (default) example** for protocol A · zscore · NC hull · clean trials:

`experiments/loo_encoding/runs/2026-08-06_35-46_resnet18_l3_zscore/protocol_A_zscore_NChull_clean/`

Each leaf has `params.yaml`, `folds_index.yaml`, `loo_summary.csv`, and per-fold
`metrics.json`, `dual_metrics_by_stimulus.csv`, `sanity_orig_recon_residual.png`,
and `model.joblib` (unless `--no-save-model`).

**Legacy deep** outputs (with `--layout deep`):
`runs/win_0035_0043/resnet18_imagenet/layer3/protocol_{A,B}/<fold_id>/`

## Stage-04 dual ROI report (existing non-LOO models)

```bash
scripts/py scripts/04_evaluate_pixel_correlation.py \
  --window configs/windows/evoked_35_42.yaml \
  --model configs/models/resnet18.yaml --feature-layer layer3
# writes plots/evaluation/.../dual_disk_vs_roi_test.csv
```

## Notes

- Missing held-out IDs are skipped at fold build time.
- Protocol A leakage audit: train/val must not contain the held-out `(date, condition)`;
  other sessions of the same `stimulus_id` may remain (expected).
- Protocol B leakage audit: no train/val trial may share the held-out `stimulus_id`.
- Protocol C: **one fold per held-out stimulus_id** (~20). Test is the first
  ``(date, condition)`` after sorting groups; train/val exclude **all** trials
  with the held-out ``stimulus_id``. See ``protocol_c_test_selection.yaml`` in
  the leaf dir. Sanity PNG default: fold-mean orig | recon | residual (mapgeog /
  ``VSD_CMAP``; same triptych as A/B). Protocol C fold-mean orig is the
  stimulus-level mean over all sessions; recon is a single ``ŷ`` from the
  held-out ``(date, condition)``. Pass ``--sanity-layout sample_trials`` for
  the old recon + random trial originals layout. Leaf dirs: ``protocol_C_*``.

### Protocol A · letters only · 201118a train-only

Letter sessions in data: **201118a, 201118c, 201118d**. **201118b** is
Control-attention (not letters) and stays in `EXCLUDED_H5_SESSIONS`.
Use `heldout_letters.yaml` (`letter_*`) plus `train_only_sessions: [201118a]`
or `--train-only-dates 201118a` so 201118a never becomes a test fold.

```bash
# Rebuild catalog + pairs after un-excluding 201118a
scripts/py scripts/01b_build_stimulus_images.py
scripts/py scripts/02b_extract_stimulus_features.py --feature-layer layer3
scripts/py scripts/01c_build_encoding_pairs.py \
  --window configs/windows/evoked_35_46.yaml --require-nc

scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46.yaml --protocol A \
  --heldout experiments/loo_encoding/heldout_letters.yaml \
  --train-only-dates 201118a --loss-roi noise_ceiling_hull \
  --no-save-model --run-root 2026-08-15_35-46_resnet18_l3_lettersA
```

### Protocol C · non-letters only (letters remain in train)

Test folds are the 12 non-letter `stimulus_id`s in
`heldout_non_letters.yaml` (held-out list minus `letter_*`). Protocol C
train/val still include letters. `train_only_sessions: [201118a]` is set
in that YAML (no non-letter stimulus currently lives on 201118a).

```bash
scripts/py experiments/loo_encoding/run_loo_encoding.py \
  --window configs/windows/evoked_35_46.yaml --protocol C \
  --heldout experiments/loo_encoding/heldout_non_letters.yaml \
  --train-only-dates 201118a --loss-roi noise_ceiling_hull \
  --no-save-model --run-root 2026-08-15_35-46_resnet18_l3_C_nonletters
```
