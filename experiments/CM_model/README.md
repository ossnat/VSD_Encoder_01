# CM model — channel×space separable encoding

Leave-one-out (Protocol A) evaluation of a **channel×space separable** Ridge
encoder: shared channel weights `a` and per-VSD-pixel spatial maps `M` in CNN
feature space.

## Model

Features stay as tensors `X ∈ R^{C×H×W}` (not flattened).

- **`a ∈ R^C`**: shared channel vector across cortex.
- **`M_p ∈ R^{H×W}`**: spatial map in **feature-map** coordinates for each
  VSD target pixel `p` (typically restricted to the noise-ceiling hull).
- Prediction:

  `y_p = Σ_{c,h,w} a_c · M_p[h,w] · X[c,h,w] + b_p`

Fit by **alternating least squares** (ALS):

1. Fix `a` → RidgeCV for all `M_p` (and bias) on channel-projected features.
2. Fix `M` → Ridge for shared `a` (trial K-fold α on the same alpha grid).
3. Renormalize `‖a‖₂ = 1` (scale absorbed into `M`).

Alpha grid / standardization follow `configs/ridge/default.yaml`.

### What `a` and `M` mean

- **`a`**: how much each CNN channel contributes to the encoding, **shared**
  across cortical pixels. Large `|a_c|` means channel `c` is globally useful.
- **`M_p`**: where in the **stimulus feature map** that cortical pixel reads
  from, after channel weighting. Upsampled overlays put `M` back onto the
  rendered stimulus for qualitative inspection.

### Caveats (ResNet channels)

ResNet ImageNet channels are **not** clean feature detectors. Channel
importance (`a`) is useful for comparing models / layers and spotting
dominant dimensions, but individual channel indices should not be
over-interpreted as “oriented edge #k”. Spatial `M` is in CNN feature
coordinates (e.g. 14×14 for ResNet18 `layer3`), not VSD pixels; overlays
upsample for visualization only.

## Required data

Under the sibling workspace `Data/` (see `configs/default.yaml`):

- `Data/VSD_Encoder_01/encoding_pairs/<monkey>/<window_id>/manifest.parquet`
- `Data/VSD_Encoder_01/DL_features_stimuli/<monkey>/<model>/<layer>/maps/*.npy`
- `Data/VSD_Encoder_01/stimuli/...` (for M-over-stimulus overlays)
- Averaged / H5 trial targets referenced by the pairs manifest
- NC hull mask: `experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`

## Data / trial filter (Protocol A defaults)

| Setting | Value |
|---------|--------|
| Window | `configs/windows/evoked_35_46_zscore.yaml` |
| Loss ROI | NC hull (`--loss-roi noise_ceiling_hull`) |
| Trials | **all** trials (no cleanliness filter) |
| Folds | every held-out **date×condition** (full Protocol A; ~69 folds) |

Optional pilot: `--one-fold-per-stimulus --seed 17` keeps one fold per stimulus.

## Smoke run

```bash
scripts/py experiments/CM_model/run_cm_loo.py \
  --window configs/windows/evoked_35_46_zscore.yaml \
  --protocol A --one-fold-per-stimulus \
  --loss-roi noise_ceiling_hull \
  --stimuli white_point_0.1 \
  --smoke
```

`--smoke` runs the first matching fold with 3 ALS iterations (override with
`--als-iters`).

## Full Protocol A LOO (all session×condition folds)

Same held-out list as `experiments/loo_encoding/heldout_list.yaml`. **Do not**
pass `--one-fold-per-stimulus` for the full stack. Existing fold dirs with
`metrics.json` + `fold_mean_*.npy` are skipped (resume-safe).

```bash
scripts/py experiments/CM_model/run_cm_loo.py \
  --window configs/windows/evoked_35_46_zscore.yaml \
  --protocol A \
  --loss-roi noise_ceiling_hull \
  --run-root 2026-08-08_35-46_resnet18_l3_CM \
  --als-iters 8
```

Subset of stimuli (still all date×condition folds for those IDs):

```bash
scripts/py experiments/CM_model/run_cm_loo.py \
  --window configs/windows/evoked_35_46_zscore.yaml \
  --protocol A \
  --loss-roi noise_ceiling_hull \
  --run-root 2026-08-08_35-46_resnet18_l3_CM \
  --stimuli white_point_0.1 black_triangle_contour_0.4 letter_A_white_1
```

One-fold-per-stimulus pilot (not the full stack):

```bash
scripts/py experiments/CM_model/run_cm_loo.py \
  --window configs/windows/evoked_35_46_zscore.yaml \
  --protocol A --one-fold-per-stimulus --seed 17 \
  --loss-roi noise_ceiling_hull
```

## After encode: pooled fold-pixel r (no refit)

Uses saved `fold_mean_orig.npy` / `fold_mean_recon.npy` only:

```bash
scripts/py experiments/CM_model/plot_pooled_cm_fold_pixel_r.py \
  --run-root experiments/CM_model/runs/2026-08-08_35-46_resnet18_l3_CM

# Fail if any folds_index entry is still missing means:
scripts/py experiments/CM_model/plot_pooled_cm_fold_pixel_r.py \
  --run-root experiments/CM_model/runs/2026-08-08_35-46_resnet18_l3_CM \
  --require-complete
```

Outputs → `…/pooled_fold_pixel_r/` (+ copy under protocol `overview/`).

## Odd–even noise corr (sibling)

Reuse the LOO helper with the CM `folds_index.yaml` (written at encode start):

```bash
scripts/py experiments/loo_encoding/compute_fold_noise_corr_odd_even.py \
  --folds-index experiments/CM_model/runs/2026-08-08_35-46_resnet18_l3_CM/protocol_A_zscore_NChull_CM/folds_index.yaml \
  --output-dir experiments/CM_model/runs/2026-08-08_35-46_resnet18_l3_CM/noise_corr_odd_even \
  --window-zscore configs/windows/evoked_35_46_zscore.yaml \
  --window-raw configs/windows/evoked_35_46.yaml
```

## Compare CM pooled r vs odd–even

```bash
scripts/py experiments/CM_model/compare_cm_vs_noise_corr.py \
  --run-root experiments/CM_model/runs/2026-08-08_35-46_resnet18_l3_CM
```

## Outputs

```
experiments/CM_model/runs/<date>_<frames>_<model>_<layer>_CM/
  protocol_A_zscore_NChull_CM/
    params.yaml
    folds_index.yaml
    loo_summary.csv
    overview/pooled_fold_pixel_r__zscore.{png,npy}
    <fold_id>/
      metrics.json
      dual_metrics_by_stimulus.csv
      channel_weights_a.npy
      mean_abs_M.npy
      mean_signed_M.npy
      fold_mean_orig.npy          # cheap unlock for pooled maps later
      fold_mean_recon.npy
      channel_a_bar.png
      channel_a_bar_top32.png
      sanity_orig_recon_residual.png
      M_overlays/*.png
  pooled_fold_pixel_r/
    pooled_fold_pixel_r__CM_zscore.{png,npy}
    summary.json
  noise_corr_odd_even/
    r_map_pooled_folds__zscore.{png,npy}
    summary.json
  cm_vs_noise_corr_comparison.{csv,json}
  cm_vs_noise_corr__zscore.png
```

No `model.joblib` by default (large; not needed for metrics / overlays).

## Code

| Path | Role |
|------|------|
| `src/encoding/cm_ridge.py` | `build_xy_maps`, ALS fit, predict |
| `src/encoding/cm_plotting.py` | channel bars + M-on-stimulus overlays |
| `experiments/CM_model/run_cm_loo.py` | LOO runner (gitignored `*.py`; still the entrypoint) |
| `experiments/CM_model/plot_pooled_cm_fold_pixel_r.py` | pooled r from saved fold means |
| `experiments/CM_model/compare_cm_vs_noise_corr.py` | CM vs odd–even table + figure |

Reusable fold logic is imported from `src/loo/folds.py` (not copied).
