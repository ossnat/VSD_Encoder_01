# Stimulus Catalog Figures

This experiment builds stimulus catalog figures from the paired
stimulus/trial manifests. The original pilot figure is still available, and the
same script now also builds the grouped full catalog figures.

## What each figure shows

- Each row is one `stimulus_id` / condition family.
- The first column is a canonical re-render from the stimulus manifest metadata,
  so synthetic stimuli share the same renderer/background treatment even if older
  cached PNGs in downstream manifests were heterogeneous.
- The remaining columns are per-trial VSD maps, where each map is the mean over
  frames `[start_frame, end_frame)`.

## Default window

This pilot uses `configs/windows/evoked_35_42.yaml`:

- `window_id`: `win_0035_0042`
- `start_frame`: `35`
- `end_frame`: `42`

That is the half-open evoked window `[35, 42)`, meaning frames `35` through
`41`, which matches the existing ROI / noise-ceiling analysis convention.

## Trial display rule

To keep the figure readable on one page, the script caps the displayed trials
per stimulus. By default:

1. Pool all available trials for the chosen `stimulus_id`.
2. Sort them by `trial_global_id`, then `date`, `condition`, `h5_session`
   ascending.
3. Show the first `N` trials, with `N=10` by default.
4. If a stimulus has fewer than `N` available displayed trials, show all of them.
5. Rows can therefore have different visible lengths; the page width is based on
   the maximum row length in the chosen subset.

## Grouped full-catalog rules

The grouped run mode uses only `stimulus_id`s that have usable paired trial data
in the encoding-pairs manifest for the selected window. It then assigns each
available stimulus deterministically from manifest metadata:

- `dots_and_circles`: `shape_type` in `point`, `circle_contour`,
  `filled_circle`
- `geometric_shapes`: `shape_type` in `bar_horizontal`, `bar_vertical`,
  `triangle_contour`
- `letters`: `shape_type == letter`

The current grouped outputs therefore include:

- `dots_and_circles`: `black_point_0.05`, `black_point_0.1`,
  `white_point_0.1`, `black_circle_contour_0.3`, `black_circle_contour_0.95`,
  `white_circle_contour_0.3`, `black_filled_circle_0.3`,
  `white_filled_circle_0.3`, `white_filled_circle_0.8`
- `geometric_shapes`: `black_bar_horizontal_0.3`, `black_bar_horizontal_1`,
  `black_bar_vertical_0.3`, `black_bar_vertical_1`,
  `black_triangle_contour_0.4`
- `letters`: `letter_A_white_1`, `letter_D_white_1`, `letter_F_white_1`,
  `letter_G_white_1`, `letter_L_white_1`, `letter_N_white_1`

Rows are ordered deterministically within each group:

- For `dots_and_circles` and `geometric_shapes`: shape family, then color, then
  `size_deg` ascending, then `stimulus_id`
- For `letters`: letter code ascending, then `stimulus_id`

Available stimuli with data that are intentionally excluded from these grouped
figures are recorded in the metadata JSON. At present that is only the blank
condition, which does not fit any requested visual group.

## Stimulus thumbnail background handling

All catalog thumbnails and model-input PNGs use a **uniform background** of **RGB (128, 128, 128)**, configured in `configs/stimuli/default.yaml` and applied by `src/stimuli/render.py` for every shape type.

- Synthetic shapes (dots, bars, circles, triangle) are drawn on gray 128.
- Letter BMP/MAT assets may carry their own field gray (e.g. 188); the renderer extracts the glyph and composites it onto gray 128.
- Contrast-curve filled circles keep catalog RGB targets; session Blank values are used only for polarity validation, not the canvas background.

## Pilot default subset

The current example uses a small, diverse subset:

- `black_bar_vertical_0.3`
- `black_circle_contour_0.3`
- `black_filled_circle_0.3`
- `black_point_0.1`
- `black_triangle_contour_0.4`
- `letter_A_white_1`

## Run

```bash
scripts/py experiments/stimulus_catalog/build_example_catalog_figure.py
```

Build the grouped full catalog:

```bash
scripts/py experiments/stimulus_catalog/build_example_catalog_figure.py --all-groups
```

Outputs are written under `experiments/stimulus_catalog/figures/`.

## SNR maps (frames 35–45, z-scored)

Per-pixel SNR catalog (stimulus | SNR map | 8 trials), window
`win_0035_0046_zscore` (`[35, 46)` after baseline z-score on `[5, 26)`).

**Per-session catalog** (default; one row per stimulus×session; full FOV):

```bash
scripts/py experiments/report_5/build_stimuli_catalog.py
```

Formula: `snr_pix = |mean_across_trials| / std_across_trials` (ddof=1),
computed **per session** (that date×condition trial set). Bottom scalar =
`mean(snr_pix)` over finite full-frame pixels. Up to 8 **distinct** trials are
shown, evenly spaced across the sorted trial list (deterministic, no RNG).
Outputs:
`experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore_per_session/`.

**Prior pooled-stimulus full-frame catalog** (one row per `stimulus_id`):
`experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore_fullframe/`.

**Legacy NC-ROI catalog** (NaN outside hull; no bottom label):

```bash
scripts/py experiments/report_5/build_stimuli_catalog.py \
  --output-dir experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore \
  --snr-mask experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy \
  --no-snr-scalar-label
```

## Clean vs outlier catalogs

Trial cleanliness (LOO full-FOV QC) is a **standalone** table, not a column on
`all_trials_index_gandalf.csv` (upstream FoundationData) or encoding-pairs
parquet (window-specific). Join at train time:

```python
qc = pd.read_csv(resolve_data_path(
    "Data/VSD_Encoder_01/qc/trial_cleanliness_gandalf__win_0035_0046_zscore.csv"
))
pairs = pairs.merge(qc[["trial_global_id", "trial_cleanliness"]], on="trial_global_id")
pairs = pairs[pairs["trial_cleanliness"] == "good"]
```

Classify / refresh labels:

```bash
scripts/py scripts/17_classify_trial_cleanliness.py
```

Build three SNR catalogs (clean / pattern / amp-edge only), same layout as the
per-session catalog:

```bash
scripts/py experiments/report_5/build_stimuli_catalog_cleanliness.py
```

Outputs:
`experiments/stimulus_catalog/figures/snr_clean_vs_outliers/`.
