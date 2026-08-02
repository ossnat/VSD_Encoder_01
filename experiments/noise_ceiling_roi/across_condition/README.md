# Across-condition split-half reliability ROI

**Supported noise-ceiling ROI method** for this project.

Global reliability convex hull where the **correlation vector length is
`n_stimuli`**, not `n_trials`. Official LOO mask =
**naive (magenta) hull at thr=0.85**.

## Method

For each stimulus / condition in the inventory:

1. Sort trials by `trial_global_id` ascending.
2. **Odd-even split** (0-based): even indices → even half, odd indices →
   odd half.
3. Average maps within each half → one **odd mean map** and one **even mean
   map** per stimulus.

Then:

4. **Stack** all odd mean maps → shape `(n_stim, H, W)`; same for even means.
5. **Per pixel**: Pearson-correlate the length-`n_stim` odd vector with the
   even vector → `r_map[y, x]`.
6. **Spearman-Brown**: `r_sb = 2r / (1 + r)` (map + whole-field pattern).
7. Threshold high-*r* pixels → **convex hull** → filled mask.
   - **Naive hull** (official / magenta outline): hull of all pixels with
     `r >= thr`. At **thr=0.85** this is installed as
     `--loss-roi noise_ceiling_hull`.
   - **Cleaned hull** (lime outline, comparison only): 8-connected components
     with area ≥ `min_component_pixels` (default 50), keep largest
     `keep_top_n` (default 2), then hull of surviving seeds.

**Whole-field pattern reliability** (single-number summary):

- `mean_odd = mean over stimuli of odd half-means`
- `mean_even = mean over stimuli of even half-means`
- `r_half_pattern = Pearson(flatten(mean_odd), flatten(mean_even))`
- `r_sb_pattern = 2 r_half / (1 + r_half)`

## Window / normalization (ROI creation only)

**Default:** raw `configs/windows/evoked_35_46.yaml` → `win_0035_0046`

| Field | Value |
|-------|--------|
| ROI window | `[35, 46)` → frames 35–45 inclusive |
| Normalization | `none` (raw F/F₀ window mean) |

This `--window` chooses the frames/normalization used to **build** the mask.
It does **not** force LOO / ridge analysis to the same window: analysis keeps
its own config, and `--loss-roi noise_ceiling_hull` loads the installed
`.npy` regardless of how that mask was built.

Example — ROI on 35–42, analysis elsewhere still fine:

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_42.yaml
```

Optional baseline z-score (requires encoding pairs for that `window_id`):

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_46_zscore.yaml
```

| Z-score field | Value |
|---------------|--------|
| Baseline | `[2, 26)` → frames 2–25 |
| Then | per-pixel z-score → mean over analysis frames in the YAML |

The script loads start/end/normalization from the window YAML (same helpers as
encoding: `load_trial_mean_maps` / pairs manifest for that `window_id`).

## Run

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py
```

Useful flags:

- `--window <yaml>` — ROI-creation window (independent of LOO analysis window)
- `--thresholds 0.50 0.60 … 0.95` — hull sweep (default includes 0.85–0.95)
- `--default-threshold 0.85` — which threshold’s mask is installed as official
- `--default-variant naive|cleaned` — install naive (default) or cleaned hull
- `--min-component-pixels 50` / `--keep-top-n 2` — cleaned-hull CC filter
- `--threshold-on r|r_sb` — threshold raw *r* or SB-corrected map
- `--sb-map` — display SB map in figures
- `--skip-placeholder` — do not copy into `global_noise_ceiling_hull__mask.npy`

Shared helpers live in `experiments/noise_ceiling_roi/nc_roi_utils.py`
(odd-even split, Spearman-Brown, convex hull / CC filter, hull overlays,
pairs loading, shuffle/blank control helpers).

## Outputs

Under `experiments/noise_ceiling_roi/across_condition/`:

| Path | Content |
|------|---------|
| `figures/<window_id>/across_condition__reliability_map.png` | Grand mean + *r* map |
| `figures/<window_id>/across_condition__reliability_hull_thr0.XX.png` | *r* map + cleaned seeds + cleaned/naive hulls |
| `figures/<window_id>/across_condition__correlation_map__naive_hull_thr0.85.png` | Clean *r* map + magenta naive thr0.85 |
| `figures/<window_id>/across_condition__naive_hull_thr0.85__on_*.png` | Naive thr0.85 overlaid on stim means |
| `rois/across_condition_r_map__<window_id>.npy` | Per-pixel *r* |
| `rois/across_condition_r_sb_map__<window_id>.npy` | Per-pixel *r_sb* |
| `rois/global_across_condition_{cleaned,naive}_hull__<window>__thr0.XX__mask.npy` | Hull masks |
| `rois/global_across_condition_{cleaned,naive}_hull__<window>__thr0.XX.yaml` | Metadata |
| `across_condition_threshold_comparison__<window_id>.csv` | Sweep table |
| `across_condition_summary__<window_id>.yaml` | Global metrics |
| `across_condition_stimuli__<window_id>.csv` | Per-stim trial / half counts |

**Default install** (naive / magenta hull at thr=0.85 unless overridden):

`experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`

(+ sidecar `global_noise_ceiling_hull__mask.yaml`). Used by
`--loss-roi noise_ceiling_hull` (see `src/evaluation/loss_roi.py`).
