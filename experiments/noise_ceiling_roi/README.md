# Noise-ceiling ROI pilot (v2)

Split-half reliability maps per **stimulus_id** to estimate the noise ceiling
before defining ROIs. This is separate from the frozen boxes in
`experiments/loo_encoding/rois/` (v1, untouched).

## Method

### Per-stimulus reliability

Per stimulus (all trials, all sessions pooled from the encoding-pairs manifest):

1. **Sort** trials by `trial_global_id` ascending.
2. **Odd-even split** (0-based indices): even → group A, odd → group B.
3. **Window-averaged maps** per trial using `[35, 42)` / `win_0035_0042`
   (half-open slice, frames 35–41).
4. **Global split-half reliability**
   - `mean_A`, `mean_B` = mean map over each half
   - `r_half = Pearson(flatten(mean_A), flatten(mean_B))`
   - `r_sb = 2 * r_half / (1 + r_half)` (Spearman-Brown)
5. **Spatial reliability heatmap** (`r_map[y, x]`)
   - At each pixel, correlate odd vs even trial vectors
   - Optional SB-corrected map: `r_sb_map = 2r/(1+r)` clipped to `[-1, 1]`
6. **Per-stimulus ROI** (updated: cleaned convex hull)

   **Old (naive) method** — still in default full run for union ROI:
   - Threshold pixels with `r >= threshold`
   - Convex hull of *all* above-threshold pixels (scatter becomes hull vertices)
   - Often too small (few seeds) or too large (speckles inflate the hull)

   **New method (cleaned hull)** — same seed cleaning as global max-r cleaned hull,
   applied per stimulus:

   1. Threshold `r_map >= thr` (per-stimulus threshold sweep; see sample script)
   2. 8-connected components; drop components with area < **35** pixels
   3. Keep largest **N** component(s) (`keep_top_n`, typically 1–2)
   4. **Cleaned convex hull** of surviving seed pixels (lime outline)
   5. Comparison overlay: **magenta** = naive hull of all r≥thr pixels

   Implemented in `cleaned_hull_roi_from_r_map()` /
   `run_per_stim_cleaned_hull_samples.py` (sample review only; full inventory
   not regenerated yet).

   Reliability figure overlays (cleaned-hull review):
     - **cyan scatter** = CC-filtered seed pixels
     - **thick lime outline** (+ light fill) = cleaned convex hull
     - **thin magenta outline** = naive hull (all r≥thr pixels)
   - If fewer than 3 cleaned seed pixels, no hull (points only; no crash)

   **Note (win_0035_0042 samples):** `min_cc=35` works for letter-like foci
   (large CCs) but leaves **empty** cleaned hulls for point / bar / triangle
   stimuli where the largest high-*r* components are only 1–6 px. See
   `per_stim_cleaned_hull_samples__win_0035_0042.md` for sensitivity at
   `min_cc=3` and comparison vs naive hull.

### Threshold selection (FOV cap) — union ROI

Default run **auto-selects** the lowest of `{0.7, 0.75, 0.8}` such that the
**union of per-stimulus hulls** covers **≤ 80%** of the FOV
(`n_union_pixels / (H*W)`). If none meet the cap, the highest candidate (0.8)
is used and all three comparison rows are still written.

- Comparison CSV: `threshold_comparison__<window_id>.csv`
- Chosen threshold is recorded in `global_summary__*.csv`, ROI YAML, and figure titles

Override with `--threshold 0.7` (fixed) or `--threshold-percentile 90`.

### Global ROI = union of per-stimulus hulls (primary)

**Primary method** for the shared analysis ROI:

1. Build each stimulus's convex-hull mask as above (same shared threshold).
2. **Global ROI mask = logical OR** of all per-stimulus hull masks.
   - Includes pixels that are high-reliability for only one stimulus
     (e.g. letter-specific foci).
   - The union may be **non-convex** or **multi-blob** if different stimuli
     activate different regions.
3. Optionally compute an **enclosing convex hull of the union** for a single
   polygon in the YAML / yellow outline in figures. That enclosing hull may
   include extra low-*r* filler pixels between disconnected blobs — it is
   **not** the ROI definition. The saved `__mask.npy` is the **union**.

**Diagnostics (not used to define the primary ROI):**

- Pixelwise **max** across per-stimulus `r_map`s (shown as a middle panel
  in the union figure, and as a dedicated hull sweep — see below)
- Mean of per-stimulus `r_sb` (secondary pattern-reliability summary)

### Max-across-stimuli reliability ROI (alternative)

Pixelwise `max_r = max_stim(r_map_stim)`. This map has **no true split-half
`r_sb`** (it is an envelope of per-stimulus maps). Use pooled whole-field
`r_sb ≈ 0.964` as the primary noise ceiling.

**Important:** a plain (naive) convex hull of all pixels with `max_r ≥ thr`
does **not** exclude scatter. Any above-threshold speck becomes a hull
vertex, so the hull expands to enclose scatter plus filler between blobs.

Shared seed cleaning (both options below):

1. Threshold `max_r ≥ thr`
2. Keep 8-connected components with area ≥ `min_component_pixels`
   (default **50**) and/or the largest `keep_top_n` (default **2**)

#### Max-r contour ROI (option 1: connected components)

Exact pipeline implemented in `contour_roi_from_seed_mask()` /
`run_max_r_threshold_sweep()` (preferred for the two central blobs):

1. **Build `max_r_map`**
   - Stack per-stimulus split-half `r_map`s (shape `(N_stim, H, W)`).
   - `max_r_map = np.nanmax(stacked, axis=0)` → `(H, W)` float32.
   - Diagnostic only: no true split-half `r_sb` on this map.

2. **Threshold → seed mask**
   - `seed_mask = np.isfinite(max_r_map) & (max_r_map >= reliability_threshold)`.
   - `reliability_threshold` = sweep value (default candidates
     `{0.5, 0.6, 0.7, 0.75, 0.8}`; override with `--max-r-thresholds`).
   - Record `n_above_threshold = seed_mask.sum()`.

3. **Connected-component labeling**
   - `scipy.ndimage.label(seed_mask.astype(bool))` → `(labeled, n_labels)`.
   - **8-connected** components (SciPy default).

4. **`min_component_pixels` filter**
   - Count pixels per label via `np.bincount(labeled.ravel())`.
   - Drop labels with area `< min_component_pixels`
     (default **50**; CLI `--min-component-pixels`).
   - Record `n_after_cc_filter` on the surviving seed pixels.

5. **`keep_top_n` largest components**
   - If `keep_top_n` is set (default **2**; CLI `--keep-top-n`, use **0** to
     keep all survivors), sort eligible labels by pixel count descending and
     keep only the top `keep_top_n`.
   - Output: `cleaned_seeds` boolean mask.

6. **Morphological closing**
   - `closed = ndimage.binary_closing(cleaned_seeds, structure=disk_structure(close_radius))`.
   - `disk_structure(radius)`: boolean disk `(xx² + yy²) <= radius²`.
   - Default `close_radius = 3` (CLI `--close-radius`; **0** = skip closing).

7. **Outer contour extraction** (marching squares; OpenCV `findContours` intent)
   - Pad `closed` by 1 pixel with zeros (so edge-touching blobs close).
   - `matplotlib.pyplot.contour(..., levels=[0.5], origin='upper')` on padded mask.
   - Take paths from `ContourSet.allsegs[0]` with ≥ 3 vertices each.
   - Vertices are `(x, y) = (col, row)` float pixel-center coords matching
     `imshow(..., origin='upper')`.

8. **Fill contour interior → ROI mask**
   - For each contour polygon:
     `MplPath(verts).contains_points(pixel_center_grid, radius=0.5)`.
   - OR all fills → `filled_mask` uint8.
   - **Union with closed mask:** `mask = np.maximum(filled_mask, closed.astype(uint8))`
     (keeps seeds / closed interior inside the ROI).

9. **Optional dilation**
   - If `dilate_radius > 0` (CLI `--dilate-radius`, default **0**):
     `ndimage.binary_dilation(mask, structure=disk_structure(dilate_radius))`.
   - Disabled by default.

10. **Saved outputs** (per threshold)
    - **Figure:** `figures/<window_id>/global_max_r__contour_thr0.XX.png`
      (cyan = CC seeds, white/cyan = outer contour fill, dashed lime =
      cleaned convex hull for comparison only).
    - **Mask:** `rois/global_max_r_contour__<window_id>__thr0.XX__mask.npy`
      (uint8 binary).
    - **YAML:** `rois/global_max_r_contour__<window_id>__thr0.XX.yaml`
      (method `noise_ceiling_max_r_cc_close_outer_contour`; records threshold,
      `min_component_pixels`, `keep_top_n`, `close_radius`, `dilate_radius`,
      pixel counts, `contour_fov_fraction`, optional `polygon_vertices`).
    - **CSV row:** appended to
      `max_r_threshold_comparison__<window_id>.csv`
      (`n_contour_pixels`, `fov_frac_contour`, etc.; merges with existing
      rows, replacing same threshold if re-run).

**Comparison (not part of option-1 mask):** cleaned convex hull =
`convex_hull_from_seed_mask(cleaned_seeds)` on the same CC-filtered seeds
(no morph-close / contour / dilate).

#### Cleaned convex hull (comparison)

3. Convex hull of the same CC-filtered seeds (no morph-close / contour).
   The old lime outline on hull figures is this cleaned hull — convex hull
   of CC-filtered seeds, not the contour ROI.

Figures: `global_max_r__reliability_hull_thr0.XX.png` — cyan cleaned seeds,
thick lime cleaned hull, thin magenta naive hull.
YAML/mask: `rois/global_max_r_convex_hull__<window>__thr0.XX.{yaml,npy}`.

Threshold sweep at `{0.5, 0.6, 0.7, 0.75, 0.8}` (override with
`--max-r-thresholds`). Comparison CSV reports
`n_contour_pixels` / `fov_frac_contour` alongside cleaned/naive hull FOV%.

**Recommended defaults (win_0035_0042):** `thr=0.70`, `keep_top_n=2`,
`min_component_pixels=50`, `close_radius=3`. Also inspect `thr=0.60`.
At `thr=0.75`, lower `min_cc≈20` to keep the smaller blob; at `thr=0.80`
components peak at size ~17 so cleaned seeds are empty for min_cc≥20.

### Alternative: pooled / concatenated global reliability

Also computed by default (disable with `--skip-pooled`; alone via `--pooled-only`):

1. **Pool** all trials across all inventory stimuli, sorted by
   `stimulus_id` then `trial_global_id`.
2. **Odd-even split** once on that pooled list.
3. **Per-pixel** correlate odd vs even trial vectors → one global `r_map`.
4. **Whole-field pattern reliability** (primary single-number ceiling):
   - `mean_A`, `mean_B` from the two halves
   - `r_half = Pearson(flatten(mean_A), flatten(mean_B))`
   - `r_sb = 2 * r_half / (1 + r_half)` ← **overall model performance bound**
     for FOV / ROI predictivity
5. **Threshold sweep** on the pooled `r_map` at
   `{0.4, 0.5, 0.6, 0.7, 0.75, 0.8}` (lower cutoffs included because pooled
   per-pixel *r* peaks well below per-stimulus *r*; requested 0.7/0.75/0.8
   are still reported):
   convex hull of above-threshold pixels; figures + YAML/mask per threshold.

**Qualitative note:** pooled per-pixel *r* is typically much lower than
per-stimulus *r* (trials mix many stimulus identities), so hulls at 0.7–0.8
may be empty or tiny. Pooled hulls (when present) tend to be more compact and
may miss stimulus-specific foci that the union-of-hulls ROI keeps. Use pooled
`r_sb` as the ceiling number; prefer union for coverage of stimulus-specific
high-*r* regions.

Secondary bound: mean of per-stimulus `r_sb` (already in
`noise_ceiling_summary__*.csv` / `global_summary__*.csv`).

## Trial count vs “20+ conditions”

The inventory has ~**20 stimulus identities** (conditions in the experimental
design sense). Reliability statistics use **trials per stimulus** (often 5–150+),
not the number of conditions. Per-pixel split-half *r* needs enough trials in
each half; with `n_trials < ~6`, `r_map` is very noisy.

## Interpretation

| Metric | Meaning |
|--------|---------|
| `r_sb_global` (per stimulus) | Ceiling for that stimulus's whole-field spatial pattern |
| **`r_sb_global` (pooled)** | **Overall noise ceiling / max predictivity for whole FOV pattern** |
| `r_map` / peak pixel *r* | **Where** trial-to-trial signal is most repeatable |
| Per-stimulus hull | High-*r* region for that stimulus alone |
| Global union mask | Shared ROI covering all stimulus-specific high-*r* hulls |
| Enclosing hull of union | Single polygon around the union (may add filler) |
| Max-*r* naive hull | Hull of all max_*r* ≥ thr pixels — **includes scatter** |
| Max-*r* cleaned hull | Convex hull of CC-filtered seeds (comparison) |
| Max-*r* contour (opt.1) | CC filter → morph close → outer contour → fill |
| `union_fov_fraction` | `n_union_pixels / (H*W)` at the chosen threshold |
| Pooled hull `fov_frac` | Compact alternative ROI from pooled `r_map` threshold |
| `mean_r_sb_per_stimulus` | Average whole-map pattern reliability across stimuli |
| `fraction_max_r_above_threshold` | Diagnostic: fraction of pixels with max_*r* ≥ threshold |

**Odd-even** is a simple deterministic split. Multi-split averaging (e.g. several
random or blocked splits, then mean *r*) would reduce split-specific noise later.

## Run

```bash
scripts/py experiments/noise_ceiling_roi/compute_noise_ceiling_rois.py
```

**Per-stimulus cleaned-hull sample review** (4 stimuli, threshold sweep, no full inventory):

```bash
scripts/py experiments/noise_ceiling_roi/run_per_stim_cleaned_hull_samples.py
```

**Per-stimulus ROI option review** (Option A reliability vs Option B combined,
4-panel figures with stimulus mean + reliability, samples only):

```bash
scripts/py experiments/noise_ceiling_roi/run_per_stim_roi_option_review.py
```

Outputs: `{stim}__roi_options_review__win_0035_0042.png` (4-panel),
`{stim}__roi_optionA_reliability__*.png`, `{stim}__roi_optionB_combined__*.png`,
`per_stim_roi_options_selected__*.csv`, `per_stim_roi_options_review__*.md`.

Option A sweeps reliability thresholds `{0.3…0.7}`, `min_cc ∈ {3,5,10}`, methods
`cleaned_hull` / `contour`. Option B combines intensity residual (v1-style local
contrast on `stim_mean − global_common`) with a reliability floor, then builds
contour or hull on the gated seed mask.

Options:

- `--window`, `--stimuli ID ...`, `--sb-map` (plot SB-corrected heatmap)
- *(default)* auto threshold among `{0.7, 0.75, 0.8}` with union ≤ 80% FOV
- `--threshold 0.7` — fixed reliability cutoff (disables auto-search)
- `--no-auto-threshold` — use default fixed 0.7 without FOV search
- `--threshold-percentile 90` — percentile of all finite per-stimulus `r_map`
  values (overrides fixed / auto)
- `--skip-global` — per-stimulus only (skip union ROI)
- `--skip-pooled` — skip pooled-concat sweep
- `--pooled-only` — load trials + run pooled threshold sweep only
- `--max-r-only` — load trials + run max-across-stimuli hull/contour sweep only
- `--skip-max-r` — skip max-r hull/contour sweep
- `--min-component-pixels 50` — CC area filter for cleaned hull / contour (default 50)
- `--keep-top-n 2` — keep largest N CCs after min-area (default 2; `0` = all)
- `--close-radius 3` — morph-close disk radius before outer contour (default 3)
- `--dilate-radius 0` — optional binary dilation after contour fill (default 0)
- `--max-r-thresholds 0.50 ...` — max-r sweep cutoffs (default 0.5–0.8)

## Outputs

| Path | Content |
|------|---------|
| `figures/<window_id>/<stimulus_id>__reliability.png` | Stimulus mean + reliability heatmap **with seed + hull overlay** (naive hull in default full run) |
| `figures/<window_id>/<stimulus_id>__cleaned_hull_thr0.XX__topN.png` | Per-stimulus cleaned vs naive hull at threshold (sample review) |
| `figures/<window_id>/<stimulus_id>__cleaned_hull_comparison__topN.png` | Multi-threshold cleaned-hull comparison panel (sample review) |
| `figures/<window_id>/<stimulus_id>__roi_options_review__<window_id>.png` | 4-panel: stim mean, reliability, Option A, Option B (sample review) |
| `figures/<window_id>/<stimulus_id>__roi_optionA_reliability__<window_id>.png` | Option A only (stim mean + reliability + polygon) |
| `figures/<window_id>/<stimulus_id>__roi_optionB_combined__<window_id>.png` | Option B only (stim mean + reliability + polygon) |
| `per_stim_roi_options_selected__<window_id>.csv` | Best Option A/B params per sample stimulus |
| `per_stim_roi_options_sweep__<window_id>.csv` | Full parameter sweep scores |
| `per_stim_roi_options_review__<window_id>.md` | Human-readable option comparison summary |
| `per_stim_cleaned_hull_samples__<window_id>.csv` | Sample threshold sweep: n_above, n_cc, naive/cleaned hull FOV% |
| `per_stim_cleaned_hull_recommendations__<window_id>.csv` | Heuristic recommended thr / keep_top_n per sample stimulus |
| `per_stim_cleaned_hull_samples__<window_id>.md` | Human-readable sample summary |
| `figures/<window_id>/global__union_of_hulls.png` | Grand mean + max-*r* diagnostic + union ROI |
| `figures/<window_id>/global__mean_reliability_map.png` | Same as above (legacy filename alias) |
| `figures/<window_id>/global__convex_hull_roi.png` | Union vs enclosing-hull filler comparison |
| `figures/<window_id>/global_pooled__reliability_map.png` | Grand mean + pooled `r_map` (no thr) |
| `figures/<window_id>/global_pooled__reliability_hull_thr0.XX.png` | Pooled `r_map` + hull at each threshold |
| `figures/<window_id>/global_max_r__reliability_hull_thr0.XX.png` | Max-*r* map + cleaned (lime) vs naive (magenta) hull |
| `figures/<window_id>/global_max_r__contour_thr0.XX.png` | Max-*r* + option-1 outer contour (+ dashed cleaned hull) |
| `noise_ceiling_summary__<window_id>.csv` | Per-stimulus metrics + hull pixel counts |
| `threshold_comparison__<window_id>.csv` | Union auto-search rows for {0.7, 0.75, 0.8} |
| `pooled_threshold_comparison__<window_id>.csv` | Pooled thr sweep + **shared** `r_half` / `r_sb` |
| `max_r_threshold_comparison__<window_id>.csv` | Max-*r* thr × naive / cleaned / contour FOV% |
| `global_summary__<window_id>.csv` | Union ROI metrics + `union_fov_fraction` |
| `rois/global_convex_hull__<window_id>.yaml` | Union metadata + optional enclosing polygon |
| `rois/global_convex_hull__<window_id>__mask.npy` | **Union** binary ROI mask |
| `rois/global_pooled_convex_hull__<window_id>__thr0.XX.yaml` | Pooled hull metadata (incl. `r_sb_global_pooled`) |
| `rois/global_pooled_convex_hull__<window_id>__thr0.XX__mask.npy` | Pooled hull mask |
| `rois/global_max_r_convex_hull__<window_id>__thr0.XX.yaml` | Cleaned max-*r* hull metadata |
| `rois/global_max_r_convex_hull__<window_id>__thr0.XX__mask.npy` | Cleaned max-*r* hull mask |
| `rois/global_max_r_contour__<window_id>__thr0.XX.yaml` | Option-1 outer-contour ROI metadata |
| `rois/global_max_r_contour__<window_id>__thr0.XX__mask.npy` | Option-1 contour ROI mask |
| `rois/<stimulus_id>__convex_hull__<window_id>.yaml` | Per-stimulus hull (optional export) |

## Next steps

- Compare encoder spatial *r* (within union ROI) to pooled `r_sb` and per-stimulus `r_sb`
- Replace proposed ROI with reviewed ROIs under `rois/` when ready
- Multi-split reliability averaging
