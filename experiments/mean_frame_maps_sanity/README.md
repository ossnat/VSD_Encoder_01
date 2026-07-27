# Trial-mean VSD maps per frame (sanity check)

For randomly chosen `(date, condition)` groups (fixed seed), plot the
**spatial** trial-mean VSD map at each frame index 24–45 (inclusive) as a
multi-panel heatmap grid — not a scalar timeseries.

## How to run

From the repo root:

```bash
scripts/py experiments/mean_frame_maps_sanity/run_mean_frame_maps.py
```

## Method

1. Load the gandalf trial table (`load_trial_table` + H5 availability filter).
2. Sample 6 `(date, condition)` pairs with ≥3 trials, preferring distinct dates
   (`seed=17`).
3. For each frame `f` in `[24, 45]`:
   - average the `(H, W)` maps across all trials of that condition
   - keep the full spatial map (no spatial reduction / no eval mask)
4. Plot all mean frames in one figure per condition with a **shared color
   scale** within the panel and the project VSD colormap (`mapgeog`).
5. Panel titles are frame indices; the figure title includes date, condition,
   and `n_trials`.

See `index.txt` for the selected conditions after a run.

## Note

This folder replaces the earlier `mean_frame_timeseries_sanity` experiment,
which incorrectly reduced each mean map to a single spatial-mean scalar and
plotted a 1D curve.
