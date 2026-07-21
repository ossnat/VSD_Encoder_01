# VSD frame-window QC (one-off)

This exploratory utility is intentionally separate from the regular pipeline
and its `plots/` outputs. It displays:

- one raw-frame figure per selected trial for frames `[32, 42)` (32–41), and
- one grid containing the corresponding per-trial means used as encoder
  targets.

Both figure sets are generated with `mapgeog` and `OrRd`. Generated files live
under `exploratory/vsd_frame_qc/results/`, which is ignored by git.

The default samples span four sessions and stimulus types. The
`290518a/condAN1` white point is included as the unseen-stimulus benchmark; all
22 of its trials are already assigned to `test` in the v3 split.

Run from the repository root:

```bash
./scripts/py exploratory/vsd_frame_qc/plot_frame_window_qc.py
```

The script verifies that each displayed mean equals the arithmetic mean of the
ten displayed raw frames. It writes `results/manifest.json` with trial IDs,
splits, frame bounds, colormaps, and output paths.
