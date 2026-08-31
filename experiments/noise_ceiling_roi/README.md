# Noise-ceiling ROI

**Only supported method:** across-condition split-half reliability → **naive
convex hull at thr=0.90** (magenta outline).

All earlier pilots (per-stimulus union-of-hulls, pooled-concat, max-*r*
contour/cleaned hull, per-stim cleaned-hull reviews) are removed. See
[`across_condition/`](across_condition/) for the method, script, figures, and
source masks.

## Official LOO mask

`--loss-roi noise_ceiling_hull` loads:

`experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`

(+ sidecar `global_noise_ceiling_hull__mask.yaml`)

This file is the **naive** across-condition hull at `r >= 0.90`, currently
built on `win_0035_0046` (raw / `normalization: none`, frames **35–45
inclusive** / `[35, 46)`). It is a **fixed installed artifact**: LOO / ridge
analysis windows do **not** have to match the ROI-creation window. Analysis
uses its own `--window`; `--loss-roi noise_ceiling_hull` loads this alias
(not a path hardcoded in ridge). Window-tagged source masks live under
`across_condition/rois/` (the previous official hull was `win_0035_0042`).

Wiring: CLI `--loss-roi noise_ceiling_hull` → `src/evaluation/loss_roi.py`
(`NOISE_CEILING_HULL_MASK_RELPATH`). NC ROI default `--window` is
`configs/windows/evoked_35_46.yaml` (`nc_roi_utils.DEFAULT_WINDOW`).

**Note:** Older LOO runs (including flatten Ridge lettersA) used the
`win_0035_0042` install. Those run directories are unchanged. Re-run LOO
after regenerating / reinstalling the mask if you need metrics under the
updated ROI.

## ROI window vs analysis window

| Flag / config | Role |
|---|---|
| NC ROI `--window` | Evoked frames + normalization used **only** when building the reliability map / hull |
| LOO / ridge `--window` | Analysis / encoding pairs window (independent) |

Default ROI `--window` is `configs/windows/evoked_35_46.yaml` (match the
current official 35–45 hull). To rebuild on a different range while leaving
analysis elsewhere:

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_42.yaml
```

(Requires encoding pairs / averaged trials for that ROI `window_id`.)

## Run / reinstall

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py
```

Defaults:

- `--window configs/windows/evoked_35_46.yaml` → `win_0035_0046`, `normalization: none`
- `--default-threshold 0.90`
- `--default-variant naive` (magenta hull)
- copies that mask → `rois/global_noise_ceiling_hull__mask.npy` (+ yaml sidecar)

For baseline z-score (`[5, 26)` baseline → z-score → mean `[35, 42)`):

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_42_zscore.yaml
```

Use `--skip-placeholder` to skip the install step. Full details:
[`across_condition/README.md`](across_condition/README.md).
