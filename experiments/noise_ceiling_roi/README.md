# Noise-ceiling ROI

**Only supported method:** across-condition split-half reliability → **naive
convex hull at thr=0.85** (magenta outline).

All earlier pilots (per-stimulus union-of-hulls, pooled-concat, max-*r*
contour/cleaned hull, per-stim cleaned-hull reviews) are removed. See
[`across_condition/`](across_condition/) for the method, script, figures, and
source masks.

## Official LOO mask

`--loss-roi noise_ceiling_hull` loads:

`experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`

(+ sidecar `global_noise_ceiling_hull__mask.yaml`)

This file is the **naive** across-condition hull at `r >= 0.85`, currently
built on `win_0035_0046` (raw / `normalization: none`). It is a **fixed
installed artifact**: LOO / ridge analysis windows do **not** have to match
the ROI-creation window. Analysis uses its own `--window`; the loss ROI is
whatever mask was last installed.

Wiring: `src/evaluation/loss_roi.py`.

**Note:** Older LOO runs may have used a hull built on `win_0035_0042` or at
thr=0.90; those run directories are unchanged. Re-run LOO after regenerating
the mask if you need metrics under the updated ROI.

## ROI window vs analysis window

| Flag / config | Role |
|---|---|
| NC ROI `--window` | Evoked frames + normalization used **only** when building the reliability map / hull |
| LOO / ridge `--window` | Analysis / encoding pairs window (independent) |

Default ROI `--window` is `configs/windows/evoked_35_46.yaml` (convenient;
same frames as the current preferred LOO analysis). To build the ROI on a
different range (e.g. 35–42) while leaving analysis on 35–46:

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
- `--default-threshold 0.85`
- `--default-variant naive` (magenta hull)
- copies that mask → `rois/global_noise_ceiling_hull__mask.npy` (+ yaml sidecar)

For baseline z-score (`[2, 26)` baseline → z-score → mean `[35, 46)`):

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_46_zscore.yaml
```

Use `--skip-placeholder` to skip the install step. Full details:
[`across_condition/README.md`](across_condition/README.md).
