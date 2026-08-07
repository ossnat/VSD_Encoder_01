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
built on `win_0035_0042` (raw / `normalization: none`). It is a **fixed
installed artifact**: LOO / ridge analysis windows do **not** have to match
the ROI-creation window. Analysis uses its own `--window`; the loss ROI is
whatever mask was last installed.

Wiring: `src/evaluation/loss_roi.py`.

**Note:** Older LOO runs may have used a hull built on `win_0035_0046` or at
thr=0.85; those run directories are unchanged. Re-run LOO after regenerating
the mask if you need metrics under the updated ROI.

## ROI window vs analysis window

| Flag / config | Role |
|---|---|
| NC ROI `--window` | Evoked frames + normalization used **only** when building the reliability map / hull |
| LOO / ridge `--window` | Analysis / encoding pairs window (independent) |

Default ROI `--window` is `configs/windows/evoked_35_42.yaml` (convenient
match to the current official hull). To build the ROI on a different range
(e.g. 35–46) while leaving analysis elsewhere:

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py \
  --window configs/windows/evoked_35_46.yaml
```

(Requires encoding pairs / averaged trials for that ROI `window_id`.)

## Run / reinstall

```bash
scripts/py experiments/noise_ceiling_roi/across_condition/compute_across_condition_reliability.py
```

Defaults:

- `--window configs/windows/evoked_35_42.yaml` → `win_0035_0042`, `normalization: none`
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
