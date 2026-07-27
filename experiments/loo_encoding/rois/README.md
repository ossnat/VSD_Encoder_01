# Frozen stimulus ROIs (analysis-ready)

Human-approved rectangular ROIs for leave-one-stimulus-out (LOO) analyses.

## Schema

ROIs are **window-independent spatial boxes** on **100×100** VSD maps
(`y` = row, `x` = col). The same box applies across trials and analysis
windows; YAML must **not** define `start_frame` / `end_frame` / `window` /
`window_id`.

## Provenance

- Approved in `experiments/loo_encoding/roi_review/` (interactive review).
- Proposal mean maps were built over half-open **`[35, 42)`** for review
  figures only — that window is not part of the ROI identity.
- All entries have `status: accepted`.

## Intended use

ROI-masked **mean pixel-r** (and related spatial metrics) in LOO encoding
evaluations — mask pixels outside each stimulus box when summarizing
correlation maps for that held-out stimulus.

This folder is a freeze of approved boxes only; it does not start LOO training.

## Contents

| File | Contents |
|------|----------|
| `all_rois.yaml` | Master index: `stimulus_id`, `x0`, `y0`, `width`, `height`, `status` |
| `{stimulus_id}.yaml` | Same coords per stimulus (+ `map_shape`, mask path) |
| `{stimulus_id}__mask.npy` | Boolean mask `(100, 100)` generated from the box |

`n_stimuli` = 20.
