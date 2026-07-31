# Stimulus-level ROI review (LOO prep)

Interactive review of **one rectangular ROI per stimulus identity**, not per
`(session, condition)`.

## Goal

For later leave-one-stimulus-out encoding, we need a loose spatial box around
the VSD response cluster for each held-out stimulus. The box should include
margin so the rise/fall of the activity wave stays inside for spatial
correlation. For elongated / multi-cluster responses (vertical bars, letters),
the box must cover **all** intense clusters, not just the single peak.

## How stimulus IDs are keyed

From stimulus catalog / `stimuli/.../manifest.parquet` fields:

| Kind | Key pattern | Example |
|------|-------------|---------|
| Shape | `{color}_{shape_type}_{size}` | `white_point_0.1` |
| Letter | `letter_{L}_{color}_{size}` | `letter_A_white_1` |

- `white_point_0.1` is the white 0.1° point (CSV: “white point 0.1…”),
  currently **test-only** via `290518a/condAN1`.
- Distinct sizes are distinct identities (e.g. `black_bar_vertical_0.3` vs
  `_1`).
- Position is **not** part of the key (letters can shift slightly across
  sessions; the ROI is still one box on the **all-session mean** map).

## ROIs are spatial only

Each ROI is a **window-independent spatial box** on **100×100** VSD maps
(`y` = row, `x` = col). The same box applies across trials and analysis
windows. ROI YAML / `all_rois.*` must **not** store `start_frame` /
`end_frame` / `window` / `window_id`.

## Proposal window (figures / placement only)

Mean maps for review use half-open frames **`[35, 42)`**
(`configs/windows/evoked_35_42.yaml` → `win_0035_0042`, frames 35–41).
That window is proposal metadata (figure annotation, `STATUS.md`,
`review_index.json`) — not part of the frozen ROI schema.

Maps are computed from raw session H5 via `average_frames`.

## Method (all stimuli)

### Common mean (global only)

**`global_common` = grand mean of ALL non-blank trials across ALL sessions**
(trial-weighted over every `date × condition` in the dataset).

This is **not** a per-session common mean. Compare each stimulus against this
single dataset-wide mean.

### Residual (box placement only)

```
residual = stim_mean − global_common
```

Used only to **place** the automatic box (threshold → connected components →
bounding box around the **union of all significant clusters**). Points stay
tighter; elongated stimuli (bars, letters, contours) get larger union boxes.

### Review figures

**One panel only:** the stimulus/condition average map (raw stim mean across
all its trials/sessions) with the ROI box overlaid. Residual / two-panel
figures are not the review product.

One global common-mean figure is saved for reference
(`figures/common_mean__global_all_conditions.png`).

## Outputs

| Path | Contents |
|------|----------|
| `stimulus_inventory.csv` / `.json` | All stimulus IDs with sessions, `n_trials`, split counts |
| `heldout_candidates.csv` | Focus subset: white point 0.1, triangles, vertical bars, letters |
| `STATUS.md` | Quick table of every `stimulus_id` + status + coords |
| `figures/{stimulus_id}__mean_map_roi.png` | **Single panel:** stim mean + box |
| `figures/common_mean__global_all_conditions.png` | Global grand mean (all trials × sessions) |
| `rois/{stimulus_id}.yaml` | Spatial box + review metadata (`x0,y0,width,height,status,method,…`; no window fields) |
| `rois/{stimulus_id}__mean_map.npy` | Cached stim mean |
| `rois/{stimulus_id}__residual_map.npy` | Cached residual (stim − global common) |
| `rois/{stimulus_id}__mask.npy` | Binary ROI mask (optional convenience) |
| `rois/all_rois.yaml` / `all_rois.json` | Master index of all ROIs |
| `review_index.json` | Summary for tooling |
| `comments/` | Drop your notes here |

Status values: `accepted` | `proposed`.

## How to regenerate

From the repo root (all inventory stimuli):

```bash
scripts/py scripts/16_propose_stimulus_rois.py
```

Subset / overrides:

```bash
scripts/py scripts/16_propose_stimulus_rois.py \
  --stimulus-ids white_point_0.1 letter_A_white_1 \
  --roi-override white_point_0.1=32,32,30,25 \
  --status-override white_point_0.1=accepted
```

Inventory only:

```bash
scripts/py scripts/16_propose_stimulus_rois.py --inventory-only
```

## Review checklist

1. Open every stimulus figure under `figures/` (skip
   `common_mean__global_all_conditions.png` for ROI shape; use it only as
   shared-structure reference).
2. Confirm the white rectangle covers the **full** response on the stimulus
   mean map — including all intense clusters for bars/letters — with a little
   rise/fall margin.
3. Leave notes under `comments/{stimulus_id}.md` if a box needs a tweak.
4. After agreement, set `status: accepted` in `rois/{stimulus_id}.yaml`
   (or re-run with `--roi-override` / `--status-override`).

Do **not** treat **proposed** boxes as final until reviewed.
