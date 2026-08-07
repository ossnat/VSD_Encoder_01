# Report 5

Report-specific figure scripts live here. Shared math / helpers go under `src/`
or `experiments/noise_ceiling_roi/nc_roi_utils.py`.

## Task 1 — Z-scored stimuli catalog (+ SNR)

Grouped stimulus catalog matching `experiments/stimulus_catalog/`, with:

| Setting | Prior catalog | Report 5 (current) |
|---|---|---|
| Window | typically `win_0035_0042` (± zscore) | `win_0035_0046_zscore` |
| Normalization | raw F/F₀ or baseline z-score | **baseline z-score** (`[5,26)` → z-score full trial → mean `[35,46)`) |
| Rows | one per `stimulus_id` (dates pooled) | **one per stimulus_id × session/date** |
| Trials shown | 10 / first 8 | **up to 8 distinct**, evenly spaced (deterministic) |
| Columns | stimulus + trials | stimulus + **SNR** + trials |
| Groups | dots/circles, geometric, letters | same |

### Columns

1. **stimulus** — canonical re-render from stimuli manifest (gray-128 background). Row label includes `stimulus_id · <session>` (e.g. `letter_D_white_1 · 201118c`).
2. **SNR map** — per-pixel `|μ|/σ` across **that session’s** trials over the **full FOV**
   (viridis; not NC-ROI masked). Bottom scalar = `mean(snr_pix)` over finite
   full-frame pixels.
3. **trial 1…8** — up to 8 distinct trials for that date×condition, evenly spaced
   across the sorted `trial_global_id` list (no RNG).

### SNR definition

For each stimulus×session row, load **all** available encoding-pair trials for
that `(stimulus_id, date)` (not only the 8 shown). Each trial map is the
analysis-window mean after baseline z-score (`[5, 26)`). Then:

```
snr_pix(x,y) = |mean_T map(x,y)| / std_T map(x,y)   # sample std, ddof=1; abs on mean
SNR (scalar / metadata) = mean of snr_pix over finite full-frame pixels
```

Note: the figure column shows `snr_pix` itself (with abs), not signed `μ/σ`.
Default display and scalar use the full FOV (no NC hull mask). Optional
`--snr-mask` can restrict the summary. Helper:
`src/evaluation/snr.py` (`map_snr_across_trials`).

### Run

From repo root:

```bash
scripts/py experiments/report_5/build_stimuli_catalog.py
```

Optional flags:

```bash
scripts/py experiments/report_5/build_stimuli_catalog.py \
  --window configs/windows/evoked_35_46_zscore.yaml \
  --trials-per-stimulus 8 \
  --group geometric_shapes
```

Optional: `--snr-mask path/to/mask.npy` (default: full frame, no mask).
Use `--no-snr-scalar-label` to omit the bottom SNR text (legacy NC-ROI style).

Default outputs (per-session full-frame):
`experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore_per_session/stimulus_catalog_<group>__win_0035_0046_zscore.{png,json}`

Prior pooled-stimulus full-frame catalog remains under:
`experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore_fullframe/`

Legacy NC-ROI catalog (masked map, no bottom label) remains under:
`experiments/stimulus_catalog/figures/snr_maps_win_0035_0046_zscore/`
---

## Task 2 — Across-condition NC ROI controls

Script: `run_roi_controls.py`. Reuses across-condition helpers in
`experiments/noise_ceiling_roi/nc_roi_utils.py`. ROI `--window` is the same
flag family as NC ROI (independent of LOO analysis window; default
`evoked_35_46.yaml`).

### Blank control — findings

**No blank VSD trials in the encoding/CHECK dataset used for analysis.**

| Source | Blank / Cond 6? |
|---|---|
| Stimulus catalog (`is_blank`, “Cond 6 Blank”) | Yes — 6 session rows (rendered PNGs) |
| Trial index / splits (`all_trials_index_gandalf`, split_v3) | **No** — `condAN6` count = 0 (only condAN1–5,7) |
| Encoding pairs (`win_0035_0046*`, etc.) | **No** — all `is_blank=False` |

Blanks were never ingested into the Gandalf processed trial index (typically
trial 6 / blank). Catalog dedupe also prefers non-blank when a session has
both blank and a real shape on the same condition, but for Gandalf that
condition is absent from the index entirely. Status note (no blank figures):

`roi_controls/blank/win_0035_0046/BLANK_CONTROL_STATUS__win_0035_0046.md`

When blanks return to encoding pairs, the same script builds an across-condition
*r* map using session-level units `blank_<date>` + naive hull at thr=0.85.

### Shuffle control

Scrambles **even** half-mean maps across the stimulus axis (fixed seed **17**)
while keeping odd maps in original order — breaks stimulus↔response pairing.
Expect mean pixel *r* ≈ 0 and thr=0.85 naive hull to collapse.

### Run

```bash
scripts/py experiments/report_5/run_roi_controls.py
# or selectively:
scripts/py experiments/report_5/run_roi_controls.py --control shuffle
scripts/py experiments/report_5/run_roi_controls.py --control blank
scripts/py experiments/report_5/run_roi_controls.py \
  --window configs/windows/evoked_35_46.yaml --threshold 0.85 --seed 17
```

### Outputs

| Path | Content |
|---|---|
| `roi_controls/shuffle/<window>/shuffle__correlation_map__naive_hull_thr0.85.png` | Shuffle *r* map + naive hull |
| `roi_controls/shuffle/<window>/shuffle__summary__<window>.yaml` | Metrics + permutation |
| `roi_controls/blank/<window>/BLANK_CONTROL_STATUS__*.md` | Blank absence note (current data) |
| `figures/roi_controls/{shuffle,blank}/…` | Copies for browsing |

---

## Task 3 — Protocol B PDF (raw vs zscore × NC ROI)

```bash
scripts/py experiments/report_5/build_protocol_B_report_pdf.py
```

Output: `experiments/report_5/figures/protocol_B_raw_vs_zscore_nc_roi.pdf`
