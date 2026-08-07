# Space-conv megapixel explorer

Local Mac GUI to browse VSD trials as a **10×10 megapixel** mosaic (each cell = mean of a 10×10 pixel block).

## Run

From the repo root:

```bash
scripts/py tools/space_conv/run_megapixel_explorer.py
```

Or:

```bash
scripts/py tools/run_space_conv.py
```

Optional non-interactive flags (still open the GUI unless `--list-only`):

```bash
scripts/py tools/space_conv/run_megapixel_explorer.py \
  --monkey gandalf --session 270618b --condition condAN1 \
  --trial all --start-frame 35 --end-frame-inclusive 45 \
  --normalization zscore
```

```bash
scripts/py tools/space_conv/run_megapixel_explorer.py --list-only --monkey gandalf
```

Requires a local display (matplotlib GUI backend). Not intended for SSH without X11/forwarding.

## Interactive flow

1. **Monkey** (default `gandalf`) → list `ProcessedData/<monkey>/session_*.h5` (excludes `*_blank.h5`; date resolve prefers `*_condsAN.h5`)
2. **Session** → conditions from H5 `trial_metadata_json`
3. **Condition** → trials, or **ALL** trials in that condition (labels include stimulus text from the stimulus catalog, e.g. `condAN1 — black point 0.1 diameter`)
4. **Analysis window**: start / end as **inclusive** frame indices (default **35–45** → code uses half-open `[35, 46)`)
5. **Normalization**: None or Z-scored (see below)
6. **Mosaic**: 10×10 tiny traces over the analysis window (`mean ± std`)
7. **Click** a cell → large detail trace of **the same megapixel series**, through frame **200** (inclusive), with the analysis window highlighted (multiple clicks → multiple figures). Y-limits follow the analysis window so early-frame outliers do not flatten the plot.

Display only — nothing is written to disk.

## Display limits

- Traces are shown for frames **`0 … 200` inclusive** (or fewer if the trial is shorter). Later frames are omitted from mosaic clamping and large figures.
- Enlarged plots use **window-based y-limits** (`mean ± std` over the analysis window). Frame 0 after baseline z-score is often a huge outlier; autoscaling to the full series made the window look like a flat line even though the underlying megapixel series matched the mosaic.

## Stimulus descriptions

Condition prompts and figure titles resolve `(monkey, h5_session, condition)` against `Data/VSD_Encoder_01/stimuli/<monkey>/parsed/conditions.parquet` (fallback: `manifest.parquet`) — the same catalog used elsewhere. The leading `condN:` prefix is stripped for display.

## Megapixel definition

- Native VSD map is **100×100** (`(10000, n_frames)` in H5).
- Each megapixel is the **block mean** of a non-overlapping **10×10** pixel patch → **10×10** megagrid.
- **Single trial**: shaded band = std **across the 100 pixels** in the block (per frame).
- **ALL trials**: time series = mean over trials of each block-mean; shaded band = std **across trials** of that block-mean (per frame).
- NaN mask pixels in the H5 (if any) are ignored via `nanmean` / `nanstd`.

## Z-score (matches `src/data/averaging.py`)

Uses `baseline_zscore_trial`:

- Per trial, per pixel
- Baseline frames **`[5, 26)`** (frames 5…25 inclusive)
- `mean` / `std` with `ddof=0`; `std = max(std, 1e-8)`
- Full trial: `(x - mean) / std`
- Applied **before** megapixel block-averaging

Same helper as averaged-trial / window configs (`normalization: baseline_zscore`).

## Data loading

- Paths via `resolve_data_path` → `Data/FoundationData/ProcessedData/<monkey>/…`
- Trials via `read_trial_by_global_id` (correct flat H5 dataset for each `trial_global_id`)
- Session/condition/trial lists come from each H5’s `trial_metadata_json` (same metadata the loaders use)
- Session discovery skips blank extracts (`session_*_blank.h5` / any `*_blank.h5`). When `--session` matches more than one file for a date, `*_condsAN.h5` is preferred.

## Layout

| Path | Role |
|------|------|
| `run_megapixel_explorer.py` | CLI + prompts entry point |
| `discovery.py` | List monkeys / sessions / conditions / trials; stimulus description lookup |
| `megapixel.py` | Normalize + block-mean stack |
| `explorer_gui.py` | Matplotlib mosaic + click-to-enlarge (frame ≤200, window y-limits) |
| `../run_space_conv.py` | Thin launcher |
