# Stimulus image rendering (stage 01b)

Rendered stimulus images are built from the experiment catalog CSV files under `Data/EncoderData/`.

## Source catalog

- Path pattern: `Data/EncoderData/{Monkey}_VSDI_*.csv`
- Example: `Gandalf_VSDI_hs27Aug20_YN.csv`
- Monkey name is inferred from the filename prefix.

## Session mapping

| H5 session file | CSV lookup |
|-----------------|------------|
| `session_270618b_condsAN.h5` | `Date = 27/6/2018`, `Session = b` |

General rule:

```
DDMMYY + session letter  ↔  Date + Session
270618b                  ↔  27/6/2018 , b
```

Condition mapping:

```
condAN1 ↔ cond1: ...
condAN2 ↔ cond2: ...
```

## Rendering rules

Configured in `configs/stimuli/default.yaml`:

| Parameter | Default |
|-----------|---------|
| Canvas | 210×210 RGB — **lower-right quadrant only** (fixation at top-left) |
| Background | gray **RGB (128, 128, 128)** — uniform for all stimuli |
| Fixation | **not drawn** |
| Quadrant extent | 6° right × 6° down from fixation |
| Scale | **1° = 35 px**; canvas = 6° × 35 px |
| Stimulus position | Catalog coords = **center** of stimulus (deg), e.g. `(0.6, −0.75)` |
| Letters | **1° × 1°** box (`size_deg` = side); center **0.5°** from each edge |
| Contour width | 1 px |
| Bar length | 1° (centered at position) |
| Bar width | 1 px |
| Size convention (shapes) | CSV values treated as **diameter** |

Example: position `(0.6, −0.75)` → center at 21 px right, 26.25 px down from fixation;
a 1° letter box spans ±17.5 px (0.5°) around that center.

Supported shapes parsed from the `stimulus (need to check r/d)` column:

- point
- filled circle
- circle contour
- triangle contour (equilateral, tip pointing **right** / +x)
- bar vertical / bar horizontal
- blank (gray screen + fixation only)

## Output layout

```
Data/VSD_Encoder_01/stimuli/
└── {monkey}/
    ├── config.json
    ├── manifest.parquet
    ├── parsed/
    │   └── conditions.parquet
    └── images/
        └── {h5_session}/
            ├── condAN1.png
            └── ...
```

QC plots are written to:

```
plots/stimuli/{monkey}/
├── all_stimuli_grid.png
├── 270618b__condAN1.png
└── ...
```

## Run

```bash
PYTHONPATH=. python scripts/01b_build_stimulus_images.py \
  --config configs/default.yaml \
  --stimuli-config configs/stimuli/default.yaml
```

Cluster:

```bash
sbatch slurm/build_stimulus_images.slurm
```

## Notes

- The CSV uses grouped blocks: session header row followed by condition rows with forward-filled `Date`, `Session`, and usually `Stimulus Position`.
- Multi-session blocks (`Session = a,b` or `a,b,c`) expand each condition to **every** listed session letter (same stimulus content; separate PNG under each `images/{h5_session}/`). Cortex-file suffixes on condition rows are **not** used to assign a single session.
- An additional catalog `Data/EncoderData/ContrastCurve_Letters_*_ExpSummary.csv` is merged in stage 01b:
  - **Contrast curves** — filled circles with RGB from parentheses; Cond6 Blank / Cond8 Error skipped; target location components are swapped to the shapes-CSV convention.
  - **Letters** — mats under `Data/EncoderData/letters_stimuli/` (session `c`/`d` prefer `2011C`/`2011D`); Control-attention rows skipped. Session **`201118b`** is excluded. **`201118a`** letters are in the catalog for **train-only** encoding (not a Protocol A/C test fold); **`201118c`** and **`201118d`** are the letter test sessions (see `src/stimuli/exclusions.py`).
- **Background:** all rendered PNGs use canonical gray **RGB (128, 128, 128)** from `configs/stimuli/default.yaml`. Letter BMP/MAT assets may have session-specific field gray; the renderer normalizes the canvas to 128. Contrast-curve session Blank RGB is used only for target polarity checks, not the canvas.
- Methods reference: `Data/EncoderData/8267.full.pdf`
- **Catalog QC:** the build script prints a warning if the same `(h5_session, condition)` appears twice (e.g. `240718*` / `condAN6` blank vs bar in the current Gandalf CSV). Encoding-pair joins prefer the non-blank row.
