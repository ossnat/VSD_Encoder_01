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
| Canvas | 224×224 RGB — **lower-right quadrant only** (fixation at top-left) |
| Background | gray (128) |
| Fixation | **not drawn** |
| Quadrant extent | 6° right × 6° down from fixation (224 px = 6°) |
| Scale | 1° diameter = `canvas_size / 6` px (~37.3 px); 0.5° = half that |
| Contour width | 1 px |
| Bar length | 0.3° (same as circle diameter; centered at `Stimulus Position`) |
| Bar width | 1 px |
| Size convention | values in CSV treated as **diameter** |

A 1° circle spans one sixth of the canvas width; a 0.5° circle is half that diameter, matching methods figures.

Supported shapes parsed from the `stimulus (need to check r/d)` column:

- point
- filled circle
- circle contour
- triangle contour
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
- Multi-session blocks (`Session = a,b,c`) use the `cortex file` suffix (`...a.1`, `...b.1`) when available.
- Methods reference: `Data/EncoderData/8267.full.pdf`
- **Catalog QC:** the build script prints a warning if the same `(h5_session, condition)` appears twice (e.g. `240718c` / `condAN6` in the current Gandalf CSV). Resolve in the source CSV if needed.
