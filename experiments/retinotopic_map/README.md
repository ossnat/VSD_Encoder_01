# Retinotopic map — corrected point-sampling plan

Probe cortical retinotopy by predicting **many small point stimuli** along a
synthetic **5×5 grid of line loci**, then **merging** the predicted VSD maps.

## Correction vs prior approach

| | Legacy (wrong for this goal) | Corrected |
|--|------------------------------|-----------|
| Stimulus | One image: full 5×5 **bar/line grid** | Many images: **0.05° black points** on those lines |
| Prediction | Single map for the whole grid | One map **per point**, then merge |
| Code kept | `render_grid_stimulus.py`, `predict_grid_vsd.py`, `figures/grid_5x5__*`, `models/.../win_0035_0043/` | New: `point_schedule.py`, `run_point_pilot.py`, `data/point_schedule.json`, `figures/point_sampling_plan.png` |

Legacy whole-grid assets are **not deleted**; they remain for reference / QC.
Do not use the single-grid prediction as the retinotopic map.

**Settings (this experiment):** raw / no z-score by default, **no loss ROI**.
Z-score twin uses the same stimuli/features with
`--window configs/windows/evoked_35_46_zscore.yaml`
(`win_0035_0046_zscore`, `baseline_zscore`, separate model — still **no loss ROI**).

## Geometry (aligned to catalog + legacy grid)

- Canvas: **224×224** = **6°×6°** lower-right quadrant (fixation top-left).
  `pixels_per_deg = 224/6 ≈ 37.333` (`configs/stimuli/default.yaml`).
- **Line loci (snapped):** **1°, 2°, 3°, 4°, 5°** (H and V). Legacy `grid_5x5`
  px **37, 74, 112, 149, 186** map to ≈ **0.99–4.98°**; we snap to integer
  degrees so **0.5°** and **1/6°** samples land on every crossing (Δ ≲ 0.01°).
- Catalog **bars** (`black_bar_*`) are **short** (0.3° / 1°) at a **single**
  locus `(0.6, −0.75)` — not a 5×5 bar field. The 5×5 lines are a **synthetic
  probe**; point spacing follows those line positions across the quadrant.
- Point size: **0.05° diameter**, black — same convention as `black_point_0.05`
  (also at `(0.6, −0.75)` in the catalog). Larger-dot variants: **0.075°** under
  `*_dense3x_p0075`, **0.1°** under `*_dense3x_p01` (same dense3x spacing;
  smaller sizes kept for comparison). Catalog also has `black_point_0.1`.

## Point schedule (concrete)

| Set | Lines | Spacing | Extent along line | Count |
|-----|-------|---------|-------------------|-------|
| **Step A** | Middle **horizontal** only (y = −3°) | **0.5°** | x = 0.5 … 5.5° | **11** |
| **Step B** | Middle **vertical** only (x = 3°) | **0.5°** | y = −0.5 … −5.5° | **11** |
| **Step A+B** | Mid-cross (H mid + V mid) | **0.5°** | as above | **22** |
| **Phase 4 full** | All **5 H + 5 V** | **0.5°** | 0.5 … 5.5° on each line | **85** unique |
| **Dense3x H-mid** | Middle H | **1/6° ≈ 0.1667°** | 0.5 … 5.5° | **31** |
| **Dense3x V-mid** | Middle V | **1/6°** | 0.5 … 5.5° | **31** |
| **Dense3x full** | All **5 H + 5 V** | **1/6°** | 0.5 … 5.5° on each line | **285** unique |
| **Dense3x full p0075** | All **5 H + 5 V** | **1/6°** | same; **0.075°** dots | **285** unique |
| **Dense3x H-mid p0075** | Middle H | **1/6°** | same; **0.075°** dots | **31** |
| **Dense3x full p01** | All **5 H + 5 V** | **1/6°** | same; **0.1°** dots | **285** unique |
| **Dense3x H-mid p01** | Middle H | **1/6°** | same; **0.1°** dots | **31** |

- 0.5° ≈ **10×** point diameter → distinct probes without a huge N.
- **Dense3x** triples points along each line (`0.5/3 = 1/6°`); same line loci.
  Intersections counted once → **285** unique (vs **85** at 0.5°).
- Schedule keys: `full_points` (0.5°), `full_dense3x_points` (1/6°, 0.05°),
  `full_dense3x_p0075_points` (1/6°, 0.075°), and `full_dense3x_p01_points`
  (1/6°, 0.1°) kept so size variants remain comparable.

Machine-readable schedule: [`data/point_schedule.json`](data/point_schedule.json).  
Illustrations: [`figures/point_sampling_plan.png`](figures/point_sampling_plan.png)
(0.5°), [`figures/point_sampling_plan_dense3x.png`](figures/point_sampling_plan_dense3x.png)
(1/6°).

```bash
scripts/py experiments/retinotopic_map/point_schedule.py
```

## Merge (predicted maps → mosaic)

| Method | Role |
|--------|------|
| **`sum` (default)** | Signed sum of per-point predictions → composite retinotopic mosaic under linear superposition |
| `mean` | `sum / N`; same pattern, count-normalized amplitude |
| `max_abs` | Per pixel keep value with largest \|pred\| — peak-dominated, less stacking |
| `stack` | Keep N maps for inspection / animation (not one mosaic) |

**Recommend `sum`** as the default mosaic; report `mean` alongside if comparing
runs with different point counts.

Merge **subsets**:

| Subset | Points | Figure role |
|--------|--------|-------------|
| `h_mid` | Step A only | First deliverable (Phases 1–3) |
| `v_mid` | Step B only | Same model; compare to H |
| `h_plus_v` | A + B | Overlay / mid-cross mosaic |
| `full` | All 5 H + 5 V unique (~85) | Phase 4 mosaic (`full_5x5/`) |
| `h_mid_dense3x` | H mid at 1/6° (31) | Dense mid-line pilot |
| `v_mid_dense3x` | V mid at 1/6° (31) | Dense mid-line pilot |
| `full_dense3x` | All H+V at 1/6° (~285), **0.05°** | Dense mosaic (`full_5x5_dense3x/`) |
| `h_mid_dense3x_p0075` | H mid dense3x, **0.075°** (31) | Larger-dot mid-line |
| `full_dense3x_p0075` | All H+V dense3x, **0.075°** (~285) | Larger-dot mosaic (`full_5x5_dense3x_p0075/`) |
| `h_mid_dense3x_p01` | H mid dense3x, **0.1°** (31) | Catalog-size mid-line |
| `full_dense3x_p01` | All H+V dense3x, **0.1°** (~285) | Catalog-size mosaic (`full_5x5_dense3x_p01/`) |

Gallery mosaics use **demeaned signed grayscale** with **symmetric clim**
`±max(p99(|d|), 2.5×median_rms)` so quiet probes stay near mid-gray instead of
noise-stretching into full contrast (old per-panel 1–99% + VSD mid-green looked
empty/noisy). Color companion: `mosaic__sum__color.png`. Replot without
re-predict: `--stage plot`.

## Phased plan

### Phase 0 — illustration + schedule *(done)*

- README + `data/point_schedule.json` + `figures/point_sampling_plan.png`.

### Step A — H mid (Phases 1–3)

1. Render **11** black 0.05° point PNGs (middle H line).
2. Extract ResNet18 ImageNet **layer3** features.
3. Train once full all-pairs ridge: **`win_0035_0046`**, raw, **no ROI**
   (or reuse matching cache under `models/win_0035_0046/.../full_all_pairs/`).
   Legacy `win_0035_0043` cache is **not** used for this run.
4. Predict per point → **sum**-merge → mosaic figure.

### Step B — V mid (same model)

1. Render / feature-extract middle V line (can be done with Step A).
2. **Reuse** the Step A encoder (no retrain).
3. Predict V-only merge + **H+V** combined merge figures.

### Phase 4 — densify to full 5×5 line sampling

- Expand to all H+V lines at 0.5° → **85** unique points.
- Same model (no retrain); predict + merge + compare to pilot mosaic.

```bash
# Full 5×5 (~85 unique): render → features → reuse model → predict + sum-merge
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046/full_5x5/`

### Dense3x — 3× denser along lines (1/6°)

- Same 5 H + 5 V loci; spacing **1/6°** → **31** pts/line → **285** unique.
- Outputs under `full_5x5_dense3x/` (0.5° `full_5x5/` kept for comparison).
- **Reuse** existing `win_0035_0046` model (do not retrain).

```bash
# Dense3x full (~285 unique): render → features → reuse model → predict + sum-merge
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x

# Optional dense mid-line subsets
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset h_mid_dense3x
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset v_mid_dense3x
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046/full_5x5_dense3x/`

### Dense3x · 0.075° dots (`*_p0075`)

- Same dense3x loci / spacing (**1/6°**, **285** unique); diameter **0.075°**.
- Outputs under `full_5x5_dense3x_p0075/` (0.05° `full_5x5_dense3x/` kept).
- **Reuse** the same `win_0035_0046` model (do not retrain).

```bash
# Dense3x full at 0.075° (~285 unique): render → features → reuse model → predict + sum-merge
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p0075

# Optional H-mid only at 0.075°
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset h_mid_dense3x_p0075
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046/full_5x5_dense3x_p0075/`

### Dense3x · 0.1° dots (`*_p01`)

- Same dense3x loci / spacing (**1/6°**, **285** unique); diameter **0.1°**
  (matches catalog `black_point_0.1`, which is in the full all-pairs train set).
- Outputs under `full_5x5_dense3x_p01/` (0.05° / 0.075° dense3x kept).
- **Reuse** the same `win_0035_0046` model (do not retrain).

```bash
# Dense3x full at 0.1° (~285 unique): render → features → reuse model → predict + sum-merge
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p01

# Optional H-mid only at 0.1°
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset h_mid_dense3x_p01
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046/full_5x5_dense3x_p01/`

### Dense3x · 0.1° · z-score window (`win_0035_0046_zscore`)

- Same **0.1°** dense3x stimuli + ResNet18/layer3 features as raw `*_p01`
  (stimulus-side; reused, not re-rendered).
- Train a **separate** full all-pairs ridge on `baseline_zscore` targets
  (`configs/windows/evoked_35_46_zscore.yaml`); **no loss ROI**.
- Figures under `point_pilot/win_0035_0046_zscore/full_5x5_dense3x_p01/`
  (and optional `h_mid_dense3x_p01/`).

```bash
# Train zscore model once, then predict full 0.1° dense3x (reuses stimuli/features)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml

# Optional H-mid only (reuses zscore model)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset h_mid_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046_zscore/full_5x5_dense3x_p01/`

### Tiny locus · 0.05° on catalog training position (`tiny_locus_p005`)

- **Why:** catalog bars / circles / triangles / `black_point_*` sit at
  **(0.6, −0.75)** — not on the synthetic 1–5° 5×5 field. Option-1 diagnostic
  asks whether predictions **shift** inside this trained locus.
- Geometry: short **H+V cross**, ~**2°** extent centered on that locus
  (H: y=−0.75, x≈0.1…1.6; V: x=0.6, y≈−0.05…−1.75), spacing **0.1°**,
  diameter **0.05°** → ~33 unique points.
- **Legacy / comparison only** — prefer `tiny_5x5_p005` below.
- **Reuse** existing `win_0035_0046_zscore` full all-pairs model (no retrain);
  no ROI; no blank subtract for primary figures.

```bash
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset tiny_locus_p005 \
  --window configs/windows/evoked_35_46_zscore.yaml

scripts/py experiments/retinotopic_map/plot_corner_stim_pred_pairs.py --subset tiny_locus_p005
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046_zscore/tiny_locus_p005/`  
Illustration: [`figures/point_sampling_plan_tiny_locus.png`](figures/point_sampling_plan_tiny_locus.png)

### Tiny 5×5 · 0.05° near catalog locus (`tiny_5x5_p005`)

- **Why:** the 2-line cross is inadequate for a retinotopic probe; this is a
  real **5 H + 5 V** grid in a train-locus window (not the OOD 1–5° field).
- **Geometry (documented):** ~**3.2°×3.2°** box with left/top pinned near
  fixation (`x∈[0.10, 3.30]`, `y∈[−3.25, −0.05]`) so all lines stay on-canvas.
  Catalog locus **(0.6, −0.75)** sits in the upper-left of the box (same
  clamp tradeoff as the old cross; right/down extend to reach full span).
  - **V lines x:** 0.10, 0.90, 1.70, 2.50, 3.30 (pitch **0.8°**)
  - **H lines y:** −3.25, −2.45, −1.65, −0.85, −0.05 (pitch **0.8°**)
  - Along-line spacing **0.1°**, diameter **0.05°** → **305** unique points
- **Reuse** `win_0035_0046_zscore` full all-pairs (no retrain); no blank subtract.

```bash
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset tiny_5x5_p005 \
  --window configs/windows/evoked_35_46_zscore.yaml

scripts/py experiments/retinotopic_map/plot_corner_stim_pred_pairs.py --subset tiny_5x5_p005
scripts/py experiments/retinotopic_map/plot_tiny_shift_diagnostic.py --subset tiny_5x5_p005
```

Figures: `experiments/retinotopic_map/figures/point_pilot/win_0035_0046_zscore/tiny_5x5_p005/`  
Illustration: [`figures/point_sampling_plan_tiny_5x5.png`](figures/point_sampling_plan_tiny_5x5.png)

## How to run

```bash
# Full pilot: schedule → render H+V → features → train win_0035_0046 → predict all
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all

# Phase 4 full 5×5 (reuse model; no retrain unless --retrain)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full

# Dense3x full 5×5 at 1/6° (reuse model)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x

# Dense3x full at 0.075° diameter (reuse model; 0.05° dense3x kept)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p0075

# Dense3x full at 0.1° diameter (reuse model; smaller-dot dense3x kept)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p01

# Dense3x full at 0.1° on z-score window (separate model; same stimuli/features)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset full_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml

# Tiny 5×5 near catalog locus (0.05°, zscore model, no retrain)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage all --subset tiny_5x5_p005 \
  --window configs/windows/evoked_35_46_zscore.yaml

# Stepwise
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage schedule
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage render
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage features
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage train
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage train \
  --window configs/windows/evoked_35_46_zscore.yaml
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset h_mid
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset v_mid
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset h_plus_v
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset full
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset full_dense3x
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset full_dense3x_p0075
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset full_dense3x_p01
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset full_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage predict --subset h_mid_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml
# Replot mosaics from saved .npy only (viz defaults; no retrain)
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage plot --subset full_dense3x_p01
scripts/py experiments/retinotopic_map/run_point_pilot.py --stage plot --subset full_dense3x_p01 \
  --window configs/windows/evoked_35_46_zscore.yaml
```

Outputs:

- Stimuli / features: `experiments/retinotopic_map/data/stimuli/`, `.../data/features/`
- Model (raw): `experiments/retinotopic_map/models/win_0035_0046/resnet18_imagenet/layer3/full_all_pairs/`
- Model (zscore): `experiments/retinotopic_map/models/win_0035_0046_zscore/resnet18_imagenet/layer3/full_all_pairs/`
- Figures: `experiments/retinotopic_map/figures/point_pilot/{win_0035_0046,win_0035_0046_zscore}/{h_mid,v_mid,h_plus_v,full_5x5,full_5x5_dense3x,full_5x5_dense3x_p0075,full_5x5_dense3x_p01,...}/`

## Layout

```
experiments/retinotopic_map/
├── README.md
├── point_schedule.py
├── run_point_pilot.py          ← Step A/B / full / dense3x / optional --window zscore
├── data/
│   ├── point_schedule.json
│   ├── stimuli/{h_mid,v_mid,full_5x5,full_5x5_dense3x,full_5x5_dense3x_p0075,full_5x5_dense3x_p01,...}/
│   └── features/resnet18_imagenet/layer3/{h_mid,v_mid,full_5x5,full_5x5_dense3x,full_5x5_dense3x_p0075,full_5x5_dense3x_p01,...}/
├── figures/
│   ├── point_sampling_plan.png
│   ├── point_sampling_plan_dense3x.png
│   ├── point_pilot/win_0035_0046/{h_mid,v_mid,h_plus_v,full_5x5,full_5x5_dense3x,full_5x5_dense3x_p0075,full_5x5_dense3x_p01}/
│   ├── point_pilot/win_0035_0046_zscore/{full_5x5_dense3x_p01,h_mid_dense3x_p01,...}/
│   └── grid_5x5__*               ← legacy whole-grid (kept)
├── render_grid_stimulus.py       ← legacy
├── predict_grid_vsd.py           ← legacy
└── models/
    ├── win_0035_0046/.../full_all_pairs/          ← raw
    ├── win_0035_0046_zscore/.../full_all_pairs/   ← baseline_zscore
    └── win_0035_0043/...                          ← legacy
```
