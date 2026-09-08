# Schira-geometry encoding (local ridge)

Predict **anchor-aligned VSD response maps** from **CNN stimulus features** that
have been resampled onto cortex with a fixed Schira retinotopy. Flatten ridge
(`scripts/03_train_ridge_encoder.py`) stays the baseline: one global weight
vector over flattened features. Here each VSD pixel gets its own small ridge
over the **C** feature channels at that pixel.

**Layout**

| What | Where |
|------|--------|
| Session → anchor transforms | `Data/.../session_register/AS_{anchor}/` |
| LUT, warped features, models, LOO, plots | `experiments/schira_encoding/` |

Config: `configs/schira_encoding/default.yaml` (defaults to raw window
`configs/windows/evoked_35_46.yaml`, `window_id: win_0035_0046`).

**Targets:** encoding uses **raw** mean maps (`normalization: none`), not
baseline z-score. Older z-score runs live under `win_0035_0046_zscore/` and
are not the default. Session→anchor registration already used raw vessel
maps (`normalization: none`); keep those transforms.

---

## Model

At each valid VSD pixel \((i,j)\):

\[
\hat{y}_{ij} = \mathbf{w}_{ij}^\top \mathbf{x}_{ij} + b_{ij}
\]

- \(\mathbf{x}_{ij} \in \mathbb{R}^{C}\): warped CNN channels at that pixel  
  (e.g. VGG16 ImageNet `block1`).
- \(\mathbf{w}_{ij} \in \mathbb{R}^{C}\), \(b_{ij}\): fit by ridge (optionally
  RidgeCV with a separate \(\alpha_{ij}\)).
- Features can be standardized **per pixel** (channel-wise mean/std over train
  trials at that pixel).

So the stored model is maps of shape `(C, H, W)` weights + `(H, W)` intercepts
(+ optional α / scale maps), not one flatten vector.

**Fit mask** (pixels that get weights): `LUT.valid ∩` flatten LOO ROI
(`noise_ceiling_hull` by default — global naive NC hull at thr 0.90).
Not the old circle r=50.

---

## Geometry pipeline

Everything is **anchor-centric**: one camera / Schira fit defines the grid;
other sessions are brought onto that grid.

### 1. Schira + affine → visual field

For Schira set `100718` and anchor session `100718a`, each anchor pixel
\((r,c)\) is mapped:

```text
(r, c)  --CorticalAffine.pixel_to_w-->  complex w
        --inverse_schira-->            (E, θ)
        --polar_to_cartesian-->        (x_deg, y_deg)
        --cartesian_deg_to_image_px--> stimulus (row, col) on the canvas
```

Stimulus canvas is the same geometry as rendering (`configs/stimuli/default.yaml`:
typically **210×210 @ 35 px/deg**). CNN ingest still resizes **210 → 224**.

### 2. Lookup table (LUT)

`scripts/10_build_schira_lut.py` builds one static `SchiraLUT` on the
**100×100** anchor grid:

- `valid` — pixels with a finite Schira/stimulus mapping  
- `x_deg`, `y_deg`, `stim_row`, `stim_col` — visual-field / canvas coords  
- At warp time, canvas coords are scaled into the CNN feature map
  (`input_size` / layer `Hf×Wf`), so **one LUT serves every layer**.

Stored under `experiments/schira_encoding/.../lut/...`.

### 3. Warp CNN features onto cortex

`scripts/11_warp_features_to_vsd.py` (via `warp_feature_map`):

- Load per-stimulus feature tensor `(C, Hf, Wf)` from DL features.
- Bilinear-sample each channel at the LUT-derived feature coordinates.
- Write warped maps `(C, 100, 100)` under `features_warped/...`.

Invalid LUT pixels are filled (typically 0).

### 4. Register sessions → anchor

`scripts/14_register_all_to_anchor.py` fits vessel / structural transforms
**moving session → fixed anchor**. Targets (VSD maps) from non-anchor days are
bilinear-warped onto the anchor grid before training (`warp_target_to_anchor`).
Anchor-session maps are used as-is.

Registrations stay in Data; encoding artifacts stay in experiments.

### 5. Assemble ridge inputs (`build_schira_xy`)

For each encoding-pair trial:

| Side | Content |
|------|---------|
| **X** | Warped features for that trial’s stimulus → `(C, H, W)` |
| **Y** | Windowed VSD map (e.g. frames `[35,46)` baseline z-score), warped to anchor if needed → `(H, W)` |

Batch shapes: `X (n, C, H, W)`, `Y (n, H, W)`.

---

## Experiment flows

### A. Full train / val / test (encoding-pair splits)

```bash
scripts/py scripts/10_build_schira_lut.py
# features: scripts/02b_extract_stimulus_features.py (if needed)
scripts/py scripts/11_warp_features_to_vsd.py ...
scripts/py scripts/14_register_all_to_anchor.py
scripts/py scripts/12_train_schira_local_ridge.py \
  --model-config configs/models/vgg16.yaml \
  --window-config configs/windows/evoked_35_46.yaml \
  --feature-layer block1
scripts/py scripts/15_evaluate_schira_pixel_correlation.py ... --split test
scripts/py scripts/16_plot_schira_test_figures.py ...
```

Train on the encoding-pairs **train** split; report masked Pearson \(r\) on
train / val / test.

### B. Leave-one-out (Protocol B)

Same fold logic as flatten LOO (`src.loo.folds`):

- **Protocol B** = one fold per held-out `stimulus_id`.  
  All trials of that stimulus are **test**; train/val never see that id
  (no stimulus leakage).
- Held-out lists: e.g. `experiments/loo_encoding/heldout_non_letters.yaml`.
- Optional `--date-prefix 100718` restricts pairs to that day.
- Default `--loss-roi noise_ceiling_hull` → fit/eval on `LUT.valid ∩` NC hull
  (same named mask as flatten LOO).

```bash
scripts/py scripts/17_run_schira_loo.py \
  --model-config configs/models/vgg16.yaml \
  --window-config configs/windows/evoked_35_46.yaml \
  --feature-layer block1 \
  --protocol B \
  --heldout experiments/loo_encoding/heldout_non_letters.yaml \
  --loss-roi noise_ceiling_hull
```

Outputs:
`experiments/schira_encoding/{monkey}/loo/{window}/{model}/{layer}/set-…__anchor-…/protocol_B_noise_ceiling_hull[__dates-…]/`

---

## Config parameters (defaults)

From `configs/schira_encoding/default.yaml` + window / model YAMLs:

| Parameter | Typical value | Role |
|-----------|---------------|------|
| `monkey` | `gandalf` | Dataset |
| `schira_set` / `anchor_session` | `100718` / `100718a` | Retinotopy + camera |
| `spatial_size` | `[100, 100]` | VSD / LUT grid |
| `input_size` | `224` | CNN ingest size |
| Stimulus canvas | 210×210, 35 px/deg | Must match stimuli config |
| Window | `[35, 46)`, **raw** (`normalization: none`) | Target map |
| Feature layer | e.g. `block1` | CNN depth |
| `ridge.alphas` | `1e-2 … 1e6` | RidgeCV grid |
| `ridge.alpha_per_pixel` | `true` | Separate α per pixel |
| `ridge.standardize_features` | `true` | Per-pixel channel z-score |
| `loss_roi` | `noise_ceiling_hull` | Fit ∩ metrics (same as flatten LOO) |
| LOO `--protocol` | `B` | Stimulus-id holdout |
| LOO `--loss-roi` | `noise_ceiling_hull` | Train mask = LUT.valid ∩ NC hull |
| LOO `--date-prefix` | optional | Subset sessions |

---

## Outputs

**Full local ridge** (`local_ridge/{window}/{model}/{layer}/set-…/`):

- `local_ridge.joblib` — weights, intercepts, α, valid mask  
- `meta.json` — counts, masked mean/median \(r\) by split  
- `valid.npy`, plots under `plots/`

**LOO** (per fold under `protocol_B_…/{fold_id}/`):

- `local_ridge.joblib`, `metrics.json`, `fold_pairs.parquet`  
- `fold_mean_orig.npy` / `fold_mean_recon.npy`, `valid.npy`  
- Root: `loo_summary.csv`, `params.yaml`

Primary scalar: **masked Pearson \(r\)** between original and reconstructed maps
(mean/median over trials), using the fit/valid mask.

---

## What the figures show

### Full-model test figures (`scripts/16_plot_schira_test_figures.py`)

- **`reconstructions_by_condition_pageXX.png`** / **`by_condition/{date}__{condition}.png`**  
  Side-by-side **original | reconstruction** for **one trial per**
  `(date, condition)` on the chosen split (usually test). Maps are already
  time-averaged over the analysis window and (if needed) warped to anchor;
  pixels outside the fit mask are NaN (blank).

- **`pixel_correlation_*.png` / `pixel_r2_*.png`**  
  Per-pixel correlation / \(R^2\) of predicted vs observed across trials
  (same style as flatten stage 04).

- **`pixel_mean_maps_*.png`**  
  Mean original, mean recon, and residual over the split.

### LOO overview figures (`scripts/18_plot_schira_loo_figures.py`)

After Protocol B LOO finishes:

```bash
scripts/py scripts/18_plot_schira_loo_figures.py \
  --loo-dir experiments/schira_encoding/gandalf/loo/win_0035_0046/\
vgg16_imagenet/block1_prepool/set-100718__anchor-100718a/protocol_B_noise_ceiling_hull
```

Writes under ``overview/``:

- ``all_shapes_orig_recon.png`` — fold-mean original | recon, one row per held-out stimulus
- ``pooled_fold_pixel_r[2]__raw.png`` — encoding pooled fold-level pixel r / R²
- ``encoding_vs_noise_corr_*.png`` — encoding vs odd/even noise correlation (same NC∩LUT mask)

Odd/even maps also under ``noise_corr_odd_even/``.

---

## Stages cheat-sheet

```bash
# 1. LUT (once per set/anchor/canvas)
scripts/py scripts/10_build_schira_lut.py

# 2. CNN features (existing; under Data/)
scripts/py scripts/02b_extract_stimulus_features.py \
  --model configs/models/vgg16.yaml --feature-layer block1

# 3. Warp features → anchor VSD grid
scripts/py scripts/11_warp_features_to_vsd.py \
  --model-config configs/models/vgg16.yaml \
  --window-config configs/windows/evoked_35_46.yaml \
  --feature-layer block1

# 4. Register all sessions → anchor (Data/)
scripts/py scripts/14_register_all_to_anchor.py

# 5. Train local ridge
scripts/py scripts/12_train_schira_local_ridge.py \
  --model-config configs/models/vgg16.yaml \
  --window-config configs/windows/evoked_35_46.yaml \
  --feature-layer block1

# 6. Pixel-r maps + test figures
scripts/py scripts/15_evaluate_schira_pixel_correlation.py ... --split test
scripts/py scripts/16_plot_schira_test_figures.py ...

# 7. Protocol B LOO
scripts/py scripts/17_run_schira_loo.py ... --protocol B --loss-roi noise_ceiling_hull
```
