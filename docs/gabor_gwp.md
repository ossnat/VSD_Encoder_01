# Gabor wavelet pyramid — parameters, evaluation, grid search

Paper reference: [Kay et al., *Nature* 452, 352–355 (2008)](https://www.nature.com/articles/nature06713) — Gabor wavelet pyramid for early visual cortex receptive-field modeling ([supplementary methods](https://www.nature.com/articles/nature06713.pdf)).

## Default model: `gabor_serre.yaml`

**Use `configs/models/gabor_serre.yaml`** for all pipeline runs (02b–06) and backbone comparison. It is tuned for our **small geometric stimuli** (0.3–1° bars, circles, triangles):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_wavelength` | 3 px | Finest scale — sensitive to thin edges/contours |
| `wavelength_factor` | √2 | Octave spacing between scales |
| `number_of_scales` | 5 | Same count as Kay / Serre-style models |
| `number_of_directions` | 8 | Standard orientation coverage |
| `pool_size` | 14 | See **Spatial pooling** below — not inherent to GWP |

Wavelengths per scale: ≈ 3, 4.2, 6, 8.5, 12 px.

`gwp.energy_mode`: `sum_squares` (default) or `sqrt_contrast` (Kay-style √(even²+odd²)).

### Spatial pooling (`pool: avg`, `pool_size: 14`)

**GWP does not require 14×14 pooling.** ResNet `layer3` on 224×224 input *natively* outputs 14×14 maps; we added pooling to GWP for a different reason:

| | No pool | `pool_size: 14` |
|---|---------|-----------------|
| Feature shape | (40, **224**, **224**) | (40, **14**, **14**) |
| Flattened dim | ~**2.0M** | ~**7,840** |
| Ridge on 376 train trials | **OOM** / ill-conditioned | Runs |

So pooling was a **practical fix** so RidgeCV could fit, not a biological requirement. It also puts GWP and ResNet on a similar spatial grid for backbone comparison, but that alignment is optional.

To experiment without pooling (e.g. for finer interpretability maps), set in a copy of the model config:

```yaml
# pool: avg      # comment out
# pool_size: 14
```

You may need stronger `alphas`, fewer scales/orientations, or a compute node with more RAM. Exploratory layer plots (`exploratory/gwp_visualization/`) already use **full 224×224** energy without pooling.

`configs/models/gabor_kay2008.yaml` is kept **for reference only** (FOV-mapped Kay bands). Do not use it as the encoding default — its coarse scales (14–224 px) target natural-image fMRI modeling, not 0.3° shapes on a 224 px canvas.

## What is “FOV-mapped wavelength”? (Kay reference)

**FOV** = field of view = stimulus width in pixels (224 here).

Kay et al. specify spatial frequency in **cycles per FOV** (*f*). Conversion to our `min_wavelength` (λ in pixels):

\[
\lambda = \frac{\mathrm{FOV}}{f}
\]

For **224×224** stimuli, Kay’s five bands are λ ≈ 224, 112, 56, 28, 14 px (coarse → fine). That mapping is documented in `gabor_kay2008.yaml` if you want to experiment, but **`gabor_serre` is the starting point** for this project.

## How evaluation works (stage 04)

Same script for ResNet and GWP — backbone only changes feature extraction (02b) and Ridge weights (03):

```bash
scripts/py scripts/04_evaluate_pixel_correlation.py \
  --window configs/windows/evoked_32_42.yaml \
  --model configs/models/gabor_serre.yaml \
  --split test
```

For each pixel \((h,w)\) on the 100×100 VSD map:

1. **Originals** — trial-averaged H5 maps for all trials in the split (`T × H × W`).
2. **Reconstructions** — Ridge prediction from stimulus GWP features (one map per trial; identical within a condition).
3. **Pearson r heatmap** — correlate original vs reconstruction **across trials** at each pixel.
4. **R² heatmap** — \(1 - \mathrm{SS}_\mathrm{res}/\mathrm{SS}_\mathrm{tot}\) across trials.
5. **Trial-mean maps** — mean original, mean reconstruction, and **mean difference** (recon − original) across trials.

Outputs:

```
plots/evaluation/{monkey}/{window}/{model_slug}/{feature_layer}/
├── pixel_correlation_{split}.png
├── pixel_r2_{split}.png
├── pixel_mean_maps_{split}.png
├── condition_mean_originals_{split}.png
└── pixel_evaluation_{split}.json
```

## How backbone comparison works (stage 05)

`05_compare_backbones.py` reads saved artifacts (no retraining):

| Column | Source |
|--------|--------|
| `r_mean_val`, `r_mean_test` | `ridge_encode/.../metrics.json` (stage 03) |
| `eval_mean_r`, `eval_mean_r2` | `pixel_evaluation_test.json` (stage 04) |

## Grid search (stage 06)

Refines hyperparameters **starting from `gabor_serre.yaml`**:

```bash
scripts/py scripts/06_grid_search_gwp.py --window configs/windows/evoked_32_42.yaml
```

- **Base model:** `configs/models/gabor_serre.yaml`
- **Objective:** `r_mean_val`
- **Grid:** `configs/grid_search/gwp.yaml` — explores `min_wavelength` around 3 px plus coarser alternatives
- **Outputs:** `Data/VSD_Encoder_01/grid_search/gwp/{monkey}/{window}/`

If a grid combo beats the default, merge the winning values into `gabor_serre.yaml` (or save a new variant config) — do not switch to `gabor_kay2008.yaml`.

## Exploratory GWP layer plots

```bash
scripts/py exploratory/gwp_visualization/scripts/plot_gwp_layers.py
```

Per-scale energy maps on circle/triangle stimuli (uses `gabor_serre.yaml`, not part of the encoding pipeline).
