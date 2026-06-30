# Backbone comparison

Compare Ridge encoding and pixel-wise evaluation metrics across stimulus feature extractors (ResNet, Gabor GWP, etc.).

## Prerequisites

Run stages **02b**, **03**, and **04** for each model config:

```bash
for m in configs/models/resnet18.yaml configs/models/gabor_serre.yaml; do
  scripts/py scripts/02b_extract_stimulus_features.py --model "$m"
  scripts/py scripts/03_train_ridge_encoder.py \
    --window configs/windows/evoked_32_42.yaml --model "$m"
  scripts/py scripts/04_evaluate_pixel_correlation.py \
    --window configs/windows/evoked_32_42.yaml --model "$m"
done
```

## Compare

```bash
scripts/py scripts/05_compare_backbones.py --window configs/windows/evoked_32_42.yaml
```

Or pass explicit models:

```bash
scripts/py scripts/05_compare_backbones.py \
  --window configs/windows/evoked_32_42.yaml \
  --model configs/models/resnet18.yaml \
  --model configs/models/gabor_serre.yaml
```

## Outputs

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/
├── backbone_comparison.csv
├── backbone_comparison.json
└── backbone_comparison.png
```

### CSV columns

| Column | Source |
|--------|--------|
| `model_slug` | Model config (`resnet18_imagenet`, `gabor_serre_gwp`, …) |
| `feature_layer` | e.g. `layer3`, `energy` |
| `alpha` | RidgeCV selected regularization |
| `r_mean_train/val/test` | Trial-wise Pearson r from stage 03 (full map) |
| `r_mean_*_masked` | Same, restricted to evaluation disk when `evaluation.use_mask` is true |
| `eval_mean_r`, `eval_mean_r2` | Pixel-wise metrics from stage 04 (full map) |
| `eval_mean_r_masked`, `eval_mean_r2_masked` | Pixel-wise metrics inside evaluation disk |
| `feature_shape` | `(C, H, W)` from stimulus feature extraction |

## Model configs

Backbones are selected via `configs/models/*.yaml`:

- **ResNet** — `type: resnet`, `preprocess: imagenet_rgb`
- **Gabor GWP** — `type: gabor_gwp`, `preprocess: grayscale_luminance`

See [`DL_feature_extraction.md`](DL_feature_extraction.md) for output layout.
