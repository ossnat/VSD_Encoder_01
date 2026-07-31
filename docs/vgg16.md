# VGG16 backbone (with early-block pooling)

ImageNet-pretrained torchvision VGG16 as a stimulus feature extractor.

## Why pooling on early blocks

Raw early maps are too large for Ridge on this dataset:

| Layer | Raw map | Pooled option | Saved shape |
|-------|---------|---------------|-------------|
| `block1` | `(64, 112, 112)` ≈ 803k | `block1_pool7` / `block1_pool14` | `(64, 7, 7)` / `(64, 14, 14)` |
| `block2` | `(128, 56, 56)` ≈ 401k | `block2_pool7` / `block2_pool14` | `(128, 7, 7)` / `(128, 14, 14)` |
| `block3` | `(256, 28, 28)` ≈ 201k | `block3_pool14` | `(256, 14, 14)` |
| `block4` | `(512, 14, 14)` | — | unpooled |
| `block5` | `(512, 7, 7)` | — | unpooled |

Pooling matches the CORnet-S V1 mega-pixel approach: `F.adaptive_avg_pool2d`.

## Layer sweep (validation selection)

```bash
scripts/py scripts/13_sweep_vgg16_layers.py \
  --window configs/windows/evoked_35_42.yaml \
  --device cpu
```

Cluster:

```bash
sbatch slurm/sweep_vgg16_layers.slurm
```

Outputs under:

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/vgg16_comparison/
```

## Test-set PDF (all VGG layers)

```bash
scripts/py scripts/14_report_vgg16_layers.py \
  --window configs/windows/evoked_35_42.yaml
```

## Best VGG vs ResNet18 vs CORnet-S

Selects the best VGG and CORnet taps on **val**, then compares on **test** against ResNet18 `layer3`:

```bash
scripts/py scripts/15_report_vgg_best_vs_baselines.py \
  --window configs/windows/evoked_35_42.yaml
```

PDF:

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/vgg_best_vs_baselines_test_report/
```

## Config

`configs/models/vgg16.yaml` — default tap `block4`.
