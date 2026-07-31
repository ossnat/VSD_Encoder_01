# CORnet-S backbone

ImageNet-pretrained [CORnet-S](https://github.com/dicarlolab/CORnet) as a stimulus feature extractor, with cortical-area taps aligned to V1 / V2 / V4 / IT.

## Why pooling on V1

Raw V1 maps are large (`64×56×56` ≈ 200k features). For Ridge we use mega-pixel adaptive average pooling:

| Layer | Saved map shape | Notes |
|-------|-----------------|-------|
| `V1_pool7` | `(64, 7, 7)` | coarse mega-pixels |
| `V1_pool14` | `(64, 14, 14)` | finer mega-pixels |
| `V2` | `(128, 28, 28)` | unpooled |
| `V4` | `(256, 14, 14)` | unpooled |
| `IT` | `(512, 7, 7)` | optional |

## Local smoke (single layer)

```bash
scripts/py scripts/11_sweep_cornet_layers.py \
  --window configs/windows/evoked_35_42.yaml \
  --layers V4 \
  --device cpu
```

## Full comparison (cluster)

```bash
cd /home/dsi/ossnat/VSD_FM/VSD_Encoder_01
mkdir -p slurm_err_out
sbatch slurm/sweep_cornet_s_layers.slurm
```

Default layers: `V1_pool7 V1_pool14 V2 V4` on validation (`--split val`).

## Outputs

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/cornet_s_comparison/
├── layer_comparison.csv
├── layer_mean_pixel_r.png
├── weight_alpha_grid.png              # ||w||₂ and log α side-by-side
├── weight_center_periphery.csv
└── weight_center_over_periphery.png   # center/periphery ||w||₂ ratio
```

## Interpreting weight maps

There are no anatomical V1/V2/V4 ROI masks in this repo yet. As a proxy, we split the evaluation disk into a **center disk** vs **peripheral annulus** (`--center-frac`, default `0.5`) and report `weight_center_over_periphery`.

If VSD FOV is mostly V1 with V2/V4 toward the periphery (or vice versa in your prep), compare whether:

- V1-pooled features put relatively more weight in the V1-dominated band
- V2 / V4 features shift weight toward their expected band

Use `weight_alpha_grid.png` for qualitative comparison of full spatial maps.

## Test-set PDF vs ResNet18

After layer runs exist:

```bash
scripts/py scripts/12_report_cornet_vs_resnet.py \
  --window configs/windows/evoked_35_42.yaml
```

This evaluates **test** pixel metrics for CORnet `V1_pool7 V1_pool14 V2 V4` and ResNet18 `layer3`, then writes:

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/cornet_vs_resnet_test_report/
├── test_metrics.csv
├── test_masked_pixel_r.png
├── center_periphery_definition.png
├── cornet_weight_alpha_grid.png
└── cornet_vs_resnet_test_report.pdf
```

## Config

`configs/models/cornet_s.yaml` — default tap `V4`. Override with `--feature-layer` / `--layers`.
