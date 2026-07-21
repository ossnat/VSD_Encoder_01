# Backbone layer sweeps (ResNet18 vs VGG16)

Resource-aware SLURM sweeps over CNN feature layers on the expanded dataset and `[35,42)` window (`win_0035_0042`). Layers are selected on **validation** masked pixel correlation; **test** is used only for the winning layers.

## Layer sets

| Backbone | Layers swept | Rationale |
|----------|--------------|-----------|
| ResNet18 | `layer2`, `layer3`, `layer4`, `avgpool` | Practical spatial sizes for RidgeCV |
| VGG16 | `block3`, `block4`, `block5` | Skip `block1`/`block2` — dense Ridge matrices are impractical on the expanded dataset |

## Local workflow

After `run_prepare_encoding.sh` for `configs/windows/evoked_35_42.yaml`:

```bash
# Validation sweeps (layer selection)
scripts/py scripts/09_sweep_feature_layers.py \
  --window configs/windows/evoked_35_42.yaml \
  --model configs/models/resnet18.yaml \
  --layers layer2 layer3 layer4 avgpool \
  --split val --device auto

scripts/py scripts/09_sweep_feature_layers.py \
  --window configs/windows/evoked_35_42.yaml \
  --model configs/models/vgg16.yaml \
  --layers block3 block4 block5 \
  --split val --device auto

# Cross-model report (test eval for winners only)
scripts/py scripts/10_report_layer_sweeps.py \
  --window configs/windows/evoked_35_42.yaml
```

### Resume / compare-only

Re-aggregate plots and CSVs without rerunning heavy stages:

```bash
scripts/py scripts/09_sweep_feature_layers.py \
  --window configs/windows/evoked_35_42.yaml \
  --model configs/models/resnet18.yaml \
  --layers layer2 layer3 layer4 avgpool \
  --split val --compare-only

scripts/py scripts/10_report_layer_sweeps.py \
  --window configs/windows/evoked_35_42.yaml \
  --compare-only
```

`09_sweep_feature_layers.py` also supports `--skip-extract`, `--skip-train`, and `--skip-eval` individually.

## Cluster workflow

Submit both validation sweeps and the report with SLURM dependencies:

```bash
bash scripts/submit_layer_sweeps.sh
```

This runs:

1. `slurm/sweep_resnet18_layers.slurm` — 8 CPU, 64G, 12h
2. `slurm/sweep_vgg16_layers.slurm` — 8 CPU, 128G, 24h
3. `slurm/report_layer_sweeps.slurm` — after both sweeps finish (`afterok`)

### Environment overrides

```bash
WINDOW_CONFIG=configs/windows/evoked_35_42.yaml \
RESNET_MEM=64G VGG_MEM=128G \
RESNET_TIME=12:00:00 VGG_TIME=24:00:00 \
DEVICE=cpu \
bash scripts/submit_layer_sweeps.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `WINDOW_CONFIG` | `configs/windows/evoked_35_42.yaml` | Frame window |
| `SPLIT` / `SELECTION_SPLIT` | `val` | Split for layer selection |
| `TEST_SPLIT` | `test` | Split for winner evaluation |
| `RESNET_LAYERS` | `layer2 layer3 layer4 avgpool` | ResNet sweep layers |
| `VGG_LAYERS` | `block3 block4 block5` | VGG sweep layers |
| `RESNET_MEM` / `VGG_MEM` | `64G` / `128G` | Memory requests |
| `COMPARE_ONLY=1` | off | Skip heavy stages; rebuild tables/plots |
| `SKIP_RESNET=1` / `SKIP_VGG=1` | — | Submit only one sweep |
| `SKIP_REPORT=1` | — | Submit sweeps only |

Monitor:

```bash
squeue -u $USER
tail -f logs/sweep_resnet18_layers_<jobid>.out
tail -f logs/sweep_vgg16_layers_<jobid>.out
tail -f logs/report_layer_sweeps_<jobid>.out
```

## Outputs

### Per-backbone validation sweep

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/layer_sweep_{model_slug}/
├── layer_comparison.csv
├── layer_comparison.json
└── layer_mean_pixel_r.png
```

Missing metrics are preserved as empty/NaN in CSV/JSON (not plotted as zero). Incomplete layers are labeled `N/A` in bar charts.

### Cross-model report

```
plots/evaluation/{monkey}/{window_id}/backbone_comparison/cross_model_report/
├── validation_layer_comparison.csv
├── test_winner_metrics.csv
├── winner_summary.json
├── validation_layers_val.png
├── winner_comparison_test.png
└── backbone_layer_sweep_report.pdf
```

Selection metric: `eval_mean_r_masked` from stage 04 on the validation split.

## Memory notes

- VGG shallow blocks (`block1`, `block2`) produce very large flattened feature vectors; Ridge training can exceed 128G on the expanded dataset.
- If a sweep job fails with OOM, increase `--mem` in the SLURM script or reduce the layer set.
- Feature extraction reuses existing `.npy` maps when complete; only missing stimuli are filled.

## Related docs

- [`backbone_comparison.md`](backbone_comparison.md) — single-layer backbone comparison (stage 05)
- [`cluster_pipeline.md`](cluster_pipeline.md) — full encoding pipeline on SLURM
- [`ridge_encoding.md`](ridge_encoding.md) — RidgeCV training and metrics
