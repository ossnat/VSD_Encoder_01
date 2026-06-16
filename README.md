# VSD_Encoder_01

Two-stage encoder project for VSD data:

1. **Feature extraction** — pretrained DNN (e.g. ResNet) on frame-averaged trials
2. **Downstream task** — linear model on cached DNN features

## Current scope: data preprocessing (stage 01)

Frame-averaged trials are built from shared session HDF5 files (see `docs/DATA_LAYOUT.md`).

### Setup

```bash
cd VSD_Encoder_01
pip install -e .
```

Workspace layout (siblings):

```
VSD_FM/
├── Data/
└── VSD_Encoder_01/
```

### Build averaged trials

```bash
PYTHONPATH=. python scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_30_50.yaml
```

Test on a handful of trials:

```bash
PYTHONPATH=. python scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_30_50.yaml \
  --max-trials 10
```

Outputs:

- `Data/VSD_Encoder_01/averaged/{monkey}/{window_id}/trials/*.nc`
- `Data/VSD_Encoder_01/averaged/{monkey}/{window_id}/manifest.parquet`
- `plots/averaged/{monkey}/{window_id}/` — 3–4 sample QC plots per run

See `docs/preprocessing.md` for schema and manifest rationale.

### SLURM (cluster)

```bash
sbatch slurm/build_averaged_trials.slurm

# custom window
WINDOW_CONFIG=configs/windows/baseline_05_25.yaml sbatch slurm/build_averaged_trials.slurm
```
