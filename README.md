# VSD_Encoder_01

Stimulus-to-VSD encoding pipeline:

1. **Average VSD trials** → target maps (Y)
2. **Render stimulus images** from EncoderData CSV catalogs
3. **Extract CNN features** from stimuli (ResNet)
4. **RidgeCV encoding** → predict VSD from stimulus features
5. **Evaluate** reconstructions on val/test splits

See `docs/encoding_pipeline.md` for the full flow.  
**Cluster:** see `docs/cluster_pipeline.md` for step-by-step setup and SLURM jobs.

## Setup

```bash
cd VSD_Encoder_01
bash scripts/cluster_setup.sh
```

**Local runs:** macOS often has no `python` command — use the project venv:

```bash
scripts/py scripts/03_train_ridge_encoder.py --help
# or: source .venv/bin/activate && PYTHONPATH=. python3 ...
```

Workspace layout (siblings):

```
VSD_FM/
├── Data/
│   ├── EncoderData/          # stimulus CSV catalogs
│   └── VSD_Encoder_01/       # processed outputs
└── VSD_Encoder_01/           # this repo
```

## Stage 01 — averaged VSD trials (targets)

```bash
scripts/py scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml
```

Outputs: `Data/VSD_Encoder_01/averaged/...`  
Docs: `docs/preprocessing.md`

## Stage 01b — render stimulus images

```bash
scripts/py scripts/01b_build_stimulus_images.py \
  --config configs/default.yaml \
  --stimuli-config configs/stimuli/default.yaml
```

Outputs:

- `Data/VSD_Encoder_01/stimuli/{monkey}/images/{h5_session}/condAN*.png`
- `Data/VSD_Encoder_01/stimuli/{monkey}/manifest.parquet`
- **QC plots:** `plots/stimuli/{monkey}/all_stimuli_grid.png` (+ one PNG per stimulus)

Docs: `docs/stimulus_rendering.md`

## Stage 01c — encoding pair manifest

Joins stimulus images with averaged VSD trials for encoder sessions:

```bash
scripts/py scripts/01c_build_encoding_pairs.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml
```

Outputs: `Data/VSD_Encoder_01/encoding_pairs/{monkey}/{window_id}/manifest.parquet`  
Docs: `docs/encoding_pairs.md`

## Stage 02b — CNN features from stimuli

```bash
scripts/py scripts/02b_extract_stimulus_features.py \
  --config configs/default.yaml \
  --model configs/models/resnet18.yaml
```

Outputs: `Data/VSD_Encoder_01/DL_features_stimuli/{monkey}/{model_slug}/{feature_layer}/`  
Docs: `docs/DL_feature_extraction.md`

## Stage 03 — RidgeCV encoder

```bash
scripts/py scripts/03_train_ridge_encoder.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml \
  --model configs/models/resnet18.yaml
```

Outputs: `Data/VSD_Encoder_01/ridge_encode/...` and QC plots in `plots/ridge_encode/...`  
Docs: `docs/ridge_encoding.md`

## Stage 02 — CNN features (legacy VSD input)

```bash
scripts/py scripts/02_extract_features.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml \
  --model configs/models/resnet18.yaml
```

Docs: `docs/DL_feature_extraction.md`

## Cluster (full pipeline)

```bash
bash scripts/cluster_setup.sh
bash scripts/verify_workspace.sh
bash scripts/run_prepare_encoding.sh      # login node: 01b, 01, 01c
bash scripts/submit_encoding_jobs.sh      # SLURM: 02b → 03
```

Details: `docs/cluster_pipeline.md`

## SLURM (individual jobs)

```bash
sbatch slurm/build_averaged_trials.slurm
sbatch slurm/build_stimulus_images.slurm
sbatch slurm/build_encoding_pairs.slurm
sbatch slurm/extract_stimulus_features.slurm
sbatch slurm/train_ridge_encoder.slurm
sbatch slurm/run_encoding_heavy.slurm      # 02b + 03 in one job
sbatch slurm/extract_features.slurm          # legacy VSD-input path
```

## Data references

- `docs/DATA_LAYOUT.md` — VSD trial H5 layout and splits
- `docs/encoding_pipeline.md` — end-to-end encoding overview
