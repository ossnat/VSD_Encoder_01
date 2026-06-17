# Cluster pipeline guide

End-to-end instructions for running the **stimulus → CNN → RidgeCV → VSD** encoding pipeline on a SLURM cluster.

Assumes the same layout as your local machine:

```
<workspace>/                    e.g. ~/VSD_FM/
├── Data/                       NOT in git — must exist on cluster
│   ├── EncoderData/            stimulus CSV catalogs
│   └── FoundationData/
│       └── ProcessedData/      session H5 files + split CSVs
└── VSD_Encoder_01/             this repo (git pull)
```

All processed outputs are written under `Data/VSD_Encoder_01/` (also not in git).

---

## Overview

| Step | What | Where to run | Script |
|------|------|--------------|--------|
| 0 | Install Python env | login node | `bash scripts/cluster_setup.sh` |
| 0b | Verify Data paths | login node | `bash scripts/verify_workspace.sh` |
| 1 | Render stimulus PNGs | login node | stage **01b** |
| 2 | Build averaged VSD `.nc` | login node* | stage **01** |
| 3 | Build encoding-pairs manifest | login node | stage **01c** |
| 3b | VSD vs stimulus QC plots | login node | `plot_vsd_vs_stimulus.py` |
| 4 | ResNet features on stimuli | SLURM | stage **02b** |
| 5 | RidgeCV train + recon plots | SLURM | stage **03** |

\*Stage 01 is I/O bound but usually fine on the login node. For very large monkeys or slow filesystems, use `sbatch slurm/build_averaged_trials.slurm` instead.

**Quick path (recommended):**

```bash
cd VSD_Encoder_01
git pull
bash scripts/cluster_setup.sh          # once, or after requirements change
bash scripts/verify_workspace.sh
bash scripts/run_prepare_encoding.sh   # stages 01b + 01 + 01c + QC
bash scripts/submit_encoding_jobs.sh   # SLURM: 02b → 03
```

---

## Step 0 — Clone / update repo

```bash
cd ~/VSD_FM/VSD_Encoder_01    # adjust path
git pull
```

---

## Step 1 — One-time environment setup

On the **login node** (needs network for pip + PyTorch download):

```bash
cd VSD_Encoder_01
bash scripts/cluster_setup.sh
```

This creates `.venv/`, installs `requirements.txt` (h5py, torch, scikit-learn, …), and installs the package in editable mode.

Optional: cache pretrained ResNet weights in the repo (avoids re-downloading on compute nodes):

```bash
export TORCH_HOME="${PWD}/.cache/torch"
```

`scripts/common_env.sh` sets `TORCH_HOME` automatically when using the wrapper scripts.

---

## Step 2 — Verify shared Data/

Data is **not** in git. After `git pull`, confirm the sibling `Data/` tree exists:

```bash
bash scripts/verify_workspace.sh
```

Required inputs:

| Path | Purpose |
|------|---------|
| `Data/EncoderData/*_VSDI_*.csv` | Stimulus catalog (stage 01b) |
| `Data/FoundationData/ProcessedData/{monkey}/session_*.h5` | Raw VSD trials |
| `Data/FoundationData/ProcessedData/splits/split_v3_*.csv` | Train/val/test splits |
| `Data/FoundationData/ProcessedData/splits/all_trials_index.csv` | Condition strings |

Set monkey if not gandalf:

```bash
MONKEY=legolas bash scripts/verify_workspace.sh
```

---

## Step 3 — Prepare data (login node, no SLURM)

Runs stages **01b → 01 → 01c** and VSD-vs-stimulus QC plots:

```bash
bash scripts/run_prepare_encoding.sh
```

### Environment overrides

```bash
MONKEY=gandalf \
WINDOW_CONFIG=configs/windows/evoked_32_42.yaml \
OVERWRITE=1 \
bash scripts/run_prepare_encoding.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `MONKEY` | from `configs/default.yaml` | Subject |
| `WINDOW_CONFIG` | `configs/windows/evoked_32_42.yaml` | Frame window for VSD averaging |
| `STIMULI_CONFIG` | `configs/stimuli/default.yaml` | Stimulus render scale |
| `OVERWRITE=1` | off | Rebuild PNGs and `.nc` files |
| `SKIP_STIMULI=1` | — | Skip 01b |
| `SKIP_AVERAGED=1` | — | Skip 01 |
| `SKIP_PAIRS=1` | — | Skip 01c |

### Outputs created

```
Data/VSD_Encoder_01/
├── stimuli/{monkey}/images/{session}/condAN*.png
├── stimuli/{monkey}/manifest.parquet
├── averaged/{monkey}/{window_id}/trials/*.nc
├── averaged/{monkey}/{window_id}/manifest.parquet
└── encoding_pairs/{monkey}/{window_id}/manifest.parquet

plots/stimuli/{monkey}/              # stimulus QC
plots/averaged/{monkey}/{window_id}/ # averaged VSD QC
plots/vsd_vs_stimulus/{monkey}/{window_id}/  # VSD vs rendered stimulus
```

### Run stages individually (optional)

```bash
source scripts/common_env.sh   # after: export REPO_ROOT=$PWD

$PYTHON scripts/01b_build_stimulus_images.py \
  --config configs/default.yaml \
  --stimuli-config configs/stimuli/default.yaml

$PYTHON scripts/01_build_averaged_trials.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml \
  --overwrite

$PYTHON scripts/01c_build_encoding_pairs.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml

$PYTHON scripts/plot_vsd_vs_stimulus.py \
  --config configs/default.yaml \
  --window configs/windows/evoked_32_42.yaml
```

### Optional: stage 01 via SLURM

If averaging all trials is slow on the login node:

```bash
OVERWRITE=1 sbatch slurm/build_averaged_trials.slurm
# then, after it finishes:
bash scripts/run_prepare_encoding.sh   # with SKIP_AVERAGED=1 SKIP_STIMULI=1 if 01b already done
```

---

## Step 4 — Heavy compute (SLURM)

After prepare stages complete, submit CNN feature extraction and Ridge training.

### Option A — two chained jobs (recommended)

```bash
bash scripts/submit_encoding_jobs.sh
```

Submits `02b` then `03` with `--dependency=afterok`.

Overrides:

```bash
MONKEY=gandalf DEVICE=cuda OVERWRITE=1 bash scripts/submit_encoding_jobs.sh
```

### Option B — single combined job

```bash
sbatch slurm/run_encoding_heavy.slurm
```

### Option C — submit jobs individually

```bash
sbatch slurm/extract_stimulus_features.slurm
# wait for completion, then:
sbatch slurm/train_ridge_encoder.slurm
```

### Monitor jobs

```bash
squeue -u $USER
tail -f logs/extract_stimulus_features_<jobid>.out
tail -f logs/train_ridge_encoder_<jobid>.out
```

### Heavy-stage outputs

```
Data/VSD_Encoder_01/
├── DL_features_stimuli/{monkey}/{model_slug}/{feature_layer}/maps/*.npy
└── ridge_encode/{monkey}/{window_id}/{model_slug}/{feature_layer}/model.joblib

plots/ridge_encode/{monkey}/{window_id}/{model_slug}/{feature_layer}/
├── bias.png
├── by_condition/{session}__{cond}.png      # original vs reconstructed
├── reconstructions_by_condition.png        # paginated grid
├── reconstructions_by_condition_recon_only.png
└── (VSD vs stimulus also in plots/vsd_vs_stimulus/ from prepare step)
```

---

## Configuration reference

| File | Controls |
|------|----------|
| `configs/default.yaml` | Monkey, paths, split CSV |
| `configs/windows/evoked_32_42.yaml` | Frames 32–42, `window_id` |
| `configs/stimuli/default.yaml` | 224 px = 6° quadrant, bar/circle scale |
| `configs/models/resnet18.yaml` | Backbone, `feature_layer` (default `layer3`) |
| `configs/ridge/default.yaml` | RidgeCV alphas, plot options |

Change subject:

```bash
MONKEY=legolas bash scripts/run_prepare_encoding.sh
```

---

## Troubleshooting

### `verify_workspace.sh` fails

- Ensure `Data/` is a **sibling** of `VSD_Encoder_01/`, not inside the repo.
- Sync or mount `FoundationData` and `EncoderData` onto the cluster.

### Encoding pairs shows few trials / missing `.nc`

- Run stage 01 with full H5s on cluster (not the partial local copy).
- Re-run `01c` after 01 completes.

### All VSD originals looked identical (fixed in code)

Session H5 files store trials sequentially; `trial_000000` in the split CSV is **per-condition**, not global. The code resolves the correct dataset via `trial_metadata_json`. Ensure you have the latest `git pull`.

### ResNet download fails on compute nodes

```bash
export TORCH_HOME=/path/to/VSD_Encoder_01/.cache/torch
# pre-download on login node:
.venv/bin/python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.DEFAULT)"
```

### Ridge job runs out of memory

Increase in `slurm/train_ridge_encoder.slurm`:

```bash
#SBATCH --mem=64G
```

---

## Full pipeline checklist

```
[ ] git pull
[ ] bash scripts/cluster_setup.sh
[ ] bash scripts/verify_workspace.sh
[ ] bash scripts/run_prepare_encoding.sh
[ ] Check plots/stimuli/ and plots/vsd_vs_stimulus/
[ ] bash scripts/submit_encoding_jobs.sh   (or sbatch slurm/run_encoding_heavy.slurm)
[ ] Check plots/ridge_encode/ after jobs finish
```

---

## Related docs

- `docs/encoding_pipeline.md` — stage overview
- `docs/stimulus_rendering.md` — stimulus PNG rules
- `docs/preprocessing.md` — VSD averaging
- `docs/encoding_pairs.md` — trial join manifest
- `docs/DL_feature_extraction.md` — CNN features
- `docs/ridge_encoding.md` — RidgeCV model
- `docs/DATA_LAYOUT.md` — H5 / split file layout
