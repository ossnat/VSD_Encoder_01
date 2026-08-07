# SLURM Protocol A full LOO pipeline

Production cluster pipeline for **all Protocol A session×condition folds**
(held-out list), comparing **zscore vs raw** × **clean vs all-data** encoding,
plus **odd/even noise correlation**, ending in `report.pdf`.

## Layout (flat)

```
experiments/loo_encoding/runs/YYYY-MM-DD_35-46_resnet18_l3/
  pipeline_manifest.yaml
  protocol_A_zscore_NChull_clean/
  protocol_A_zscore_NChull_all/
  protocol_A_raw_NChull_clean/
  protocol_A_raw_NChull_all/
  noise_corr_odd_even/
  pooled_fold_pixel_r__*.png
  corr_summary_encoding.csv
  corr_summary_table.csv
  report.pdf
```

## Prerequisites (cluster)

1. Repo + venv (`bash scripts/cluster_setup.sh`).
2. Averaged trials + encoding pairs for **both** windows:
   - `configs/windows/evoked_35_46.yaml`
   - `configs/windows/evoked_35_46_zscore.yaml`
3. ResNet18 ImageNet **layer3** stimulus features extracted (window-independent).
4. NC hull mask present:
   `experiments/noise_ceiling_roi/rois/global_noise_ceiling_hull__mask.npy`
5. Trial cleanliness CSV (for clean leaves), default:
   `Data/VSD_Encoder_01/qc/trial_cleanliness_gandalf__win_0035_0046_zscore.csv`
   (generate via `scripts/17_classify_trial_cleanliness.py` if missing).
6. Edit `protocol_A_full.yaml` → `slurm.partition` / submit with `PARTITION=` / `ACCOUNT=`.

## Recommended approach: array over folds

One **SLURM array task = one Protocol A fold** inside one leaf. Four arrays
(one per leaf) keep concurrent writers off shared `folds_index.yaml` /
`loo_summary.csv` (`--array-worker`). Finalize rebuilds summaries.

Noise corr only needs the **all-data** fold list from prepare — it can run
**in parallel** with encoding.

## Quick start — ResNet18 / layer3

```bash
# On the login node, from repo root:
cd /path/to/VSD_Encoder_01

# Optional path dry-run (laptop-safe; no training):
scripts/py experiments/loo_encoding/prepare_protocol_A_pipeline.py \
  --config experiments/loo_encoding/slurm/protocol_A_full.yaml \
  --paths-only

# Optional: pin the run-root date
export RUN_DATE=2026-08-07
export PARTITION=generic          # CHANGE ME
# export ACCOUNT=mylab            # if required

bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
```

Prepare-only (writes leaves + `folds.txt`, no encode):

```bash
PREPARE_ONLY=1 bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
```

Print sbatch lines without submitting:

```bash
DRY_RUN_SUBMIT=1 bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
```

## Stages

| Stage | Script | What |
|------:|--------|------|
| 0 | `00_prepare.slurm` | Dry-run 4 leaves → `params.yaml`, `folds_index.yaml`, `folds.txt`, `pipeline_manifest.yaml` |
| 1 | `01_encode_array.slurm` | Array: `run_loo_encoding.py --fold-id … --array-worker --no-save-model` (models not saved by default to avoid quota blowups) |
| 2 | `02_finalize_and_maps.slurm` | `finalize_loo_leaf.py` + triplet overviews + pooled encoding r maps |
| 3 | `03_noise_corr.slurm` | `compute_fold_noise_corr_odd_even.py` → `noise_corr_odd_even/` |
| 4 | `04_report.slurm` | `build_protocol_A_report_pdf.py` → `report.pdf` |

Master: `submit_full_protocol_A.sh` wires dependencies.

## Manual / single-leaf encode

After prepare:

```bash
export RUN_ROOT=experiments/loo_encoding/runs/2026-08-07_35-46_resnet18_l3
export LEAF_KEY=zscore_clean
N=$(wc -l < "${RUN_ROOT}/protocol_A_zscore_NChull_clean/folds.txt")
sbatch --array=0-$((N-1))%50 \
  --export=ALL,LEAF_KEY,RUN_ROOT,PIPELINE_CONFIG \
  experiments/loo_encoding/slurm/01_encode_array.slurm
```

## Swap model / layer

1. Copy config:
   ```bash
   cp experiments/loo_encoding/slurm/protocol_A_full.yaml \
      experiments/loo_encoding/slurm/protocol_A_vgg16_l3.yaml
   ```
2. Edit:
   ```yaml
   model: configs/models/vgg16.yaml
   feature_layer: <layer>
   ```
3. Ensure stimulus features for that model/layer exist.
4. Submit:
   ```bash
   PIPELINE_CONFIG=experiments/loo_encoding/slurm/protocol_A_vgg16_l3.yaml \
     bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
   ```

Run-root name auto-shortens (e.g. `resnet18_imagenet`→`resnet18`, `layer3`→`l3`).

## Held-out list / fold count

Default: `experiments/loo_encoding/heldout_list.yaml` (~20 stimulus IDs).
Protocol A expands each ID to **all session×condition** rows present in
encoding pairs → often **tens to ~100+ folds** (not the 12-fold triangle/bar/F
smoke subset).

Missing IDs are skipped at fold build. Clean leaves may have **fewer** folds
than all-data if a `(date, condition)` loses all trials after QC.

To restrict stimuli for a pilot, pass a custom YAML via `heldout_list:` or
temporarily edit the shared list (prefer a copy).

## Corr summary table (in PDF)

| | clean encoding | all-data encoding | odd-even noise |
|--|----------------|-------------------|----------------|
| zscore | mean r (NC hull) | … | … |
| raw | … | … | … |

Encoding = pooled fold-level pixel-r (fold-mean orig vs recon).  
Noise = odd/even trial means per fold, same pooling (all trials; no clean filter).

## Files in this directory

| File | Purpose |
|------|---------|
| `protocol_A_full.yaml` | Default model/windows/QC/SLURM params |
| `common.sh` | Shared venv / PYTHONPATH helpers |
| `submit_full_protocol_A.sh` | Master multi-stage submitter |
| `00_prepare.slurm` | Stage 0 |
| `01_encode_array.slurm` | Stage 1 array worker |
| `02_finalize_and_maps.slurm` | Stage 2 |
| `03_noise_corr.slurm` | Stage 3 |
| `04_report.slurm` | Stage 4 |
| `logs/` | SLURM stdout/stderr (created on submit) |

Python helpers live one level up under `experiments/loo_encoding/`.
