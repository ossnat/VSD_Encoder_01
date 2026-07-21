#!/bin/bash
#SBATCH --job-name=vsd_encoding
#SBATCH --output=/home/dsi/ossnat/VSD_FM/VSD_Encoder_01/slurm_err_out/vsd_encoding_%j.out
#SBATCH --error=/home/dsi/ossnat/VSD_FM/VSD_Encoder_01/slurm_err_out/vsd_encoding_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_FM/VSD_Encoder_01

# Site-specific single-allocation wrapper for stages 02b + 03.
#
# Before first submission:
#   mkdir -p /home/dsi/ossnat/VSD_FM/VSD_Encoder_01/slurm_err_out
#
# Submit (from the repo root on the cluster):
#   sbatch slurm/run_encoding_heavy_generic.slurm
#
# Optional overrides:
#   sbatch --export=ALL,WINDOW_CONFIG=configs/windows/evoked_35_42.yaml \
#     slurm/run_encoding_heavy_generic.slurm

set -euo pipefail

REPO_ROOT="/home/dsi/ossnat/VSD_FM/VSD_Encoder_01"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_35_42.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/models/resnet18.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
DEVICE="${DEVICE:-cuda}"
MONKEY="${MONKEY:-}"
OVERWRITE="${OVERWRITE:-}"

mkdir -p logs slurm_err_out
export MPLBACKEND=Agg
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  exit 1
fi

echo "REPO_ROOT=${REPO_ROOT}"
echo "PWD=$(pwd)"
echo "PYTHON=${PYTHON}"
echo "WINDOW_CONFIG=${WINDOW_CONFIG}"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "DEVICE=${DEVICE}"

echo ""
echo "=== Stage 02b: stimulus CNN features ==="
FEAT_CMD=("${PYTHON}" scripts/02b_extract_stimulus_features.py
  --config "${CONFIG}"
  --model "${MODEL_CONFIG}"
  --device "${DEVICE}"
)
if [[ -n "${MONKEY}" ]]; then
  FEAT_CMD+=(--monkey "${MONKEY}")
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  FEAT_CMD+=(--overwrite)
fi
echo "Running: ${FEAT_CMD[*]}"
"${FEAT_CMD[@]}"

echo ""
echo "=== Stage 03: RidgeCV encoder ==="
RIDGE_CMD=("${PYTHON}" scripts/03_train_ridge_encoder.py
  --config "${CONFIG}"
  --window "${WINDOW_CONFIG}"
  --ridge-config "${RIDGE_CONFIG}"
  --model "${MODEL_CONFIG}"
)
if [[ -n "${MONKEY}" ]]; then
  RIDGE_CMD+=(--monkey "${MONKEY}")
fi
echo "Running: ${RIDGE_CMD[*]}"
"${RIDGE_CMD[@]}"

echo "Encoding heavy stages complete."
