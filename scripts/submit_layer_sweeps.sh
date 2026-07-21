#!/usr/bin/env bash
# Submit ResNet18 and VGG16 validation layer sweeps, then the cross-model report.
#
# Usage (from repo root, after run_prepare_encoding.sh):
#   bash scripts/submit_layer_sweeps.sh
#
# Environment overrides (forwarded to sbatch scripts):
#   MONKEY=gandalf
#   CONFIG=configs/default.yaml
#   WINDOW_CONFIG=configs/windows/evoked_35_42.yaml
#   RIDGE_CONFIG=configs/ridge/default.yaml
#   RESNET_MODEL=configs/models/resnet18.yaml
#   VGG_MODEL=configs/models/vgg16.yaml
#   DEVICE=auto|cpu|cuda
#   SPLIT=val
#   RESNET_LAYERS="layer2 layer3 layer4 avgpool"
#   VGG_LAYERS="block3 block4 block5"
#   RESNET_MEM=64G
#   VGG_MEM=128G
#   RESNET_TIME=12:00:00
#   VGG_TIME=24:00:00
#   REPORT_TIME=04:00:00
#   PARTITION=cpu
#   SKIP_RESNET=1
#   SKIP_VGG=1
#   SKIP_REPORT=1
#   COMPARE_ONLY=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p logs

export CONFIG="${CONFIG:-configs/default.yaml}"
export WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_35_42.yaml}"
export RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
export RESNET_MODEL="${RESNET_MODEL:-configs/models/resnet18.yaml}"
export VGG_MODEL="${VGG_MODEL:-configs/models/vgg16.yaml}"
export DEVICE="${DEVICE:-auto}"
export MONKEY="${MONKEY:-}"
export SPLIT="${SPLIT:-val}"
export SELECTION_SPLIT="${SELECTION_SPLIT:-val}"
export TEST_SPLIT="${TEST_SPLIT:-test}"
export PARTITION="${PARTITION:-cpu}"

JOB_RESNET=""
JOB_VGG=""

if [[ "${SKIP_RESNET:-}" != "1" ]]; then
  echo "Submitting ResNet18 validation layer sweep..."
  JOB_RESNET="$(
    sbatch --parsable \
      --partition="${PARTITION}" \
      --mem="${RESNET_MEM:-64G}" \
      --time="${RESNET_TIME:-12:00:00}" \
      --export=ALL,LAYERS="${RESNET_LAYERS:-layer2 layer3 layer4 avgpool}" \
      slurm/sweep_resnet18_layers.slurm
  )"
  echo "  Job ID: ${JOB_RESNET}"
fi

if [[ "${SKIP_VGG:-}" != "1" ]]; then
  echo "Submitting VGG16 validation layer sweep..."
  JOB_VGG="$(
    sbatch --parsable \
      --partition="${PARTITION}" \
      --mem="${VGG_MEM:-128G}" \
      --time="${VGG_TIME:-24:00:00}" \
      --export=ALL,LAYERS="${VGG_LAYERS:-block3 block4 block5}" \
      slurm/sweep_vgg16_layers.slurm
  )"
  echo "  Job ID: ${JOB_VGG}"
fi

if [[ "${SKIP_REPORT:-}" != "1" ]]; then
  DEPENDENCY=""
  if [[ -n "${JOB_RESNET}" && -n "${JOB_VGG}" ]]; then
    DEPENDENCY="--dependency=afterok:${JOB_RESNET}:${JOB_VGG}"
  elif [[ -n "${JOB_RESNET}" ]]; then
    DEPENDENCY="--dependency=afterok:${JOB_RESNET}"
  elif [[ -n "${JOB_VGG}" ]]; then
    DEPENDENCY="--dependency=afterok:${JOB_VGG}"
  fi

  echo "Submitting cross-model layer-sweep report..."
  JOB_REPORT="$(
    sbatch --parsable \
      ${DEPENDENCY} \
      --partition="${PARTITION}" \
      --mem="${REPORT_MEM:-32G}" \
      --time="${REPORT_TIME:-04:00:00}" \
      slurm/report_layer_sweeps.slurm
  )"
  echo "  Job ID: ${JOB_REPORT}"
fi

echo ""
echo "Jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: ${REPO_ROOT}/logs/"
