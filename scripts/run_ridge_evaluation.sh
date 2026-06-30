#!/usr/bin/env bash
# Re-run Ridge training (03) and pixel evaluation (04) with current ridge config
# (including evaluation mask). Optionally refresh backbone comparison (05).
#
# Usage (from repo root):
#   bash scripts/run_ridge_evaluation.sh
#
# Environment overrides:
#   MONKEY=gandalf
#   CONFIG=configs/default.yaml
#   WINDOW_CONFIG=configs/windows/evoked_32_42.yaml
#   RIDGE_CONFIG=configs/ridge/default.yaml
#   MODELS="configs/models/resnet18.yaml configs/models/gabor_serre.yaml"
#   SPLITS="test"                    # space-separated splits for stage 04
#   SKIP_RIDGE=1                     # skip stage 03
#   SKIP_EVAL=1                      # skip stage 04
#   SKIP_COMPARE=1                   # skip stage 05
#   RUN_COMPARE=1                    # run stage 05 after 03+04 (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
RIDGE_CONFIG="${RIDGE_CONFIG:-configs/ridge/default.yaml}"
MODELS="${MODELS:-configs/models/resnet18.yaml configs/models/gabor_serre.yaml}"
SPLITS="${SPLITS:-test}"
MONKEY="${MONKEY:-}"
RUN_COMPARE="${RUN_COMPARE:-1}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"

MONKEY_ARGS=()
if [[ -n "${MONKEY}" ]]; then
  MONKEY_ARGS=(--monkey "${MONKEY}")
fi

read -r -a MODEL_LIST <<< "${MODELS}"

echo ""
echo "=== Ridge evaluation re-run ==="
echo "CONFIG=${CONFIG}"
echo "WINDOW=${WINDOW_CONFIG}"
echo "RIDGE_CONFIG=${RIDGE_CONFIG}"
echo "MODELS=${MODELS}"
echo "SPLITS=${SPLITS}"

for model in "${MODEL_LIST[@]}"; do
  if [[ ! -f "${model}" ]]; then
    echo "ERROR: model config not found: ${model}" >&2
    exit 1
  fi

  if [[ "${SKIP_RIDGE:-}" != "1" ]]; then
    echo ""
    echo "=== Stage 03: RidgeCV | ${model} ==="
    "${PYTHON}" scripts/03_train_ridge_encoder.py \
      --config "${CONFIG}" \
      --window "${WINDOW_CONFIG}" \
      --ridge-config "${RIDGE_CONFIG}" \
      --model "${model}" \
      "${MONKEY_ARGS[@]}"
  fi

  if [[ "${SKIP_EVAL:-}" != "1" ]]; then
    read -r -a SPLIT_LIST <<< "${SPLITS}"
    for split in "${SPLIT_LIST[@]}"; do
      echo ""
      echo "=== Stage 04: pixel evaluation (${split}) | ${model} ==="
      "${PYTHON}" scripts/04_evaluate_pixel_correlation.py \
        --config "${CONFIG}" \
        --window "${WINDOW_CONFIG}" \
        --ridge-config "${RIDGE_CONFIG}" \
        --model "${model}" \
        --split "${split}" \
        "${MONKEY_ARGS[@]}"
    done
  fi
done

if [[ "${RUN_COMPARE}" == "1" && "${SKIP_COMPARE:-}" != "1" ]]; then
  echo ""
  echo "=== Stage 05: backbone comparison ==="
  COMPARE_ARGS=(
    --config "${CONFIG}"
    --window "${WINDOW_CONFIG}"
  )
  for model in "${MODEL_LIST[@]}"; do
    COMPARE_ARGS+=(--model "${model}")
  done
  read -r -a SPLIT_LIST <<< "${SPLITS}"
  for split in "${SPLIT_LIST[@]}"; do
    echo "--- compare split=${split} ---"
    "${PYTHON}" scripts/05_compare_backbones.py \
      "${COMPARE_ARGS[@]}" \
      --split "${split}"
  done
fi

echo ""
echo "Ridge evaluation re-run complete."
