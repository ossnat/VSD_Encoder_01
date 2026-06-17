#!/usr/bin/env bash
# Check that workspace Data/ paths exist before running the encoding pipeline.
#
# Usage (from repo root):
#   bash scripts/verify_workspace.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

MONKEY="${MONKEY:-gandalf}"

echo "=== Workspace layout ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "MONKEY=${MONKEY}"
echo ""

check_path() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    echo "OK  ${label}: ${path}"
  else
    echo "MISSING  ${label}: ${path}"
    return 1
  fi
}

ERR=0
check_path "FoundationData root" "${WORKSPACE_ROOT}/Data/FoundationData/ProcessedData" || ERR=1
check_path "EncoderData root" "${WORKSPACE_ROOT}/Data/EncoderData" || ERR=1
check_path "Split CSV" "${WORKSPACE_ROOT}/Data/FoundationData/ProcessedData/splits/split_v3_seed17_session_condition_group.csv" || ERR=1
check_path "Trials index" "${WORKSPACE_ROOT}/Data/FoundationData/ProcessedData/splits/all_trials_index.csv" || ERR=1

N_H5=$(find "${WORKSPACE_ROOT}/Data/FoundationData/ProcessedData/${MONKEY}" -maxdepth 1 -name 'session_*.h5' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${N_H5}" -gt 0 ]]; then
  echo "OK  Session H5 files (${MONKEY}): ${N_H5} under ProcessedData/${MONKEY}/"
else
  echo "MISSING  Session H5 files for monkey=${MONKEY}"
  ERR=1
fi

ENC_CSV=$(ls "${WORKSPACE_ROOT}/Data/EncoderData/"*VSDI*.csv 2>/dev/null | head -1 || true)
if [[ -n "${ENC_CSV}" ]]; then
  echo "OK  Encoder catalog CSV: ${ENC_CSV}"
else
  echo "MISSING  Encoder catalog CSV under Data/EncoderData/"
  ERR=1
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  echo "OK  Python venv: ${REPO_ROOT}/.venv"
else
  echo "MISSING  venv — run: bash scripts/cluster_setup.sh"
  ERR=1
fi

echo ""
if [[ "${ERR}" -eq 0 ]]; then
  echo "Workspace check passed."
else
  echo "Workspace check FAILED — fix missing paths before running the pipeline."
  exit 1
fi
