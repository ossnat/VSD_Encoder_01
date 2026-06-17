#!/usr/bin/env bash
# Run lightweight encoding-pipeline stages on the login node (no SLURM).
#
# Stages:
#   01b  render stimulus images from EncoderData CSV
#   01   build frame-averaged VSD .nc targets
#   01c  build encoding-pairs manifest
#   QC   VSD vs stimulus comparison plots
#
# Usage (from repo root):
#   bash scripts/run_prepare_encoding.sh
#
# Environment overrides:
#   MONKEY=gandalf
#   CONFIG=configs/default.yaml
#   WINDOW_CONFIG=configs/windows/evoked_32_42.yaml
#   STIMULI_CONFIG=configs/stimuli/default.yaml
#   OVERWRITE=1              # rebuild stimuli + averaged .nc
#   SKIP_STIMULI=1           # skip 01b
#   SKIP_AVERAGED=1          # skip 01
#   SKIP_PAIRS=1             # skip 01c
#   SKIP_VSD_STIMULUS_PLOTS=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-configs/default.yaml}"
WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_32_42.yaml}"
STIMULI_CONFIG="${STIMULI_CONFIG:-configs/stimuli/default.yaml}"
MONKEY="${MONKEY:-}"
OVERWRITE="${OVERWRITE:-}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"

echo ""
echo "=== Verify workspace ==="
MONKEY="${MONKEY}" bash scripts/verify_workspace.sh

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi
MONKEY_ARGS=()
if [[ -n "${MONKEY}" ]]; then
  MONKEY_ARGS=(--monkey "${MONKEY}")
fi

if [[ "${SKIP_STIMULI:-}" != "1" ]]; then
  echo ""
  echo "=== Stage 01b: render stimulus images ==="
  "${PYTHON}" scripts/01b_build_stimulus_images.py \
    --config "${CONFIG}" \
    --stimuli-config "${STIMULI_CONFIG}" \
    "${MONKEY_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"
fi

if [[ "${SKIP_AVERAGED:-}" != "1" ]]; then
  echo ""
  echo "=== Stage 01: build averaged VSD trials ==="
  "${PYTHON}" scripts/01_build_averaged_trials.py \
    --config "${CONFIG}" \
    --window "${WINDOW_CONFIG}" \
    "${MONKEY_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}"
fi

if [[ "${SKIP_PAIRS:-}" != "1" ]]; then
  echo ""
  echo "=== Stage 01c: build encoding pairs manifest ==="
  "${PYTHON}" scripts/01c_build_encoding_pairs.py \
    --config "${CONFIG}" \
    --window "${WINDOW_CONFIG}" \
    "${MONKEY_ARGS[@]}"
fi

if [[ "${SKIP_VSD_STIMULUS_PLOTS:-}" != "1" ]]; then
  echo ""
  echo "=== QC: VSD vs stimulus plots ==="
  "${PYTHON}" scripts/plot_vsd_vs_stimulus.py \
    --config "${CONFIG}" \
    --window "${WINDOW_CONFIG}" \
    "${MONKEY_ARGS[@]}"
fi

echo ""
echo "Prepare stages complete."
echo "Next (compute-heavy): bash scripts/submit_encoding_jobs.sh"
