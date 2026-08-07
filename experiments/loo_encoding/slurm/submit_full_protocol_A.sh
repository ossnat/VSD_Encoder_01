#!/usr/bin/env bash
# Submit full Protocol A LOO pipeline on SLURM (flat run layout).
#
# Stages:
#   0) prepare   — dry-run 4 leaves + folds.txt + pipeline_manifest.yaml
#   1) encode    — 4 array jobs (zscore/raw × clean/all), one task per fold
#   2) finalize  — rebuild summaries, triplet overviews, pooled encoding maps
#   3) noise     — odd/even pooled noise corr (can start after prepare)
#   4) report    — report.pdf (after finalize + noise)
#
# Usage (from repo root on the cluster):
#   bash experiments/loo_encoding/slurm/submit_full_protocol_A.sh
#
# Useful env overrides:
#   PIPELINE_CONFIG=experiments/loo_encoding/slurm/protocol_A_full.yaml
#   RUN_DATE=2026-08-07
#   PARTITION=generic
#   ACCOUNT=mylab
#   PREPARE_ONLY=1          # only stage 0
#   SKIP_NOISE=1
#   SKIP_REPORT=1
#   FORCE=1                 # re-run existing folds
#   DRY_RUN_SUBMIT=1        # print sbatch commands only
#
# Path dry-run (no SLURM, no training):
#   scripts/py experiments/loo_encoding/prepare_protocol_A_pipeline.py \
#     --config experiments/loo_encoding/slurm/protocol_A_full.yaml --paths-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export REPO_ROOT
export PIPELINE_CONFIG="${PIPELINE_CONFIG:-experiments/loo_encoding/slurm/protocol_A_full.yaml}"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common.sh"

PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
THROTTLE="$("${PYTHON}" - <<'PY'
import os, yaml
from pathlib import Path
from src.paths import project_root
repo = project_root()
cfg = yaml.safe_load((repo / os.environ["PIPELINE_CONFIG"]).read_text())
print((cfg.get("slurm") or {}).get("array_throttle") or 50)
PY
)"

SBATCH_EXTRA=()
if [[ -n "${PARTITION}" ]]; then
  SBATCH_EXTRA+=(--partition="${PARTITION}")
fi
if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_EXTRA+=(--account="${ACCOUNT}")
fi

submit() {
  local desc="$1"
  shift
  if [[ "${DRY_RUN_SUBMIT:-}" == "1" ]]; then
    echo "[dry-run] ${desc}: sbatch ${SBATCH_EXTRA[*]-} $*" >&2
    echo "DRYRUN_${desc}"
    return 0
  fi
  local jid
  jid="$(sbatch --parsable "${SBATCH_EXTRA[@]}" "$@")"
  echo "${desc}: job ${jid}" >&2
  echo "${jid}"
}

echo "=== Protocol A full pipeline ==="
echo "REPO_ROOT=${REPO_ROOT}"
echo "PIPELINE_CONFIG=${PIPELINE_CONFIG}"

# Stage 0: prepare (blocking if we need fold counts for arrays)
PREPARE_ARGS=()
if [[ -n "${RUN_DATE:-}" ]]; then
  PREPARE_ARGS+=(--export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}",RUN_DATE="${RUN_DATE}")
else
  PREPARE_ARGS+=(--export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}")
fi

JOB_PREP="$(submit "prepare" "${PREPARE_ARGS[@]}" "${SCRIPT_DIR}/00_prepare.slurm")"

if [[ "${PREPARE_ONLY:-}" == "1" ]]; then
  echo "PREPARE_ONLY=1 — stopping after prepare job ${JOB_PREP}"
  exit 0
fi

# Wait for prepare so we can read fold counts (unless dry-run submit).
if [[ "${DRY_RUN_SUBMIT:-}" != "1" ]]; then
  echo "Waiting for prepare job ${JOB_PREP} …"
  while squeue -j "${JOB_PREP}" -h 2>/dev/null | grep -q .; do
    sleep 20
  done
  # Confirm success via sacct if available
  if command -v sacct >/dev/null 2>&1; then
    state="$(sacct -j "${JOB_PREP}" -n -o State -X | head -1 | tr -d ' ')"
    if [[ "${state}" != "COMPLETED" ]]; then
      echo "ERROR: prepare job ${JOB_PREP} ended with state=${state}" >&2
      exit 1
    fi
  fi
fi

# Resolve RUN_ROOT from freshly written manifest / plan
export RUN_ROOT
RUN_ROOT="$(
  if [[ -n "${RUN_ROOT_OVERRIDE:-}" ]]; then
    echo "${RUN_ROOT_OVERRIDE}"
  else
    resolve_run_root
  fi
)"
echo "RUN_ROOT=${RUN_ROOT}"
MANIFEST="${RUN_ROOT}/pipeline_manifest.yaml"
if [[ "${DRY_RUN_SUBMIT:-}" != "1" && ! -f "${MANIFEST}" ]]; then
  echo "ERROR: missing ${MANIFEST}" >&2
  exit 1
fi

LEAF_KEYS=(zscore_clean zscore_all raw_clean raw_all)
ENCODE_JIDS=()

for key in "${LEAF_KEYS[@]}"; do
  if [[ "${DRY_RUN_SUBMIT:-}" == "1" ]]; then
    N=12
  else
    N="$("${PYTHON}" - <<PY
import yaml
from pathlib import Path
man = yaml.safe_load(Path("${MANIFEST}").read_text())
leaf = next(L for L in man["leaves"] if L["key"] == "${key}")
print(int(leaf.get("n_folds") or 0))
PY
)"
  fi
  if (( N <= 0 )); then
    echo "WARNING: leaf ${key} has 0 folds — skipping array"
    continue
  fi
  LAST=$((N - 1))
  ARRAY_SPEC="0-${LAST}%${THROTTLE}"
  jid="$(submit "encode_${key}" \
    --dependency="afterok:${JOB_PREP}" \
    --export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}",LEAF_KEY="${key}",RUN_ROOT="${RUN_ROOT}",FORCE="${FORCE:-}" \
    --array="${ARRAY_SPEC}" \
    "${SCRIPT_DIR}/01_encode_array.slurm")"
  ENCODE_JIDS+=("${jid}")
done

# Dependency string for all encode arrays
DEP_ENCODE=""
if ((${#ENCODE_JIDS[@]} > 0)); then
  DEP_ENCODE="afterok:$(IFS=:; echo "${ENCODE_JIDS[*]}")"
fi

# Stage 3 noise can run after prepare (fold list known); parallel with encode
JOB_NOISE=""
if [[ "${SKIP_NOISE:-}" != "1" ]]; then
  JOB_NOISE="$(submit "noise_corr" \
    --dependency="afterok:${JOB_PREP}" \
    --export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}",RUN_ROOT="${RUN_ROOT}" \
    "${SCRIPT_DIR}/03_noise_corr.slurm")"
fi

# Stage 2 finalize after all encode arrays
JOB_FIN=""
if [[ -n "${DEP_ENCODE}" ]]; then
  JOB_FIN="$(submit "finalize_maps" \
    --dependency="${DEP_ENCODE}" \
    --export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}",RUN_ROOT="${RUN_ROOT}" \
    "${SCRIPT_DIR}/02_finalize_and_maps.slurm")"
else
  echo "WARNING: no encode jobs; skipping finalize"
fi

# Stage 4 report after finalize + noise
if [[ "${SKIP_REPORT:-}" != "1" && -n "${JOB_FIN}" ]]; then
  DEP_REPORT="afterok:${JOB_FIN}"
  if [[ -n "${JOB_NOISE}" ]]; then
    DEP_REPORT="afterok:${JOB_FIN}:${JOB_NOISE}"
  fi
  JOB_PDF="$(submit "report_pdf" \
    --dependency="${DEP_REPORT}" \
    --export=ALL,PIPELINE_CONFIG="${PIPELINE_CONFIG}",RUN_ROOT="${RUN_ROOT}" \
    "${SCRIPT_DIR}/04_report.slurm")"
  echo "report job: ${JOB_PDF}"
fi

echo ""
echo "Submitted."
echo "  prepare:  ${JOB_PREP}"
echo "  encode:   ${ENCODE_JIDS[*]:-none}"
echo "  noise:    ${JOB_NOISE:-skipped}"
echo "  finalize: ${JOB_FIN:-skipped}"
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${REPO_ROOT}/experiments/loo_encoding/slurm/logs/"
echo "Output:  ${RUN_ROOT}/report.pdf (when complete)"
