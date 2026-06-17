# Shared environment for VSD_Encoder_01 shell scripts.
# Source from repo root:  source scripts/common_env.sh

: "${REPO_ROOT:?Set REPO_ROOT before sourcing, or run via scripts/run_prepare_encoding.sh}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.cache/torch}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: ${PYTHON} not found. Run: bash scripts/cluster_setup.sh" >&2
  return 1 2>/dev/null || exit 1
fi

WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "PYTHON=$("${PYTHON}" --version)"
