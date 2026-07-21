#!/bin/bash
#SBATCH --job-name=vsd_encoding
#SBATCH --output=/home/dsi/ossnat/VSD_Encoder_01/slurm_err_out/vsd_encoding_%j.out
#SBATCH --error=/home/dsi/ossnat/VSD_Encoder_01/slurm_err_out/vsd_encoding_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/dsi/ossnat/VSD_Encoder_01

# Site-specific single-allocation wrapper for stages 02b + 03.
#
# Before first submission:
#   mkdir -p /home/dsi/ossnat/VSD_Encoder_01/slurm_err_out
#
# Submit:
#   sbatch slurm/run_encoding_heavy_generic.slurm
#
# Optional overrides:
#   sbatch --export=ALL,WINDOW_CONFIG=configs/windows/evoked_35_42.yaml \
#     slurm/run_encoding_heavy_generic.slurm

set -euo pipefail

REPO_ROOT="/home/dsi/ossnat/VSD_Encoder_01"
cd "${REPO_ROOT}"

export WINDOW_CONFIG="${WINDOW_CONFIG:-configs/windows/evoked_35_42.yaml}"
export DEVICE="${DEVICE:-cuda}"

# Run the heavy stages directly in this allocation. SBATCH directives in the
# delegated script are comments when it is invoked with bash.
exec bash slurm/run_encoding_heavy.slurm
