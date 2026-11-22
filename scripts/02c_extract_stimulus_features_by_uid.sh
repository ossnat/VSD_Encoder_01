#!/usr/bin/env bash
# Re-extract CNN / Gabor features once per stimulus-library image (maps_by_uid/<stimulus_uid>.npy) for every
# model and layer that exists under Data/VSD_Encoder_01/DL_features_stimuli/<monkey>/ today.
# The old per-session maps (maps/<session>__<condition>.npy) are left untouched.
#
# Usage (repo root, macOS project venv):   bash scripts/02c_extract_stimulus_features_by_uid.sh [--device mps|cpu|cuda]
# Afterwards (any machine):                scripts/py tools/stimulus_audit/check_uid_features.py
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${2:-auto}"; [[ "${1:-}" == "--device" ]] || DEVICE="auto"

run () {  # model_yaml layer
  echo "=== $1  $2"
  scripts/py scripts/02b_extract_stimulus_features.py --model "configs/models/$1.yaml" --feature-layer "$2" --overwrite --device "$DEVICE"
}

for L in layer1 layer2 layer3 layer4 avgpool; do run resnet18 "$L"; done
for L in block1_prepool block1 block2 block3 block4 block5 avgpool block1_pool14 block1_pool7 block2_pool14 block2_pool7 block3_pool14; do run vgg16 "$L"; done
for L in V1_pool14 V1_pool7 V2 V4; do run cornet_s "$L"; done
run gabor_serre energy
echo
echo "Done. Not re-extracted here: the Gabor grid-search variants (gabor_serre_gwp_<params>); rerun scripts/06_grid_search_gwp.py if needed."
