#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/qlora_math.yaml}"

echo "Using config: ${CONFIG_PATH}"
echo "This script is intended for AutoDL or another cloud GPU environment."

python -m src.train.train_sft --config "${CONFIG_PATH}"
