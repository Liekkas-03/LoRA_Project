#!/usr/bin/env bash
set -euo pipefail

# 本脚本用于云端启动 QLoRA/AdaQLoRA 训练，本地不应运行重依赖训练。
CONFIG_PATH="${1:-configs/qlora_math.yaml}"

echo "Using config: ${CONFIG_PATH}"
echo "This script is intended for AutoDL or another cloud GPU environment."

python -m src.train.train_sft --config "${CONFIG_PATH}"
