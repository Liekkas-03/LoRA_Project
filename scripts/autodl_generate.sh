#!/usr/bin/env bash
set -euo pipefail

# 本脚本用于云端加载 base model + LoRA adapter，并生成 MATH prediction JSONL。
BASE_MODEL="${1:?Usage: bash scripts/autodl_generate.sh BASE_MODEL ADAPTER_PATH INPUT_JSONL OUTPUT_JSONL [MAX_NEW_TOKENS] [PROMPT_TYPE]}"
ADAPTER_PATH="${2:?Usage: bash scripts/autodl_generate.sh BASE_MODEL ADAPTER_PATH INPUT_JSONL OUTPUT_JSONL [MAX_NEW_TOKENS] [PROMPT_TYPE]}"
INPUT_JSONL="${3:?Usage: bash scripts/autodl_generate.sh BASE_MODEL ADAPTER_PATH INPUT_JSONL OUTPUT_JSONL [MAX_NEW_TOKENS] [PROMPT_TYPE]}"
OUTPUT_JSONL="${4:?Usage: bash scripts/autodl_generate.sh BASE_MODEL ADAPTER_PATH INPUT_JSONL OUTPUT_JSONL [MAX_NEW_TOKENS] [PROMPT_TYPE]}"
MAX_NEW_TOKENS="${5:-512}"
PROMPT_TYPE="${6:-qwen25-math-cot}"

echo "Base model: ${BASE_MODEL}"
echo "Adapter: ${ADAPTER_PATH}"
echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_JSONL}"
echo "Prompt type: ${PROMPT_TYPE}"

python -m src.infer.generate_answers \
  --base-model "${BASE_MODEL}" \
  --adapter "${ADAPTER_PATH}" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt-type "${PROMPT_TYPE}"
