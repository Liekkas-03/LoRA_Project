"""Classify coarse math reasoning errors from prediction JSONL files."""

# 本文件负责给数学题预测结果添加粗粒度错误类型标签，
# 为后续 verifier-guided replay 和错误分析提供基础数据。

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.extract_answer import extract_answer
from src.eval.math_verifier import equivalent, looks_like_number


def classify_error(prediction: str, reference: str) -> str:
    # 对单条预测进行错误分类：先抽答案，再判断正确、格式错误或答案错误等。
    pred_answer = extract_answer(prediction)
    ref_answer = extract_answer(reference)

    if pred_answer is None:
        return "format_error"
    if ref_answer is None:
        return "missing_reference"
    if equivalent(pred_answer, ref_answer):
        return "correct"
    if looks_like_number(pred_answer) and looks_like_number(ref_answer):
        return "answer_error"
    lowered = prediction.lower()
    if any(token in lowered for token in ("cannot", "can't", "unable", "not enough")):
        return "refusal_or_invalid"
    return "reasoning_error"


def transform_records(
    input_path: Path,
    output_path: Path,
    prediction_field: str,
    answer_field: str,
) -> dict[str, int]:
    # 处理整个 JSONL 文件，为每条样本添加 error_type 字段，并统计各类错误数量。
    counts: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record: dict[str, Any] = json.loads(line)
            # 根据模型预测和标准答案生成错误类型标签。
            error_type = classify_error(str(record.get(prediction_field, "")), str(record.get(answer_field, "")))
            record["error_type"] = error_type
            counts[error_type] = counts.get(error_type, 0) + 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return counts


def main() -> None:
    # 命令行入口：读取预测文件，输出带 error_type 的新 JSONL 文件。
    parser = argparse.ArgumentParser(description="Add coarse error_type labels to math prediction JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--answer-field", default="answer")
    args = parser.parse_args()

    counts = transform_records(
        Path(args.input),
        Path(args.output),
        args.prediction_field,
        args.answer_field,
    )
    print(json.dumps({"error_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
