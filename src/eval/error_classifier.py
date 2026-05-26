"""给数学预测结果添加粗粒度错误类型。

该脚本用于后续错误分析和 verifier-guided replay。当前只做轻量规则：
正确、格式错误、答案错误、推理错误和拒答/无效输出。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.eval.extract_answer import extract_answer
from src.eval.math_verifier import equivalent, looks_like_number


def classify_error(prediction: str, reference: str) -> str:
    # 对单条预测分类，先抽最终答案，再判断正误和错误形态。
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
    # 处理整个 JSONL，为每条样本写入 error_type。
    counts: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
            error_type = classify_error(str(record.get(prediction_field, "")), str(record.get(answer_field, "")))
            record["error_type"] = error_type
            counts[error_type] = counts.get(error_type, 0) + 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return counts


def main() -> None:
    # 命令行入口：输出带 error_type 的新 JSONL。
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
