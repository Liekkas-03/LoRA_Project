"""从 prediction JSONL 计算数学最终答案准确率。

本文件读取云端生成的预测结果，抽取最终答案，与数据集自带标准
answer 比较，并输出整体与分难度准确率。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.eval.extract_answer import extract_answer
from src.eval.math_verifier import equivalent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 读取预测 JSONL 文件。
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return records


def evaluate_records(
    records: list[dict[str, Any]],
    prediction_field: str,
    answer_field: str,
    group_field: str,
) -> dict[str, Any]:
    # 逐条抽答案、判等价，并统计整体与分组指标。
    total = 0
    correct = 0
    missing_prediction = 0
    missing_reference = 0
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for record in records:
        pred_answer = extract_answer(str(record.get(prediction_field, "")))
        ref_answer = extract_answer(str(record.get(answer_field, "")))
        group = str(record.get(group_field, "unknown"))

        if pred_answer is None:
            missing_prediction += 1
        if ref_answer is None:
            missing_reference += 1

        is_correct = equivalent(pred_answer, ref_answer)
        total += 1
        correct += int(is_correct)
        grouped[group]["total"] += 1
        grouped[group]["correct"] += int(is_correct)

    group_metrics = {
        group: {
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
        }
        for group, stats in sorted(grouped.items())
    }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "missing_prediction": missing_prediction,
        "missing_reference": missing_reference,
        "by_group": group_metrics,
    }


def main() -> None:
    # 命令行入口：打印评测报告，并可选写入 JSON 文件。
    parser = argparse.ArgumentParser(description="Evaluate math final-answer accuracy.")
    parser.add_argument("--input", required=True, help="JSONL file with predictions and references.")
    parser.add_argument("--prediction-field", default="prediction")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--group-field", default="difficulty_group")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    metrics = evaluate_records(records, args.prediction_field, args.answer_field, args.group_field)
    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
