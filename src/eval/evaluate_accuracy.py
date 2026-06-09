"""从 prediction JSONL 计算数学最终答案准确率。

本文件读取云端生成的预测结果，抽取最终答案，与数据集自带标准
answer 比较，并输出整体与分难度准确率。默认会在安装了
Math-Verify 时优先使用它，否则回退到本地轻量判分逻辑。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from src.eval.math_verifier import compare_answers, resolve_backend


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
    backend: str,
) -> dict[str, Any]:
    # 逐条抽答案、判等价，并统计整体与分组指标。
    resolved_backend = resolve_backend(backend)  # type: ignore[arg-type]
    total = 0
    correct = 0
    missing_prediction = 0
    missing_reference = 0
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for record in records:
        prediction = str(record.get(prediction_field, ""))
        reference = str(record.get(answer_field, ""))
        group = str(record.get(group_field, "unknown"))

        is_correct, has_prediction, has_reference = compare_answers(
            prediction=prediction,
            reference=reference,
            backend=resolved_backend,
        )

        if not has_prediction:
            missing_prediction += 1
        if not has_reference:
            missing_reference += 1

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
        "backend": resolved_backend,
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
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "simple", "math_verify"),
        help="Scoring backend: auto prefers Math-Verify when installed, otherwise falls back to simple.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    records = read_jsonl(Path(args.input))
    metrics = evaluate_records(
        records,
        args.prediction_field,
        args.answer_field,
        args.group_field,
        args.backend,
    )
    report = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
