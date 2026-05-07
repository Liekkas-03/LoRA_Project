"""Prepare GSM8K raw JSONL files into the project JSONL schema.

本文件负责把 GSM8K 官方 JSONL 数据转换成项目统一格式，
并补充 final_answer、difficulty_score 和 difficulty_group 字段。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.difficulty_scorer import difficulty_group, difficulty_score
from src.eval.extract_answer import extract_answer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 读取 GSM8K 原始 JSONL，每行是一道题。
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


def convert_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    # 将一条 GSM8K 样本转换成项目统一字段。
    question = str(record.get("question", "")).strip()
    solution = str(record.get("answer", "")).strip()
    final_answer = extract_answer(solution)

    converted: dict[str, Any] = {
        "id": f"gsm8k-{split}-{index:06d}",
        "dataset": "gsm8k",
        "split": split,
        "question": question,
        "solution": solution,
        "answer": final_answer or solution,
        "level": "",
        "type": "grade_school_math",
    }
    score = difficulty_score(converted)
    converted["difficulty_score"] = score
    converted["difficulty_group"] = difficulty_group(score)
    return converted


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    # 将转换后的统一格式样本写成 JSONL。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_file(input_path: Path, output_path: Path, split: str) -> dict[str, int]:
    # 转换单个 GSM8K split，并返回样本数量和难度分布。
    raw_records = read_jsonl(input_path)
    converted = [convert_record(record, split, index) for index, record in enumerate(raw_records)]
    write_jsonl(converted, output_path)

    counts = {"total": len(converted), "easy": 0, "medium": 0, "hard": 0}
    for record in converted:
        counts[record["difficulty_group"]] += 1
    return counts


def main() -> None:
    # 命令行入口：把指定 GSM8K JSONL 转成项目统一 JSONL。
    parser = argparse.ArgumentParser(description="Prepare GSM8K JSONL into the project schema.")
    parser.add_argument("--input", required=True, help="Raw GSM8K JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    args = parser.parse_args()

    counts = prepare_file(Path(args.input), Path(args.output), args.split)
    print(json.dumps({"dataset": "gsm8k", "split": args.split, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

