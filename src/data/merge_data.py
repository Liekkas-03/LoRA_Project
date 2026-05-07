"""Merge multiple unified JSONL files into one training or evaluation file.

本文件负责把 GSM8K、MATH 等已经转换成统一格式的 JSONL 合并，
方便后续 QLoRA/AdaQLoRA 训练脚本读取一个总的 train/dev 文件。
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 读取一个统一格式 JSONL 文件。
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


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    # 将合并后的样本写入输出 JSONL。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    # 汇总数据集来源、split 和难度分布，便于检查合并结果。
    return {
        "total": len(records),
        "by_dataset": dict(Counter(str(record.get("dataset", "unknown")) for record in records)),
        "by_split": dict(Counter(str(record.get("split", "unknown")) for record in records)),
        "by_difficulty": dict(Counter(str(record.get("difficulty_group", "unknown")) for record in records)),
    }


def merge_files(input_paths: list[Path], output_path: Path) -> dict[str, Any]:
    # 按输入顺序合并多个 JSONL 文件，并返回合并统计。
    merged: list[dict[str, Any]] = []
    for path in input_paths:
        merged.extend(read_jsonl(path))
    write_jsonl(merged, output_path)
    summary = summarize(merged)
    summary["output"] = str(output_path)
    summary["inputs"] = [str(path) for path in input_paths]
    return summary


def main() -> None:
    # 命令行入口：把多个统一格式 JSONL 合并成一个文件。
    parser = argparse.ArgumentParser(description="Merge project JSONL files.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files.")
    parser.add_argument("--output", required=True, help="Merged output JSONL path.")
    args = parser.parse_args()

    summary = merge_files([Path(path) for path in args.inputs], Path(args.output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

