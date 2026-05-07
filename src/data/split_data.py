"""Split unified JSONL data into train/dev files with a fixed random seed.

本文件负责把项目统一 JSONL 数据按固定随机种子划分为 train/dev，
保证后续 replay、调参和最终测试之间有清晰的数据边界。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 读取统一格式 JSONL 文件，每行一条样本。
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
    # 将划分后的样本写成 JSONL，便于后续训练或评测脚本读取。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_records(
    records: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
    stratify_field: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # 按比例划分 train/dev；如果指定字段，则在每个分组内分别抽 dev。
    if not 0 < dev_ratio < 1:
        raise ValueError(f"dev_ratio must be between 0 and 1, got {dev_ratio}")

    rng = random.Random(seed)
    if not stratify_field:
        shuffled = list(records)
        rng.shuffle(shuffled)
        dev_size = max(1, round(len(shuffled) * dev_ratio)) if shuffled else 0
        return shuffled[dev_size:], shuffled[:dev_size]

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get(stratify_field, "unknown"))
        groups.setdefault(key, []).append(record)

    train_records: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    for key in sorted(groups):
        group_records = list(groups[key])
        rng.shuffle(group_records)
        dev_size = round(len(group_records) * dev_ratio)
        if len(group_records) > 1:
            dev_size = max(1, dev_size)
        dev_records.extend(group_records[:dev_size])
        train_records.extend(group_records[dev_size:])

    rng.shuffle(train_records)
    rng.shuffle(dev_records)
    return train_records, dev_records


def split_file(
    input_path: Path,
    train_output: Path,
    dev_output: Path,
    dev_ratio: float,
    seed: int,
    stratify_field: str | None,
) -> dict[str, Any]:
    # 读取输入文件，完成划分并输出统计信息。
    records = read_jsonl(input_path)
    train_records, dev_records = split_records(records, dev_ratio, seed, stratify_field)
    write_jsonl(train_records, train_output)
    write_jsonl(dev_records, dev_output)
    return {
        "total": len(records),
        "train": len(train_records),
        "dev": len(dev_records),
        "dev_ratio": dev_ratio,
        "seed": seed,
        "stratify_field": stratify_field or "",
    }


def main() -> None:
    # 命令行入口：从统一 JSONL 划分出 train/dev 两个文件。
    parser = argparse.ArgumentParser(description="Split project JSONL data into train/dev files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--dev-output", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-field", default="difficulty_group")
    args = parser.parse_args()

    summary = split_file(
        input_path=Path(args.input),
        train_output=Path(args.train_output),
        dev_output=Path(args.dev_output),
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        stratify_field=args.stratify_field or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

