"""为 MATH 数据集样本写入官方难度标签。

本文件只服务于当前 MATH-only 主线：读取样本中的官方 `level`
字段，并转换成论文中使用的 easy/medium/hard 分组。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_math_level(record: dict[str, Any]) -> int:
    # 从 "Level 3"、3 或 math_level 字段中解析 MATH 官方难度等级。
    raw = record.get("math_level", record.get("level", ""))
    match = re.search(r"([1-5])", str(raw))
    if not match:
        raise ValueError(f"Missing valid MATH level in record: {record.get('id', '<no id>')}")
    return int(match.group(1))


def level_to_group(level: int) -> str:
    # 将 MATH Level 1-5 合并成三档，方便论文主表和分难度分析。
    if level in {1, 2}:
        return "easy"
    if level == 3:
        return "medium"
    if level in {4, 5}:
        return "hard"
    raise ValueError(f"MATH level must be 1-5, got {level}")


def assign_math_difficulty(record: dict[str, Any]) -> dict[str, Any]:
    # 给单条样本补充统一难度字段，保留官方 level 作为可追溯依据。
    updated = dict(record)
    level = parse_math_level(updated)
    updated["dataset"] = "math"
    updated["math_level"] = level
    updated["difficulty_score"] = float(level)
    updated["difficulty_group"] = level_to_group(level)
    updated["difficulty_source"] = "math_official_level"
    return updated


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # 读取 JSONL，每一行是一道 MATH 题或一条预测结果。
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
    # 写出带官方难度字段的 JSONL 文件。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_groups(records: list[dict[str, Any]]) -> dict[str, int]:
    # 汇总 easy/medium/hard 以及 Level 1-5 的样本数量。
    counts = {
        "total": len(records),
        "easy": 0,
        "medium": 0,
        "hard": 0,
        "level_1": 0,
        "level_2": 0,
        "level_3": 0,
        "level_4": 0,
        "level_5": 0,
    }
    for record in records:
        group = str(record["difficulty_group"])
        level = int(record["math_level"])
        counts[group] += 1
        counts[f"level_{level}"] += 1
    return counts


def score_file(input_path: Path, output_path: Path) -> dict[str, int]:
    # 处理整个文件，用官方 MATH level 重写难度字段。
    records = [assign_math_difficulty(record) for record in read_jsonl(input_path)]
    write_jsonl(records, output_path)
    return count_groups(records)


def main() -> None:
    # 命令行入口：把已有 MATH JSONL 转成带官方难度字段的版本。
    parser = argparse.ArgumentParser(description="Assign official MATH difficulty groups.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    counts = score_file(Path(args.input), Path(args.output))
    print(json.dumps({"difficulty_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
