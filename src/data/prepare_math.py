"""将 MATH 官方目录转换成项目统一 JSONL 格式。

本文件读取 MATH/train 或 MATH/test 下的题目 JSON，抽取题目、解答、
最终答案、题型和官方 Level，并直接写入 difficulty_group。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.difficulty_scorer import assign_math_difficulty
from src.eval.extract_answer import extract_answer


def iter_problem_files(input_dir: Path) -> list[Path]:
    # 递归收集 MATH 数据集中的所有 JSON 题目文件。
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def read_problem(path: Path) -> dict[str, Any]:
    # 读取单个 MATH 原始题目文件。
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def infer_subject(path: Path, input_dir: Path) -> str:
    # 从路径推断题型，例如 algebra、geometry、number_theory。
    relative = path.relative_to(input_dir)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return "unknown"


def convert_record(record: dict[str, Any], path: Path, input_dir: Path, split: str, index: int) -> dict[str, Any]:
    # 将 MATH 原始字段转换成训练和评测都能使用的统一字段。
    solution = str(record.get("solution", "")).strip()
    final_answer = extract_answer(solution)
    subject = str(record.get("type") or infer_subject(path, input_dir))

    converted: dict[str, Any] = {
        "id": f"math-{split}-{index:06d}",
        "dataset": "math",
        "split": split,
        "source_path": str(path.relative_to(input_dir)).replace("\\", "/"),
        "question": str(record.get("problem", "")).strip(),
        "solution": solution,
        "answer": final_answer or solution,
        "level": str(record.get("level", "")),
        "type": subject,
    }
    return assign_math_difficulty(converted)


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    # 将统一格式样本写成 JSONL，便于 datasets.load_dataset("json") 读取。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_groups(records: list[dict[str, Any]]) -> dict[str, int]:
    # 统计转换后各难度组和官方 Level 的样本数量。
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
        counts[str(record["difficulty_group"])] += 1
        counts[f"level_{int(record['math_level'])}"] += 1
    return counts


def prepare_dir(input_dir: Path, output_path: Path, split: str) -> dict[str, int]:
    # 转换一个 MATH split 目录，并返回难度分布。
    problem_files = iter_problem_files(input_dir)
    converted = [
        convert_record(read_problem(path), path, input_dir, split, index)
        for index, path in enumerate(problem_files)
    ]
    write_jsonl(converted, output_path)
    return count_groups(converted)


def main() -> None:
    # 命令行入口：把 MATH 官方目录转换为项目 JSONL。
    parser = argparse.ArgumentParser(description="Prepare MATH JSON files into project JSONL.")
    parser.add_argument("--input-dir", required=True, help="Raw MATH split directory, e.g. MATH/train.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    args = parser.parse_args()

    counts = prepare_dir(Path(args.input_dir), Path(args.output), args.split)
    print(json.dumps({"dataset": "math", "split": args.split, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
