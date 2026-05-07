"""Prepare the MATH dataset directory into the project JSONL schema.

本文件负责把 MATH 官方目录结构中的 JSON 文件转换成项目统一格式，
保留题型、Level、标准解答，并补充 final_answer 与难度字段。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.difficulty_scorer import difficulty_group, difficulty_score
from src.eval.extract_answer import extract_answer


def iter_problem_files(input_dir: Path) -> list[Path]:
    # 递归收集 MATH 数据集中的 JSON 题目文件，并保持稳定排序。
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def read_problem(path: Path) -> dict[str, Any]:
    # 读取单个 MATH JSON 题目文件。
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def infer_subject(path: Path, input_dir: Path) -> str:
    # 从文件相对路径推断题目类型，例如 algebra 或 number_theory。
    relative = path.relative_to(input_dir)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return "unknown"


def convert_record(record: dict[str, Any], path: Path, input_dir: Path, split: str, index: int) -> dict[str, Any]:
    # 将一条 MATH 样本转换成项目统一字段。
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


def prepare_dir(input_dir: Path, output_path: Path, split: str) -> dict[str, int]:
    # 转换 MATH 的一个 split 目录，并统计难度分布。
    problem_files = iter_problem_files(input_dir)
    converted = [
        convert_record(read_problem(path), path, input_dir, split, index)
        for index, path in enumerate(problem_files)
    ]
    write_jsonl(converted, output_path)

    counts = {"total": len(converted), "easy": 0, "medium": 0, "hard": 0}
    for record in converted:
        counts[record["difficulty_group"]] += 1
    return counts


def main() -> None:
    # 命令行入口：把 MATH split 目录转成项目统一 JSONL。
    parser = argparse.ArgumentParser(description="Prepare MATH JSON files into the project schema.")
    parser.add_argument("--input-dir", required=True, help="Raw MATH split directory, e.g. MATH/train.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--split", required=True, choices=("train", "dev", "test"))
    args = parser.parse_args()

    counts = prepare_dir(Path(args.input_dir), Path(args.output), args.split)
    print(json.dumps({"dataset": "math", "split": args.split, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

