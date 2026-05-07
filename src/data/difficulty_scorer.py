"""Heuristic difficulty scoring for math reasoning datasets."""

# 本文件负责给数学推理样本打启发式难度分，并划分为 easy/medium/hard，
# 后续用于分难度评测、课程学习和自适应 LoRA rank 分配。

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


OPERATOR_RE = re.compile(r"[\+\-\*/=^]|\\frac|\\sqrt|\\binom")


def _text(record: dict[str, Any], *fields: str) -> str:
    # 从样本中取多个可能的文本字段，并拼成统一字符串。
    return " ".join(str(record.get(field, "")) for field in fields if record.get(field))


def parse_math_level(record: dict[str, Any]) -> int:
    # 解析 MATH 数据集里的 Level 字段，得到 0-5 的难度等级。
    raw = str(record.get("level", ""))
    match = re.search(r"(\d+)", raw)
    if not match:
        return 0
    return max(0, min(5, int(match.group(1))))


def estimate_solution_steps(solution: str) -> int:
    # 根据标准解答的行数或句子数，粗略估计解题步骤数量。
    if not solution:
        return 0
    line_steps = len([line for line in solution.splitlines() if line.strip()])
    sentence_steps = len(re.findall(r"[.;]\s+", solution))
    return max(line_steps, sentence_steps)


def expression_complexity(text: str) -> int:
    # 统计数字、运算符和长 token 数量，作为表达式复杂度信号。
    numbers = len(re.findall(r"-?\d+(?:\.\d+)?", text))
    operators = len(OPERATOR_RE.findall(text))
    long_tokens = len([token for token in re.split(r"\s+", text) if len(token) > 12])
    return numbers + operators + long_tokens


def dataset_base_score(dataset: str) -> float:
    # 不同数据集给不同基础难度，MATH 默认比 GSM8K 更难。
    name = dataset.lower()
    if name == "math":
        return 2.0
    if name == "gsm8k":
        return 0.8
    return 1.0


def difficulty_score(record: dict[str, Any]) -> float:
    # 综合数据集来源、官方等级、步骤数和表达式复杂度，得到连续难度分。
    dataset = str(record.get("dataset", ""))
    question = _text(record, "question", "problem")
    solution = _text(record, "solution", "answer")
    level = parse_math_level(record)
    steps = estimate_solution_steps(solution)
    complexity = expression_complexity(question + " " + solution)

    score = dataset_base_score(dataset)
    score += 0.55 * level
    score += min(steps, 12) * 0.12
    score += min(complexity, 40) * 0.035
    return round(score, 4)


def difficulty_group(score: float) -> str:
    # 将连续难度分映射成 easy/medium/hard 三档。
    if score < 2.0:
        return "easy"
    if score < 3.6:
        return "medium"
    return "hard"


def score_file(input_path: Path, output_path: Path) -> dict[str, int]:
    # 处理整个 JSONL 文件，为每条样本写入 difficulty_score 和 difficulty_group。
    counts = {"easy": 0, "medium": 0, "hard": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
            score = difficulty_score(record)
            group = difficulty_group(score)
            record["difficulty_score"] = score
            record["difficulty_group"] = group
            counts[group] += 1
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    return counts


def main() -> None:
    # 命令行入口：读取原始 JSONL，输出带难度字段的新 JSONL。
    parser = argparse.ArgumentParser(description="Add heuristic difficulty scores to JSONL math records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    counts = score_file(Path(args.input), Path(args.output))
    print(json.dumps({"difficulty_counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
