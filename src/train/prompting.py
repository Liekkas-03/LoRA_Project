"""Prompt formatting utilities for math SFT data.

本文件负责把统一 JSONL 样本转换成监督微调使用的 text 字段，
让训练脚本不用关心 GSM8K/MATH 原始字段差异。
"""

from __future__ import annotations

from typing import Any


def format_math_cot(record: dict[str, Any]) -> str:
    # 构造数学推理 SFT 文本：题目、分步解答和最终答案。
    question = str(record.get("question", "")).strip()
    solution = str(record.get("solution", "")).strip()
    answer = str(record.get("answer", "")).strip()

    parts = [
        "Question:",
        question,
        "",
        "Solution:",
        solution,
    ]
    if answer and answer not in solution:
        parts.extend(["", f"Final Answer: {answer}"])
    return "\n".join(parts).strip()


def format_example(record: dict[str, Any], template: str = "math_cot") -> str:
    # 根据配置选择 prompt 模板，目前默认使用数学 CoT 监督微调格式。
    if template == "math_cot":
        return format_math_cot(record)
    raise ValueError(f"Unknown prompt template: {template}")

