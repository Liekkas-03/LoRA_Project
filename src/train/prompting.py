"""数学推理 SFT 的 prompt 格式化工具。

本文件将 MATH 样本转换成训练文本。训练时包含标准解题过程；
推理评测时则由 generate_answers.py 只给题目，不泄露答案。
"""

from __future__ import annotations

from typing import Any


def format_math_cot(record: dict[str, Any]) -> str:
    # 构造监督微调文本：题目、标准解题过程、最终答案。
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
    # 根据配置选择 prompt 模板，目前主线只保留 MATH CoT 模板。
    if template == "math_cot":
        return format_math_cot(record)
    raise ValueError(f"Unknown prompt template: {template}")
