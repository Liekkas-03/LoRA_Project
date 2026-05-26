"""数学答案抽取工具。

本文件只使用 Python 标准库，方便在本地轻量环境运行。它负责从模型
输出或标准解答中抽取最终答案，例如 `\boxed{32}` 或 `Final Answer: 32`。
"""

from __future__ import annotations

import argparse
import re


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_RE = re.compile(r"####\s*([^\n\r]+)")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer|therefore|so)\s*(?:is|=|:)?\s*([^\n\r.]+)",
    re.IGNORECASE,
)


def strip_latex_wrappers(text: str) -> str:
    # 去掉常见 LaTeX 外壳，降低格式差异对判分的影响。
    value = text.strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "")
    return value.strip()


def extract_boxed(text: str) -> str | None:
    # 优先抽取 MATH 标准解答中最常见的 \boxed{...}。
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_hash_answer(text: str) -> str | None:
    # 兼容模型可能输出的 #### answer 格式。
    matches = HASH_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_after_final_answer(text: str) -> str | None:
    # 抽取自然语言提示后的答案，例如 "Final Answer: 7"。
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_last_number(text: str) -> str | None:
    # 兜底策略：没有明确答案标记时，取最后一个数字。
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def extract_answer(text: str | None) -> str | None:
    # 按可信度从高到低尝试不同抽取规则。
    if not text:
        return None

    for extractor in (
        extract_boxed,
        extract_hash_answer,
        extract_after_final_answer,
        extract_last_number,
    ):
        answer = extractor(text)
        if answer:
            return answer.strip()
    return None


def main() -> None:
    # 命令行入口：快速查看一段文本会被抽成什么答案。
    parser = argparse.ArgumentParser(description="Extract a math final answer.")
    parser.add_argument("text", help="Prediction or reference text.")
    args = parser.parse_args()
    answer = extract_answer(args.text)
    print("" if answer is None else answer)


if __name__ == "__main__":
    main()
