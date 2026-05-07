"""Answer extraction helpers for math reasoning outputs.

These utilities intentionally use only the Python standard library so they can
run in the local lightweight workspace.
"""

from __future__ import annotations

import re


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
GSM8K_RE = re.compile(r"####\s*([^\n\r]+)")
FINAL_ANSWER_RE = re.compile(
    r"(?:final answer|answer|therefore|so)\s*(?:is|=|:)?\s*([^\n\r.]+)",
    re.IGNORECASE,
)


def strip_latex_wrappers(text: str) -> str:
    # 清理答案外层常见的 LaTeX 包装，便于后续做字符串或数值比较。
    value = text.strip()
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("$", "")
    return value.strip()


def extract_boxed(text: str) -> str | None:
    # 抽取数学题里常见的最终答案格式：\boxed{...}。
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_gsm8k_hash_answer(text: str) -> str | None:
    # 抽取 GSM8K 标准答案格式：#### 7。
    matches = GSM8K_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_after_final_answer(text: str) -> str | None:
    # 抽取自然语言里的答案提示，例如 "final answer is 7"。
    matches = FINAL_ANSWER_RE.findall(text)
    if not matches:
        return None
    return strip_latex_wrappers(matches[-1])


def extract_last_number(text: str) -> str | None:
    # 兜底策略：如果没有明确答案标记，就取文本中的最后一个数字。
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text)
    if not numbers:
        return None
    return numbers[-1].replace(",", "")


def extract_answer(text: str | None) -> str | None:
    """Extract a final answer from a model prediction or reference solution."""

    if not text:
        return None

    # 按可信度从高到低尝试抽取答案，最后一个数字只作为兜底。
    for extractor in (
        extract_gsm8k_hash_answer,
        extract_boxed,
        extract_after_final_answer,
        extract_last_number,
    ):
        answer = extractor(text)
        if answer:
            return answer.strip()
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a math final answer.")
    parser.add_argument("text", help="Prediction or reference text.")
    args = parser.parse_args()
    answer = extract_answer(args.text)
    print("" if answer is None else answer)


if __name__ == "__main__":
    main()
