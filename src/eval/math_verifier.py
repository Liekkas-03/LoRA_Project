"""Lightweight math answer verification.

The verifier first tries normalized exact match, then numeric equivalence.
If SymPy is available it also tries symbolic equivalence, but SymPy is optional.

本文件负责判断模型预测答案和标准答案是否等价。它会先做格式归一化，
再尝试字符串匹配、数值匹配，最后在可用时尝试 SymPy 符号等价判断。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re


def normalize_answer(value: str | None) -> str:
    # 统一答案格式，去掉逗号、美元符号和部分 LaTeX 标记。
    if value is None:
        return ""
    normalized = value.strip()
    normalized = normalized.replace(",", "")
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\%", "%")
    normalized = normalized.replace("\\cdot", "*")
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\left", "")
    normalized = normalized.replace("\\right", "")
    normalized = normalized.strip(". ")
    return normalized


def _parse_number(value: str) -> Decimal | None:
    # 尝试把答案解析成数字，支持整数、小数和简单分数。
    cleaned = normalize_answer(value)
    cleaned = cleaned.rstrip("%")
    try:
        if "/" in cleaned and not any(op in cleaned for op in ("+", "*", "^")):
            return Decimal(Fraction(cleaned).numerator) / Decimal(Fraction(cleaned).denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def numeric_equal(prediction: str, reference: str, tolerance: Decimal = Decimal("1e-6")) -> bool:
    # 判断两个答案在数值上是否相等，允许极小误差。
    pred_number = _parse_number(prediction)
    ref_number = _parse_number(reference)
    if pred_number is None or ref_number is None:
        return False
    return abs(pred_number - ref_number) <= tolerance


def symbolic_equal(prediction: str, reference: str) -> bool:
    # 如果本地或云端安装了 SymPy，就尝试判断两个符号表达式是否等价。
    try:
        import sympy as sp  # type: ignore
    except Exception:
        return False

    pred = normalize_answer(prediction).replace("^", "**")
    ref = normalize_answer(reference).replace("^", "**")
    try:
        return bool(sp.simplify(sp.sympify(pred) - sp.sympify(ref)) == 0)
    except Exception:
        return False


def equivalent(prediction: str | None, reference: str | None) -> bool:
    # 统一的判等入口：依次尝试字符串、数值和符号等价。
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    if pred.lower() == ref.lower():
        return True
    if numeric_equal(pred, ref):
        return True
    if symbolic_equal(pred, ref):
        return True
    return False


def looks_like_number(value: str | None) -> bool:
    # 粗略判断文本里是否包含数字，供错误分类脚本使用。
    if not value:
        return False
    return bool(re.search(r"-?\d", value))
