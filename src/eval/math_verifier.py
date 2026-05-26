"""数学答案等价判断工具。

评测时先做字符串标准化，再尝试数值等价；如果环境中安装了 SymPy，
还会额外尝试符号等价判断。SymPy 是可选依赖，本地不强制安装。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re


def normalize_answer(value: str | None) -> str:
    # 统一答案格式，去掉逗号、美元符号和部分 LaTeX 命令。
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
    cleaned = normalize_answer(value).rstrip("%")
    try:
        if "/" in cleaned and not any(op in cleaned for op in ("+", "*", "^")):
            fraction = Fraction(cleaned)
            return Decimal(fraction.numerator) / Decimal(fraction.denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def numeric_equal(prediction: str, reference: str, tolerance: Decimal = Decimal("1e-6")) -> bool:
    # 判断两个答案在数值上是否相等。
    pred_number = _parse_number(prediction)
    ref_number = _parse_number(reference)
    if pred_number is None or ref_number is None:
        return False
    return abs(pred_number - ref_number) <= tolerance


def symbolic_equal(prediction: str, reference: str) -> bool:
    # 如果 SymPy 可用，尝试判断两个符号表达式是否等价。
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
    # 统一判等入口：字符串、数值、符号等价依次尝试。
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
    # 粗略判断文本中是否包含数字，供错误分类使用。
    if not value:
        return False
    return bool(re.search(r"-?\d", value))
