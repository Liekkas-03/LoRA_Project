"""数学答案等价判断工具。

评测时先做字符串标准化，再尝试数值等价；如果环境中安装了 SymPy，
还会额外尝试符号等价判断。SymPy 是可选依赖，本地不强制安装。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Any, Literal


VerifierBackend = Literal["simple", "math_verify"]


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


def math_verify_available() -> bool:
    # 轻量探测 Math-Verify 是否已安装，避免本地强依赖。
    try:
        import math_verify  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def resolve_backend(requested: Literal["auto", "simple", "math_verify"]) -> VerifierBackend:
    # 根据环境自动选择评测后端；云端装了 Math-Verify 就优先使用它。
    if requested == "simple":
        return "simple"
    if requested == "math_verify":
        if not math_verify_available():
            raise ImportError(
                "Math-Verify is not installed. Install it on AutoDL with "
                "`pip install math-verify[antlr4_13_2]`."
            )
        return "math_verify"
    return "math_verify" if math_verify_available() else "simple"


def _looks_like_bare_latex(text: str) -> bool:
    # 对像 \frac{1}{2} 这种裸 LaTeX 做一次保守识别，便于 Math-Verify 补救解析。
    if "\\" not in text:
        return False
    latex_markers = ("\\frac", "\\sqrt", "\\pi", "\\theta", "\\alpha", "\\beta", "\\gamma", "\\boxed")
    if not any(marker in text for marker in latex_markers):
        return False
    wrapped_markers = ("$", "\\(", "\\[", "\\boxed")
    return not any(marker in text for marker in wrapped_markers)


def _load_math_verify_symbols() -> tuple[Any, Any, Any, Any]:
    # 延迟导入 Math-Verify，避免本地轻量环境被重依赖拖慢。
    from math_verify import parse, verify  # type: ignore
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # type: ignore

    return parse, verify, LatexExtractionConfig, ExprExtractionConfig


def _parse_with_math_verify(value: str | None, is_reference: bool) -> tuple[Any | None, bool]:
    # 用 Math-Verify 解析答案；参考答案和预测答案使用略有区别的提取配置。
    if value is None:
        return None, False

    text = str(value).strip()
    if not text:
        return None, False

    parse, _verify, latex_config_cls, expr_config_cls = _load_math_verify_symbols()

    if is_reference:
        extraction_config = [latex_config_cls(), expr_config_cls()]
    else:
        extraction_config = [latex_config_cls(boxed_match_priority=0), expr_config_cls()]

    candidates = [text]
    if _looks_like_bare_latex(text):
        candidates.append(f"${text}$")

    for candidate in candidates:
        try:
            parsed = parse(candidate, extraction_config=extraction_config)
        except Exception:
            continue
        if parsed:
            return parsed, True
    return None, False


def compare_answers(
    prediction: str | None,
    reference: str | None,
    backend: VerifierBackend = "simple",
) -> tuple[bool, bool, bool]:
    # 统一比较入口：返回是否正确、预测是否成功抽取、参考答案是否成功抽取。
    if backend == "simple":
        from src.eval.extract_answer import extract_answer

        pred_answer = extract_answer(prediction)
        ref_answer = extract_answer(reference)
        return equivalent(pred_answer, ref_answer), pred_answer is not None, ref_answer is not None

    reference_parsed, has_reference = _parse_with_math_verify(reference, is_reference=True)
    prediction_parsed, has_prediction = _parse_with_math_verify(prediction, is_reference=False)

    if not has_prediction or not has_reference:
        return False, has_prediction, has_reference

    try:
        _parse, verify, _latex_config_cls, _expr_config_cls = _load_math_verify_symbols()
        return bool(verify(reference_parsed, prediction_parsed)), True, True
    except Exception:
        from src.eval.extract_answer import extract_answer

        pred_answer = extract_answer(prediction)
        ref_answer = extract_answer(reference)
        return equivalent(pred_answer, ref_answer), pred_answer is not None, ref_answer is not None
