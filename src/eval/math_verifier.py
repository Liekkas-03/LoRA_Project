"""Lightweight math answer verification.

The verifier first tries normalized exact match, then numeric equivalence.
If SymPy is available it also tries symbolic equivalence, but SymPy is optional.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re


def normalize_answer(value: str | None) -> str:
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
    cleaned = normalize_answer(value)
    cleaned = cleaned.rstrip("%")
    try:
        if "/" in cleaned and not any(op in cleaned for op in ("+", "*", "^")):
            return Decimal(Fraction(cleaned).numerator) / Decimal(Fraction(cleaned).denominator)
        return Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def numeric_equal(prediction: str, reference: str, tolerance: Decimal = Decimal("1e-6")) -> bool:
    pred_number = _parse_number(prediction)
    ref_number = _parse_number(reference)
    if pred_number is None or ref_number is None:
        return False
    return abs(pred_number - ref_number) <= tolerance


def symbolic_equal(prediction: str, reference: str) -> bool:
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
    if not value:
        return False
    return bool(re.search(r"-?\d", value))

