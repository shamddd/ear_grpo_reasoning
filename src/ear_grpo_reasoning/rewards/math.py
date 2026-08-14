"""Conservative numeric answer extraction for math verification."""

from __future__ import annotations

import math
import re
from fractions import Fraction

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+(?:\.\d+)?)?"


def _clean(candidate: str) -> str | None:
    match = re.search(_NUMBER, candidate.replace("$", ""))
    if not match:
        return None
    return match.group(0).replace(",", "")


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer using explicit markers before fallback."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if "####" in text:
        return _clean(text.rsplit("####", maxsplit=1)[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return _clean(boxed[-1])
    explicit = re.findall(
        rf"(?:final\s+answer|answer(?:\s+is)?|result\s+is)\s*[:=]?\s*({_NUMBER})",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return _clean(explicit[-1])
    candidates = re.findall(_NUMBER, text)
    return _clean(candidates[-1]) if candidates else None


def _as_fraction(value: str) -> Fraction:
    if "/" in value:
        numerator, denominator = value.split("/", maxsplit=1)
        if float(denominator) == 0:
            raise ValueError("division by zero")
        return Fraction(numerator) / Fraction(denominator)
    return Fraction(value)


def compute_math_reward(completion: str, ground_truth: str, *, tolerance: float = 1e-9) -> float:
    """Return a binary reward for finite numeric answers.

    This verifier does not claim symbolic equivalence. Unsupported or malformed
    expressions receive zero reward rather than being guessed from.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    predicted = extract_answer(completion)
    target = extract_answer(ground_truth)
    if predicted is None or target is None:
        return 0.0
    try:
        predicted_value = _as_fraction(predicted)
        target_value = _as_fraction(target)
        predicted_float = float(predicted_value)
        target_float = float(target_value)
    except (ValueError, ZeroDivisionError):
        return 0.0
    if not math.isfinite(predicted_float) or not math.isfinite(target_float):
        return 0.0
    return float(math.isclose(predicted_float, target_float, rel_tol=0.0, abs_tol=tolerance))
