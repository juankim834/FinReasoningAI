"""Reusable utilities for extracting, normalizing, and comparing numeric outputs."""

from __future__ import annotations

import re
from typing import Optional


_SCALE_MAP = {
    "K": 1_000.0,
    "M": 1_000_000.0,
    "B": 1_000_000_000.0,
    "THOUSAND": 1_000.0,
    "MILLION": 1_000_000.0,
    "BILLION": 1_000_000_000.0,
}


def extract_number(text: str) -> Optional[float]:
    """Extract the primary numeric value from text.

    Supported patterns include currency, percentages, shorthand scale suffixes
    (K/M/B), word scale suffixes (thousand/million/billion), accounting-style
    negatives, and embedded numbers in longer strings.

    Examples:
        "$1,234.56" -> 1234.56
        "12.5%" -> 0.125
        "1.2B" -> 1200000000.0
        "(1,234.56)" -> -1234.56
        "approximately 15%" -> 0.15

    Args:
        text: Input string to scan.

    Returns:
        The extracted numeric value as float, or None if no number is found.
    """
    if text is None:
        return None

    source = text.strip()
    if not source:
        return None

    pattern = re.compile(
        r"""
        (?P<neg_paren>\()?
        \s*
        (?P<currency>[$€£])?
        \s*
        (?P<number>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)
        \s*
        (?P<scale>[KkMmBb]|thousand|million|billion)?
        \s*
        (?P<percent>%?)
        \s*
        (?P<close_paren>\))?
        """,
        re.VERBOSE,
    )

    match = pattern.search(source)
    if not match:
        return None

    raw_num = match.group("number")
    if raw_num is None:
        return None

    try:
        value = float(raw_num.replace(",", ""))
    except ValueError:
        return None

    neg_paren = match.group("neg_paren")
    close_paren = match.group("close_paren")
    if neg_paren and close_paren:
        value = -abs(value)

    scale_token = match.group("scale")
    if scale_token:
        scale = _SCALE_MAP.get(scale_token.upper())
        if scale is not None:
            value *= scale

    percent_token = match.group("percent")
    if percent_token == "%":
        value /= 100.0

    return value


def normalize_number(value: float, decimals: int = 4) -> float:
    """Round a numeric value to a fixed number of decimal places.

    Args:
        value: Number to normalize.
        decimals: Number of decimal places to keep.

    Returns:
        The rounded float value.
    """
    return float(round(float(value), int(decimals)))


def numbers_match(pred: str, gold: str, tolerance: float = 0.01) -> bool:
    """Compare extracted numeric values with relative-error tolerance.

    Args:
        pred: Predicted output string.
        gold: Ground-truth output string.
        tolerance: Maximum allowed relative error.

    Returns:
        True when both are non-numeric, or both numeric within tolerance.
        False otherwise.
    """
    pred_num = extract_number(pred)
    gold_num = extract_number(gold)

    if pred_num is None and gold_num is None:
        return True
    if pred_num is None or gold_num is None:
        return False

    relative_error = abs(pred_num - gold_num) / max(abs(gold_num), 1e-9)
    return relative_error <= tolerance


def score_prediction(pred: str, gold: str) -> dict:
    """Score prediction quality using exact and numerical matching.

    Args:
        pred: Predicted text output.
        gold: Ground-truth label text.

    Returns:
        A dictionary containing exact string match, numerical match,
        extracted numbers, and relative error when available.
    """
    pred_clean = pred.strip().lower()
    gold_clean = gold.strip().lower()

    pred_num = extract_number(pred)
    gold_num = extract_number(gold)

    if pred_num is not None and gold_num is not None:
        relative_error = abs(pred_num - gold_num) / max(abs(gold_num), 1e-9)
    else:
        relative_error = None

    return {
        "exact_match": pred_clean == gold_clean,
        "numerical_match": numbers_match(pred, gold),
        "pred_number": pred_num,
        "gold_number": gold_num,
        "relative_error": relative_error,
    }


if __name__ == "__main__":
    # Core extraction cases
    assert extract_number("$1,234.56") == 1234.56
    assert extract_number("12.5%") == 0.125
    assert extract_number("1.2B") == 1_200_000_000.0
    assert extract_number("1.2M") == 1_200_000.0
    assert extract_number("1.2K") == 1200.0
    assert extract_number("(1,234.56)") == -1234.56
    assert extract_number("1.23 million") == 1_230_000.0
    assert extract_number("approximately 15%") == 0.15
    assert extract_number("the answer is 42") == 42.0
    assert extract_number("no number here") is None

    # Normalization
    assert normalize_number(3.1415926) == 3.1416
    assert normalize_number(3.1415926, 2) == 3.14

    # Matching behavior
    assert numbers_match("12.5%", "0.125") is True
    assert numbers_match("1.21B", "1.2B", tolerance=0.01) is True
    assert numbers_match("1.3B", "1.2B", tolerance=0.01) is False
    assert numbers_match("no number", "still no number") is True
    assert numbers_match("42", "no number") is False

    # Scoring
    s1 = score_prediction(" 42 ", "42")
    assert s1["exact_match"] is True
    assert s1["numerical_match"] is True
    assert s1["pred_number"] == 42.0
    assert s1["gold_number"] == 42.0
    assert s1["relative_error"] == 0.0

    s2 = score_prediction("about 10%", "0.11")
    assert s2["exact_match"] is False
    assert s2["numerical_match"] is False
    assert s2["pred_number"] == 0.1
    assert s2["gold_number"] == 0.11
    assert s2["relative_error"] is not None

    print("All number_parser tests passed.")
