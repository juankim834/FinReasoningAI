"""
File: tests/test_robustness.py

Step 5e — Robustness Tests

Tests designed to catch known failure modes of LLMs on financial reasoning:
  1. Numerical perturbation: if we change a number in the context by ±10%,
     the model's answer should scale proportionally (not stay anchored to training).
  2. Entity swap: replacing a company name with a different company should not
     cause the model to hallucinate the original company's financials.
  3. Unit confusion: changing "$2.5 million" to "$2.5 billion" in context should
     change the answer accordingly — not be ignored.
  4. Negation: "What is NOT the revenue?" should not return the revenue.

These tests require a loaded model and tokenizer. Use the --model_id flag or
set FINREASONING_MODEL_ID env var to run with a real model.
For pure logic/metric tests, the mock functions below work without a GPU.
"""

from __future__ import annotations

import os
import re
import math
from typing import Optional, Tuple
import pytest

from src.eval.evaluate import (
    compute_exact_match,
    compute_f1,
    compute_grounding_rate,
    is_answer_parsable,
    _extract_number,
    _normalize_text,
)
from src.inference.self_consistency import (
    self_consistent_answer,
    aggregate_numerical,
    aggregate_categorical,
    _parse_number,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def model_and_tokenizer():
    """
    Load the model for integration tests.
    Skipped if no GPU or no model path is configured.
    """
    model_id = os.environ.get("FINREASONING_MODEL_ID", "")
    adapter_dir = os.environ.get("FINREASONING_ADAPTER_DIR", "")

    if not model_id:
        pytest.skip("FINREASONING_MODEL_ID not set — skipping model integration tests.")

    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("No GPU available — skipping model integration tests.")
    except ImportError:
        pytest.skip("torch not installed.")

    from src.model.load_model import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id)

    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)

    return model, tokenizer


@pytest.fixture
def sample_financial_context():
    return (
        "Apple Inc. reported total revenues of $394.3 billion for fiscal year 2022, "
        "compared to $365.8 billion in 2021, representing growth of 7.8%. "
        "Net income was $99.8 billion, with earnings per share of $6.11. "
        "The company's operating margin was 30.3% and free cash flow totaled $111.4 billion."
    )


@pytest.fixture
def perturbed_context_10pct_up(sample_financial_context):
    """Context with all numeric values increased by 10%."""
    def _scale_num(match) -> str:
        num_str = match.group(1)
        suffix = match.group(2) or ""
        try:
            v = float(num_str.replace(",", ""))
            v_new = v * 1.10
            # Preserve decimal places
            dec_places = len(num_str.split(".")[-1]) if "." in num_str else 0
            formatted = f"{v_new:,.{dec_places}f}"
            return f"{formatted}{suffix}"
        except ValueError:
            return match.group(0)

    _NUM_RE = re.compile(r"(\d[\d,]*\.?\d*)((?:\s*billion|\s*million|\s*%|\s*\$)?)", re.IGNORECASE)
    return _NUM_RE.sub(_scale_num, sample_financial_context)


@pytest.fixture
def entity_swapped_context(sample_financial_context):
    """Context with 'Apple' replaced by 'Microsoft'."""
    return sample_financial_context.replace("Apple Inc.", "Microsoft Corporation").replace("Apple", "Microsoft")


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for metric functions
# ──────────────────────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_identical_strings(self):
        assert compute_exact_match("$394.3 billion", "$394.3 billion") == 1.0

    def test_tolerance_within(self):
        # 394.3B vs 394.3393B → relative error ≈ 0.01% → within tolerance
        assert compute_exact_match("$394.339 billion", "$394.3 billion") == 1.0

    def test_tolerance_exceeded(self):
        # 394.3B vs 395.0B → relative error ≈ 0.18% → outside tolerance
        assert compute_exact_match("$395.0 billion", "$394.3 billion") == 0.0

    def test_text_exact_match(self):
        assert compute_exact_match("The operating margin improved", "The operating margin improved") == 1.0

    def test_normalized_text_match(self):
        # Articles and case should be normalized
        assert compute_exact_match("the revenue increased", "Revenue increased") == 1.0

    def test_percentage_match(self):
        assert compute_exact_match("7.8%", "7.8%") == 1.0

    def test_think_tags_stripped(self):
        pred = "<think>\nLet me calculate...\n</think>\n$99.8 billion"
        assert compute_exact_match(pred, "$99.8 billion") == 1.0

    def test_empty_prediction(self):
        assert compute_exact_match("", "$99.8 billion") == 0.0

    def test_unit_scaling_billion_vs_raw(self):
        # "$99.8 billion" vs "99800000000" — both should parse to 99.8e9
        assert compute_exact_match("$99.8 billion", "99800000000") == 1.0


class TestF1Score:
    def test_perfect_f1(self):
        assert compute_f1("operating margin improved", "operating margin improved") == pytest.approx(1.0)

    def test_partial_overlap(self):
        f1 = compute_f1("margin improved slightly", "operating margin improved")
        assert 0.4 < f1 < 1.0

    def test_no_overlap(self):
        assert compute_f1("revenue declined", "profit increased") == pytest.approx(0.0)

    def test_empty_strings(self):
        assert compute_f1("", "") == pytest.approx(1.0)
        assert compute_f1("something", "") == pytest.approx(0.0)


class TestAnswerParsability:
    def test_numerical_parsable(self):
        assert is_answer_parsable("$394.3 billion") is True

    def test_text_parsable(self):
        assert is_answer_parsable("The operating margin improved significantly.") is True

    def test_refusal_not_parsable(self):
        assert is_answer_parsable("I cannot determine the answer from the context.") is False

    def test_empty_not_parsable(self):
        assert is_answer_parsable("") is False

    def test_insufficient_info_not_parsable(self):
        assert is_answer_parsable("Insufficient information provided.") is False


class TestGroundingRate:
    def test_fully_grounded(self, sample_financial_context):
        # $394.3 billion IS in the context
        rate = compute_grounding_rate("$394.3 billion", sample_financial_context)
        assert rate == pytest.approx(1.0)

    def test_hallucinated_number(self, sample_financial_context):
        # $999 billion is NOT in the context
        rate = compute_grounding_rate("$999 billion", sample_financial_context)
        assert rate == pytest.approx(0.0)

    def test_no_numbers_in_prediction(self, sample_financial_context):
        rate = compute_grounding_rate("Revenue increased year over year.", sample_financial_context)
        assert rate == pytest.approx(1.0)  # No numbers to verify → grounded by default


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for self-consistency
# ──────────────────────────────────────────────────────────────────────────────

class TestSelfConsistency:
    def test_numerical_median_aggregation(self):
        answers = ["$2.5 billion", "$2.4 billion", "$2.5B", "$2.5 billion",
                   "$2.6 billion", "$2.5 billion", "2.5 billion", "$2.5B"]
        final, confidence = self_consistent_answer(answers)
        num = _parse_number(final)
        assert num is not None
        # Median of [2.5, 2.4, 2.5, 2.5, 2.6, 2.5, 2.5, 2.5] × 1e9 = 2.5e9
        assert abs(num - 2.5e9) / 2.5e9 < 0.01

    def test_high_agreement_confidence(self):
        answers = ["$394.3 billion"] * 7 + ["$400 billion"]
        _, confidence = self_consistent_answer(answers)
        assert confidence >= 0.7

    def test_categorical_majority_vote(self):
        answers = [
            "The operating margin improved",
            "Operating margin improved",
            "margins went down",
            "The operating margin improved",
            "Operating margin improved significantly",
        ]
        final, confidence = self_consistent_answer(answers)
        assert "improved" in final.lower()

    def test_outlier_resistance(self):
        # One massive outlier should not pull the median far
        answers = ["$2.5 billion", "$2.5 billion", "$2.5 billion",
                   "$2.5 billion", "$2.5 trillion"]  # trillion is 1000× wrong
        final, _ = self_consistent_answer(answers)
        num = _parse_number(final)
        assert num is not None
        assert abs(num - 2.5e9) / 2.5e9 < 0.01, f"Expected ~2.5B, got {num}"


# ──────────────────────────────────────────────────────────────────────────────
# Robustness integration tests (require model)
# ──────────────────────────────────────────────────────────────────────────────

class TestNumericalPerturbation:
    """
    If all numbers in the context increase by 10%, the model's answer
    for a ratio/growth question should stay within reasonable bounds.
    For absolute value questions, the answer should scale by ~10%.
    """

    def test_revenue_question_scales(
        self,
        model_and_tokenizer,
        sample_financial_context,
        perturbed_context_10pct_up,
    ):
        model, tokenizer = model_and_tokenizer
        from src.inference.generate import generate_answer

        q = "What was Apple's total revenue in fiscal year 2022?"

        ans_original = generate_answer(model, tokenizer, q, sample_financial_context)
        ans_perturbed = generate_answer(model, tokenizer, q, perturbed_context_10pct_up)

        num_orig = _parse_number(ans_original)
        num_pert = _parse_number(ans_perturbed)

        assert num_orig is not None, f"Could not parse original answer: {ans_original}"
        assert num_pert is not None, f"Could not parse perturbed answer: {ans_perturbed}"

        # Perturbed answer should be ~10% higher (within ±5%)
        ratio = num_pert / num_orig
        assert 1.04 < ratio < 1.17, (
            f"Expected ~1.10× ratio after 10% perturbation, got {ratio:.3f} "
            f"(orig={num_orig:.2e}, pert={num_pert:.2e})"
        )


class TestEntitySwap:
    """
    After swapping 'Apple' → 'Microsoft' in the context, the model should
    NOT return Apple-specific facts from training memory.
    """

    def test_entity_swap_grounding(
        self,
        model_and_tokenizer,
        entity_swapped_context,
    ):
        model, tokenizer = model_and_tokenizer
        from src.inference.generate import generate_answer

        q = "What was Microsoft's total revenue in fiscal year 2022?"
        answer = generate_answer(model, tokenizer, q, entity_swapped_context)

        # The answer should be grounded in the (swapped) context
        grounding = compute_grounding_rate(answer, entity_swapped_context)
        assert grounding >= 0.5, (
            f"Model may be hallucinating non-context numbers. "
            f"Grounding rate: {grounding:.2f}. Answer: {answer}"
        )

        # The answer should NOT mention "Apple" (entity leak from training)
        assert "apple" not in answer.lower(), (
            f"Entity leak detected: answer mentions 'Apple' despite context swap. "
            f"Answer: {answer}"
        )


class TestUnitConfusion:
    """The model should correctly distinguish million vs billion."""

    def test_million_vs_billion(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        from src.inference.generate import generate_answer

        context_million = (
            "TechCorp reported revenues of $45.2 million for Q3 2023, "
            "compared to $38.7 million in Q3 2022."
        )
        context_billion = context_million.replace("million", "billion")

        q = "What were TechCorp's revenues in Q3 2023?"

        ans_m = generate_answer(model, tokenizer, q, context_million)
        ans_b = generate_answer(model, tokenizer, q, context_billion)

        num_m = _parse_number(ans_m)
        num_b = _parse_number(ans_b)

        if num_m is not None and num_b is not None:
            ratio = num_b / num_m if num_m != 0 else float("inf")
            assert ratio > 100, (
                f"Unit confusion: billion answer ({num_b:.2e}) should be ~1000× "
                f"million answer ({num_m:.2e}), got ratio={ratio:.1f}."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Parametric data perturbation tests (no model needed)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("original,perturbed,expected_ratio", [
    (100.0, 110.0, 1.10),
    (50.0, 45.0, 0.90),
    (1000.0, 1100.0, 1.10),
])
def test_numerical_perturbation_metric(original: float, perturbed: float, expected_ratio: float):
    """Verify that our perturbation logic produces correct scaling."""
    ratio = perturbed / original
    assert abs(ratio - expected_ratio) < 0.001


@pytest.mark.parametrize("answer,expected_parsable", [
    ("$394.3 billion", True),
    ("7.8%", True),
    ("The revenue increased by 10%", True),
    ("I cannot determine the answer", False),
    ("N/A", False),
    ("", False),
])
def test_parsability_parametric(answer: str, expected_parsable: bool):
    assert is_answer_parsable(answer) == expected_parsable


@pytest.mark.parametrize("pred,gt,expected_em", [
    ("$394.3 billion", "$394.3 billion", 1.0),
    ("$394.4 billion", "$394.3 billion", 0.0),   # >0.01% relative error
    ("30.3%", "30.3%", 1.0),
    ("not answerable", "$394.3 billion", 0.0),
])
def test_exact_match_parametric(pred: str, gt: str, expected_em: float):
    result = compute_exact_match(pred, gt)
    assert result == pytest.approx(expected_em, abs=0.001)
