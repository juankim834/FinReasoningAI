"""Load and classify FinCoT SFT samples from Hugging Face."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DATASET_NAME = "TheFinAI/FinCoT"
PREFERRED_SFT_SPLITS = ("SFT Training", "SFT")
NUMERICAL_REASONING = "Numerical Reasoning"
NON_NUMERICAL_REASONING = "Non-Numerical Reasoning"

QUESTION_KEYS = (
    "question",
    "Question",
    "instruction",
    "Instruction",
    "input",
    "Input",
    "query",
    "prompt",
)
ANSWER_KEYS = (
    "answer",
    "Answer",
    "Final_response",
    "final_response",
    "output",
    "Output",
    "response",
)
REASONING_KEYS = (
    "reasoning",
    "Reasoning",
    "Reasoning_process",
    "reasoning_process",
    "chain_of_thought",
    "cot",
    "analysis",
    "rationale",
)
CONTEXT_KEYS = (
    "context",
    "Context",
    "document",
    "Document",
    "passage",
    "news",
)
EXPLICIT_REASONING_TYPE_KEYS = (
    "reasoning_category",
    "reasoning_type",
    "reasoning_kind",
    "reasoning_task",
    "task_type",
    "task",
    "category",
    "type",
)

NUMERICAL_VALUE_HINTS = (
    "numerical",
    "quantitative",
    "calculation",
    "math",
    "arithmetic",
    "ratio",
    "percentage",
    "growth",
)
NON_NUMERICAL_VALUE_HINTS = (
    "non-numerical",
    "non numerical",
    "qualitative",
    "textual",
    "descriptive",
    "sentiment",
    "summarization",
    "classification",
)
NUMERICAL_KEYWORDS = (
    "calculate",
    "calculation",
    "compute",
    "what is the percentage",
    "percentage",
    "percent",
    "ratio",
    "growth rate",
    "cagr",
    "compound annual growth",
    "basis points",
    "bps",
    "increase",
    "decrease",
    "gross margin",
    "operating margin",
    "net margin",
    "return on assets",
    "return on equity",
    "eps",
    "earnings per share",
    "price-to-earnings",
    "p/e",
    "dividend yield",
    "current ratio",
    "quick ratio",
    "debt-to-equity",
    "free cash flow",
    "present value",
    "future value",
    "discount rate",
    "annualized",
    "weighted average",
    "sum of",
    "total of",
)
NON_NUMERICAL_KEYWORDS = (
    "summarize",
    "summary",
    "explain qualitatively",
    "qualitative",
    "describe",
    "discuss",
    "interpret the statement",
    "what does this mean",
    "identify the risk",
    "sentiment",
)
TABLE_HINTS = ("|", "\t", "fiscal year", "years ended", "three months ended")
MONEY_OR_UNIT_PATTERN = re.compile(
    r"(?:\$|usd|eur|million|billion|thousand|%|percent|basis points|bps)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")
ARITHMETIC_PATTERN = re.compile(r"[\d\)\]]\s*[-+/*=]\s*[\d\(\[]")


def _stringify(value: Any) -> str:
    """Convert a dataset value into a readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(item) for item in value if _stringify(item))
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            return str(value).strip()
    return str(value).strip()


def _first_present(sample: dict[str, Any], keys: Iterable[str]) -> str:
    """Return the first non-empty string from the provided keys."""
    for key in keys:
        value = _stringify(sample.get(key))
        if value:
            return value
    return ""


def _extract_messages_text(messages: Any) -> str:
    """Flatten conversational message content into plain text for heuristics."""
    if not isinstance(messages, list):
        return ""

    chunks: list[str] = []
    for item in messages:
        if isinstance(item, dict):
            for key in ("content", "text", "value"):
                value = _stringify(item.get(key))
                if value:
                    chunks.append(value)
                    break
        else:
            value = _stringify(item)
            if value:
                chunks.append(value)
    return "\n".join(chunks)


def _resolve_sft_split(split_names: Iterable[str]) -> str:
    """Resolve the preferred FinCoT SFT split name."""
    split_list = list(split_names)
    normalized = {name.casefold(): name for name in split_list}

    for preferred in PREFERRED_SFT_SPLITS:
        if preferred.casefold() in normalized:
            return normalized[preferred.casefold()]

    for name in split_list:
        if "sft" in name.casefold():
            return name

    raise ValueError(
        f"Unable to find an SFT split in {split_list}. "
        f"Expected one of {list(PREFERRED_SFT_SPLITS)}."
    )


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Attach canonical fields while preserving the original sample fields."""
    normalized = dict(sample)
    normalized["question"] = _first_present(sample, QUESTION_KEYS)
    normalized["answer"] = _first_present(sample, ANSWER_KEYS)
    normalized["reasoning"] = _first_present(sample, REASONING_KEYS)
    normalized["context"] = _first_present(sample, CONTEXT_KEYS)

    if not normalized["question"]:
        messages_text = _extract_messages_text(sample.get("messages"))
        if messages_text:
            normalized["question"] = messages_text

    return normalized


def _classify_from_explicit_field(sample: dict[str, Any]) -> str | None:
    """Use an explicit task/category field when it clearly maps to a category."""
    for key in EXPLICIT_REASONING_TYPE_KEYS:
        raw_value = _stringify(sample.get(key))
        if not raw_value:
            continue

        value = raw_value.casefold()
        if any(hint in value for hint in NUMERICAL_VALUE_HINTS):
            return NUMERICAL_REASONING
        if any(hint in value for hint in NON_NUMERICAL_VALUE_HINTS):
            return NON_NUMERICAL_REASONING
    return None


def classify_reasoning_category(sample: dict[str, Any]) -> str:
    """
    Classify a sample as numerical vs non-numerical reasoning.

    The function first checks for an explicit dataset field such as `task_type`
    or `category`. If the dataset does not expose a usable label, it falls back
    to a score-based heuristic over question/context/reasoning/answer text.
    """
    explicit_category = _classify_from_explicit_field(sample)
    if explicit_category is not None:
        return explicit_category

    text_parts = [
        _stringify(sample.get("question")),
        _stringify(sample.get("context")),
        _stringify(sample.get("reasoning")),
        _stringify(sample.get("answer")),
        _extract_messages_text(sample.get("messages")),
    ]
    text = "\n".join(part for part in text_parts if part)
    lowered = text.casefold()

    score = 0

    number_matches = NUMBER_PATTERN.findall(text)
    if len(number_matches) >= 4:
        score += 2
    elif len(number_matches) >= 2:
        score += 1

    if ARITHMETIC_PATTERN.search(text):
        score += 3

    unit_matches = MONEY_OR_UNIT_PATTERN.findall(lowered)
    if len(unit_matches) >= 2:
        score += 2
    elif unit_matches:
        score += 1

    if sum(1 for keyword in NUMERICAL_KEYWORDS if keyword in lowered) >= 2:
        score += 2
    elif any(keyword in lowered for keyword in NUMERICAL_KEYWORDS):
        score += 1

    if any(marker in lowered for marker in TABLE_HINTS) and len(number_matches) >= 3:
        score += 2

    if "step 1" in lowered or "first," in lowered or "then," in lowered:
        if len(number_matches) >= 2:
            score += 1

    if any(keyword in lowered for keyword in NON_NUMERICAL_KEYWORDS) and score < 3:
        score -= 1

    return NUMERICAL_REASONING if score >= 3 else NON_NUMERICAL_REASONING


def load_fincot_sft_split(
    dataset_name: str = DATASET_NAME,
) -> tuple[Any, str, list[str]]:
    """Load the FinCoT SFT split and return the dataset, split name, and columns."""
    from datasets import load_dataset

    dataset_dict = load_dataset(dataset_name)
    split_name = _resolve_sft_split(dataset_dict.keys())
    dataset = dataset_dict[split_name]
    columns = list(dataset.column_names)
    return dataset, split_name, columns


def load_fincot_samples(
    dataset_name: str = DATASET_NAME,
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load FinCoT SFT samples, preserve original fields, and add canonical ones."""
    dataset, split_name, columns = load_fincot_sft_split(dataset_name=dataset_name)
    raw_rows = dataset.to_list()
    normalized_rows = [_normalize_sample(row) for row in raw_rows]
    if max_samples is not None:
        normalized_rows = normalized_rows[:max_samples]

    metadata = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "columns": columns,
        "original_sample_count": len(raw_rows),
    }
    logger.info(
        "Loaded %d rows from %s split '%s'.",
        len(normalized_rows),
        dataset_name,
        split_name,
    )
    return normalized_rows, metadata


def summarize_reasoning_categories(samples: list[dict[str, Any]]) -> Counter[str]:
    """Count reasoning categories in a sample list."""
    return Counter(str(sample.get("reasoning_category", "")) for sample in samples)


__all__ = [
    "DATASET_NAME",
    "NON_NUMERICAL_REASONING",
    "NUMERICAL_REASONING",
    "classify_reasoning_category",
    "load_fincot_samples",
    "load_fincot_sft_split",
    "summarize_reasoning_categories",
]
