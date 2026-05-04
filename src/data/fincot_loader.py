"""Load and normalize FinCoT-style financial reasoning datasets."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

PRIMARY_DATASETS = (
    "Duxiaoman-DI/FinCoT",
    "IDEA-FinAI/fingpt-forecasting",
    "gbharti/finance-alpaca",
)

TASK_LABELS = ("financial_qa", "numerical_reasoning", "structured_analysis")


def _stringify(value: Any) -> str:
    """Convert any dataset value into a trimmed string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


QUESTION_KEYS = (
    "question",
    "instruction",
    "input",
    "query",
    "prompt",
)
ANSWER_KEYS = (
    "answer",
    "output",
    "response",
    "label",
    "target",
)
COT_KEYS = (
    "chain_of_thought",
    "cot",
    "reasoning",
    "analysis",
    "thought",
    "rationale",
)
CONTEXT_KEYS = (
    "context",
    "input_context",
    "document",
    "passage",
    "news",
    "instruction_context",
)
TASK_KEYS = (
    "task",
    "task_type",
    "category",
    "type",
)


def _first_present(sample: dict[str, Any], keys: Iterable[str]) -> str:
    """Return the first non-empty string value found for the provided keys."""
    for key in keys:
        value = _stringify(sample.get(key))
        if value:
            return value
    return ""


def _infer_task(question: str, answer: str, chain_of_thought: str, context: str) -> str:
    """Infer one of the supported task labels from sample content."""
    blob = " ".join(part.lower() for part in (question, answer, chain_of_thought, context) if part)
    numerical_markers = (
        "calculate",
        "ratio",
        "margin",
        "growth rate",
        "cagr",
        "percentage",
        "basis points",
        "bps",
    )
    structured_markers = (
        "table",
        "json",
        "balance sheet",
        "income statement",
        "cash flow",
        "financial data",
    )
    if any(marker in blob for marker in numerical_markers):
        return "numerical_reasoning"
    if any(marker in blob for marker in structured_markers):
        return "structured_analysis"
    return "financial_qa"


def _normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw sample from any supported source."""
    question = _first_present(sample, QUESTION_KEYS)
    answer = _first_present(sample, ANSWER_KEYS)
    chain_of_thought = _first_present(sample, COT_KEYS)
    context = _first_present(sample, CONTEXT_KEYS)
    task = _first_present(sample, TASK_KEYS).lower().replace(" ", "_")
    if task not in TASK_LABELS:
        task = _infer_task(question, answer, chain_of_thought, context)

    if not question or not answer:
        raise ValueError("Sample is missing a question or answer after normalization.")

    return {
        "question": question,
        "answer": answer,
        "chain_of_thought": chain_of_thought,
        "context": context or None,
        "task": task,
    }


def _load_huggingface_rows() -> list[dict[str, Any]]:
    """Try candidate Hugging Face datasets in priority order."""
    from datasets import load_dataset

    last_error: Exception | None = None
    for dataset_name in PRIMARY_DATASETS:
        try:
            dataset = load_dataset(dataset_name)
            logger.info("Loaded dataset from Hugging Face: %s", dataset_name)
            rows: list[dict[str, Any]] = []
            for split in dataset.values():
                rows.extend(split.to_list())
            return rows
        except Exception as exc:  # pragma: no cover - depends on external availability
            last_error = exc
            logger.warning("Failed to load dataset %s: %s", dataset_name, exc)
    raise RuntimeError("Unable to load a supported FinCoT dataset from Hugging Face.") from last_error


def _load_local_rows(local_path: str) -> list[dict[str, Any]]:
    """Load JSONL rows from a local FinCoT export."""
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset not found: {local_path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d: %s", line_number, exc)
    return rows


def load_fincot_samples(
    source: str = "huggingface",
    local_path: str = "data/raw/fincot.jsonl",
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load FinCoT-style samples and normalize them to a common schema."""
    source = source.lower()
    if source not in {"huggingface", "local"}:
        raise ValueError("source must be either 'huggingface' or 'local'.")

    raw_rows = _load_huggingface_rows() if source == "huggingface" else _load_local_rows(local_path)

    normalized: list[dict[str, Any]] = []
    for row in raw_rows:
        try:
            normalized.append(_normalize_sample(row))
        except ValueError as exc:
            logger.debug("Skipping unusable row: %s", exc)

    rng = random.Random(seed)
    rng.shuffle(normalized)
    if max_samples is not None:
        normalized = normalized[:max_samples]

    task_counts = Counter(sample["task"] for sample in normalized)
    summary = ", ".join(f"{task}={count}" for task, count in sorted(task_counts.items()))
    print(f"Loaded {len(normalized)} FinCoT samples.")
    print(f"Task distribution: {summary}")
    return normalized


__all__ = ["load_fincot_samples"]
