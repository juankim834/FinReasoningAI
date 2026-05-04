"""Preprocess FinCoT SFT samples into train/test prompt-completion datasets."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizerBase

from src.data.fincot_loader import (
    DATASET_NAME,
    NON_NUMERICAL_REASONING,
    NUMERICAL_REASONING,
    classify_reasoning_category,
    load_fincot_samples,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are FinReasoningAI, a financial reasoning assistant."
COT_SYSTEM_PROMPT = SYSTEM_PROMPT
DEFAULT_OUTPUT_DIR = "data/processed_fincot_sft"
DEFAULT_FINAL_SAMPLE_SIZE = 5000
DEFAULT_TRAIN_SIZE = 4500
DEFAULT_TEST_SIZE = 500
DEFAULT_SEED = 42
NEGATIVE_FIELD_KEYS = (
    "Negative_reasoning_process",
    "negative_reasoning_process",
    "Negative_response",
    "negative_response",
)


def _build_messages(
    sample: dict[str, Any],
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> tuple[str, str]:
    """Build a raw FinCoT prompt and completion from canonical fields."""
    del tool_definitions
    context = str(sample.get("context") or "").strip()
    question = str(sample.get("question") or "").strip()
    answer = str(sample.get("answer") or "").strip()
    reasoning = str(sample.get("reasoning") or sample.get("chain_of_thought") or "").strip()

    if not question:
        raise ValueError("Sample is missing a question.")
    if not answer:
        raise ValueError("Sample is missing an answer.")

    prompt = f"{context}\n\n{question}" if context else question
    completion = f"{reasoning}\n\n{answer}" if include_cot and reasoning else answer
    return prompt, completion


def _drop_negative_fields(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove negative training targets from the saved sample payload."""
    return {
        key: value
        for key, value in sample.items()
        if key not in NEGATIVE_FIELD_KEYS
    }


def format_sample_as_chat(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Convert a sample into raw FinCoT prompt/completion text."""
    del tokenizer
    prompt, completion = _build_messages(sample, include_cot=include_cot, tool_definitions=tool_definitions)

    row = _drop_negative_fields(dict(sample))
    row.update(
        {
            "prompt": prompt,
            "completion": completion,
        }
    )
    return row


def format_as_prompt_completion(
    sample: dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compatibility wrapper used by evaluation and older callers."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return format_sample_as_chat(
        sample,
        tokenizer=tokenizer,
        include_cot=include_cot,
        tool_definitions=tool_definitions,
    )


def format_sample(
    sample: dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Backward-compatible alias for older callers."""
    return format_as_prompt_completion(
        sample,
        tokenizer=tokenizer,
        include_cot=include_cot,
        tool_definitions=tool_definitions,
    )


def _annotate_reasoning_categories(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add the reasoning_category field to each sample."""
    annotated: list[dict[str, Any]] = []
    for sample in samples:
        enriched = dict(sample)
        enriched["reasoning_category"] = classify_reasoning_category(enriched)
        annotated.append(enriched)
    return annotated


def _shuffle_copy(samples: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return a shuffled copy of the sample list."""
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _sample_final_subset(
    samples: list[dict[str, Any]],
    final_sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """
    Build the final 5,000-sample subset with a balanced target when possible.

    If both classes have at least half of the requested size, sample 50/50.
    Otherwise include all of the smaller class and fill from the larger class.
    """
    if len(samples) < final_sample_size:
        raise ValueError(
            f"Requested {final_sample_size} samples, but only {len(samples)} are available."
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["reasoning_category"])].append(sample)

    numerical = _shuffle_copy(grouped.get(NUMERICAL_REASONING, []), seed)
    non_numerical = _shuffle_copy(grouped.get(NON_NUMERICAL_REASONING, []), seed + 1)
    target_each = final_sample_size // 2

    numerical_take = min(len(numerical), target_each)
    non_numerical_take = min(len(non_numerical), target_each)

    selected: list[dict[str, Any]] = []
    selected.extend(numerical[:numerical_take])
    selected.extend(non_numerical[:non_numerical_take])

    remaining_needed = final_sample_size - len(selected)
    if remaining_needed > 0:
        remaining_pool = numerical[numerical_take:] + non_numerical[non_numerical_take:]
        remaining_pool = _shuffle_copy(remaining_pool, seed + 2)
        selected.extend(remaining_pool[:remaining_needed])

    if len(selected) != final_sample_size:
        raise ValueError(
            "Unable to assemble the requested final sample size after balancing."
        )

    selected = _shuffle_copy(selected, seed + 3)
    return selected, Counter(sample["reasoning_category"] for sample in selected)


def _allocate_split_counts(
    category_counts: dict[str, int],
    target_size: int,
) -> dict[str, int]:
    """Allocate an exact split size while staying close to stratified proportions."""
    total = sum(category_counts.values())
    if target_size < 0 or target_size > total:
        raise ValueError("target_size must be between 0 and the total category count.")
    if total == 0:
        return {category: 0 for category in category_counts}

    exact = {
        category: (count / total) * target_size
        for category, count in category_counts.items()
    }
    allocated = {
        category: min(math.floor(value), category_counts[category])
        for category, value in exact.items()
    }

    remaining = target_size - sum(allocated.values())
    remainders = sorted(
        (
            (exact[category] - allocated[category], category)
            for category in category_counts
            if allocated[category] < category_counts[category]
        ),
        reverse=True,
    )
    index = 0
    while remaining > 0 and remainders:
        _, category = remainders[index % len(remainders)]
        if allocated[category] < category_counts[category]:
            allocated[category] += 1
            remaining -= 1
        index += 1
        if index > 10000:
            raise RuntimeError("Unexpected allocation loop while creating stratified split.")
    return allocated


def _stratified_train_test_split(
    samples: list[dict[str, Any]],
    train_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create an exact-size train/test split with approximate stratification."""
    if train_size + test_size != len(samples):
        raise ValueError("train_size + test_size must equal len(samples).")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample["reasoning_category"])].append(sample)

    for index, category in enumerate(sorted(grouped)):
        grouped[category] = _shuffle_copy(grouped[category], seed + index)

    category_counts = {category: len(rows) for category, rows in grouped.items()}
    test_allocations = _allocate_split_counts(category_counts, test_size)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for category, rows in grouped.items():
        category_test = test_allocations.get(category, 0)
        test_rows.extend(rows[:category_test])
        train_rows.extend(rows[category_test:])

    train_rows = _shuffle_copy(train_rows, seed + 11)
    test_rows = _shuffle_copy(test_rows, seed + 12)

    if len(train_rows) != train_size or len(test_rows) != test_size:
        raise ValueError(
            f"Split sizes are incorrect: train={len(train_rows)} test={len(test_rows)}."
        )
    return train_rows, test_rows


def build_dataset(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> DatasetDict:
    """Format and save a provided train/test sample dict mapping."""
    if not samples:
        raise ValueError("No samples were provided to build_dataset.")

    dataset_rows = [
        format_sample_as_chat(
            sample,
            tokenizer=tokenizer,
            include_cot=include_cot,
            tool_definitions=tool_definitions,
        )
        for sample in samples
    ]
    dataset_dict = DatasetDict({"train": Dataset.from_list(dataset_rows)})

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    return dataset_dict


def build_train_test_dataset_dict(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> DatasetDict:
    """Format and save the final train/test DatasetDict."""
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    format_sample_as_chat(
                        sample,
                        tokenizer=tokenizer,
                        include_cot=include_cot,
                        tool_definitions=tool_definitions,
                    )
                    for sample in train_samples
                ]
            ),
            "test": Dataset.from_list(
                [
                    format_sample_as_chat(
                        sample,
                        tokenizer=tokenizer,
                        include_cot=include_cot,
                        tool_definitions=tool_definitions,
                    )
                    for sample in test_samples
                ]
            ),
        }
    )

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    return dataset_dict


def prepare_fincot_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = DATASET_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    final_sample_size: int = DEFAULT_FINAL_SAMPLE_SIZE,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
    seed: int = DEFAULT_SEED,
) -> tuple[DatasetDict, dict[str, Any]]:
    """End-to-end FinCoT SFT preparation pipeline."""
    if train_size + test_size != final_sample_size:
        raise ValueError("train_size + test_size must equal final_sample_size.")

    loaded_samples, metadata = load_fincot_samples(dataset_name=dataset_name)
    annotated_samples = _annotate_reasoning_categories(loaded_samples)
    category_counts = Counter(sample["reasoning_category"] for sample in annotated_samples)

    final_samples, sampled_counts = _sample_final_subset(
        annotated_samples,
        final_sample_size=final_sample_size,
        seed=seed,
    )
    train_samples, test_samples = _stratified_train_test_split(
        final_samples,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
    )

    dataset_dict = build_train_test_dataset_dict(
        train_samples=train_samples,
        test_samples=test_samples,
        tokenizer=tokenizer,
        output_dir=output_dir,
        include_cot=include_cot,
        tool_definitions=tool_definitions,
    )

    summary = {
        **metadata,
        "numerical_count": category_counts.get(NUMERICAL_REASONING, 0),
        "non_numerical_count": category_counts.get(NON_NUMERICAL_REASONING, 0),
        "final_sample_size": len(final_samples),
        "sampled_distribution": dict(sampled_counts),
        "train_size": len(train_samples),
        "test_size": len(test_samples),
        "train_distribution": dict(Counter(sample["reasoning_category"] for sample in train_samples)),
        "test_distribution": dict(Counter(sample["reasoning_category"] for sample in test_samples)),
    }
    return dataset_dict, summary


def print_preparation_summary(summary: dict[str, Any], output_dir: str) -> None:
    """Print the requested dataset preparation summary."""
    print(f"Dataset: {summary['dataset_name']}")
    print(f"Loaded split: {summary['split_name']}")
    print(f"Original number of samples in SFT Training: {summary['original_sample_count']}")
    print(f"Available dataset columns: {summary['columns']}")
    print(f"Number of Numerical Reasoning samples: {summary['numerical_count']}")
    print(f"Number of Non-Numerical Reasoning samples: {summary['non_numerical_count']}")
    print(f"Final training set size: {summary['train_size']}")
    print(f"Final test set size: {summary['test_size']}")
    print(f"Category distribution in train: {summary['train_distribution']}")
    print(f"Category distribution in test: {summary['test_distribution']}")
    print(f"Saved processed dataset to: {output_dir}")


def load_and_format_dataset(
    data_path: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 2048,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = DEFAULT_SEED,
    num_proc: int = 4,
) -> DatasetDict:
    """Backward-compatible wrapper; prefers the FinCoT SFT pipeline."""
    del data_path, max_length, train_frac, val_frac, num_proc
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    dataset_dict, _ = prepare_fincot_sft_dataset(
        tokenizer=tokenizer,
        seed=seed,
    )
    return dataset_dict


def load_eval_test_samples(
    data_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[dict[str, Any]]:
    """Load the saved FinCoT held-out test rows for evaluation."""
    dataset = load_from_disk(str(Path(data_dir)))
    if "test" not in dataset:
        raise ValueError(f"Dataset at {data_dir} does not contain a 'test' split.")
    return dataset["test"].to_list()


def print_dataset_stats(
    dataset_dict: DatasetDict,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> None:
    """Print rough length statistics for each split."""
    for split_name, dataset in dataset_dict.items():
        lengths: list[int] = []
        for row in dataset:
            text = f"{row['prompt']}{row['completion']}"
            if tokenizer is None:
                lengths.append(len(text) // 4)
            else:
                lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        average = sum(lengths) / len(lengths) if lengths else 0
        logger.info(
            "Split %s | n=%d | avg_len=%.1f | max=%d",
            split_name,
            len(dataset),
            average,
            max(lengths, default=0),
        )


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build train/test datasets from TheFinAI/FinCoT SFT split")
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final_sample_size", type=int, default=DEFAULT_FINAL_SAMPLE_SIZE)
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude_cot", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_dict, summary = prepare_fincot_sft_dataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        final_sample_size=args.final_sample_size,
        train_size=args.train_size,
        test_size=args.test_size,
        include_cot=not args.exclude_cot,
        seed=args.seed,
    )
    print_preparation_summary(summary, args.output_dir)
    print_dataset_stats(dataset_dict, tokenizer)
