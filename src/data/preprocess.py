"""Preprocess normalized financial reasoning samples into chat-style datasets."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are FinReasoningAI, a financial reasoning assistant. "
    "Think step by step before giving your final answer. "
    "For numerical questions, show your calculation chain. "
    "Your final answer must be on a line starting with 'Answer:'."
)


def _tool_prompt_suffix(tool_definitions: Optional[list[dict[str, Any]]]) -> str:
    """Render tool definitions as a JSON block appended to the system prompt."""
    if not tool_definitions:
        return ""
    return "\n\n## Available Tools\n" + json.dumps(tool_definitions, indent=2, ensure_ascii=False)


def _build_messages(
    sample: dict[str, Any],
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, str]], str]:
    """Build chat messages plus the expected assistant completion."""
    system_prompt = SYSTEM_PROMPT + _tool_prompt_suffix(tool_definitions)
    context = (sample.get("context") or "").strip()
    question = str(sample.get("question", "")).strip()
    if not question:
        raise ValueError("Sample is missing a question.")

    user_message = f"{context}\n\n{question}" if context else question
    answer = str(sample.get("answer", "")).strip()
    if not answer:
        raise ValueError("Sample is missing an answer.")

    chain_of_thought = str(sample.get("chain_of_thought", "") or "").strip()
    if include_cot and chain_of_thought:
        completion = f"{chain_of_thought}\nAnswer: {answer}"
    else:
        completion = f"Answer: {answer}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return messages, completion


def _split_prompt_completion(rendered_with_prompt: str, rendered_full: str, completion: str) -> tuple[str, str]:
    """Recover the prompt boundary after apply_chat_template rendering."""
    if rendered_full.endswith(completion):
        prompt = rendered_full[: -len(completion)]
    else:
        prompt = rendered_with_prompt
    if not prompt:
        prompt = rendered_with_prompt
    return prompt, completion


def format_sample_as_chat(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, str]:
    """Convert a normalized sample into prompt/completion chat text."""
    messages, completion = _build_messages(sample, include_cot=include_cot, tool_definitions=tool_definitions)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt, completion = _split_prompt_completion(prompt, full_text, completion)
    return {
        "prompt": prompt,
        "completion": completion,
        "task": str(sample.get("task", "financial_qa")),
    }


def format_as_prompt_completion(
    sample: dict[str, Any],
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
) -> dict[str, str]:
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


def _stratified_partition(
    samples: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Create train/val/test splits while preserving task proportions."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if not 0 <= val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1.")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.get("task", "financial_qa"))].append(sample)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for task_samples in grouped.values():
        rng.shuffle(task_samples)
        total = len(task_samples)
        train_count = int(total * train_frac)
        val_count = int(total * val_frac)
        if total >= 2:
            train_count = max(train_count, 1)
        if total >= 5 and val_frac > 0:
            val_count = max(val_count, 1)
        if train_count + val_count >= total:
            overflow = train_count + val_count - total + 1
            if val_count >= overflow:
                val_count -= overflow
            else:
                train_count = max(1, train_count - (overflow - val_count))
                val_count = 0
        splits["train"].extend(task_samples[:train_count])
        splits["val"].extend(task_samples[train_count: train_count + val_count])
        splits["test"].extend(task_samples[train_count + val_count:])

    for split_name, split_samples in splits.items():
        counts = Counter(sample.get("task", "financial_qa") for sample in split_samples)
        logger.info("Split %s: %d samples %s", split_name, len(split_samples), dict(counts))
    return splits


def build_dataset(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str = "data/processed",
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    include_cot: bool = True,
    tool_definitions: Optional[list[dict[str, Any]]] = None,
    seed: int = 42,
) -> DatasetDict:
    """Format normalized samples, split them, and save a DatasetDict to disk."""
    splits = _stratified_partition(samples, train_frac=train_frac, val_frac=val_frac, seed=seed)
    dataset_dict = DatasetDict()
    for split_name, split_samples in splits.items():
        rows = [
            format_sample_as_chat(
                sample,
                tokenizer=tokenizer,
                include_cot=include_cot,
                tool_definitions=tool_definitions,
            )
            for sample in split_samples
        ]
        dataset_dict[split_name] = Dataset.from_list(rows)

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))
    return dataset_dict


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %d: %s", line_number, exc)
    return rows


def _normalize_eval_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw eval sample while preserving optional legacy fields."""
    normalized = {
        "question": str(sample.get("question") or sample.get("instruction") or sample.get("input") or "").strip(),
        "answer": str(sample.get("answer") or sample.get("output") or sample.get("response") or "").strip(),
        "chain_of_thought": str(sample.get("chain_of_thought") or sample.get("reasoning") or "").strip(),
        "context": (sample.get("context") or sample.get("document") or "") or "",
        "task": str(sample.get("task") or sample.get("task_type") or "financial_qa"),
        "expression": sample.get("expression", ""),
        "variables": sample.get("variables") if isinstance(sample.get("variables"), dict) else {},
        "financial_data": sample.get("financial_data") if isinstance(sample.get("financial_data"), dict) else {},
        "id": sample.get("id"),
        "instruction": sample.get("instruction", ""),
    }
    return normalized


def load_eval_test_samples(
    data_dir: str,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load raw JSONL samples and return the held-out test portion."""
    path = Path(data_dir)
    raw_samples: list[dict[str, Any]] = []
    if path.is_dir():
        for jsonl_path in sorted(path.glob("*.jsonl")):
            raw_samples.extend(load_jsonl(str(jsonl_path)))
    elif path.is_file():
        raw_samples = load_jsonl(str(path))
    else:
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    normalized = [_normalize_eval_sample(sample) for sample in raw_samples]
    splits = _stratified_partition(normalized, train_frac=train_frac, val_frac=val_frac, seed=seed)
    return splits["test"]


def load_and_format_dataset(
    data_path: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    max_length: int = 2048,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
    num_proc: int = 4,
) -> DatasetDict:
    """Backward-compatible wrapper that loads raw JSONL and builds a DatasetDict."""
    del max_length, num_proc
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    path = Path(data_path)
    raw_samples: list[dict[str, Any]] = []
    if path.is_dir():
        for jsonl_path in sorted(path.glob("*.jsonl")):
            raw_samples.extend(load_jsonl(str(jsonl_path)))
    elif path.is_file():
        raw_samples = load_jsonl(str(path))
    else:
        raise FileNotFoundError(f"data_path not found: {data_path}")

    normalized = [_normalize_eval_sample(sample) for sample in raw_samples]
    return build_dataset(
        normalized,
        tokenizer=tokenizer,
        output_dir="data/processed",
        train_frac=train_frac,
        val_frac=val_frac,
        include_cot=True,
        seed=seed,
    )


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
        logger.info("Split %s | n=%d | avg_len=%.1f | max=%d", split_name, len(dataset), average, max(lengths, default=0))


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build prompt/completion datasets for FinReasoningAI")
    parser.add_argument("--data_path", default="data/raw/fincot.jsonl")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--train_frac", type=float, default=0.90)
    parser.add_argument("--val_frac", type=float, default=0.05)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    path = Path(args.data_path)
    raw_samples: list[dict[str, Any]] = []
    if path.is_dir():
        for jsonl_path in sorted(path.glob("*.jsonl")):
            raw_samples.extend(load_jsonl(str(jsonl_path)))
    else:
        raw_samples = load_jsonl(str(path))

    normalized = [_normalize_eval_sample(sample) for sample in raw_samples]
    dataset = build_dataset(
        normalized,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    print_dataset_stats(dataset, tokenizer)
    print(f"Saved DatasetDict to {args.output_dir}: {{split: len(ds) for split, ds in dataset.items()}}")
