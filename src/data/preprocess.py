"""
File: src/data/preprocess.py

Step 2d — Preprocessing Pipeline

Converts raw JSONL data into tokenized HuggingFace DatasetDicts.

Key decisions:
  - ChatML template (Qwen2.5 native format): uses <|im_start|> / <|im_end|> tokens.
    This ensures the tokenizer's special tokens are used correctly and inference
    prompts match the training format exactly.
  - max_length=2048: balances context quality vs VRAM. Most FinQA contexts fit
    in 512–1024 tokens; 10-K paragraphs may reach 1800–2000 tokens.
  - Stratified split: preserves task-type proportions across train/val/test to
    avoid evaluation being dominated by the majority class (Type A).

[WARN] TRADE-OFF: Truncating to 2048 tokens may cut off the tail of long 10-K
passages. Monitor the truncation rate; if >20% of Type A samples are truncated,
consider increasing to 3072 (costs ~12% more VRAM).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from datasets import DatasetDict, Dataset, Features, Value
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# ChatML prompt templates (Qwen2.5-native format)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are FinReasoning AI, an expert financial analyst. "
    "Answer questions accurately using only the provided context. "
    "Give concise, precise answers. Do not include chain-of-thought unless asked."
)

COT_SYSTEM_PROMPT = (
    "You are FinReasoning AI, an expert financial analyst. "
    "Think step by step inside <think>...</think> tags, then give your final answer."
)


def format_financial_qa(sample: dict, include_reasoning: bool = True) -> str:
    """
    Format a Type A (financial_qa) sample into ChatML.

    If reasoning is present and include_reasoning=True, the <think> block is
    included in the assistant turn — this is what we train on.
    At inference, we strip <think>...</think> from the output.
    """
    has_cot = sample.get("reasoning") is not None
    system = COT_SYSTEM_PROMPT if has_cot else SYSTEM_PROMPT

    user_content = (
        f"{sample['instruction']}\n\n"
        f"Context:\n{sample['context']}\n\n"
        f"Question: {sample['question']}"
    )

    if has_cot and include_reasoning:
        reasoning = sample["reasoning"]
        if not reasoning.startswith("<think>"):
            reasoning = f"<think>\n{reasoning}\n</think>"
        assistant_content = f"{reasoning}\n\n{sample['answer']}"
    else:
        assistant_content = sample["answer"]

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )


def format_numerical_reasoning(sample: dict) -> str:
    """Format a Type B (numerical_reasoning) sample into ChatML."""
    var_lines = "\n".join(f"  {k} = {v:,.4f}" for k, v in sample["variables"].items())
    user_content = (
        f"{sample['instruction']}\n\n"
        f"Context:\n{sample.get('context', '')}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Expression: {sample['expression']}\n"
        f"Variables:\n{var_lines}"
    )
    answer = str(sample["answer"])
    if sample.get("unit"):
        answer = f"{answer} {sample['unit']}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


def format_structured_analysis(sample: dict) -> str:
    """Format a Type C (structured_analysis) sample into ChatML."""
    fin_data_str = json.dumps(sample["financial_data"], indent=2)
    user_content = (
        f"{sample['instruction']}\n\n"
        f"Financial Data:\n{fin_data_str}\n\n"
        f"Question: {sample['question']}"
    )

    has_cot = sample.get("reasoning") is not None
    system = COT_SYSTEM_PROMPT if has_cot else SYSTEM_PROMPT

    if has_cot:
        reasoning = sample["reasoning"]
        if not reasoning.startswith("<think>"):
            reasoning = f"<think>\n{reasoning}\n</think>"
        assistant_content = f"{reasoning}\n\n{sample['answer']}"
    else:
        assistant_content = sample["answer"]

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_content}<|im_end|>"
    )


FORMATTERS = {
    "financial_qa": format_financial_qa,
    "numerical_reasoning": format_numerical_reasoning,
    "structured_analysis": format_structured_analysis,
}


def format_sample(sample: dict) -> str:
    """Dispatch to the appropriate formatter based on task type."""
    task = sample.get("task", "financial_qa")
    formatter = FORMATTERS.get(task)
    if formatter is None:
        raise ValueError(f"Unknown task type: {task!r}")
    return formatter(sample)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization
# ──────────────────────────────────────────────────────────────────────────────

def tokenize_sample(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
) -> dict:
    """
    Tokenize a single formatted sample.
    Returns input_ids, attention_mask, and labels (identical to input_ids for CLM).

    Truncation warning: logged when any sample exceeds max_length.
    """
    text = format_sample(sample)
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    original_len = len(tokenizer(text, truncation=False)["input_ids"])
    if original_len > max_length:
        logger.warning(
            "Sample %s truncated: %d → %d tokens (task=%s). "
            "Consider reducing context or increasing max_length.",
            sample.get("id", "?"),
            original_len,
            max_length,
            sample.get("task", "?"),
        )

    tokens["labels"] = tokens["input_ids"].copy()
    tokens["task"] = sample.get("task", "financial_qa")
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading & splitting
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping malformed JSONL line: %s", e)
    return records


def stratified_split(
    samples: List[dict],
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """
    Stratified split by task type: 90% train / 5% val / 5% test.
    Ensures each split has proportional representation of all task types.
    """
    import random

    rng = random.Random(seed)

    # Group by task
    by_task: Dict[str, List[dict]] = {}
    for s in samples:
        t = s.get("task", "financial_qa")
        by_task.setdefault(t, []).append(s)

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for task, task_samples in by_task.items():
        rng.shuffle(task_samples)
        n = len(task_samples)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        splits["train"].extend(task_samples[:n_train])
        splits["val"].extend(task_samples[n_train : n_train + n_val])
        splits["test"].extend(task_samples[n_train + n_val :])

        logger.info(
            "Task %-22s — train: %4d, val: %3d, test: %3d",
            task, n_train, n_val, n - n_train - n_val,
        )

    # Final shuffle within each split
    for key in splits:
        rng.shuffle(splits[key])

    return splits


def _normalize_sample(sample: dict) -> dict:
    """
    Flatten a raw sample into a schema that PyArrow can build a uniform
    Arrow table from.

    Root cause of ArrowTypeError:
      - `answer` can be float (numerical_reasoning) or str (all others).
        PyArrow infers the column type from the first row; a later float
        in a str column (or vice-versa) raises ArrowTypeError.
      - `variables` and `financial_data` are dicts, which are absent in
        most rows (None). PyArrow cannot unify dict and NullType columns.
      - `reasoning` is sometimes None and sometimes str; that is fine
        because Arrow has a nullable string type, but we convert explicitly
        to be safe.

    Strategy: convert every field to a plain Python str, preserving the
    data needed by format_sample() while giving PyArrow a uniform schema.
    The tokenization step calls format_sample() on the raw dict, so
    dict-typed fields must be reconstructed there — we JSON-serialize them
    here and deserialize inside format_sample()-aware paths.
    """
    out = {
        "id":          str(sample.get("id", "")),
        "task":        str(sample.get("task", "financial_qa")),
        "instruction": str(sample.get("instruction", "")),
        "context":     str(sample.get("context", "")),
        "question":    str(sample.get("question", "")),
        # answer may be float or str — always coerce to str
        "answer":      str(sample.get("answer", "")),
        "reasoning":   str(sample.get("reasoning", "") or ""),
        "unit":        str(sample.get("unit", "") or ""),
        # dict fields: JSON-serialize so every row is a str
        "variables":   json.dumps(sample.get("variables") or {}),
        "financial_data": json.dumps(sample.get("financial_data") or {}),
        "expression":  str(sample.get("expression", "") or ""),
        "format":      str(sample.get("format", "") or ""),
    }
    return out


def _denormalize_sample(row: dict) -> dict:
    """
    Reverse _normalize_sample: restore JSON-serialized fields back to
    Python dicts so that format_sample() receives the expected types.
    """
    row = dict(row)
    try:
        row["variables"] = json.loads(row.get("variables") or "{}")
    except (json.JSONDecodeError, TypeError):
        row["variables"] = {}
    try:
        row["financial_data"] = json.loads(row.get("financial_data") or "{}")
    except (json.JSONDecodeError, TypeError):
        row["financial_data"] = {}
    # Restore None for empty optional fields
    if not row.get("reasoning"):
        row["reasoning"] = None
    if not row.get("unit"):
        row["unit"] = None
    return row


def load_and_format_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 2048,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
    num_proc: int = 4,
) -> DatasetDict:
    """
    Load raw JSONL data, apply ChatML formatting, tokenize, and split.

    Args:
        data_path:   Path to a JSONL file (can also be a directory; all .jsonl
                     files in the directory will be concatenated).
        tokenizer:   Qwen2.5 tokenizer (must already be loaded).
        max_length:  Maximum token length (2048 recommended for A100 80GB).
        train_frac:  Fraction for training (default 0.90).
        val_frac:    Fraction for validation (default 0.05); test gets remainder.
        seed:        Reproducibility seed.
        num_proc:    Parallel workers for tokenization.

    Returns:
        DatasetDict with keys "train", "val", "test".

    Memory: tokenized dataset at 2048 tokens × ~100K samples ≈ ~1.6 GB disk.
    """
    path = Path(data_path)
    raw_samples: List[dict] = []

    if path.is_dir():
        for jl_file in sorted(path.glob("*.jsonl")):
            raw_samples.extend(load_jsonl(str(jl_file)))
            logger.info("Loaded %s", jl_file)
    elif path.is_file():
        raw_samples = load_jsonl(str(path))
    else:
        raise FileNotFoundError(f"data_path not found: {data_path}")

    logger.info("Total raw samples loaded: %d", len(raw_samples))

    splits = stratified_split(raw_samples, train_frac=train_frac, val_frac=val_frac, seed=seed)

    dataset_dict = {}
    truncation_counts: Dict[str, int] = {}

    for split_name, split_samples in splits.items():
        if not split_samples:
            continue

        # Normalize to a uniform flat string schema so PyArrow can build
        # a consistent Arrow table regardless of task type.
        # (answer=float vs str, variables=dict vs absent, etc. all become str)
        normalized = [_normalize_sample(s) for s in split_samples]
        hf_dataset = Dataset.from_list(normalized)

        # Use map with batched=True for speed
        tokenized = hf_dataset.map(
            lambda batch: _tokenize_batch(batch, tokenizer, max_length),
            batched=True,
            batch_size=32,
            num_proc=1,  # tokenizer is not process-safe; keep at 1
            remove_columns=hf_dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )

        dataset_dict[split_name] = tokenized
        logger.info(
            "Split '%s': %d samples tokenized.", split_name, len(tokenized)
        )

    return DatasetDict(dataset_dict)


def _tokenize_batch(
    batch: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """
    Batch tokenization helper for HuggingFace .map().

    Rows arrive from Dataset as normalized flat strings (_normalize_sample).
    We denormalize each row back to the expected Python types before passing
    to tokenize_sample() -> format_sample(), which needs:
      - variables:      dict[str, float]   (numerical_reasoning)
      - financial_data: dict               (structured_analysis)
      - reasoning:      str | None
    """
    input_ids_list, attention_mask_list, labels_list, task_list = [], [], [], []

    n = len(next(iter(batch.values())))
    for i in range(n):
        raw = {k: v[i] for k, v in batch.items()}
        sample = _denormalize_sample(raw)
        tok = tokenize_sample(sample, tokenizer, max_length)
        input_ids_list.append(tok["input_ids"])
        attention_mask_list.append(tok["attention_mask"])
        labels_list.append(tok["labels"])
        task_list.append(tok["task"])

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "task": task_list,
    }


def print_dataset_stats(dataset_dict: DatasetDict, tokenizer: PreTrainedTokenizerBase) -> None:
    """Print dataset statistics including token length distribution."""
    import numpy as np

    for split, ds in dataset_dict.items():
        lengths = [len(ids) for ids in ds["input_ids"]]
        logger.info(
            "Split %-8s | n=%6d | avg_len=%6.0f | p50=%5d | p95=%5d | max=%5d",
            split, len(ds),
            float(np.mean(lengths)),
            int(np.percentile(lengths, 50)),
            int(np.percentile(lengths, 95)),
            max(lengths),
        )
        trunc_rate = sum(1 for l in lengths if l >= 2048) / len(lengths) * 100
        if trunc_rate > 20:
            warnings.warn(
                f"High truncation rate in split '{split}': {trunc_rate:.1f}% of samples "
                f"hit max_length=2048. Consider increasing max_length.",
                UserWarning,
                stacklevel=2,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Preprocess FinReasoning dataset")
    parser.add_argument("--data_path", default="data/raw/synthetic.jsonl")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_and_format_dataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    print_dataset_stats(dataset, tokenizer)

    # Save to disk
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))

    print(f"\n[OK] Step 2d complete — DatasetDict saved to {args.output_dir}")
    print(f"   Splits: { {k: len(v) for k, v in dataset.items()} }")
