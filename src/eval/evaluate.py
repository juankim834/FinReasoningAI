"""
File: src/eval/evaluate.py

Step 5a & 5b — Evaluation Framework

Metrics rationale:
  - Exact Match (EM) with ±0.01% tolerance: the primary metric for numerical
    answers. Financial numbers must be precise; loose matching (e.g., ±1%) can
    mask systematic rounding errors that compound in multi-step reasoning.
  - F1 token overlap: for qualitative/textual answers where exact match is too
    strict (e.g., "EBITDA margin increased" vs "the EBITDA margin went up").
  - Answer parsability rate: measures whether the model produces an extractable
    answer at all — a model that says "I cannot determine this" fails this metric.
  - Hallucination proxy (citation grounding): checks whether the numeric values
    in the model's answer appear in the provided context, within tolerance.
    If the answer contains a number not derivable from context, flag it.
"""

from __future__ import annotations

import csv
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"-?\$?\s*(\d[\d,]*\.?\d*)\s*(%|billion|million|thousand|bps|x|M|B|K)?",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalize_text(text: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_number(text: str) -> Optional[float]:
    """
    Extract the first numeric value from text, handling:
      - Currency: $1.2B → 1.2e9
      - Percentages: 12.5% → 0.125
      - Basis points: 50 bps → 50 (returned as-is, unit disambiguation left to caller)
      - Comma-separated: 1,234,567 → 1234567.0
    """
    text = _THINK_RE.sub("", text).strip()
    # Remove currency symbols for parsing
    text_clean = text.replace("$", "").replace(",", "")

    match = _NUMBER_RE.search(text_clean)
    if not match:
        return None

    num_str = match.group(1).replace(",", "")
    unit = (match.group(2) or "").lower()

    try:
        value = float(num_str)
    except ValueError:
        return None

    # Scale by unit
    scale = {
        "billion": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
    }
    if unit in scale:
        value *= scale[unit]
    elif unit == "%":
        value /= 100.0

    return value


def compute_exact_match(prediction: str, ground_truth: str, tol: float = 1e-4) -> float:
    """
    Exact match metric for financial answers.
    Returns 1.0 if answers match (with ±tol tolerance for numbers), 0.0 otherwise.

    tol=1e-4 = 0.01% — matches the Agent.md spec "±0.01% tolerance".
    """
    pred_clean = _THINK_RE.sub("", prediction).strip()
    gt_clean = ground_truth.strip()

    # Normalize text
    if _normalize_text(pred_clean) == _normalize_text(gt_clean):
        return 1.0

    # Try numeric comparison
    pred_num = _extract_number(pred_clean)
    gt_num = _extract_number(gt_clean)

    if pred_num is not None and gt_num is not None and gt_num != 0:
        relative_error = abs(pred_num - gt_num) / abs(gt_num)
        return 1.0 if relative_error <= tol else 0.0

    return 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score for qualitative/text answers.
    Inspired by the SQuAD evaluation metric.
    """
    pred_tokens = _normalize_text(_THINK_RE.sub("", prediction)).split()
    gt_tokens = _normalize_text(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gt_counter = Counter(gt_tokens)

    common = sum((pred_counter & gt_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_answer_parsable(prediction: str) -> bool:
    """
    Check if the model's answer contains an extractable number or clear entity.
    Returns False for refusals like "I cannot determine" or empty strings.
    """
    pred_clean = _THINK_RE.sub("", prediction).strip()
    if not pred_clean:
        return False

    refusal_patterns = [
        r"cannot (determine|calculate|find|answer)",
        r"insufficient (information|data|context)",
        r"not (enough|sufficient|available)",
        r"i('m| am) (not sure|unable)",
        r"n/?a",
    ]
    for pat in refusal_patterns:
        if re.search(pat, pred_clean, re.IGNORECASE):
            return False

    return bool(_NUMBER_RE.search(pred_clean)) or len(pred_clean.split()) >= 2


def compute_grounding_rate(
    prediction: str,
    context: str,
    tol: float = 0.02,
) -> float:
    """
    Hallucination proxy: check whether numeric values in the prediction
    appear in or are derivable from the context (within tol relative error).

    Returns the fraction of predicted numbers that are grounded in context.
    A score of 1.0 means all numbers in the answer can be verified against context.

    [WARN] TRADE-OFF: This is a proxy metric — a model can hallucinate qualitative
    claims while still grounding all numbers. Treat it as a necessary-but-not-
    sufficient hallucination check.
    """
    pred_clean = _THINK_RE.sub("", prediction).strip()
    pred_nums_raw = _NUMBER_RE.findall(pred_clean)
    if not pred_nums_raw:
        return 1.0  # No numbers to verify

    # Extract all numbers from context
    context_nums_raw = _NUMBER_RE.findall(context)
    context_nums = set()
    for num_str, unit in context_nums_raw:
        try:
            v = float(num_str.replace(",", ""))
            if unit.lower() in ("billion", "b"):
                v *= 1e9
            elif unit.lower() in ("million", "m"):
                v *= 1e6
            elif unit.lower() == "%":
                v /= 100.0
            context_nums.add(v)
        except ValueError:
            continue

    if not context_nums:
        return 0.5  # Context has no numbers — inconclusive

    grounded = 0
    total = len(pred_nums_raw)

    for num_str, unit in pred_nums_raw:
        try:
            v = float(num_str.replace(",", ""))
            if unit.lower() in ("billion", "b"):
                v *= 1e9
            elif unit.lower() in ("million", "m"):
                v *= 1e6
            elif unit.lower() == "%":
                v /= 100.0
        except ValueError:
            continue

        # Check if any context number is within tol of this predicted number
        for cv in context_nums:
            if cv == 0:
                continue
            if abs(v - cv) / abs(cv) <= tol:
                grounded += 1
                break

    return grounded / total if total > 0 else 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def generate_prediction(
    model,
    tokenizer,
    sample: dict,
    max_new_tokens: int = 128,
) -> str:
    """Generate a single prediction for a sample using greedy decoding."""
    from src.data.preprocess import format_sample

    # Build prompt (no assistant turn)
    from src.inference.generate import build_prompt
    prompt = build_prompt(
        question=sample.get("question", ""),
        context=sample.get("context", ""),
        instruction=sample.get("instruction", ""),
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1900,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    output_csv: str = "outputs/eval_results.csv",
    max_new_tokens: int = 128,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on a test dataset.

    Args:
        model:          The fine-tuned model (PEFT or merged).
        tokenizer:      Matching tokenizer.
        test_dataset:   HuggingFace Dataset or list of raw sample dicts.
        output_csv:     Path to save the per-sample results CSV.
        max_new_tokens: Max tokens to generate per sample.
        max_samples:    If set, cap evaluation at this many samples.

    Returns:
        Dict of aggregate metrics: em, f1, parsability_rate, grounding_rate.
    """
    if hasattr(test_dataset, "to_list"):
        samples = test_dataset.to_list()
    else:
        samples = list(test_dataset)

    if max_samples is not None:
        samples = samples[:max_samples]

    model.eval()

    results: List[Dict[str, Any]] = []
    em_scores, f1_scores, parsable_flags, grounding_scores = [], [], [], []

    logger.info("Evaluating %d samples...", len(samples))

    for i, sample in enumerate(samples):
        try:
            prediction = generate_prediction(model, tokenizer, sample, max_new_tokens)
        except Exception as exc:
            logger.warning("Generation failed for sample %d: %s", i, exc)
            prediction = ""

        ground_truth = str(sample.get("answer", ""))
        context = sample.get("context", "")

        em = compute_exact_match(prediction, ground_truth)
        f1 = compute_f1(prediction, ground_truth)
        parsable = is_answer_parsable(prediction)
        grounding = compute_grounding_rate(prediction, context)

        em_scores.append(em)
        f1_scores.append(f1)
        parsable_flags.append(float(parsable))
        grounding_scores.append(grounding)

        results.append({
            "id": sample.get("id", str(i)),
            "task": sample.get("task", "unknown"),
            "question": sample.get("question", "")[:100],
            "ground_truth": ground_truth[:100],
            "prediction": prediction[:200],
            "exact_match": em,
            "f1": f1,
            "parsable": parsable,
            "grounding_rate": grounding,
        })

        if (i + 1) % 50 == 0:
            logger.info(
                "Progress: %d/%d | EM=%.3f | F1=%.3f | Parse=%.3f | Ground=%.3f",
                i + 1, len(samples),
                sum(em_scores) / len(em_scores),
                sum(f1_scores) / len(f1_scores),
                sum(parsable_flags) / len(parsable_flags),
                sum(grounding_scores) / len(grounding_scores),
            )

    # Aggregate
    aggregate: Dict[str, float] = {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "parsability_rate": sum(parsable_flags) / len(parsable_flags) if parsable_flags else 0.0,
        "grounding_rate": sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0,
        "n_samples": float(len(samples)),
    }

    # Per-task breakdown
    by_task: Dict[str, List] = {}
    for r in results:
        t = r["task"]
        by_task.setdefault(t, []).append(r)

    for task, task_results in by_task.items():
        task_em = sum(r["exact_match"] for r in task_results) / len(task_results)
        task_f1 = sum(r["f1"] for r in task_results) / len(task_results)
        aggregate[f"{task}_em"] = task_em
        aggregate[f"{task}_f1"] = task_f1
        logger.info("Task %-25s | EM=%.3f | F1=%.3f | n=%d",
                    task, task_em, task_f1, len(task_results))

    # Save CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        if results:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    logger.info("Results CSV saved to %s", output_csv)

    # Print summary
    logger.info("=" * 55)
    logger.info("EVALUATION SUMMARY (n=%d samples)", len(samples))
    logger.info("  Exact Match      : %.4f", aggregate["exact_match"])
    logger.info("  F1 Score         : %.4f", aggregate["f1"])
    logger.info("  Parsability Rate : %.4f", aggregate["parsability_rate"])
    logger.info("  Grounding Rate   : %.4f", aggregate["grounding_rate"])
    logger.info("=" * 55)

    return aggregate


# ──────────────────────────────────────────────────────────────────────────────
# Standalone
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate FinReasoning model")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/sft_qlora/final_adapter",
                        help="Path to LoRA adapter (or 'none' to use base model)")
    parser.add_argument("--test_data", default="data/processed",
                        help="Path to preprocessed DatasetDict or raw JSONL")
    parser.add_argument("--output_csv", default="outputs/eval_results.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    from src.model.load_model import load_model_and_tokenizer, DEFAULT_BNB_CONFIG
    from datasets import load_from_disk

    model, tokenizer = load_model_and_tokenizer(args.model_id)

    if args.adapter_dir != "none" and Path(args.adapter_dir).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        logger.info("Loaded adapter from %s", args.adapter_dir)

    test_path = Path(args.test_data)
    if test_path.is_dir():
        ds = load_from_disk(str(test_path))
        test_ds = ds.get("test", ds.get("val", list(ds.values())[0]))
    else:
        import json
        with open(test_path) as f:
            test_ds = [json.loads(l) for l in f if l.strip()]

    metrics = evaluate_model(
        model, tokenizer, test_ds,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
    )
    print("\n[OK] Step 5b complete — evaluation results:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
