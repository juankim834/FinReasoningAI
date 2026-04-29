"""
File: src/eval/evaluate.py

Step 5a & 5b — Evaluation Framework

Metrics rationale:
  - Exact Match (EM) with configurable fractional tolerance (default 1%): primary
    metric for numerical answers unless a stricter ``numeric_tolerance`` is passed.
  - F1: SQuAD-style token overlap for ``financial_qa`` only; for numerical and
    structured tasks, F1 is binary (1.0 iff EM passes) so token overlap does not
    inflate scores on numeric outputs.
  - Answer parsability rate: whether the model produces an extractable answer.
  - Grounding: for ``financial_qa``, numeric literals in the answer are checked
    against the context string. For ``numerical_reasoning`` / ``structured_analysis``,
    the first extracted prediction number is compared to the value obtained by
    evaluating ``expression`` with ``variables`` (same bindings as data generation).
"""

from __future__ import annotations

import csv
import logging
import re
import string
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(
    r"-?\$?\s*(\d[\d,]*\.?\d*)\s*(%|billion|million|thousand|trillion|bps|x|T|B|M|K)?",
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

    scale = {
        "billion": 1e9, "b": 1e9,
        "million": 1e6, "m": 1e6,
        "thousand": 1e3, "k": 1e3,
        "trillion": 1e12, "t": 1e12,
    }
    if unit in scale:
        value *= scale[unit]
    elif unit == "%":
        value /= 100.0

    return value


def _evaluate_expression_numeric(expression: str, variables: Optional[dict]) -> Optional[float]:
    """
    Evaluate ``expression`` with ``variables`` as the only global names (controlled
    ``eval``, same strategy as synthetic data generation).
    """
    if not expression or not str(expression).strip():
        return None
    ns: dict = {}
    for k, v in (variables or {}).items():
        if isinstance(v, (int, float)):
            ns[str(k).replace("-", "_")] = float(v)
    try:
        result = eval(str(expression).strip(), {"__builtins__": {}}, ns)  # noqa: S307
        return float(result)
    except Exception:
        return None


def compute_exact_match(
    prediction: str,
    ground_truth: str,
    tol: float = 0.01,
) -> float:
    """
    Exact match metric for financial answers.
    Returns 1.0 if answers match (with relative numeric error ≤ ``tol``), else 0.0.

    ``tol`` is a **fractional** tolerance (e.g. ``0.01`` = 1%).
    """
    pred_clean = _THINK_RE.sub("", prediction).strip()
    gt_clean = ground_truth.strip()

    if _normalize_text(pred_clean) == _normalize_text(gt_clean):
        return 1.0

    pred_num = _extract_number(pred_clean)
    gt_num = _extract_number(gt_clean)

    if pred_num is not None and gt_num is not None and gt_num != 0:
        relative_error = abs(pred_num - gt_num) / abs(gt_num)
        return 1.0 if relative_error <= tol else 0.0

    return 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score for qualitative/text answers (SQuAD-style).
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


def compute_f1_for_task(
    task: str,
    prediction: str,
    ground_truth: str,
    exact_match_score: float,
) -> Tuple[float, str]:
    """
    Return (f1_value, metric_mode) where ``metric_mode`` is ``token_f1`` or ``numeric_em``.
    """
    if task == "financial_qa":
        return compute_f1(prediction, ground_truth), "token_f1"
    return (1.0 if exact_match_score >= 1.0 else 0.0), "numeric_em"


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
    task: str = "financial_qa",
    expression: Optional[str] = None,
    variables: Optional[dict] = None,
    numeric_tolerance: float = 0.01,
) -> float:
    """
    Hallucination proxy for numeric grounding.

    - ``financial_qa``: fraction of numbers in the prediction that match some
      number in ``context`` within relative error ``tol``.
    - ``numerical_reasoning`` / ``structured_analysis``: the first extracted
      prediction number is compared to ``expression`` evaluated with ``variables``;
      returns 1.0 if within ``numeric_tolerance``, else 0.0 (no literal context match).
    """
    pred_clean = _THINK_RE.sub("", prediction).strip()

    if task in ("numerical_reasoning", "structured_analysis"):
        pred_num = _extract_number(pred_clean)
        if pred_num is None:
            return 1.0
        expected = _evaluate_expression_numeric(expression or "", variables)
        if expected is None or (expected != expected):  # NaN
            return 0.5
        if expected == 0:
            return 1.0 if abs(pred_num) <= numeric_tolerance else 0.0
        rel = abs(pred_num - expected) / abs(expected)
        return 1.0 if rel <= numeric_tolerance else 0.0

    pred_nums_raw = _NUMBER_RE.findall(pred_clean)
    if not pred_nums_raw:
        return 1.0

    context_nums_raw = _NUMBER_RE.findall(context)
    context_nums = set()
    for num_str, unit in context_nums_raw:
        try:
            v = float(num_str.replace(",", ""))
            u = unit.lower()
            if u in ("billion", "b"):
                v *= 1e9
            elif u in ("million", "m"):
                v *= 1e6
            elif u in ("trillion", "t"):
                v *= 1e12
            elif u == "%":
                v /= 100.0
            context_nums.add(v)
        except ValueError:
            continue

    if not context_nums:
        return 0.5

    grounded = 0
    total = len(pred_nums_raw)

    for num_str, unit in pred_nums_raw:
        try:
            v = float(num_str.replace(",", ""))
            u = unit.lower()
            if u in ("billion", "b"):
                v *= 1e9
            elif u in ("million", "m"):
                v *= 1e6
            elif u in ("trillion", "t"):
                v *= 1e12
            elif u == "%":
                v /= 100.0
        except ValueError:
            continue

        for cv in context_nums:
            if cv == 0:
                continue
            if abs(v - cv) / abs(cv) <= tol:
                grounded += 1
                break

    return grounded / total if total > 0 else 1.0


def _validate_eval_samples(samples: List[dict], n: int = 5) -> None:
    """
    Validate the first ``n`` samples before running inference.

    Raises:
        ValueError: with guidance to re-run ``load_eval_test_samples`` if required
        fields are missing or empty.
    """
    hint = (
        "Re-run `load_eval_test_samples` from your raw JSONL path so each row "
        "includes the fields required for evaluation."
    )
    for i, sample in enumerate(samples[:n]):
        sid = sample.get("id", f"index_{i}")
        for key in ("question", "answer"):
            val = sample.get(key)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ValueError(
                    f"Sample id={sid!r} is missing or has empty {key!r}. {hint}"
                )
        task = sample.get("task", "financial_qa")
        ctx_val = sample.get("context")
        has_context = ctx_val is not None and (
            (isinstance(ctx_val, str) and ctx_val.strip() != "")
            or (not isinstance(ctx_val, str) and ctx_val != "")
        )
        if not has_context:
            if task == "structured_analysis" and sample.get("financial_data"):
                has_context = True
        if not has_context:
            raise ValueError(
                f"Sample id={sid!r} is missing or has empty 'context' "
                f"(for structured_analysis, provide non-empty 'financial_data' if context is absent). {hint}"
            )
        if task in ("numerical_reasoning", "structured_analysis"):
            expr = sample.get("expression")
            if expr is None or (isinstance(expr, str) and not str(expr).strip()):
                raise ValueError(
                    f"Sample id={sid!r} (task={task!r}) requires non-empty 'expression'. {hint}"
                )


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────────────────────────────────────

_INFERENCE_MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "greedy": {
        "use_cot": False,
        "self_consistency_n": 1,
        "temperature": 0.0,
        "max_new_tokens": 128,
    },
    "cot": {
        "use_cot": True,
        "self_consistency_n": 1,
        "temperature": 0.0,
        "max_new_tokens": 300,
    },
    "self_consistency": {
        "use_cot": False,
        "self_consistency_n": 5,
        "temperature": 0.7,
        "max_new_tokens": 128,
    },
}


def generate_prediction(
    model,
    tokenizer,
    sample: dict,
    max_new_tokens: int = 128,
    use_cot: bool = False,
    temperature: float = 0.0,
    do_sample: bool = False,
    self_consistency_n: int = 1,
) -> str:
    """
    Generate a prediction for one sample.

    Greedy decoding is used when ``do_sample`` is False and ``self_consistency_n`` is 1.
    For ``self_consistency_n`` > 1, delegates to ``sample_with_self_consistency``.
    """
    from src.data.preprocess import format_as_prompt_completion
    from src.inference.self_consistency import sample_with_self_consistency

    sample_prompt = deepcopy(sample)
    if use_cot:
        sample_prompt["_eval_use_cot"] = True
        if sample_prompt.get("reasoning") is None:
            sample_prompt["reasoning"] = ""

    pc = format_as_prompt_completion(sample_prompt)
    prompt = pc["prompt"]
    if not prompt.strip():
        logger.warning("Empty prompt for sample id=%s task=%s", sample.get("id"), sample.get("task"))

    if self_consistency_n > 1:
        final, _conf, _raw = sample_with_self_consistency(
            model,
            tokenizer,
            prompt,
            n=self_consistency_n,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return final.strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1900,
    ).to(model.device)

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample or temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temperature, 1e-5)
        gen_kwargs["top_p"] = 0.9
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _evaluate_one_pass(
    model,
    tokenizer,
    samples: List[dict],
    numeric_tolerance: float,
    max_new_tokens: int,
    inference_mode: str,
    tolerance_used: float,
    run_baseline: bool,
    base_is_peft: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[Dict[str, float]]]:
    """
    Run one inference configuration on ``samples``.

    When ``run_baseline`` and the model is a ``PeftModel``, runs generations with
    adapters disabled first, then with adapters enabled (default).
    """
    from peft import PeftModel

    mode_cfg = _INFERENCE_MODE_DEFAULTS[inference_mode]
    use_cot = mode_cfg["use_cot"]
    sc_n = mode_cfg["self_consistency_n"]
    temperature = mode_cfg["temperature"]
    gen_tokens = max_new_tokens if inference_mode == "greedy" else int(mode_cfg["max_new_tokens"])

    baseline_agg: Optional[Dict[str, float]] = None
    b_pred: List[str] = []
    b_em: List[float] = []
    b_f1: List[float] = []
    b_par: List[float] = []
    b_gr: List[float] = []

    if run_baseline and base_is_peft:
        if not isinstance(model, PeftModel):
            logger.warning("run_baseline=True but model is not a PeftModel; skipping baseline pass.")
        else:
            with model.disable_adapter():
                for i, sample in enumerate(samples):
                    try:
                        bp = generate_prediction(
                            model,
                            tokenizer,
                            sample,
                            max_new_tokens=gen_tokens,
                            use_cot=use_cot,
                            temperature=temperature,
                            do_sample=(temperature > 0),
                            self_consistency_n=sc_n,
                        )
                    except Exception as exc:
                        logger.warning("Baseline generation failed for sample %d: %s", i, exc)
                        bp = ""
                    b_pred.append(bp)
                    gt = str(sample.get("answer", ""))
                    ctx = str(sample.get("context", ""))
                    task = str(sample.get("task", "financial_qa"))
                    expr = sample.get("expression")
                    variables = sample.get("variables") if isinstance(sample.get("variables"), dict) else {}
                    em_b = compute_exact_match(bp, gt, tol=numeric_tolerance)
                    f1_b, _ = compute_f1_for_task(task, bp, gt, em_b)
                    par_b = is_answer_parsable(bp)
                    gr_b = compute_grounding_rate(
                        bp,
                        ctx,
                        task=task,
                        expression=str(expr) if expr is not None else None,
                        variables=variables,
                        numeric_tolerance=numeric_tolerance,
                    )
                    b_em.append(em_b)
                    b_f1.append(f1_b)
                    b_par.append(float(par_b))
                    b_gr.append(gr_b)
            if b_em:
                baseline_agg = {
                    "baseline_exact_match": sum(b_em) / len(b_em),
                    "baseline_f1": sum(b_f1) / len(b_f1),
                    "baseline_parsability_rate": sum(b_par) / len(b_par),
                    "baseline_grounding_rate": sum(b_gr) / len(b_gr),
                    "baseline_n_samples": float(len(samples)),
                }

    results: List[Dict[str, Any]] = []
    em_scores: List[float] = []
    f1_scores: List[float] = []
    parsable_flags: List[float] = []
    grounding_scores: List[float] = []

    for i, sample in enumerate(samples):
        try:
            pred = generate_prediction(
                model,
                tokenizer,
                sample,
                max_new_tokens=gen_tokens,
                use_cot=use_cot,
                temperature=temperature,
                do_sample=(temperature > 0),
                self_consistency_n=sc_n,
            )
        except Exception as exc:
            logger.warning("Generation failed for sample %d: %s", i, exc)
            pred = ""

        gt = str(sample.get("answer", ""))
        ctx = str(sample.get("context", ""))
        task = str(sample.get("task", "financial_qa"))
        expr = sample.get("expression")
        variables = sample.get("variables") if isinstance(sample.get("variables"), dict) else {}

        em = compute_exact_match(pred, gt, tol=numeric_tolerance)
        f1_val, metric_mode = compute_f1_for_task(task, pred, gt, em)
        parsable = is_answer_parsable(pred)
        grounding = compute_grounding_rate(
            pred,
            ctx,
            task=task,
            expression=str(expr) if expr is not None else None,
            variables=variables,
            numeric_tolerance=numeric_tolerance,
        )

        em_scores.append(em)
        f1_scores.append(f1_val)
        parsable_flags.append(float(parsable))
        grounding_scores.append(grounding)

        row: Dict[str, Any] = {
            "id": sample.get("id", str(i)),
            "task": task,
            "inference_mode": inference_mode,
            "tolerance_used": tolerance_used,
            "metric_mode": metric_mode,
            "question": sample.get("question", "")[:100],
            "ground_truth": gt[:100],
            "prediction": pred[:500],
            "exact_match": em,
            "f1": f1_val,
            "parsable": parsable,
            "grounding_rate": grounding,
        }
        if run_baseline and base_is_peft and isinstance(model, PeftModel) and b_pred:
            row["baseline_prediction"] = b_pred[i][:500]
            row["baseline_exact_match"] = b_em[i]
            row["baseline_f1"] = b_f1[i]
            row["baseline_parsable"] = bool(b_par[i])
            row["baseline_grounding_rate"] = b_gr[i]
        results.append(row)

    aggregate: Dict[str, float] = {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "parsability_rate": sum(parsable_flags) / len(parsable_flags) if parsable_flags else 0.0,
        "grounding_rate": sum(grounding_scores) / len(grounding_scores) if grounding_scores else 0.0,
        "n_samples": float(len(samples)),
    }

    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    for task, task_results in by_task.items():
        aggregate[f"{task}_em"] = sum(x["exact_match"] for x in task_results) / len(task_results)
        aggregate[f"{task}_f1"] = sum(x["f1"] for x in task_results) / len(task_results)

    return results, aggregate, baseline_agg


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    output_csv: str = "outputs/eval_results.csv",
    max_new_tokens: int = 128,
    max_samples: Optional[int] = None,
    numeric_tolerance: float = 0.01,
    run_baseline: bool = False,
    inference_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the model on a test dataset (optionally with baseline and multiple
    inference modes).

    Args:
        model:              Fine-tuned model (often ``PeftModel`` wrapped).
        tokenizer:        Matching tokenizer.
        test_dataset:     HuggingFace Dataset or list of sample dicts.
        output_csv:       Path for per-sample CSV (stacked rows per inference mode).
        max_new_tokens:   Default max new tokens when a mode does not override it.
        max_samples:      Cap evaluation to this many samples (after validation).
        numeric_tolerance: Fractional tolerance for EM and expression-based grounding
            (e.g. ``0.01`` = 1%).
        run_baseline:     If True and ``model`` is a ``PeftModel``, evaluate the
            base weights (adapter disabled) and add ``baseline_*`` columns and metrics.
        inference_modes:  Subset of ``\"greedy\"``, ``\"cot\"``, ``\"self_consistency\"``.

    Returns:
        Dict mapping each inference mode name to its aggregate metric dict for that run
        (including ``baseline_*`` and ``delta_*`` keys when ``run_baseline`` is True).
    """
    if inference_modes is None:
        inference_modes = ["greedy"]

    for m in inference_modes:
        if m not in _INFERENCE_MODE_DEFAULTS:
            raise ValueError(
                f"Unknown inference_mode {m!r}. Expected one of {list(_INFERENCE_MODE_DEFAULTS)}."
            )

    if hasattr(test_dataset, "to_list"):
        samples = test_dataset.to_list()
    else:
        samples = list(test_dataset)

    _validate_eval_samples(samples)

    if max_samples is not None:
        samples = samples[:max_samples]

    model.eval()

    from peft import PeftModel

    base_is_peft = isinstance(model, PeftModel)

    all_rows: List[Dict[str, Any]] = []
    out_by_mode: Dict[str, Any] = {}

    tolerance_used = float(numeric_tolerance)

    for mode in inference_modes:
        rows, aggregate, baseline_agg = _evaluate_one_pass(
            model,
            tokenizer,
            samples,
            numeric_tolerance=numeric_tolerance,
            max_new_tokens=max_new_tokens,
            inference_mode=mode,
            tolerance_used=tolerance_used,
            run_baseline=run_baseline,
            base_is_peft=base_is_peft,
        )

        merged = dict(aggregate)
        if baseline_agg:
            merged.update(baseline_agg)
            delta: Dict[str, float] = {}
            baseline_key_map = {
                "exact_match": "baseline_exact_match",
                "f1": "baseline_f1",
                "parsability_rate": "baseline_parsability_rate",
                "grounding_rate": "baseline_grounding_rate",
            }
            for k, v in aggregate.items():
                bk = baseline_key_map.get(k)
                if bk is not None and isinstance(v, float) and bk in merged:
                    delta[f"delta_{k}"] = v - float(merged[bk])
            merged.update(delta)
            print(f"[evaluate] delta vs baseline ({mode}): {delta}")
        all_rows.extend(rows)
        out_by_mode[mode] = merged

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            for r in all_rows[1:]:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    logger.info("Results CSV saved to %s", output_csv)

    for mode, merged in out_by_mode.items():
        logger.info("=" * 55)
        logger.info("EVALUATION SUMMARY mode=%s (n=%d samples)", mode, len(samples))
        for label, key in (
            ("Exact Match", "exact_match"),
            ("F1 Score", "f1"),
            ("Parsability Rate", "parsability_rate"),
            ("Grounding Rate", "grounding_rate"),
        ):
            if key in merged:
                logger.info("  %-18s: %.4f", label, merged[key])
        logger.info("=" * 55)

    return out_by_mode


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
    parser.add_argument("--numeric_tolerance", type=float, default=0.01)
    parser.add_argument("--run_baseline", action="store_true")
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

    metrics_by_mode = evaluate_model(
        model, tokenizer, test_ds,
        output_csv=args.output_csv,
        max_samples=args.max_samples,
        numeric_tolerance=args.numeric_tolerance,
        run_baseline=args.run_baseline,
    )
    print("\n[OK] Step 5b complete — evaluation results:")
    for mode, metrics in metrics_by_mode.items():
        print(f"\n--- {mode} ---")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
