"""
File: src/inference/self_consistency.py

Step 5c — Self-Consistency Decoding

Self-consistency rationale:
  - Sample N=5–10 independent generations at temperature=0.7 and aggregate.
  - For categorical / qualitative answers: majority vote.
  - For numerical answers: median (more robust to outlier hallucinations than mean).
  - This exploits the model's internal uncertainty: if 8/10 samples agree,
    confidence is high; if samples are evenly split, the question is genuinely hard.
  - Median aggregation is specifically motivated by financial use cases:
    a model hallucinating "$2.5 trillion" (vs the correct "$2.5 billion") produces
    a massive outlier; median is resistant to this while mean would be distorted.

[WARN] TRADE-OFF: Self-consistency multiplies latency by N. For real-time APIs,
N=3 is a reasonable compromise; for batch/async workflows, N=8–10 maximizes
accuracy. Consider caching results for repeated questions.

Optional verifier model (Step 5d):
  - A small Qwen2.5-1.5B verifier can re-rank the N candidates before aggregation.
  - Training: fine-tune on (question, candidate_answer, score) triples where
    score = EM against ground truth. Output a scalar 0–1 via regression head.
  - At inference: score each candidate, take argmax instead of majority vote.
  - This adds ~2 GB VRAM for the 1.5B verifier (negligible on A100).
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBER_RE = re.compile(
    r"-?\$?\s*(\d[\d,]*\.?\d*)\s*(%|billion|million|thousand|trillion|bps|x|T|B|M|K)?",
    re.IGNORECASE,
)

# Word / letter suffix multipliers (case-insensitive) for _normalize_to_float
_SCALE_SUFFIX: dict[str, float] = {
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


def _extract_final_answer(text: str) -> str:
    """
    Strip <think> tags and extract the final answer portion.
    If no </think> tag, return the full text (model answered directly).
    """
    # Remove think blocks
    cleaned = _THINK_RE.sub("", text).strip()
    # If empty after stripping, the model only produced think content — use raw
    return cleaned if cleaned else text.strip()


def _is_numerical(answer: str) -> bool:
    """Heuristic: does this answer look like a number?"""
    return bool(_NUMBER_RE.search(answer))


def _normalize_to_float(s: str) -> Optional[float]:
    """
    Parse a single scalar from a free-form answer string.

    Strips currency symbols, thousands separators, and percent signs, applies
    K/M/B/T (and billion/million/thousand/trillion) suffix multipliers, and
    returns ``None`` on failure so callers can exclude bad parses from aggregates.
    """
    s = _THINK_RE.sub("", s).strip()
    if not s:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    if s.endswith("%"):
        s = s[:-1].strip()
        pct_mode = True
    else:
        pct_mode = False

    match = _NUMBER_RE.search(s.replace(" ", ""))
    if not match:
        return None
    try:
        v = float(match.group(1).replace(",", ""))
    except ValueError:
        return None
    unit = (match.group(2) or "").lower()
    if unit in _SCALE_SUFFIX:
        v *= _SCALE_SUFFIX[unit]
    elif unit == "%" or pct_mode:
        v /= 100.0
    return v


def _parse_number(answer: str) -> Optional[float]:
    """Parse the primary numeric value from an answer string (alias for median path)."""
    return _normalize_to_float(answer)


def aggregate_numerical(answers: List[str]) -> str:
    """
    Aggregate numerical answers via median after ``_normalize_to_float`` on each
    candidate so formats like ``$111.4 billion`` and ``111.4B`` map to the same
    scale. If no value normalizes, falls back to majority vote on raw strings.
    """
    nums = []
    for a in answers:
        n = _normalize_to_float(a)
        if n is not None:
            nums.append((n, a))

    if not nums:
        return aggregate_categorical(answers) if answers else ""

    median_val = statistics.median(v for v, _ in nums)

    # Return the answer string whose numeric value is closest to the median
    closest = min(nums, key=lambda x: abs(x[0] - median_val))
    return closest[1]


def aggregate_categorical(answers: List[str]) -> str:
    """
    Aggregate categorical/text answers via majority vote.
    Normalizes whitespace and lowercases before voting.
    """
    normalized = [" ".join(a.lower().split()) for a in answers]
    counter = Counter(normalized)
    majority_normalized, _ = counter.most_common(1)[0]

    # Return original-case version of the majority answer
    for a, n in zip(answers, normalized):
        if n == majority_normalized:
            return a

    return answers[0]


def self_consistent_answer(
    answers: List[str],
    force_numerical: bool = False,
) -> Tuple[str, float]:
    """
    Aggregate N sampled answers into a single final answer.

    Args:
        answers:         List of N raw model outputs.
        force_numerical: Force numerical aggregation even if heuristic disagrees.

    Returns:
        (final_answer, confidence) where confidence = fraction of answers agreeing.
    """
    if not answers:
        return "", 0.0

    cleaned = [_extract_final_answer(a) for a in answers]

    # Determine if this is a numerical question
    numerical_count = sum(1 for a in cleaned if _is_numerical(a))
    is_num = force_numerical or (numerical_count > len(cleaned) / 2)

    if is_num:
        final = aggregate_numerical(cleaned)
        # Confidence: fraction of answers within 10% of the final numeric value
        final_num = _normalize_to_float(final)
        if final_num is not None and final_num != 0:
            agreeing = sum(
                1 for a in cleaned
                if (n := _normalize_to_float(a)) is not None
                and abs(n - final_num) / abs(final_num) <= 0.10
            )
            confidence = agreeing / len(cleaned)
        else:
            confidence = 1.0 / len(cleaned)
    else:
        final = aggregate_categorical(cleaned)
        norm_final = " ".join(final.lower().split())
        agreeing = sum(
            1 for a in cleaned if " ".join(a.lower().split()) == norm_final
        )
        confidence = agreeing / len(cleaned)

    return final, confidence


def sample_with_self_consistency(
    model,
    tokenizer,
    prompt: str,
    n: int = 8,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    min_confidence: float = 0.4,
) -> Tuple[str, float, List[str]]:
    """
    Run self-consistency sampling: generate N completions and aggregate.

    Args:
        model:           The fine-tuned model.
        tokenizer:       Matching tokenizer.
        prompt:          Full formatted prompt string (including system + user turns).
        n:               Number of samples.
        temperature:     Sampling temperature (0.7 gives diverse but coherent outputs).
        max_new_tokens:  Max tokens per generation.
        min_confidence:  If confidence < min_confidence, log a warning.

    Returns:
        (final_answer, confidence, all_raw_answers)

    Memory: each generation holds one beam in memory; N sequential generations
    peak at ~2× inference VRAM = ~24 GB for Qwen2.5-14B at 4-bit. [OK]
    """
    from src.inference.generate import _is_vllm_model, _vllm_generate_texts

    if _is_vllm_model(model):
        raw_answers = _vllm_generate_texts(
            model=model,
            prompts=[prompt],
            temperature=temperature,
            top_p=0.9,
            max_new_tokens=max_new_tokens,
            n=n,
        )[0]
        for i, answer in enumerate(raw_answers):
            logger.debug("Sample %d/%d: %s", i + 1, n, answer[:80])

        final, confidence = self_consistent_answer(raw_answers)
        if confidence < min_confidence:
            logger.warning(
                "Low self-consistency confidence (%.2f < %.2f). "
                "Consider returning 'Insufficient information' for this query.",
                confidence, min_confidence,
            )
        return final, confidence, raw_answers

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1900,
    ).to(model.device)

    raw_answers: List[str] = []

    model.eval()
    for i in range(n):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        raw_answers.append(answer)
        logger.debug("Sample %d/%d: %s", i + 1, n, answer[:80])

    final, confidence = self_consistent_answer(raw_answers)

    if confidence < min_confidence:
        logger.warning(
            "Low self-consistency confidence (%.2f < %.2f). "
            "Consider returning 'Insufficient information' for this query.",
            confidence, min_confidence,
        )

    return final, confidence, raw_answers


# ──────────────────────────────────────────────────────────────────────────────
# Optional verifier model (Step 5d)
# ──────────────────────────────────────────────────────────────────────────────

class AnswerVerifier:
    """
    A small verifier model (Qwen2.5-1.5B) that scores answer plausibility on 0–1.

    Usage:
        verifier = AnswerVerifier("outputs/verifier")
        scores = verifier.score_candidates(question, context, candidates)
        best = candidates[scores.index(max(scores))]

    Fine-tuning the verifier:
        1. Collect (question, context, candidate_answer, label) tuples where
           label = EM against ground truth (0.0 or 1.0), or soft score from F1.
        2. Fine-tune Qwen2.5-1.5B with a regression head (linear layer on [CLS]
           equivalent — use last token hidden state for decoder-only models).
        3. Training: MSE loss, lr=1e-4, 1–2 epochs on ~10K pairs.
        4. Expected VRAM: ~3 GB (1.5B model in BF16, no quantization needed).

    [WARN] TRADE-OFF: The verifier adds ~3 GB VRAM and latency proportional to N.
    At N=8 and 128-token answers, typical overhead is ~200ms per query.
    """

    def __init__(self, verifier_path: str, device: str = "auto") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(verifier_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            verifier_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

    def score_candidates(
        self,
        question: str,
        context: str,
        candidates: List[str],
    ) -> List[float]:
        """
        Score each candidate answer on a 0–1 plausibility scale.

        Input format: "[QUESTION] {question} [CONTEXT] {context} [ANSWER] {candidate}"
        """
        scores = []
        for candidate in candidates:
            text = (
                f"[QUESTION] {question}\n"
                f"[CONTEXT] {context[:512]}\n"
                f"[ANSWER] {candidate}"
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.model.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                score = torch.sigmoid(logits).item()
            scores.append(score)

        return scores

    def rerank_with_verifier(
        self,
        question: str,
        context: str,
        candidates: List[str],
    ) -> Tuple[str, float]:
        """
        Re-rank self-consistency candidates using verifier scores.
        Returns (best_answer, verifier_score).
        """
        scores = self.score_candidates(question, context, candidates)
        best_idx = scores.index(max(scores))
        return candidates[best_idx], scores[best_idx]


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Demo with mock answers (no GPU needed)
    parser = argparse.ArgumentParser(description="Self-consistency demo")
    parser.add_argument("--demo", action="store_true", default=True)
    args = parser.parse_args()

    print("=== Self-Consistency Demo (mock answers) ===\n")

    mock_answers = [
        "$2.5 billion",
        "2.5B",
        "$2.4 billion",
        "approximately $2.5 billion",
        "$2.5B",
        "$2.6 billion",
        "2.5 billion dollars",
        "$2.5 billion",
    ]
    final, confidence = self_consistent_answer(mock_answers)
    print(f"Answers: {mock_answers}")
    print(f"Final:   {final}")
    print(f"Confidence: {confidence:.2%}\n")

    cat_answers = [
        "The operating margin improved significantly",
        "Operating margins improved significantly",
        "Margins declined",
        "The operating margin improved significantly",
        "Operating margin improved",
    ]
    final_cat, conf_cat = self_consistent_answer(cat_answers)
    print(f"Categorical answers: {cat_answers}")
    print(f"Final:   {final_cat}")
    print(f"Confidence: {conf_cat:.2%}")
    print("\n[OK] Step 5c complete — self-consistency aggregation verified.")
