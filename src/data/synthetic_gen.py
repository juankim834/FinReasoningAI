"""
Deprecated: synthetic data generation is retained for legacy experiments.

New training runs should use `src.data.fincot_loader.load_fincot_samples`
with the public FinCoT-style datasets described in the Colab notebook.

File: src/data/synthetic_gen.py

Step 2c — Synthetic Data Generation

Generates 5,000 synthetic FinQA pairs using two methods:
  1. Template-based generation for numerical reasoning tasks (Type B) — fast,
     deterministic, zero GPU cost. Produces arithmetically correct ground truth.
  2. LLM-driven generation using a local Qwen2.5-7B model for Type A financial
     QA pairs. Constrained prompts keep outputs in the financial domain.

Quality filtering:
  - Length filter: context ≥ 50 tokens, answer ≥ 1 token, answer ≤ 100 tokens.
  - Answer parsability: numerical answers must be extractable by regex.
  - Deduplication: SimHash on (question, answer) pairs; drop if similarity > 0.9.

[WARN] TRADE-OFF: LLM-generated data quality depends on the generator model. Using
Qwen2.5-7B as generator for 14B training can introduce systematic biases if the
7B model has incorrect financial knowledge. Always review a random 5% sample
manually before adding to the final training mix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Template-based numerical generation (Type B)
# ──────────────────────────────────────────────────────────────────────────────

NUMERICAL_TEMPLATES = [
    # (template_str, expression, unit)
    (
        "What is the year-over-year revenue growth rate for {company} "
        "from {year_a} to {year_b}?",
        "(revenue_{year_b} - revenue_{year_a}) / revenue_{year_a}",
        "%",
    ),
    (
        "Calculate the gross profit margin for {company} in {year_b} "
        "given revenue of ${revenue_b:,.0f}M and COGS of ${cogs_b:,.0f}M.",
        "(revenue_{year_b} - cogs_{year_b}) / revenue_{year_b}",
        "%",
    ),
    (
        "What is the EBITDA margin of {company} in {year_b}?",
        "ebitda_{year_b} / revenue_{year_b}",
        "%",
    ),
    (
        "By how many basis points did the operating margin change for {company} "
        "between {year_a} and {year_b}?",
        "((op_income_{year_b} / revenue_{year_b}) - (op_income_{year_a} / revenue_{year_a})) * 10000",
        "bps",
    ),
    (
        "What is the debt-to-equity ratio for {company} in {year_b}?",
        "total_debt_{year_b} / total_equity_{year_b}",
        "x",
    ),
    (
        "Calculate the compound annual growth rate (CAGR) of {company}'s revenue "
        "over {n_years} years.",
        "(revenue_{year_b} / revenue_{year_a}) ** (1 / {n_years}) - 1",
        "%",
    ),
    (
        "What percentage of revenue did {company} spend on R&D in {year_b}?",
        "rnd_{year_b} / revenue_{year_b}",
        "%",
    ),
    (
        "What is the free cash flow of {company} in {year_b}?",
        "operating_cf_{year_b} - capex_{year_b}",
        "$M",
    ),
]

COMPANIES = [
    "Apple", "Microsoft", "Amazon", "Alphabet", "Meta",
    "Tesla", "NVIDIA", "Berkshire Hathaway", "Johnson & Johnson",
    "JPMorgan Chase", "Goldman Sachs", "Morgan Stanley", "Visa", "Mastercard",
]

CONTEXTS = [
    (
        "{company} reported revenue of ${revenue_b:,.0f}M in {year_b}, up from "
        "${revenue_a:,.0f}M in {year_a}. Cost of goods sold was ${cogs_b:,.0f}M in {year_b}. "
        "Operating income reached ${op_income_b:,.0f}M, and EBITDA was ${ebitda_b:,.0f}M."
    ),
    (
        "From {company}'s {year_b} Annual Report: Total revenues were ${revenue_b:,.0f}M "
        "(vs ${revenue_a:,.0f}M in {year_a}). R&D expenses totaled ${rnd_b:,.0f}M. "
        "Free cash flow was ${fcf_b:,.0f}M (operating CF: ${ocf_b:,.0f}M, capex: ${capex_b:,.0f}M). "
        "Total debt: ${debt_b:,.0f}M, total equity: ${equity_b:,.0f}M."
    ),
]


def _random_financials(seed: Optional[int] = None) -> dict:
    """Generate internally consistent random financial figures."""
    rng = random.Random(seed)
    year_a = rng.choice([2020, 2021, 2022])
    year_b = year_a + rng.randint(1, 2)
    n_years = year_b - year_a

    revenue_a = rng.uniform(1_000, 200_000)   # $M
    growth = rng.uniform(-0.05, 0.30)
    revenue_b = revenue_a * (1 + growth)

    gross_margin = rng.uniform(0.25, 0.80)
    cogs_b = revenue_b * (1 - gross_margin)
    ebitda_margin = rng.uniform(0.10, 0.45)
    ebitda_b = revenue_b * ebitda_margin
    op_margin_b = ebitda_margin - rng.uniform(0.02, 0.08)
    op_income_b = revenue_b * op_margin_b
    op_income_a = revenue_a * (op_margin_b + rng.uniform(-0.05, 0.05))
    rnd_b = revenue_b * rng.uniform(0.02, 0.20)
    capex_b = revenue_b * rng.uniform(0.03, 0.12)
    ocf_b = ebitda_b * rng.uniform(0.70, 0.95)
    fcf_b = ocf_b - capex_b
    debt_b = revenue_b * rng.uniform(0.20, 1.50)
    equity_b = revenue_b * rng.uniform(0.30, 2.00)

    return {
        "year_a": year_a, "year_b": year_b, "n_years": n_years,
        "revenue_a": revenue_a, "revenue_b": revenue_b,
        "cogs_b": cogs_b, "ebitda_b": ebitda_b,
        "op_income_a": op_income_a, "op_income_b": op_income_b,
        "rnd_b": rnd_b, "capex_b": capex_b, "ocf_b": ocf_b, "fcf_b": fcf_b,
        "debt_b": debt_b, "equity_b": equity_b,
    }


def _evaluate_expression(expr: str, variables: dict) -> float:
    """Safely evaluate a financial expression given variable bindings."""
    # Build a local namespace from flattened variables
    ns: dict = {}
    for k, v in variables.items():
        ns[k.replace("-", "_")] = v
    try:
        result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307 — controlled namespace
        return float(result)
    except Exception as exc:
        logger.debug("Expression eval failed: %s — %s", expr, exc)
        return float("nan")


def generate_numerical_samples(n: int = 1500, seed: int = 42) -> List[dict]:
    """
    Generate n Type B (numerical reasoning) samples via templates.
    Returns a list of raw dicts matching the NumericalReasoning schema.
    """
    samples = []
    rng = random.Random(seed)
    template_pool = NUMERICAL_TEMPLATES

    for i in range(n):
        company = rng.choice(COMPANIES)
        fin = _random_financials(seed=seed + i)

        tmpl_q, expr_tmpl, unit = rng.choice(template_pool)

        # Substitute human-readable fields into question
        fin_with_company = {"company": company, **fin}
        try:
            question = tmpl_q.format(**fin_with_company)
        except KeyError:
            continue

        # Build variable dict with year-suffixed keys matching the expression
        ya, yb = fin["year_a"], fin["year_b"]
        variables = {
            f"revenue_{ya}": fin["revenue_a"],
            f"revenue_{yb}": fin["revenue_b"],
            f"cogs_{yb}": fin["cogs_b"],
            f"ebitda_{yb}": fin["ebitda_b"],
            f"op_income_{ya}": fin["op_income_a"],
            f"op_income_{yb}": fin["op_income_b"],
            f"rnd_{yb}": fin["rnd_b"],
            f"capex_{yb}": fin["capex_b"],
            f"operating_cf_{yb}": fin["ocf_b"],
            f"total_debt_{yb}": fin["debt_b"],
            f"total_equity_{yb}": fin["equity_b"],
        }

        # Resolve year placeholders in expression
        expr = expr_tmpl.replace("{year_a}", str(ya)).replace("{year_b}", str(yb))
        expr = expr.replace("{n_years}", str(fin["n_years"]))

        answer_val = _evaluate_expression(expr, variables)
        if answer_val != answer_val:  # NaN check
            continue

        # Format answer with unit-aware rounding
        if unit == "%":
            answer = f"{answer_val * 100:.2f}%"
        elif unit == "bps":
            answer = f"{answer_val:.1f} bps"
        else:
            answer = f"{answer_val:.2f}"

        context_tmpl = rng.choice(CONTEXTS)
        context = context_tmpl.format(**fin_with_company)

        samples.append({
            "id": str(uuid.uuid4()),
            "task": "numerical_reasoning",
            "instruction": "Evaluate the following financial expression and provide the numerical result.",
            "expression": expr,
            "variables": variables,
            "answer": answer_val if unit not in ("%", "bps") else answer,
            "unit": unit,
            # Include context so preprocessing can form a complete prompt
            "context": context,
            "question": question,
        })

    logger.info("Generated %d numerical reasoning samples.", len(samples))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Template-based structured analysis (Type C)
# ──────────────────────────────────────────────────────────────────────────────


def generate_structured_analysis_samples(n: int = 500, seed: int = 42) -> List[dict]:
    """
    Generate Type C (structured_analysis) samples from markdown tables + KPI dicts.
    No LLM required — templates only, aligned with StructuredAnalysis / preprocess.
    """
    rng = random.Random(seed)
    samples: List[dict] = []

    for i in range(n):
        company = rng.choice(COMPANIES)
        fin = _random_financials(seed=seed + i + 50000)
        ya, yb = fin["year_a"], fin["year_b"]
        ebitda_a = fin["revenue_a"] * (fin["ebitda_b"] / max(fin["revenue_b"], 1e-9))
        cogs_a = fin["revenue_a"] * (fin["cogs_b"] / max(fin["revenue_b"], 1e-9))
        gp_a = (1.0 - cogs_a / max(fin["revenue_a"], 1e-9)) * 100
        gp_b = (1.0 - fin["cogs_b"] / max(fin["revenue_b"], 1e-9)) * 100
        debt_eq_a = fin["debt_b"] / max(fin["equity_b"], 1e-9) * rng.uniform(0.85, 1.05)
        debt_eq_b = fin["debt_b"] / max(fin["equity_b"], 1e-9)

        financial_data = {
            "company": company,
            "years": [ya, yb],
            "revenue_m": {str(ya): round(fin["revenue_a"], 2), str(yb): round(fin["revenue_b"], 2)},
            "ebitda_m": {str(ya): round(ebitda_a, 2), str(yb): round(fin["ebitda_b"], 2)},
            "gross_margin_pct": {str(ya): round(gp_a, 2), str(yb): round(gp_b, 2)},
            "debt_to_equity": {str(ya): round(debt_eq_a, 3), str(yb): round(debt_eq_b, 3)},
        }

        md_lines = [
            f"### {company} — summary financials ($M except margins and ratios)",
            "",
            f"| Metric | {ya} | {yb} |",
            "|---|---:|---:|",
            f"| Revenue | {fin['revenue_a']:,.1f} | {fin['revenue_b']:,.1f} |",
            f"| EBITDA | {ebitda_a:,.1f} | {fin['ebitda_b']:,.1f} |",
            f"| Gross margin (%) | {gp_a:.2f} | {gp_b:.2f} |",
            f"| Debt / equity (x) | {debt_eq_a:.3f} | {debt_eq_b:.3f} |",
        ]
        context = "\n".join(md_lines)

        variant = rng.randint(0, 3)
        fmt: str = rng.choice(["paragraph", "bullet"])
        if variant == 0:
            higher_rev = yb if fin["revenue_b"] >= fin["revenue_a"] else ya
            question = (
                f"Using the table, which fiscal year had higher reported revenue for {company}, "
                f"{ya} or {yb}, and what were the approximate revenue figures ($M)?"
            )
            answer = (
                f"{higher_rev} had higher revenue: approximately "
                f"${fin['revenue_a']:,.0f}M in {ya} vs ${fin['revenue_b']:,.0f}M in {yb}."
            )
            unit = "$M"
            reasoning = (
                f"Compare the Revenue row for {ya} and {yb}. "
                f"The larger value corresponds to the year with higher revenue."
            )
        elif variant == 1:
            margin_trend = "improved" if gp_b >= gp_a else "weakened"
            question = (
                f"How did {company}'s gross margin (%) move from {ya} to {yb} according to the table?"
            )
            answer = (
                f"Gross margin {margin_trend}: from {gp_a:.2f}% in {ya} to {gp_b:.2f}% in {yb}."
            )
            unit = "%"
            reasoning = (
                f"Read Gross margin row: {gp_a:.2f}% vs {gp_b:.2f}%. "
                f"Determine direction of change and state both levels."
            )
        elif variant == 2:
            eb_a, eb_b = ebitda_a, fin["ebitda_b"]
            higher_ebitda = yb if eb_b >= eb_a else ya
            diff = abs(eb_b - eb_a)
            question = (
                f"In which year was EBITDA higher for {company}, and by about how many $M?"
            )
            answer = (
                f"EBITDA was higher in {higher_ebitda}: "
                f"${eb_a:,.0f}M in {ya} vs ${eb_b:,.0f}M in {yb} "
                f"(difference ~${diff:,.0f}M)."
            )
            unit = "$M"
            reasoning = (
                "Compare EBITDA values across the two columns, identify the larger year, "
                "then subtract to approximate the spread."
            )
        else:
            question = (
                f"What is the direction of the debt-to-equity ratio for {company} from {ya} to {yb}?"
            )
            direction = "flat to slightly lower" if debt_eq_b <= debt_eq_a else "higher"
            answer = (
                f"Debt/equity moved from {debt_eq_a:.3f}x in {ya} to {debt_eq_b:.3f}x in {yb} "
                f"— overall {direction} versus the prior year in this synthetic snapshot."
            )
            unit = "x"
            reasoning = (
                "Inspect the Debt / equity row and compare the two fiscal years numerically."
            )

        samples.append({
            "id": str(uuid.uuid4()),
            "task": "structured_analysis",
            "instruction": "Analyze the following financial data and answer the question.",
            "context": context,
            "question": question,
            "answer": answer,
            "reasoning": reasoning,
            "financial_data": financial_data,
            "format": fmt,
            "unit": unit,
        })

    logger.info("Generated %d structured_analysis samples.", len(samples))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# LLM-driven FinQA generation (Type A) using Qwen2.5-7B
# ──────────────────────────────────────────────────────────────────────────────

FINQA_GENERATOR_PROMPT = """\
<|im_start|>system
You are a financial analyst creating training data. Generate a realistic financial question and its concise answer based on the context below. Output ONLY valid JSON with keys: "question", "answer". Do not include any chain-of-thought or explanation — only the final answer.
<|im_end|>
<|im_start|>user
Context:
{context}

Generate one question-answer pair about a specific numerical fact or relationship in this context. The answer must be extractable from the context. Output JSON only.
<|im_end|>
<|im_start|>assistant
"""

SYNTHETIC_CONTEXTS = [
    "{company} reported total revenues of ${rev_b:.1f} billion for fiscal year {year_b}, "
    "compared to ${rev_a:.1f} billion in {year_a}, representing growth of {growth:.1f}%. "
    "Net income was ${ni_b:.1f} billion, with earnings per share of ${eps_b:.2f}. "
    "The company's operating margin was {op_margin:.1f}% and free cash flow totaled "
    "${fcf_b:.1f} billion.",

    "In its {year_b} annual report, {company} disclosed that its gross profit margin "
    "expanded by {margin_exp:.0f} basis points to {gp_margin:.1f}%, driven by cost "
    "efficiencies. Revenue grew {growth:.1f}% year-over-year to ${rev_b:.1f} billion. "
    "Capital expenditures were ${capex_b:.1f} billion, up {capex_growth:.1f}% from {year_a}.",

    "{company}'s balance sheet as of December 31, {year_b} showed total assets of "
    "${assets_b:.1f} billion and total debt of ${debt_b:.1f} billion. The debt-to-equity "
    "ratio stood at {de_ratio:.2f}x. Cash and equivalents were ${cash_b:.1f} billion, "
    "providing {months:.1f} months of operating expense coverage.",
]


def _generate_llm_samples(
    n: int = 3500,
    generator_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 8,
    seed: int = 42,
) -> List[dict]:
    """
    Generate Type A FinQA samples using a local Qwen2.5-7B model.
    Falls back to template-only mode if no GPU is available.

    [WARN] TRADE-OFF: Requires ~16 GB VRAM for the 7B generator (separate from the
    14B training model). Run generation on a separate machine or time-share the GPU.
    """

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if not torch.cuda.is_available():
        logger.warning(
            "No GPU available for LLM-based generation. "
            "Returning empty list — use template generation only."
        )
        return []

    logger.info("Loading generator model: %s", generator_model_id)
    tokenizer = AutoTokenizer.from_pretrained(generator_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        generator_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    rng = random.Random(seed)
    samples = []

    for i in range(n):
        company = rng.choice(COMPANIES)
        fin = _random_financials(seed=seed + i + 10000)
        ya, yb = fin["year_a"], fin["year_b"]

        ctx_tmpl = rng.choice(SYNTHETIC_CONTEXTS)
        growth = (fin["revenue_b"] - fin["revenue_a"]) / fin["revenue_a"] * 100
        gp_margin = (1 - fin["cogs_b"] / fin["revenue_b"]) * 100
        op_margin = fin["op_income_b"] / fin["revenue_b"] * 100

        try:
            context = ctx_tmpl.format(
                company=company,
                year_a=ya, year_b=yb,
                rev_a=fin["revenue_a"] / 1000,
                rev_b=fin["revenue_b"] / 1000,
                growth=growth,
                ni_b=fin["ebitda_b"] * 0.65 / 1000,
                eps_b=rng.uniform(1.0, 15.0),
                op_margin=op_margin,
                fcf_b=fin["fcf_b"] / 1000,
                margin_exp=rng.uniform(50, 300),
                gp_margin=gp_margin,
                capex_b=fin["capex_b"] / 1000,
                capex_growth=rng.uniform(-10, 30),
                assets_b=(fin["revenue_b"] * 1.5) / 1000,
                debt_b=fin["debt_b"] / 1000,
                de_ratio=fin["debt_b"] / max(fin["equity_b"], 1),
                cash_b=fin["ocf_b"] * 0.3 / 1000,
                months=rng.uniform(3, 18),
            )
        except (KeyError, ZeroDivisionError):
            continue

        prompt = FINQA_GENERATOR_PROMPT.format(context=context)

        try:
            output = gen_pipeline(prompt)[0]["generated_text"]
            # Extract only the assistant turn
            assistant_text = output[len(prompt):].strip()
            # Parse JSON
            json_match = re.search(r"\{.*?\}", assistant_text, re.DOTALL)
            if not json_match:
                continue
            qa = json.loads(json_match.group())
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
        except (json.JSONDecodeError, Exception):
            continue

        if not _passes_quality_filter(question, answer, context):
            continue

        samples.append({
            "id": str(uuid.uuid4()),
            "task": "financial_qa",
            "instruction": "Answer the following financial question based on the provided context.",
            "context": context,
            "question": question,
            "answer": answer,
            "reasoning": None,
        })

        if (i + 1) % 100 == 0:
            logger.info("Generated %d / %d LLM samples", len(samples), n)

    logger.info("LLM generation complete: %d samples produced.", len(samples))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Quality filtering & deduplication
# ──────────────────────────────────────────────────────────────────────────────

_ANSWER_PARSABLE_RE = re.compile(
    r"[\$]?\s*-?\d[\d,]*\.?\d*\s*(%|billion|million|thousand|bps|x|M|B)?",
    re.IGNORECASE,
)


def _passes_quality_filter(question: str, answer: str, context: str) -> bool:
    """
    Apply quality filters:
      1. Length filter: context ≥ 50 chars, answer between 1 and 200 chars.
      2. Answer parsability: must contain a number or clear entity.
      3. Basic overlap: question tokens must have ≥1 match in context.
    """
    if len(context) < 50:
        return False
    if not (1 <= len(answer) <= 200):
        return False
    if not question.strip().endswith("?"):
        return False
    # At least one extractable numeric or entity in the answer
    if not (_ANSWER_PARSABLE_RE.search(answer) or len(answer.split()) <= 8):
        return False
    return True


def _simhash(text: str) -> int:
    """64-bit SimHash fingerprint for near-duplicate detection."""
    tokens = text.lower().split()
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324
        for i in range(64):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(64) if v[i] > 0)


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate(samples: List[dict], threshold: int = 10) -> List[dict]:
    """
    Remove near-duplicate samples using SimHash on (question + answer).
    threshold: max Hamming distance to be considered a duplicate (lower = stricter).
    """
    fingerprints: List[int] = []
    unique: List[dict] = []
    for s in samples:
        fp = _simhash(s.get("question", "") + " " + str(s.get("answer", "")))
        is_dup = any(_hamming_distance(fp, existing) <= threshold for existing in fingerprints)
        if not is_dup:
            fingerprints.append(fp)
            unique.append(s)
    removed = len(samples) - len(unique)
    logger.info("Deduplication: removed %d duplicates, kept %d samples.", removed, len(unique))
    return unique


def deduplicate_exact_question_answer(samples: List[dict]) -> List[dict]:
    """
    Drop exact duplicates on (question, answer) only. Template-heavy synthetic data
    is often over-collapsed by SimHash; this keeps numerically distinct rows.
    """
    seen: Set[Tuple[str, str]] = set()
    unique: List[dict] = []
    for s in samples:
        key = (s.get("question", "").strip().lower(), str(s.get("answer", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    removed = len(samples) - len(unique)
    logger.info(
        "Exact-QA dedup: removed %d duplicate rows, kept %d samples.", removed, len(unique)
    )
    return unique


# ──────────────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_synthetic_data(
    output_path: str = "data/raw/synthetic.jsonl",
    total_target: int = 5000,
    numerical_fraction: float = 0.30,
    structured_analysis_fraction: float = 0.10,
    use_llm: bool = True,
    generator_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    seed: int = 42,
) -> List[dict]:
    """
    Generate a full synthetic dataset and save to JSONL.

    Args:
        output_path:                  Where to write the JSONL file.
        total_target:                 Target number of samples after assembly.
        numerical_fraction:           Fraction of samples that are numerical (Type B).
        structured_analysis_fraction: Fraction for Type C (structured_analysis).
        use_llm:                      Whether to use the LLM generator for Type A samples.
        generator_model_id:         Qwen2.5-7B model ID for LLM generation.
        seed:                         Random seed for reproducibility.

    Returns:
        List of sample dicts.
    """
    n_structured = int(round(total_target * structured_analysis_fraction))
    n_numerical = int(round(total_target * numerical_fraction))
    n_finqa = total_target - n_numerical - n_structured
    if n_finqa < 0:
        raise ValueError(
            "numerical_fraction + structured_analysis_fraction must not exceed 1.0 "
            f"(got {numerical_fraction} + {structured_analysis_fraction})."
        )

    oversample = 5
    logger.info(
        "Target mix: %d financial_qa + %d numerical_reasoning + %d structured_analysis (= %d).",
        n_finqa, n_numerical, n_structured, total_target,
    )

    numerical_samples = generate_numerical_samples(n=max(n_numerical * oversample, 100), seed=seed)

    structured_samples: List[dict] = []
    try:
        structured_samples = generate_structured_analysis_samples(
            n=max(n_structured * oversample, 100), seed=seed
        )
    except Exception as exc:
        print(
            f"[structured_analysis] generation raised: {type(exc).__name__}: {exc}"
        )
        import traceback
        traceback.print_exc()

    if use_llm:
        llm_samples = _generate_llm_samples(
            n=max(n_finqa * oversample, 100),
            generator_model_id=generator_model_id,
            seed=seed,
        )
    else:
        llm_samples = _template_finqa_fallback(n=max(n_finqa * oversample, 100), seed=seed)

    def _qf(s: dict) -> bool:
        return _passes_quality_filter(
            s.get("question", ""),
            str(s.get("answer", "")),
            s.get("context", ""),
        )

    num_f = [s for s in numerical_samples if _qf(s)]
    str_f = [s for s in structured_samples if _qf(s)]
    fin_f = [s for s in llm_samples if _qf(s)]

    print(
        "After quality filter — "
        f"numerical_reasoning: {len(num_f)}, structured_analysis: {len(str_f)}, "
        f"financial_qa: {len(fin_f)} | "
        f"{dict(Counter(s['task'] for s in num_f + str_f + fin_f))}"
    )

    num_d = deduplicate_exact_question_answer(num_f)
    str_d = deduplicate_exact_question_answer(str_f)
    fin_d = deduplicate_exact_question_answer(fin_f)

    print(
        "After dedup — "
        f"numerical_reasoning: {len(num_d)}, structured_analysis: {len(str_d)}, "
        f"financial_qa: {len(fin_d)} | "
        f"{dict(Counter(s['task'] for s in num_d + str_d + fin_d))}"
    )

    rng = random.Random(seed)
    for pool in (num_d, str_d, fin_d):
        rng.shuffle(pool)

    num_take = num_d[:n_numerical]
    str_take = str_d[:n_structured]
    fin_take = fin_d[:n_finqa]
    all_samples = num_take + str_take + fin_take
    rng.shuffle(all_samples)

    task_counts = Counter(s["task"] for s in all_samples)
    expected_tasks = {"financial_qa", "numerical_reasoning", "structured_analysis"}
    missing = expected_tasks - set(task_counts.keys())

    if missing:
        raise RuntimeError(
            f"Data generation produced 0 samples for task type(s): {sorted(missing)}. "
            "Check the generator for that task type before continuing. "
            f"Got: {dict(task_counts)}"
        )

    shortfall_pct = (total_target - len(all_samples)) / total_target
    if shortfall_pct > 0.10:
        raise RuntimeError(
            f"Data generation shortfall: expected {total_target} samples, got {len(all_samples)} "
            f"({shortfall_pct:.0%} short). Check generator logs above for errors."
        )

    if len(all_samples) != total_target:
        raise RuntimeError(
            f"Stratified assembly could not fill {total_target} samples (got {len(all_samples)}). "
            f"Pool sizes after dedup: numerical={len(num_d)}, structured={len(str_d)}, "
            f"financial_qa={len(fin_d)}; targets: {n_numerical}, {n_structured}, {n_finqa}."
        )

    print(f"Generation complete: {len(all_samples)} samples — {dict(task_counts)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, default=str) + "\n")

    written = 0
    read_back_tasks: Counter = Counter()
    with Path(output_path).open("r", encoding="utf-8") as rf:
        for line in rf:
            if not line.strip():
                continue
            written += 1
            read_back_tasks[json.loads(line)["task"]] += 1
    assert written == len(all_samples), (
        f"Write verification failed: wrote {len(all_samples)} samples but "
        f"file contains {written} lines. Possible disk/symlink issue."
    )
    print(f"Verified: {written} samples written to {output_path}")
    print(f"After read-back — task mix from file: {dict(read_back_tasks)}")

    logger.info("Saved %d synthetic samples to %s", len(all_samples), output_path)
    return all_samples


def _template_finqa_fallback(n: int = 3500, seed: int = 42) -> List[dict]:
    """
    Fallback template-based FinQA generation when no GPU is available.
    Creates simple fill-in-the-blank QA pairs from financial contexts.
    """
    rng = random.Random(seed)
    samples = []

    fallback_questions = [
        ("What was {company}'s total revenue in {year_b}?",
         "${rev_b:.1f} billion"),
        ("How much did {company}'s revenue grow from {year_a} to {year_b}?",
         "{growth:.1f}%"),
        ("What was {company}'s operating margin in {year_b}?",
         "{op_margin:.1f}%"),
        ("What was {company}'s free cash flow in {year_b}?",
         "${fcf_b:.1f} billion"),
        ("What was {company}'s gross profit margin in {year_b}?",
         "{gp_margin:.1f}%"),
    ]

    for i in range(n):
        company = rng.choice(COMPANIES)
        fin = _random_financials(seed=seed + i + 20000)
        ya, yb = fin["year_a"], fin["year_b"]
        growth = (fin["revenue_b"] - fin["revenue_a"]) / fin["revenue_a"] * 100
        gp_margin = (1 - fin["cogs_b"] / fin["revenue_b"]) * 100
        op_margin = fin["op_income_b"] / fin["revenue_b"] * 100

        ctx_tmpl = rng.choice(SYNTHETIC_CONTEXTS)
        try:
            context = ctx_tmpl.format(
                company=company, year_a=ya, year_b=yb,
                rev_a=fin["revenue_a"] / 1000, rev_b=fin["revenue_b"] / 1000,
                growth=growth, ni_b=fin["ebitda_b"] * 0.65 / 1000,
                eps_b=rng.uniform(1.0, 15.0), op_margin=op_margin,
                fcf_b=fin["fcf_b"] / 1000, margin_exp=rng.uniform(50, 300),
                gp_margin=gp_margin, capex_b=fin["capex_b"] / 1000,
                capex_growth=rng.uniform(-10, 30),
                assets_b=(fin["revenue_b"] * 1.5) / 1000,
                debt_b=fin["debt_b"] / 1000,
                de_ratio=fin["debt_b"] / max(fin["equity_b"], 1),
                cash_b=fin["ocf_b"] * 0.3 / 1000, months=rng.uniform(3, 18),
            )
        except (KeyError, ZeroDivisionError):
            continue

        q_tmpl, a_tmpl = rng.choice(fallback_questions)
        try:
            question = q_tmpl.format(
                company=company, year_a=ya, year_b=yb,
                rev_b=fin["revenue_b"] / 1000, growth=growth,
                op_margin=op_margin, fcf_b=fin["fcf_b"] / 1000, gp_margin=gp_margin,
            )
            answer = a_tmpl.format(
                company=company, year_a=ya, year_b=yb,
                rev_b=fin["revenue_b"] / 1000, growth=growth,
                op_margin=op_margin, fcf_b=fin["fcf_b"] / 1000, gp_margin=gp_margin,
            )
        except KeyError:
            continue

        samples.append({
            "id": str(uuid.uuid4()),
            "task": "financial_qa",
            "instruction": "Answer the following financial question based on the provided context.",
            "context": context,
            "question": question,
            "answer": answer,
            "reasoning": None,
        })

    logger.info("Template fallback: generated %d FinQA samples.", len(samples))
    return samples


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic FinReasoning training data")
    parser.add_argument("--output", default="data/raw/synthetic.jsonl")
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--numerical-fraction", type=float, default=0.30)
    parser.add_argument("--structured-fraction", type=float, default=0.10)
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM generation (use template fallback only)")
    parser.add_argument("--generator", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test-structured",
        type=int,
        metavar="N",
        help="Generate N structured_analysis samples only, print JSON, and exit.",
    )
    args = parser.parse_args()

    if args.test_structured is not None:
        print(f"--- Isolation test: {args.test_structured} structured_analysis samples ---\n")
        for row in generate_structured_analysis_samples(n=args.test_structured, seed=args.seed):
            print(json.dumps(row, indent=2, default=str))
            print()
        raise SystemExit(0)

    samples = generate_all_synthetic_data(
        output_path=args.output,
        total_target=args.total,
        numerical_fraction=args.numerical_fraction,
        structured_analysis_fraction=args.structured_fraction,
        use_llm=not args.no_llm,
        generator_model_id=args.generator,
        seed=args.seed,
    )
    print(f"\n[OK] Step 2c complete — {len(samples)} synthetic samples saved to {args.output}")
