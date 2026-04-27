"""
File: src/train/dpo_train.py

Step 3e / 4e — Optional DPO (Direct Preference Optimization) Second Stage

What DPO adds for financial QA:
  - SFT teaches the model to mimic the training distribution. DPO additionally
    teaches it to PREFER grounded, accurate answers over hallucinated ones.
  - In financial contexts, the most dangerous failure mode is a confident but
    wrong number (e.g., quoting 2022 revenue for a 2023 question). DPO can
    explicitly penalize these by providing (chosen=correct, rejected=hallucinated)
    pairs derived from SFT model outputs.
  - DPO is more data-efficient than RLHF for this setting — no reward model
    needed, and preference pairs can be constructed automatically via:
      1. Sample N outputs from the SFT model for each question
      2. Use exact-match or verifier scoring to rank outputs
      3. Take best-ranked as "chosen", worst-ranked as "rejected"

[WARN] TRADE-OFF: DPO can cause the model to become overly conservative — it may
learn to output short, safe answers rather than detailed correct ones. Monitor
F1 (not just EM) after DPO. If F1 drops >5 points, reduce beta or add a KL
penalty weight.

Memory estimate (DPO with reference model):
  - SFT model (4-bit):    ~8.5 GB
  - Reference model (4-bit): ~8.5 GB (separate frozen copy)
  - LoRA adapters + grads:  ~0.6 GB
  - Optimizer + activations: ~10 GB
  Total: ~28 GB — fits A100 80GB [OK]

beta=0.1 rationale: low beta keeps the DPO policy close to the SFT reference,
avoiding catastrophic deviation. Higher beta (0.3–0.5) can be used if hallucination
is severe but risks losing fluency.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoTokenizer, TrainingArguments
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_SFT_ADAPTER_DIR = "outputs/sft_qlora/final_adapter"
DEFAULT_DPO_OUTPUT_DIR = "outputs/dpo_qlora"
DEFAULT_PREF_DATA_PATH = "data/raw/dpo_preferences.jsonl"

QLORA_CONFIG = LoraConfig(
    r=32,   # smaller r for DPO — less adaptation needed since SFT already did heavy lifting
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


# ──────────────────────────────────────────────────────────────────────────────
# Preference dataset construction
# ──────────────────────────────────────────────────────────────────────────────

def construct_preference_pairs_from_sft(
    sft_model: PeftModel,
    tokenizer: AutoTokenizer,
    eval_samples: List[dict],
    n_samples: int = 8,
    temperature: float = 0.8,
    max_new_tokens: int = 128,
    output_path: str = DEFAULT_PREF_DATA_PATH,
) -> List[dict]:
    """
    Construct DPO preference pairs by:
      1. Sampling n_samples completions per question from the SFT model.
      2. Scoring each completion against the ground-truth answer.
      3. Selecting the best completion as "chosen" and worst as "rejected".

    This requires no human annotation — fully automatic.

    Args:
        sft_model:      The trained SFT model (with LoRA adapter loaded).
        tokenizer:      Matching tokenizer.
        eval_samples:   List of raw sample dicts with "question", "answer", "context".
        n_samples:      Number of completions to sample per question.
        temperature:    Sampling temperature.
        max_new_tokens: Max tokens to generate per completion.
        output_path:    Where to save the preference JSONL.

    Returns:
        List of preference dicts with keys: prompt, chosen, rejected.
    """
    from src.data.preprocess import format_sample, SYSTEM_PROMPT
    from src.eval.evaluate import compute_exact_match, compute_f1

    sft_model.eval()
    pref_pairs = []

    logger.info("Constructing preference pairs from %d eval samples...", len(eval_samples))

    for idx, sample in enumerate(eval_samples):
        prompt = _build_prompt_only(sample, tokenizer)
        if prompt is None:
            continue

        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1800).input_ids.to(sft_model.device)

        completions = []
        for _ in range(n_samples):
            with torch.no_grad():
                output_ids = sft_model.generate(
                    prompt_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_ids = output_ids[0, prompt_ids.shape[1]:]
            completion = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            # Strip think tags for scoring
            completion_clean = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
            completions.append(completion_clean)

        # Score against ground truth
        gt_answer = str(sample.get("answer", ""))
        scores = []
        for c in completions:
            em = compute_exact_match(c, gt_answer)
            f1 = compute_f1(c, gt_answer)
            scores.append(em * 0.6 + f1 * 0.4)

        if max(scores) == min(scores):
            # All completions are equally good/bad — not useful for DPO
            continue

        best_idx = scores.index(max(scores))
        worst_idx = scores.index(min(scores))

        if scores[best_idx] <= scores[worst_idx]:
            continue

        pref_pairs.append({
            "id": str(uuid.uuid4()),
            "prompt": prompt,
            "chosen": completions[best_idx],
            "rejected": completions[worst_idx],
            "chosen_score": scores[best_idx],
            "rejected_score": scores[worst_idx],
            "task": sample.get("task", "financial_qa"),
        })

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d / %d samples, %d pairs so far.",
                        idx + 1, len(eval_samples), len(pref_pairs))

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in pref_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info("Saved %d preference pairs to %s", len(pref_pairs), output_path)
    return pref_pairs


def _build_prompt_only(sample: dict, tokenizer) -> Optional[str]:
    """Build the user+system portion of the ChatML prompt (no assistant turn)."""
    from src.data.preprocess import SYSTEM_PROMPT, COT_SYSTEM_PROMPT

    task = sample.get("task", "financial_qa")
    context = sample.get("context", "")
    question = sample.get("question", "")
    instruction = sample.get("instruction", "Answer the following financial question.")

    if not question:
        return None

    if task == "numerical_reasoning":
        var_lines = "\n".join(
            f"  {k} = {v:,.4f}"
            for k, v in sample.get("variables", {}).items()
        )
        user_content = (
            f"{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Expression: {sample.get('expression', '')}\n"
            f"Variables:\n{var_lines}"
        )
    else:
        user_content = f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}"

    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_preference_dataset(pref_data_path: str) -> Dataset:
    """Load DPO preference pairs from JSONL into a HuggingFace Dataset."""
    records = []
    with open(pref_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        raise ValueError(f"No preference pairs found in {pref_data_path}")

    # DPOTrainer expects columns: prompt, chosen, rejected
    dataset = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in records
    ])
    logger.info("Loaded %d preference pairs for DPO.", len(dataset))
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# DPO training
# ──────────────────────────────────────────────────────────────────────────────

def main(
    model_id: str = DEFAULT_MODEL_ID,
    sft_adapter_dir: str = DEFAULT_SFT_ADAPTER_DIR,
    output_dir: str = DEFAULT_DPO_OUTPUT_DIR,
    pref_data_path: str = DEFAULT_PREF_DATA_PATH,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 768,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    use_wandb: bool = False,
) -> None:
    """
    DPO fine-tuning on top of the SFT model.

    Args:
        model_id:             Base model HuggingFace ID.
        sft_adapter_dir:      Path to saved SFT LoRA adapter.
        output_dir:           Where to save DPO adapter.
        pref_data_path:       JSONL file of preference pairs.
        beta:                 DPO temperature (0.1 = stay close to reference).
        max_length:           Max total sequence length (prompt + response).
        max_prompt_length:    Max prompt length in tokens.
        num_train_epochs:     DPO epochs (1 is usually enough).

    Memory: ~28 GB VRAM (two 4-bit model copies + adapters). Fits A100 80GB [OK]
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not torch.cuda.is_available():
        raise EnvironmentError("DPO training requires a CUDA GPU.")

    # ── Load tokenizer ────────────────────────────────────────────────────
    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load SFT model (policy) ───────────────────────────────────────────
    from src.model.load_model import load_model_and_tokenizer, DEFAULT_BNB_CONFIG
    from src.model.apply_lora import apply_qlora

    policy_model, _ = load_model_and_tokenizer(model_id=model_id, bnb_config=DEFAULT_BNB_CONFIG)
    policy_model = apply_qlora(policy_model, lora_config=QLORA_CONFIG, gradient_checkpointing=True)

    # Load SFT adapter weights into the policy model
    if Path(sft_adapter_dir).exists():
        logger.info("Loading SFT adapter from %s", sft_adapter_dir)
        from peft import PeftModel
        policy_model = PeftModel.from_pretrained(
            policy_model.base_model.model, sft_adapter_dir, is_trainable=True
        )
    else:
        logger.warning(
            "SFT adapter not found at %s. Training DPO from scratch (not recommended).",
            sft_adapter_dir,
        )

    # ── Load preference dataset ───────────────────────────────────────────
    pref_dataset = load_preference_dataset(pref_data_path)

    # Split 90/10 for train/eval
    split = pref_dataset.train_test_split(test_size=0.10, seed=42)
    train_pref = split["train"]
    eval_pref = split["test"]

    # ── DPO config ────────────────────────────────────────────────────────
    import os
    report_to = ["wandb"] if use_wandb and os.environ.get("WANDB_API_KEY") else ["none"]

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        # Training
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        # Precision
        fp16=False,
        bf16=True,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        # Eval & save
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # Logging
        logging_steps=10,
        report_to=report_to,
        run_name="finreasoningai-dpo",
        seed=42,
    )

    # ── DPOTrainer ─────────────────────────────────────────────────────────
    trainer = DPOTrainer(
        model=policy_model,
        ref_model=None,   # None = TRL uses a frozen copy of the policy at start
        args=dpo_config,
        train_dataset=train_pref,
        eval_dataset=eval_pref,
        tokenizer=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────────────
    logger.info("Starting DPO training (beta=%.2f, max_length=%d)", beta, max_length)
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    final_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("[OK] DPO training complete. Adapter saved to %s", final_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DPO training for FinReasoning AI")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sft_adapter_dir", default=DEFAULT_SFT_ADAPTER_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_DPO_OUTPUT_DIR)
    parser.add_argument("--pref_data_path", default=DEFAULT_PREF_DATA_PATH)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    main(
        model_id=args.model_id,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=args.output_dir,
        pref_data_path=args.pref_data_path,
        beta=args.beta,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_wandb=args.use_wandb,
    )
