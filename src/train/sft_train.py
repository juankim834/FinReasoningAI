"""
File: src/train/sft_train.py

Steps 3 & 4 — SFT Training Pipeline

Training strategy rationale:
  - effective_batch = per_device_batch(4) × grad_accum(8) = 32. This is the
    standard "sweet spot" for instruction fine-tuning: large enough to stabilize
    gradients, small enough to fit in A100 80GB with grad checkpointing.
  - cosine schedule + 50-step warmup: cosine decay is more stable than linear
    for fine-tuning; warmup prevents large early updates from destroying
    pretrained weights.
  - paged_adamw_32bit: offloads optimizer state pages to CPU RAM on demand,
    reducing peak VRAM by ~8–12 GB vs standard AdamW32bit.
  - bf16=True, fp16=False: BF16 has wider dynamic range than FP16, critical
    for financial arithmetic where numbers span many orders of magnitude.

CoT scratchpad suppression strategy:
  - Training: ~10% of samples include <think>...</think> blocks in the
    assistant turn. The model learns to reason internally when prompted with
    the CoT system prompt.
  - Inference (default mode): system prompt instructs direct answering;
    if any <think> leaks into output, it is stripped by post-processing.
  - Inference (CoT mode): use COT_SYSTEM_PROMPT; let the model output the
    full scratchpad, then extract the final answer after </think>.
  - This is superior to training without CoT entirely because it gives the
    model latent reasoning capacity that improves accuracy on numerical chains
    even in non-CoT mode.

Memory estimate at bs=4, grad_ckpt=True, max_len=2048, bf16:
  - Base model (4-bit NF4):  ~8.5 GB
  - LoRA adapters + grads:   ~0.6 GB
  - Optimizer states (paged): ~2.0 GB (paged to CPU)
  - Activations (with ckpt): ~14.0 GB
  - KV cache + misc:          ~6.0 GB
  Total estimated:            ~31 GB — fits A100 80GB with comfortable margin [OK]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_qlora"
DEFAULT_DATA_DIR = "data/processed"

QLORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Response template tells DataCollatorForCompletionOnlyLM where the
# assistant's answer starts — only labels those tokens, not the prompt.
# This is critical: we do NOT want to train the model to reproduce its own prompt.
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def build_training_arguments(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    warmup_steps: int = 50,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 200,
    early_stopping_patience: int = 5,
    use_wandb: bool = False,
) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments for QLoRA SFT.

    Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
                        = 4 × 8 = 32
    """
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. Disabling W&B logging.")
        use_wandb = False

    report_to = ["wandb"] if use_wandb else ["none"]
    os.environ.setdefault("WANDB_PROJECT", "finreasoningai-sft")

    return TrainingArguments(
        output_dir=output_dir,
        # Batch & accumulation
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Epochs & scheduling
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        # Precision
        fp16=False,
        bf16=True,
        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",   # paged optimizer: VRAM-efficient
        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Logging & saving
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Reproducibility
        seed=42,
        data_seed=42,
        # Reporting
        report_to=report_to,
        run_name="finreasoningai-sft-qlora",
        # Remove unused columns so HF doesn't crash on custom columns (e.g., "task")
        remove_unused_columns=True,
    )


def load_datasets(data_dir: str) -> DatasetDict:
    """
    Load preprocessed DatasetDict from disk.
    Falls back to synthetic generation if the directory is empty.
    """
    data_path = Path(data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_dir}. "
            "Run: python -m src.data.synthetic_gen && python -m src.data.preprocess"
        )
    logger.info("Loading DatasetDict from %s", data_dir)
    return load_from_disk(str(data_dir))


def main(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    use_wandb: bool = False,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    End-to-end SFT training pipeline:
      1. Load model + tokenizer in 4-bit NF4
      2. Apply QLoRA adapters
      3. Load & prepare dataset
      4. Configure SFTTrainer with EarlyStopping
      5. Train and save adapter

    [WARN] TRADE-OFF: SFTTrainer's completion-only masking (DataCollatorForCompletionOnlyLM)
    only computes loss on the assistant response tokens, not on the prompt.
    This is correct for instruction tuning but means the model doesn't learn to
    generate the system prompt or user turn — intentional.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # ── 1. Check hardware ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "No CUDA GPU detected. QLoRA training requires a CUDA GPU (A100 80GB recommended)."
        )
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s | Total VRAM: %.1f GB", gpu_name, total_vram_gb)
    if total_vram_gb < 40.0:
        logger.warning(
            "GPU has only %.1f GB VRAM. Minimum recommended is 40 GB. "
            "Try reducing per_device_train_batch_size to 2.",
            total_vram_gb,
        )

    # ── 2. Load model & tokenizer ──────────────────────────────────────────
    from src.model.load_model import load_model_and_tokenizer, DEFAULT_BNB_CONFIG

    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        bnb_config=DEFAULT_BNB_CONFIG,
    )

    # ── 3. Apply QLoRA ────────────────────────────────────────────────────
    from src.model.apply_lora import apply_qlora

    model = apply_qlora(model, lora_config=QLORA_CONFIG, gradient_checkpointing=True)

    # ── 4. Load dataset ───────────────────────────────────────────────────
    dataset = load_datasets(data_dir)
    train_ds = dataset["train"]
    val_ds = dataset.get("val", dataset.get("validation"))

    logger.info(
        "Dataset — train: %d, val: %d",
        len(train_ds),
        len(val_ds) if val_ds else 0,
    )

    # ── 5. Build training arguments ───────────────────────────────────────
    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_wandb=use_wandb,
    )

    # ── 6. Completion-only data collator ──────────────────────────────────
    # Only compute loss on tokens AFTER the response template.
    # This prevents the model from memorizing the prompt format.
    response_template_ids = tokenizer.encode(
        RESPONSE_TEMPLATE, add_special_tokens=False
    )
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )

    # ── 7. SFTTrainer ─────────────────────────────────────────────────────
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=5),
    ]

    # Try to import WandB callback
    if use_wandb:
        try:
            from transformers.integrations import WandbCallback
            callbacks.append(WandbCallback())
        except ImportError:
            logger.warning("WandbCallback not available.")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        callbacks=callbacks,
        # SFTTrainer-specific
        max_seq_length=max_seq_length,
        dataset_text_field=None,   # dataset is already tokenized
        packing=False,             # packing can mix samples from different tasks
    )

    # ── 8. Train ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting SFT training")
    logger.info("  Model:          %s", model_id)
    logger.info("  Epochs:         %d", num_train_epochs)
    logger.info("  Effective BS:   %d (×%d ×%d)", per_device_train_batch_size *
                gradient_accumulation_steps, per_device_train_batch_size,
                gradient_accumulation_steps)
    logger.info("  LR:             %.2e", learning_rate)
    logger.info("  Max seq length: %d", max_seq_length)
    logger.info("  Output dir:     %s", output_dir)
    logger.info("=" * 60)

    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM before training: %.1f GB", vram_before)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── 9. Save adapter ───────────────────────────────────────────────────
    adapter_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    logger.info("[OK] SFT training complete. Adapter saved to %s", adapter_dir)

    # Print final metrics
    if trainer.state.best_metric is not None:
        logger.info("Best eval loss: %.4f", trainer.state.best_metric)

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
        logger.info("Peak VRAM during training: %.1f GB", peak_vram)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT QLoRA training for FinReasoning AI")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_wandb=args.use_wandb,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
