"""
File: src/train/sft_train.py

Steps 3 & 4 -- SFT Training Pipeline

TRL version compatibility:
  - TRL >= 0.20.0: DataCollatorForCompletionOnlyLM was removed.
    Use SFTConfig + prompt-completion dataset format.
    Completion-only loss is automatic when the dataset has "prompt"/"completion"
    columns (no DataCollator needed).
  - TRL < 0.20.0: Use TrainingArguments + DataCollatorForCompletionOnlyLM.
    Dataset must have pre-tokenized "input_ids" column.

This file detects the TRL version at import time and selects the right path.

Training config summary:
  - Effective batch size: 4 x 8 = 32
  - cosine LR schedule + 50-step warmup
  - paged_adamw_32bit: offloads optimizer state pages to CPU (~8-12 GB savings)
  - bf16: wider dynamic range than fp16 for financial number magnitudes
  - gradient_checkpointing: ~30% VRAM savings at ~25% compute cost

Memory estimate at bs=4, grad_ckpt=True, max_len=2048, bf16:
  - Base model (4-bit NF4):     ~8.5 GB
  - LoRA adapters + grads:      ~0.6 GB
  - Optimizer (paged to CPU):   ~2.0 GB
  - Activations (grad ckpt):    ~14.0 GB
  - KV cache + misc:            ~6.0 GB
  Total:                        ~31 GB -- fits A100 80GB
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TRL version detection
# ---------------------------------------------------------------------------

def _parse_version(v: str) -> Tuple[int, ...]:
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())

_TRL_VERSION = _parse_version(importlib.metadata.version("trl"))
_NEW_TRL = _TRL_VERSION >= (0, 20, 0)   # DataCollatorForCompletionOnlyLM removed
# TRL 0.20+ renamed max_seq_length -> max_length in SFTConfig
_TRL_USES_MAX_LENGTH = _TRL_VERSION >= (0, 20, 0)

logger.debug("TRL version: %s  (new_api=%s)", _TRL_VERSION, _NEW_TRL)

if _NEW_TRL:
    # TRL >= 0.20: use SFTConfig (inherits from TrainingArguments)
    from trl import SFTTrainer, SFTConfig as _SFTOrTrainingArgs
else:
    # TRL < 0.20: use TrainingArguments + DataCollatorForCompletionOnlyLM
    try:
        from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    except ImportError:
        from trl import SFTTrainer
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    from transformers import TrainingArguments as _SFTOrTrainingArgs

from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

# Used only by the old TRL path to detect the assistant response start token
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------------
# Training arguments builder
# ---------------------------------------------------------------------------

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
    use_wandb: bool = False,
):
    """
    Build SFTConfig (new TRL) or TrainingArguments (old TRL).
    The returned object is passed directly to SFTTrainer(args=...).
    """
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set. Disabling W&B logging.")
        use_wandb = False

    report_to = ["wandb"] if use_wandb else ["none"]
    os.environ.setdefault("WANDB_PROJECT", "finreasoningai-sft")

    common_kwargs = dict(
        output_dir=output_dir,
        # Batch
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Schedule
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        # Precision
        fp16=False,
        bf16=True,
        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Logging / saving
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
        seed=42,
        data_seed=42,
        report_to=report_to,
        run_name="finreasoningai-sft-qlora",
        remove_unused_columns=False,   # keep "task" column for logging
    )

    if _NEW_TRL:
        # SFTConfig-specific parameters.
        # TRL 0.20+ renamed max_seq_length -> max_length; use the right name dynamically.
        import inspect as _inspect
        _sft_params = _inspect.signature(_SFTOrTrainingArgs.__init__).parameters
        _len_kwarg = "max_length" if "max_length" in _sft_params else "max_seq_length"
        _extra: dict = {_len_kwarg: max_seq_length, "packing": False}
        # completion_only_loss and dataset_text_field were added in 0.20 but may be
        # absent in some builds; only pass them when accepted.
        if "completion_only_loss" in _sft_params:
            _extra["completion_only_loss"] = True
        if "dataset_text_field" in _sft_params:
            _extra["dataset_text_field"] = None
        return _SFTOrTrainingArgs(**_extra, **common_kwargs)
    else:
        # Old TRL: max_seq_length goes to SFTTrainer directly, not here
        return _SFTOrTrainingArgs(**common_kwargs)


# ---------------------------------------------------------------------------
# Dataset loader (with auto-migration from old pre-tokenized format)
# ---------------------------------------------------------------------------

def _needs_migration(ds: DatasetDict) -> bool:
    """Return True if the dataset is in the old pre-tokenized format."""
    first = next(iter(ds.values()))
    return "prompt" not in first.column_names


def _find_raw_jsonl(data_dir: str) -> Optional[str]:
    """
    Try to locate the source JSONL file by searching a few common locations
    relative to the processed data directory.
    """
    data_path = Path(data_dir)
    candidates = [
        data_path.parent.parent / "data" / "raw" / "synthetic.jsonl",
        data_path.parent / "raw" / "synthetic.jsonl",
        Path("data") / "raw" / "synthetic.jsonl",
        Path("data/raw/synthetic.jsonl"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def load_datasets(data_dir: str) -> DatasetDict:
    """
    Load the preprocessed DatasetDict.

    If the saved dataset is in the old pre-tokenized format (has 'input_ids'
    instead of 'prompt'/'completion'), this function automatically re-runs
    preprocessing from the raw JSONL and saves the result back to disk.
    No manual intervention required.
    """
    data_path = Path(data_dir)

    if not data_path.exists() or not any(data_path.iterdir()):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_dir}. "
            "Run: python -m src.data.synthetic_gen  then  python -m src.data.preprocess"
        )

    logger.info("Loading DatasetDict from %s", data_dir)
    ds = load_from_disk(str(data_dir))

    if not _needs_migration(ds):
        return ds

    # ---- Auto-migration: old format detected --------------------------------
    logger.warning(
        "Dataset at '%s' is in the old pre-tokenized format (has 'input_ids'). "
        "Automatically re-running preprocessing to generate the new "
        "prompt-completion format.",
        data_dir,
    )

    raw_jsonl = _find_raw_jsonl(data_dir)
    if raw_jsonl is None:
        raise FileNotFoundError(
            f"Dataset at '{data_dir}' is in the old format and the source JSONL "
            "could not be found automatically.\n"
            "Fix: re-run the preprocessing cell in the notebook, or run:\n"
            "  python -m src.data.preprocess --data_path data/raw/synthetic.jsonl"
        )

    logger.info("Found raw JSONL at %s. Re-preprocessing now...", raw_jsonl)

    from src.data.preprocess import load_and_format_dataset

    new_ds = load_and_format_dataset(
        data_path=raw_jsonl,
        tokenizer=None,   # no tokenizer needed for prompt-completion format
        seed=42,
    )

    # Save back to disk (overwrites old format)
    new_ds.save_to_disk(str(data_path))
    logger.info(
        "Re-preprocessing complete. New format saved to %s. "
        "Splits: %s",
        data_dir,
        {k: len(v) for k, v in new_ds.items()},
    )
    return new_ds


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

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
    End-to-end SFT training:
      1. Load Qwen2.5-14B in 4-bit NF4
      2. Apply QLoRA adapters
      3. Load prompt-completion dataset
      4. Build SFTTrainer (TRL version-aware)
      5. Train and save adapter
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # -- Hardware check -------------------------------------------------------
    if not torch.cuda.is_available():
        raise EnvironmentError(
            "No CUDA GPU detected. QLoRA training requires a GPU (A100 80GB recommended)."
        )
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s | VRAM: %.1f GB", gpu_name, total_vram_gb)
    if total_vram_gb < 40.0:
        logger.warning(
            "GPU has only %.1f GB VRAM. Try per_device_train_batch_size=2.", total_vram_gb
        )

    trl_ver_str = importlib.metadata.version("trl")
    logger.info("TRL version: %s  (using %s API)",
                trl_ver_str, "SFTConfig" if _NEW_TRL else "TrainingArguments")

    # -- Load model + tokenizer -----------------------------------------------
    from src.model.load_model import load_model_and_tokenizer, DEFAULT_BNB_CONFIG

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"

    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        bnb_config=DEFAULT_BNB_CONFIG,
        attn_implementation=attn_impl,
    )

    # -- Apply QLoRA ----------------------------------------------------------
    from src.model.apply_lora import apply_qlora

    model = apply_qlora(model, lora_config=QLORA_CONFIG, gradient_checkpointing=True)

    # -- Load dataset ---------------------------------------------------------
    dataset = load_datasets(data_dir)
    train_ds = dataset["train"]
    val_ds = dataset.get("val", dataset.get("validation"))

    logger.info(
        "Dataset -- train: %d, val: %d",
        len(train_ds), len(val_ds) if val_ds else 0,
    )

    # -- Training arguments ---------------------------------------------------
    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_wandb=use_wandb,
    )

    # -- SFTTrainer -----------------------------------------------------------
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    if _NEW_TRL:
        # TRL >= 0.20: pass args=SFTConfig; dataset has "prompt"/"completion" columns;
        # completion-only loss is handled automatically.
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=callbacks,
        )
    else:
        # TRL < 0.20: use response-template-based DataCollator for masking
        response_template_ids = tokenizer.encode(
            RESPONSE_TEMPLATE, add_special_tokens=False
        )
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
            mlm=False,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            callbacks=callbacks,
            max_seq_length=max_seq_length,
            dataset_text_field=None,
            packing=False,
        )

    # -- Train ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting SFT training")
    logger.info("  Model       : %s", model_id)
    logger.info("  TRL version : %s", trl_ver_str)
    logger.info("  Epochs      : %d", num_train_epochs)
    logger.info("  Eff. batch  : %d (bs=%d x accum=%d)",
                per_device_train_batch_size * gradient_accumulation_steps,
                per_device_train_batch_size, gradient_accumulation_steps)
    logger.info("  LR          : %.2e", learning_rate)
    logger.info("  Max seq len : %d", max_seq_length)
    logger.info("  Output      : %s", output_dir)
    logger.info("=" * 60)

    if torch.cuda.is_available():
        logger.info("VRAM before training: %.1f GB", torch.cuda.memory_allocated() / 1e9)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # -- Save adapter ---------------------------------------------------------
    adapter_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("SFT training complete. Adapter saved to %s", adapter_dir)

    if trainer.state.best_metric is not None:
        logger.info("Best eval loss: %.4f", trainer.state.best_metric)
    if torch.cuda.is_available():
        logger.info("Peak VRAM: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
