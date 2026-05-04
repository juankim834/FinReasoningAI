"""Supervised fine-tuning entry point for 1-epoch FinCoT LoRA training."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer
from src.model.load_model import get_preferred_torch_dtype

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs/sft_qlora"
DEFAULT_DATA_DIR = "data/processed_fincot_sft"
RESPONSE_TEMPLATE = "Answer:"
# Minimum number of completion tokens that must remain after truncation.
# Samples whose prompt alone would consume >= max_seq_length - this value are dropped.
_MIN_COMPLETION_TOKENS = 32


def _parse_version(version: str) -> tuple[int, ...]:
    """Convert a version string into a comparable tuple."""
    parts = []
    for token in version.split("."):
        if token.isdigit():
            parts.append(int(token))
    return tuple(parts)


_TRL_VERSION = _parse_version(importlib.metadata.version("trl"))
_NEW_TRL = _TRL_VERSION >= (0, 20, 0)

if _NEW_TRL:
    from trl import SFTConfig, SFTTrainer
    DataCollatorForCompletionOnlyLM = None
else:
    try:
        from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
    except ImportError:  # pragma: no cover - old TRL fallback
        from trl import SFTTrainer
        from trl.trainer.utils import DataCollatorForCompletionOnlyLM
    from transformers import TrainingArguments as SFTConfig



def build_lora_config(
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
) -> LoraConfig:
    """Construct the LoRA configuration used for QLoRA training."""
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )



def build_training_arguments(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    gradient_checkpointing: bool = True,
    warmup_steps: int = 50,
    logging_steps: int = 10,
    eval_steps: int = 100,
    save_steps: int = 200,
    use_wandb: bool = False,
) -> Any:
    """Build trainer arguments for either old or new TRL releases."""
    preferred_dtype = get_preferred_torch_dtype()
    use_bf16 = preferred_dtype == torch.bfloat16
    report_to = ["wandb"] if use_wandb and os.environ.get("WANDB_API_KEY") else ["none"]
    common_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else {},
        optim="paged_adamw_32bit",
        report_to=report_to,
        remove_unused_columns=_NEW_TRL is True,
        seed=42,
        data_seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if _NEW_TRL:
        params = inspect.signature(SFTConfig.__init__).parameters
        extra = {"packing": False}
        if "max_length" in params:
            extra["max_length"] = max_seq_length
        elif "max_seq_length" in params:
            extra["max_seq_length"] = max_seq_length
        if "completion_only_loss" in params:
            extra["completion_only_loss"] = True
        if "dataset_text_field" in params:
            extra["dataset_text_field"] = None
        return SFTConfig(**common_kwargs, **extra)
    return SFTConfig(**common_kwargs)



def load_datasets(data_dir: str = DEFAULT_DATA_DIR) -> DatasetDict:
    """Load a preprocessed DatasetDict from disk."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {data_dir}")
    dataset = load_from_disk(str(data_path))
    if "train" not in dataset:
        raise ValueError("DatasetDict must contain a 'train' split.")
    return dataset



def _build_old_trl_collator(tokenizer: AutoTokenizer) -> Any:
    """Build a completion-only collator for older TRL versions.

    Root-cause fix: using "Answer:" as the response_template string is fragile
    because DataCollatorForCompletionOnlyLM scans tokenized IDs for that
    subsequence.  When long prompts (e.g. full tool-JSON schema + table context)
    push the combined text close to max_seq_length, right-side truncation cuts
    "Answer:" off before the collator ever sees it, producing the warning
    "Could not find response key `Answer:` in the following instance".

    Using the ChatML assistant-turn marker token IDs instead anchors the
    response mask to a position that is structurally part of the *prompt*
    (produced by apply_chat_template), so it is never affected by
    completion-side truncation.
    """
    if DataCollatorForCompletionOnlyLM is None:
        raise RuntimeError(
            "TRL does not provide DataCollatorForCompletionOnlyLM in this environment. "
            "Upgrade TRL or use the prompt/completion dataset path with TRL >= 0.20."
        )
    # Encode the ChatML assistant-turn opening as a token-ID list.
    # Everything after these tokens is treated as the response and receives loss.
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    return DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )


_ASSISTANT_MARKER = "<|im_start|>assistant"
# Matches a bare "Answer:" that some flat-format datasets append to the prompt.
_TRAILING_ANSWER_RE = re.compile(r"\s*Answer:\s*$")


def _prepare_old_trl_dataset(
    dataset: Any,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Any:
    """
    Convert a prompt/completion dataset into a single text-field dataset for TRL 0.8.x.

    Two issues are addressed here:

    1. **Non-ChatML prompts** – Some external samples (e.g. FinQA-style flat text
       starting with "Please answer the given financial question…") are stored
       without the ChatML ``<|im_start|>assistant\\n`` marker.  The collator's
       response template can never match in those sequences.  When a tokenizer
       is supplied, such prompts are re-wrapped with ``apply_chat_template``
       (``add_generation_prompt=True``) so the marker is always present.

    2. **Double "Answer:" prefix** – Flat-format prompts often end with
       "Answer:" (as part of the question template), while the stored completion
       also starts with "Answer:".  Concatenating them naively produces
       "Answer:Answer:…".  The trailing "Answer:" is stripped from the prompt
       before concatenation.
    """
    required_columns = {"prompt", "completion"}
    if not required_columns.issubset(set(dataset.column_names)):
        return dataset

    def add_text(example: dict[str, Any]) -> dict[str, str]:
        completion = str(example["completion"]).lstrip()
        if not completion.startswith(RESPONSE_TEMPLATE):
            completion = f"{RESPONSE_TEMPLATE} {completion}"

        prompt = str(example["prompt"])

        if _ASSISTANT_MARKER not in prompt:
            # Strip trailing "Answer:" that flat-format prompts append.
            clean_prompt = _TRAILING_ANSWER_RE.sub("", prompt).strip()
            if tokenizer is not None:
                # Re-wrap in ChatML so the response template is always present.
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": clean_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                # Minimal fallback when no tokenizer is available.
                prompt = (
                    f"<|im_start|>user\n{clean_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

        text = f"{prompt}{completion}"
        return {"text": text}

    text_dataset = dataset.map(add_text)
    keep_columns = ["text"]
    return text_dataset.remove_columns(
        [column for column in text_dataset.column_names if column not in keep_columns]
    )


def _tokenize_old_trl_dataset(
    dataset: Any,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    max_train_length: int = 16384,
) -> Any:
    """
    Tokenize text examples for the legacy TRL/Trainer path.

    Two-stage length handling:
    1. Hard-drop any sample whose full tokenized length exceeds ``max_train_length``
       (default 4096).  These samples are almost entirely prompt; even after
       truncation to ``max_seq_length`` they would leave almost no completion
       tokens, wasting a training step and potentially skewing the loss.
    2. Samples between ``max_seq_length`` and ``max_train_length`` are *kept*
       and right-truncated to ``max_seq_length`` during tokenisation.  With the
       ChatML assistant-turn marker as the response template this is safe: the
       marker lives in the prompt portion and is never truncated away.
    """
    text_dataset = _prepare_old_trl_dataset(dataset, tokenizer=tokenizer)

    def _within_hard_limit(example: dict[str, Any]) -> bool:
        """Drop samples longer than max_train_length tokens."""
        ids = tokenizer(
            example["text"],
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        if len(ids) > max_train_length:
            logger.warning(
                "Dropping training sample: tokenized length %d > max_train_length %d.",
                len(ids),
                max_train_length,
            )
            return False
        return True

    # Filter only when the dataset is small enough that per-sample tokenization
    # is cheap; skip for very large datasets to avoid double-tokenization cost.
    if len(text_dataset) <= 20_000:
        before = len(text_dataset)
        text_dataset = text_dataset.filter(_within_hard_limit)
        dropped = before - len(text_dataset)
        if dropped:
            logger.info("Dropped %d / %d samples exceeding max_train_length=%d.",
                        dropped, before, max_train_length)

    def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    return text_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=text_dataset.column_names,
    )



def main(
    model_id: str = DEFAULT_MODEL_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    gradient_checkpointing: bool = True,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    max_train_length: int = 16384,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
    use_wandb: bool = False,
    resume_from_checkpoint: Optional[str] = None,
    eval_steps: int = 100,
    save_steps: int = 100,
) -> Any:
    """Train a QLoRA adapter on the processed prompt/completion dataset."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if not torch.cuda.is_available():
        raise EnvironmentError("No CUDA GPU detected. QLoRA training requires a GPU.")

    from src.model.apply_lora import apply_qlora
    from src.model.load_model import get_default_bnb_config, get_preferred_torch_dtype, load_model_and_tokenizer

    preferred_dtype = get_preferred_torch_dtype()
    logger.info("Preferred training dtype: %s", preferred_dtype)
    model, tokenizer = load_model_and_tokenizer(model_id=model_id, bnb_config=get_default_bnb_config())
    lora_config = build_lora_config(
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    model = apply_qlora(model, lora_config=lora_config, gradient_checkpointing=gradient_checkpointing)

    dataset = load_datasets(data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("val") or dataset.get("validation") or dataset.get("test")
    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_wandb=use_wandb,
        eval_steps=eval_steps,
        save_steps=save_steps,
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]

    if _NEW_TRL:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    else:
        train_dataset = _tokenize_old_trl_dataset(
            train_dataset, tokenizer,
            max_seq_length=max_seq_length, max_train_length=max_train_length,
        )
        eval_dataset = (
            _tokenize_old_trl_dataset(
                eval_dataset, tokenizer,
                max_seq_length=max_seq_length, max_train_length=max_train_length,
            )
            if eval_dataset is not None
            else None
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_build_old_trl_collator(tokenizer),
            callbacks=callbacks,
        )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    adapter_dir = Path(output_dir) / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("Adapter saved to %s", adapter_dir)
    return trainer



def save_adapter_for_vllm(
    trainer: Optional[Any] = None,
    output_dir: Optional[str] = None,
    adapter_dir: Optional[str] = None,
) -> None:
    """Save a standalone PEFT adapter directory and print vLLM loading instructions."""
    target_dir = adapter_dir or output_dir
    if trainer is not None:
        if target_dir is None:
            raise ValueError("output_dir or adapter_dir is required when saving from a trainer.")
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(target_path))
        if hasattr(trainer, "tokenizer") and trainer.tokenizer is not None:
            trainer.tokenizer.save_pretrained(str(target_path))
    elif target_dir is None:
        raise ValueError("adapter_dir or output_dir must be provided.")

    print(f"Adapter saved to: {target_dir}")
    print()
    print("To serve with vLLM:")
    print("  vllm serve Qwen/Qwen2.5-14B-Instruct \\")
    print("    --enable-lora \\")
    print(f"    --lora-modules fin-reasoning={target_dir} \\")
    print("    --max-lora-rank 64")
    print()
    print("To call with LoRARequest:")
    print("  from vllm import LLM, SamplingParams")
    print("  from vllm.lora.request import LoRARequest")
    print('  llm = LLM("Qwen/Qwen2.5-14B-Instruct", enable_lora=True)')
    print("  outputs = llm.generate(prompts, SamplingParams(...),")
    print(f'                         lora_request=LoRARequest("fin-reasoning", 1, "{target_dir}"))')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a FinReasoningAI QLoRA adapter")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (trades memory for speed)")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_train_length", type=int, default=16384,
                        help="Hard-drop samples longer than this many tokens before training.")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    trainer = main(
        model_id=args.model_id,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        max_train_length=args.max_train_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    save_adapter_for_vllm(trainer=trainer, output_dir=str(Path(args.output_dir) / "final_adapter"))
