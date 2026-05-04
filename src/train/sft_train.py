"""Supervised fine-tuning entry point for 1-epoch FinCoT LoRA training."""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import os
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
    """Build a completion-only collator for older TRL versions."""
    if DataCollatorForCompletionOnlyLM is None:
        raise RuntimeError(
            "TRL does not provide DataCollatorForCompletionOnlyLM in this environment. "
            "Upgrade TRL or use the prompt/completion dataset path with TRL >= 0.20."
        )
    return DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False,
    )


def _prepare_old_trl_dataset(dataset: Any) -> Any:
    """
    Convert a prompt/completion dataset into a single text-field dataset for TRL 0.8.x.

    Older SFTTrainer versions require `dataset_text_field` or `formatting_func`
    even when a custom completion-only collator is supplied.
    """
    required_columns = {"prompt", "completion"}
    if not required_columns.issubset(set(dataset.column_names)):
        return dataset

    def add_text(example: dict[str, Any]) -> dict[str, str]:
        completion = str(example["completion"])
        if completion.lstrip().startswith(RESPONSE_TEMPLATE):
            text = f"{example['prompt']}{completion}"
        else:
            text = f"{example['prompt']}\n{RESPONSE_TEMPLATE} {completion}"
        return {"text": text}

    text_dataset = dataset.map(add_text)
    keep_columns = ["text"]
    return text_dataset.remove_columns(
        [column for column in text_dataset.column_names if column not in keep_columns]
    )


def _tokenize_old_trl_dataset(dataset: Any, tokenizer: AutoTokenizer, max_seq_length: int) -> Any:
    """
    Tokenize text examples for the legacy TRL/Trainer path.

    This avoids relying on old SFTTrainer preprocessing behavior, which can leave
    raw string fields in the collator path and trigger tensor conversion errors.
    """
    text_dataset = _prepare_old_trl_dataset(dataset)

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
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
    use_wandb: bool = False,
    resume_from_checkpoint: Optional[str] = None,
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
    model = apply_qlora(model, lora_config=lora_config, gradient_checkpointing=True)

    dataset = load_datasets(data_dir)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("val") or dataset.get("validation") or dataset.get("test")
    training_args = build_training_arguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_wandb=use_wandb,
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
        train_dataset = _tokenize_old_trl_dataset(train_dataset, tokenizer, max_seq_length=max_seq_length)
        eval_dataset = (
            _tokenize_old_trl_dataset(eval_dataset, tokenizer, max_seq_length=max_seq_length)
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
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
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
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_wandb=args.use_wandb,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    save_adapter_for_vllm(trainer=trainer, output_dir=str(Path(args.output_dir) / "final_adapter"))
