"""
File: src/model/apply_lora.py

Applies QLoRA (LoRA on 4-bit quantized weights) to the loaded model.

Design rationale:
  - r=64 gives a large rank that improves expressiveness for complex financial
    reasoning tasks. Larger r = more trainable params but better task adaptation.
  - alpha=128 (= 2×r) is a common stable starting point; effective LR scales as
    alpha/r, so alpha=128, r=64 → scale factor 2.
  - Targeting all projection matrices (q/k/v/o + MLP gate/up/down) ensures the
    full attention + feed-forward pathway can be adapted — important for a domain
    shift as large as general → financial NLP.
  - lora_dropout=0.05 is light regularization; heavier dropout hurts numerical
    tasks where precision matters.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default LoRA config matching the specification in Agent.md Step 3a
# ──────────────────────────────────────────────────────────────────────────────

QLORA_TARGET_MODULES: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention projections
    "gate_proj", "up_proj", "down_proj",        # MLP / FFN projections (SwiGLU)
]

DEFAULT_LORA_CONFIG = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=QLORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # Merge adapter weights into base at inference time for speed
    # (done via model.merge_and_unload() after training)
)


def apply_qlora(
    model: PreTrainedModel,
    lora_config: Optional[LoraConfig] = None,
    gradient_checkpointing: bool = True,
) -> PeftModel:
    """
    Prepare a 4-bit quantized model for QLoRA training and attach LoRA adapters.

    Args:
        model:                  A model loaded with BitsAndBytesConfig (4-bit).
        lora_config:            LoraConfig instance. Defaults to DEFAULT_LORA_CONFIG.
        gradient_checkpointing: Enable gradient checkpointing to reduce VRAM.
                                Required to keep total memory under 80 GB.

    Returns:
        PeftModel with LoRA adapters attached and unfrozen; base weights frozen.

    [WARN] TRADE-OFF: gradient_checkpointing recomputes activations during backward,
    trading ~20–30% extra compute for ~30–40% VRAM reduction. Essential on A100.

    Memory impact of LoRA adapters:
        r=64, 7 modules × 2 layers each × 14B model → ~80M trainable params
        ~80M × 4 bytes (bf16) × 2 (param + grad) ≈ ~640 MB — negligible vs base.
    """
    if lora_config is None:
        lora_config = DEFAULT_LORA_CONFIG

    # prepare_model_for_kbit_training:
    #   1. Casts layer norms to float32 (stability)
    #   2. Enables gradient checkpointing if requested
    #   3. Disables input_require_grads issues for 4-bit layers
    logger.info("Preparing model for k-bit (4-bit) training...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=gradient_checkpointing,
    )

    logger.info(
        "Applying QLoRA: r=%d, alpha=%d, dropout=%.2f, target_modules=%s",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
        lora_config.target_modules,
    )
    peft_model = get_peft_model(model, lora_config)

    _print_trainable_parameters(peft_model)
    return peft_model


def _print_trainable_parameters(model: PeftModel) -> None:
    """Log the count and percentage of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    pct = 100.0 * trainable / total if total > 0 else 0.0

    logger.info("=" * 60)
    logger.info("Trainable parameters : %12.3fM  (%5.2f%%)", trainable / 1e6, pct)
    logger.info("Frozen  parameters   : %12.3fB", frozen / 1e9)
    logger.info("Total   parameters   : %12.3fB", total / 1e9)
    logger.info("=" * 60)

    # [WARN] TRADE-OFF: ~0.5% trainable params is typical for QLoRA. This is intentional —
    # training only adapters prevents catastrophic forgetting of general knowledge while
    # allowing the model to adapt to financial domain patterns.
    if pct > 5.0:
        logger.warning(
            "Trainable parameter ratio (%.1f%%) is unusually high for QLoRA. "
            "Consider reducing r to save VRAM.", pct
        )


def save_lora_adapter(peft_model: PeftModel, output_dir: str) -> None:
    """Save only the LoRA adapter weights (not the full model)."""
    peft_model.save_pretrained(output_dir)
    logger.info("LoRA adapter saved to %s", output_dir)


def load_lora_adapter(
    base_model: PreTrainedModel,
    adapter_path: str,
) -> PeftModel:
    """Load a saved LoRA adapter onto a base model."""
    from peft import PeftModel as _PeftModel
    peft_model = _PeftModel.from_pretrained(base_model, adapter_path)
    logger.info("LoRA adapter loaded from %s", adapter_path)
    return peft_model


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not torch.cuda.is_available():
        print("No GPU detected — skipping live test. Config printed below.")
        print(DEFAULT_LORA_CONFIG)
        sys.exit(0)

    from src.model.load_model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer()
    peft_model = apply_qlora(model)

    print("\n[OK] QLoRA applied successfully.")
    print(f"   Type: {type(peft_model)}")
    print(f"   VRAM after LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
