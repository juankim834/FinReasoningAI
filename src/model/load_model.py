"""
File: src/model/load_model.py

Step 1 — Model Selection & Loading

WHY Qwen2.5-14B over alternatives:
  - vs LLaMA-3.1-8B / Mistral-7B: 14B parameters significantly outperform 7–8B
    on multi-step numerical reasoning. FinQA benchmarks show ~8–12pt EM gain.
  - vs Phi-3-medium (14B): Qwen2.5-14B has a 128k token context window vs Phi-3's
    4k, essential for long 10-K / earnings transcript passages.
  - Financial tokenization: Qwen2.5 was trained on a large multilingual corpus
    including financial documents; its tokenizer handles "$", "%", "bps", "EBITDA"
    as single or minimal subword units — reducing representation fragmentation.
  - Numeric handling: Qwen2.5 shows strong arithmetic consistency via its
    structured pretraining on code + math data, critical for numerical reasoning.
  - A100 80GB compatibility: 14B in 4-bit NF4 ≈ 8–9 GB base model VRAM.
    Adding LoRA adapters, optimizer states, and activations stays well within 80GB.

VRAM estimates (rough):
  Inference (4-bit):   ~8–9 GB for weights + ~2–4 GB KV cache @ 2048 ctx = ~12 GB
  Training  (QLoRA):   ~8–9 GB weights + ~6–8 GB LoRA grads/optimizer +
                        ~14–16 GB activations (grad ckpt) + batch buffer ≈ 38–42 GB
  Headroom on A100:    80 GB − 42 GB ≈ 38 GB → safe margin
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# 4-bit NF4 double-quantization config (saves ~15% VRAM vs single quant)
DEFAULT_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 is optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # BF16 compute preserves dynamic range
    bnb_4bit_use_double_quant=True,       # double quantization: quantize the quantization constants
)


def _check_vram(required_gb: float = 12.0) -> None:
    """Warn if available VRAM is below the required threshold."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA device detected. Model will load on CPU — this is extremely slow.")
        return
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    logger.info("VRAM available: %.1f GB / %.1f GB total", free_gb, total_gb)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {free_gb:.1f} GB free, need ≥{required_gb:.1f} GB. "
            "Ensure no other processes are using the GPU."
        )


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    bnb_config: BitsAndBytesConfig | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load Qwen2.5-14B-Instruct in 4-bit NF4 quantization, ready for QLoRA.

    Args:
        model_id:            HuggingFace model ID. Default: Qwen/Qwen2.5-14B-Instruct
        bnb_config:          BitsAndBytesConfig. If None, uses DEFAULT_BNB_CONFIG.
        device_map:          Accelerate device map. "auto" works for single GPU.
        trust_remote_code:   Required for Qwen models (custom attention code).
        attn_implementation: "flash_attention_2" (preferred) or "eager".

    Returns:
        (model, tokenizer) ready for PEFT / SFT training.

    Memory: ~38–42 GB VRAM during training with bs=4, grad_ckpt=True.
    """
    if bnb_config is None:
        bnb_config = DEFAULT_BNB_CONFIG

    _check_vram(required_gb=10.0)

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        padding_side="right",   # right-padding required by SFTTrainer's packing
    )
    # Qwen2.5 uses <|endoftext|> as pad; set explicitly to avoid warnings
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token: %s", tokenizer.eos_token)

    logger.info("Loading model from %s (4-bit NF4, double_quant=True)", model_id)

    # Flash Attention 2 requires compute_dtype=bfloat16 and CUDA ≥ 8.0 (A100 = 8.0 ✓)
    # Fall back to "eager" if flash-attn is not installed
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    except (ImportError, ValueError) as exc:
        if "flash" in str(exc).lower() or "attn_implementation" in str(exc).lower():
            logger.warning(
                "Flash Attention 2 not available (%s). Falling back to eager attention.", exc
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
            )
        else:
            raise

    # Disable cache during training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.pretraining_tp = 1  # avoid tensor-parallel issues with PEFT

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded. Total parameters: %.2fB (stored in 4-bit NF4)", total_params / 1e9
    )

    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after model load: %.1f GB allocated", used_gb)

    return model, tokenizer


def get_default_bnb_config() -> BitsAndBytesConfig:
    """Return the default 4-bit NF4 double-quantization config."""
    return DEFAULT_BNB_CONFIG


# ──────────────────────────────────────────────────────────────────────────────
# Standalone test / demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Load Qwen2.5-14B and print model info")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--attn", default="flash_attention_2",
                        choices=["flash_attention_2", "eager"])
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        attn_implementation=args.attn,
    )

    # Quick sanity-check forward pass
    sample = tokenizer("What is the EBITDA margin?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**sample)
    print(f"\n[OK] Forward pass OK — logits shape: {out.logits.shape}")
    print(f"   Vocab size: {out.logits.shape[-1]}")
    print(f"   VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
