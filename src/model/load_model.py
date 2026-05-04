"""Model loading utilities for training and inference."""

from __future__ import annotations

import importlib
from importlib import import_module
import logging
from pathlib import Path
from typing import Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MIN_BNB_VERSION = "0.46.1"

DEFAULT_BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


def _parse_version(version: str) -> tuple[int, ...]:
    """Convert a dotted version string into an integer tuple."""
    return tuple(int(part) for part in version.split(".") if part.isdigit())



def _check_vram(required_gb: float = 12.0) -> None:
    """Warn or fail when available VRAM is below the requested threshold."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA device detected. Model will load on CPU and be extremely slow.")
        return
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    logger.info("VRAM available: %.1f GB / %.1f GB total", free_gb, total_gb)
    if free_gb < required_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {free_gb:.1f} GB free, need at least {required_gb:.1f} GB. "
            "Ensure no other GPU processes are active."
        )



def _validate_bitsandbytes_installation() -> None:
    """
    Validate that bitsandbytes imports cleanly and has CUDA support.

    Colab Python 3.12 runtimes can end up with a CPU-only wheel or an older
    bitsandbytes build that crashes on Triton 3.x with `triton.ops` import errors.
    """
    try:
        bnb = importlib.import_module("bitsandbytes")
    except Exception as exc:  # pragma: no cover - depends on runtime package state
        msg = str(exc)
        if "triton.ops" in msg:
            raise RuntimeError(
                "bitsandbytes failed to import because this runtime has a newer Triton layout and "
                "the installed bitsandbytes wheel is too old.\n"
                f"Fix in Colab: pip uninstall -y bitsandbytes && pip install -U bitsandbytes=={MIN_BNB_VERSION}\n"
                "Then restart the runtime and rerun the notebook from the install cell."
            ) from exc
        raise RuntimeError(
            "bitsandbytes could not be imported. Reinstall it in Colab with:\n"
            f"  pip uninstall -y bitsandbytes && pip install -U bitsandbytes=={MIN_BNB_VERSION}\n"
            "Then restart the runtime."
        ) from exc

    bnb_version = getattr(bnb, "__version__", "0.0.0")
    if _parse_version(bnb_version) < _parse_version(MIN_BNB_VERSION):
        raise RuntimeError(
            f"bitsandbytes {bnb_version} is too old for the current Colab Triton stack.\n"
            f"Install bitsandbytes=={MIN_BNB_VERSION} or newer, then restart the runtime."
        )

    try:
        cextension = importlib.import_module("bitsandbytes.cextension")
        compiled_with_cuda = getattr(cextension, "COMPILED_WITH_CUDA", None)
    except Exception:
        compiled_with_cuda = None

    if compiled_with_cuda is False:
        raise RuntimeError(
            "bitsandbytes imported, but the installed wheel does not have GPU support.\n"
            f"Fix in Colab: pip uninstall -y bitsandbytes && pip install -U bitsandbytes=={MIN_BNB_VERSION}\n"
            "Then restart the runtime before loading the model."
        )



def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    bnb_config: BitsAndBytesConfig | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a quantized Qwen model and tokenizer for QLoRA training."""
    if bnb_config is None:
        bnb_config = DEFAULT_BNB_CONFIG

    _check_vram(required_gb=10.0)
    _validate_bitsandbytes_installation()

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token: %s", tokenizer.eos_token)

    logger.info("Loading model from %s (4-bit NF4, double_quant=True)", model_id)

    flash_keywords = ("flash", "attn_implementation", "flash_attn")

    def is_flash_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(keyword in msg for keyword in flash_keywords)

    def is_bnb_triton_error(exc: Exception) -> bool:
        msg = str(exc)
        return "triton.ops" in msg or "triton_based_modules" in msg or (
            "bitsandbytes" in msg and "import" in msg.lower()
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        if is_bnb_triton_error(exc):
            raise RuntimeError(
                "bitsandbytes failed to import due to a Triton version mismatch.\n"
                f"This Colab runtime needs bitsandbytes>={MIN_BNB_VERSION}.\n"
                f"Fix: run pip uninstall -y bitsandbytes && pip install -U bitsandbytes=={MIN_BNB_VERSION}\n"
                "Then restart the runtime."
            ) from exc
        if is_flash_error(exc):
            logger.warning("Flash Attention 2 not available (%s). Falling back to eager attention.", exc)
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

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    total_params = sum(parameter.numel() for parameter in model.parameters())
    logger.info("Model loaded. Total parameters: %.2fB", total_params / 1e9)

    if torch.cuda.is_available():
        used_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("VRAM after model load: %.1f GB allocated", used_gb)

    return model, tokenizer



def get_default_bnb_config() -> BitsAndBytesConfig:
    """Return the default 4-bit bitsandbytes config."""
    return DEFAULT_BNB_CONFIG



def load_vllm_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.90,
    enable_lora: bool = False,
    **llm_kwargs: Any,
) -> Tuple[Any, PreTrainedTokenizerBase]:
    """Load the model through vLLM for inference."""
    _check_vram(required_gb=10.0)

    logger.info("Loading tokenizer from %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token: %s", tokenizer.eos_token)

    logger.info("Loading vLLM engine from %s", model_id)
    LLM = import_module("vllm").LLM
    model = LLM(
        model=model_id,
        tokenizer=model_id,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_lora=enable_lora,
        **llm_kwargs,
    )
    return model, tokenizer


def load_model_with_adapter(
    model_id: str = DEFAULT_MODEL_ID,
    adapter_dir: str | None = None,
    *,
    bnb_config: BitsAndBytesConfig | None = None,
    device_map: str = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "flash_attention_2",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a base model and optionally attach a PEFT adapter."""
    model, tokenizer = load_model_and_tokenizer(
        model_id=model_id,
        bnb_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    if adapter_dir and Path(adapter_dir).exists():
        PeftModel = import_module("peft").PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
        logger.info("Loaded adapter from %s", adapter_dir)
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Load Qwen2.5-14B and print model info")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--attn", default="flash_attention_2", choices=["flash_attention_2", "eager"])
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, attn_implementation=args.attn)
    sample = tokenizer("What is the EBITDA margin?", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**sample)
    print(f"\n[OK] Forward pass OK - logits shape: {out.logits.shape}")
    print(f"   Vocab size: {out.logits.shape[-1]}")
    print(f"   VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
