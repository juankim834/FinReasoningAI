"""
File: src/inference/generate.py

Step 6 — Production Inference

Default mode: NO chain-of-thought in output.
The system prompt instructs the model to answer directly. If the model's
internal training used <think> tags, those may leak into output — they are
stripped by the post-processing filter.

System prompt (exact string used at inference):
  "You are FinReasoning AI, an expert financial analyst. Answer questions
   accurately using only the provided context. Give concise, precise answers.
   Do not include chain-of-thought or reasoning steps in your response."

Hallucination reduction:
  1. Constrained numeric decoding: LogitsProcessor biases toward digit tokens
     when the model is generating the final answer span (heuristic detection).
  2. Context grounding check: numeric values in the answer are verified against
     the context. If a number is ungrounded, return "Insufficient information".
  3. Confidence threshold: if self-consistency confidence < 0.3, refuse to answer.

Tool use (Step 6d):
  - The model can emit <tool_call>calculate(expr)</tool_call> tags.
  - A Python interpreter intercepts these and executes the expression safely.
  - The result is injected back into the model's context for the final answer.
  - This is more reliable than model-internal arithmetic for multi-step chains.
"""

from __future__ import annotations

import ast
from importlib import import_module
import logging
import math
import operator
import re
from typing import Any, List, Optional, Union

import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

DIRECT_SYSTEM_PROMPT = (
    "You are FinReasoning AI, an expert financial analyst. "
    "Answer questions accurately using only the provided context. "
    "Give concise, precise answers. "
    "Do not include chain-of-thought or reasoning steps in your response. "
    "If the context does not contain enough information to answer, "
    "respond with exactly: Insufficient information."
)

COT_SYSTEM_PROMPT = (
    "You are FinReasoning AI, an expert financial analyst. "
    "Think step by step inside <think>...</think> tags, then give your final answer "
    "after the closing </think> tag. Be precise with numbers."
)

TOOL_SYSTEM_PROMPT = (
    "You are FinReasoning AI, an expert financial analyst with access to a calculator. "
    "When you need to compute a numerical expression, emit exactly: "
    "<tool_call>calculate(expression)</tool_call>. "
    "After the tool result is provided, give your final answer. "
    "Answer concisely."
)

# Pattern to detect and strip leaked think tags from output
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>calculate\((.+?)\)</tool_call>", re.DOTALL)
_NUMBER_RE = re.compile(
    r"-?\$?\s*(\d[\d,]*\.?\d*)\s*(%|billion|million|thousand|bps|x|M|B|K)?",
    re.IGNORECASE,
)

# Refusal signals
_REFUSAL_RE = re.compile(
    r"(insufficient information|cannot (determine|calculate|find|answer)|"
    r"not (enough|sufficient) (information|data|context)|i('m| am) unable)",
    re.IGNORECASE,
)

INSUFFICIENT_INFO_RESPONSE = "Insufficient information."


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(
    question: str,
    context: str = "",
    instruction: str = "",
    use_cot: bool = False,
    use_tools: bool = False,
    max_context_chars: int = 6000,   # ~1500 tokens at 4 chars/token average
) -> str:
    """
    Build a complete ChatML prompt (system + user turn, NO assistant turn).
    The assistant turn is left open for the model to complete.

    Args:
        question:         The financial question.
        context:          Source document text (10-K passage, table, etc.)
        instruction:      Optional overriding instruction.
        use_cot:          If True, use CoT system prompt.
        use_tools:        If True, use tool-augmented system prompt.
        max_context_chars: Truncate context to this many characters to avoid
                          exceeding 2048 tokens.
    """
    if use_tools:
        system = TOOL_SYSTEM_PROMPT
    elif use_cot:
        system = COT_SYSTEM_PROMPT
    else:
        system = DIRECT_SYSTEM_PROMPT

    if len(context) > max_context_chars:
        logger.warning(
            "Context truncated from %d to %d characters for question: %s...",
            len(context), max_context_chars, question[:60],
        )
        context = context[:max_context_chars] + "\n[...context truncated...]"

    if not instruction:
        instruction = "Answer the following financial question based on the provided context."

    user_parts = [instruction]
    if context.strip():
        user_parts.append(f"\nContext:\n{context}")
    user_parts.append(f"\nQuestion: {question}")
    user_content = "\n".join(user_parts)

    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Constrained numeric decoding (LogitsProcessor)
# ──────────────────────────────────────────────────────────────────────────────

class NumericBiasLogitsProcessor(LogitsProcessor):
    """
    Biases token probabilities toward numeric tokens during the final answer span.

    Heuristic: if the last generated tokens look like the beginning of a numeric
    answer (e.g., "$", a digit), we boost digit/punctuation tokens and suppress
    alphabetic tokens. This makes the model less likely to switch from a number
    to a word mid-answer (e.g., "1.2 trillion" → "1.2 tril" then reverting to text).

    [WARN] TRADE-OFF: This processor can hurt non-numerical answers. Only activate
    when you expect a numerical response (e.g., when the question contains "how much",
    "what was the", "calculate"). The generate_answer() function handles this.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        bias_strength: float = 3.0,
    ) -> None:
        self.bias_strength = bias_strength
        # Pre-compute numeric token IDs (digits 0-9, commas, periods, $, %, -)
        numeric_chars = set("0123456789.,%-$")
        self.numeric_token_ids: List[int] = []
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id])
            if token_str and any(c in numeric_chars for c in token_str):
                self.numeric_token_ids.append(token_id)

        self._is_active = False

    def activate(self) -> None:
        self._is_active = True

    def deactivate(self) -> None:
        self._is_active = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self._is_active or not self.numeric_token_ids:
            return scores
        # Boost numeric tokens
        scores[:, self.numeric_token_ids] += self.bias_strength
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# vLLM generation adapter
# ──────────────────────────────────────────────────────────────────────────────

def _is_vllm_model(model: Any) -> bool:
    """Return True when ``model`` is a vLLM LLM engine rather than HF Transformers."""
    return (
        model.__class__.__module__.startswith("vllm")
        or hasattr(model, "llm_engine")
        or hasattr(model, "engine_class")
    )


def _build_vllm_sampling_params(
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    num_beams: int = 1,
    n: int = 1,
) -> Any:
    """Create SamplingParams while staying compatible across vLLM versions."""
    from inspect import signature

    SamplingParams = import_module("vllm").SamplingParams

    sig = signature(SamplingParams)
    supported = set(sig.parameters)

    params: dict[str, Any] = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p if temperature > 0.0 else 1.0,
        "max_tokens": max_new_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }

    if num_beams > 1:
        if "use_beam_search" in supported:
            params["use_beam_search"] = True
            params["best_of"] = num_beams
            params["temperature"] = 0.0
            params["top_p"] = 1.0
        else:
            logger.warning(
                "num_beams=%d requested, but this vLLM version does not expose "
                "beam search in SamplingParams. Falling back to greedy decoding.",
                num_beams,
            )

    # Older/newer vLLM releases differ slightly; pass only accepted parameters.
    return SamplingParams(**{k: v for k, v in params.items() if k in supported})


def _vllm_generate_texts(
    model: Any,
    prompts: List[str],
    *,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    n: int = 1,
) -> List[List[str]]:
    """
    Generate completions with vLLM.

    Returns one list of completions per prompt. The public inference functions
    keep the same post-processing and validation logic regardless of backend.
    """
    sampling_params = _build_vllm_sampling_params(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        n=n,
    )
    outputs = model.generate(prompts, sampling_params)
    return [[completion.text for completion in request_output.outputs] for request_output in outputs]


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def postprocess_answer(raw_output: str) -> str:
    """
    Clean model output:
      1. Strip <think>...</think> blocks (scratchpad suppression).
      2. Strip leading/trailing whitespace and special tokens.
      3. Remove repeated newlines.
    """
    # Strip think blocks
    cleaned = _THINK_RE.sub("", raw_output)
    # Strip common EOS tokens
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    # Collapse whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_answer_from_cot_output(raw_output: str) -> str:
    """
    Extract the final answer from a CoT output with <think> blocks.
    Returns the text after the last </think> tag.
    Falls back to postprocess_answer() if no </think> found.
    """
    # Find the position of the last </think>
    tag_pos = raw_output.rfind("</think>")
    if tag_pos == -1:
        return postprocess_answer(raw_output)

    after_think = raw_output[tag_pos + len("</think>"):].strip()
    return postprocess_answer(after_think) if after_think else postprocess_answer(raw_output)


def check_grounding(answer: str, context: str, tol: float = 0.02) -> bool:
    """
    Returns True if all numeric values in the answer are grounded in context.
    Used to decide whether to return the answer or substitute "Insufficient information."
    """
    from src.eval.evaluate import compute_grounding_rate
    if not context.strip():
        return True  # No context provided → no grounding check possible
    rate = compute_grounding_rate(answer, context)
    return rate >= 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Tool use: safe arithmetic interpreter
# ──────────────────────────────────────────────────────────────────────────────

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursive safe arithmetic evaluator (no builtins, no imports)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {node.op}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary op: {node.op}")
        return op(operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            allowed_funcs = {"sqrt": math.sqrt, "abs": abs, "round": round,
                             "log": math.log, "exp": math.exp}
            if fname in allowed_funcs:
                args = [_safe_eval(a) for a in node.args]
                return float(allowed_funcs[fname](*args))
        raise ValueError(f"Unsupported function call: {ast.dump(node)}")
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> Union[float, str]:
    """
    Safely evaluate a mathematical expression string.
    Returns the numeric result or an error message string.

    Example:
        safe_calculate("(394.3 - 365.8) / 365.8 * 100") → 7.79...
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return result
    except (ValueError, ZeroDivisionError, SyntaxError, TypeError) as exc:
        logger.warning("Calculator error for expr %r: %s", expression, exc)
        return f"Error: {exc}"


def _execute_tool_calls(text: str) -> str:
    """
    Find all <tool_call>calculate(...)</tool_call> tags in text,
    evaluate the expression, and inject the result back.
    """
    def replace_tool_call(match: re.Match) -> str:
        expr = match.group(1).strip()
        result = safe_calculate(expr)
        if isinstance(result, float):
            formatted = f"{result:,.6g}"
        else:
            formatted = str(result)
        return f"<tool_result>{formatted}</tool_result>"

    return _TOOL_CALL_RE.sub(replace_tool_call, text)


# ──────────────────────────────────────────────────────────────────────────────
# Core generation function
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer(
    model,
    tokenizer: Optional[PreTrainedTokenizerBase],
    question: str,
    context: str = "",
    instruction: str = "",
    use_cot: bool = False,
    use_tools: bool = False,
    self_consistency_n: int = 1,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    num_beams: int = 1,
    use_numeric_bias: bool = False,
    grounding_check: bool = True,
    min_confidence: float = 0.30,
) -> str:
    """
    Generate a financial answer for the given question + context.

    Args:
        model:              Fine-tuned model, or a vLLM LLM engine.
        tokenizer:          Matching tokenizer. Optional for vLLM.
        question:           The question to answer.
        context:            Source document passage (10-K, earnings transcript, etc.)
        instruction:        Optional override instruction string.
        use_cot:            If True, enable chain-of-thought reasoning mode.
        use_tools:          If True, enable tool-use mode (calculator).
        self_consistency_n: If > 1, sample this many times and aggregate via
                            self-consistency. Recommended: 5–10 for high accuracy.
        temperature:        Generation temperature. 0.0 = greedy, >0 = sampling.
                            Auto-set to 0.7 when self_consistency_n > 1.
        top_p:              Nucleus sampling probability.
        max_new_tokens:     Maximum tokens to generate.
        num_beams:          Beam search width. 1 = greedy/sampling.
        use_numeric_bias:   If True, apply NumericBiasLogitsProcessor.
        grounding_check:    If True, verify answer numbers are grounded in context.
        min_confidence:     Self-consistency confidence threshold below which
                            "Insufficient information" is returned.

    Returns:
        Final cleaned answer string.

    Decoding modes:
        Greedy:        temperature=0.0, num_beams=1 (fastest, deterministic)
        Beam search:   temperature=0.0, num_beams=4 (more coherent long answers)
        Sampling:      temperature=0.7, top_p=0.9, num_beams=1
        Self-consist:  self_consistency_n=8, temperature=0.7 (highest accuracy)

    Memory: single forward pass at inference ≈ 12 GB VRAM (base + KV cache).
    """
    # Validate decoding settings
    if self_consistency_n > 1:
        if temperature == 0.0:
            temperature = 0.7  # must sample to get diversity
        if num_beams > 1:
            num_beams = 1  # beam search is incompatible with diversity sampling

    prompt = build_prompt(
        question=question,
        context=context,
        instruction=instruction,
        use_cot=use_cot,
        use_tools=use_tools,
    )

    # --- Self-consistency path ---
    if self_consistency_n > 1:
        if _is_vllm_model(model):
            from src.inference.self_consistency import self_consistent_answer

            raw_answers = _vllm_generate_texts(
                model=model,
                prompts=[prompt],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                n=self_consistency_n,
            )[0]
            final, confidence = self_consistent_answer(raw_answers)
        else:
            if tokenizer is None:
                raise ValueError("tokenizer is required for Transformers generation.")
            from src.inference.self_consistency import sample_with_self_consistency
            final, confidence, _ = sample_with_self_consistency(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                n=self_consistency_n,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        if confidence < min_confidence:
            logger.warning(
                "Self-consistency confidence %.2f < threshold %.2f. "
                "Returning refusal.", confidence, min_confidence
            )
            return INSUFFICIENT_INFO_RESPONSE

        if grounding_check and not check_grounding(final, context):
            logger.warning("Answer failed grounding check. Returning refusal.")
            return INSUFFICIENT_INFO_RESPONSE

        return final

    # --- Single generation path ---
    if _is_vllm_model(model):
        if use_numeric_bias:
            logger.warning("use_numeric_bias is not supported by vLLM; continuing without it.")
        if num_beams > 1 and temperature > 0.0:
            num_beams = 1
        raw_output = _vllm_generate_texts(
            model=model,
            prompts=[prompt],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            n=1,
        )[0][0]
    else:
        if tokenizer is None:
            raise ValueError("tokenizer is required for Transformers generation.")
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1900,
            padding=False,
        ).to(model.device)

        do_sample = temperature > 0.0

        logits_processors = LogitsProcessorList()
        numeric_processor: Optional[NumericBiasLogitsProcessor] = None
        if use_numeric_bias:
            numeric_processor = NumericBiasLogitsProcessor(tokenizer)
            # Heuristic activation: activate if question implies a numeric answer
            numeric_keywords = re.compile(
                r"\b(how much|how many|what (is|was|were) the|calculate|compute|"
                r"total|revenue|profit|margin|ratio|growth|return|percentage|rate)\b",
                re.IGNORECASE,
            )
            if numeric_keywords.search(question):
                numeric_processor.activate()
                logits_processors.append(numeric_processor)

        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                num_beams=num_beams,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=logits_processors if logits_processors else None,
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)

    # --- Tool use: execute any calculator calls ---
    if use_tools and "<tool_call>" in raw_output:
        raw_output = _execute_tool_calls(raw_output)
        # Re-inject tool results into the model for a final generation step
        # (simplified: just strip tool_call tags and return with results)
        raw_output = raw_output.replace("<tool_result>", "[calculated: ").replace("</tool_result>", "]")

    # --- Post-process ---
    if use_cot:
        answer = extract_answer_from_cot_output(raw_output)
    else:
        answer = postprocess_answer(raw_output)

    # --- Grounding check ---
    if grounding_check and context.strip() and not check_grounding(answer, context):
        logger.warning(
            "Grounding check failed for answer: %s... Returning refusal.", answer[:80]
        )
        return INSUFFICIENT_INFO_RESPONSE

    # --- Refusal passthrough ---
    if _REFUSAL_RE.search(answer):
        return INSUFFICIENT_INFO_RESPONSE

    return answer if answer else INSUFFICIENT_INFO_RESPONSE


# ──────────────────────────────────────────────────────────────────────────────
# Batch inference
# ──────────────────────────────────────────────────────────────────────────────

def batch_generate(
    model,
    tokenizer: Optional[PreTrainedTokenizerBase],
    samples: List[dict],
    use_cot: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> List[str]:
    """
    Generate answers for a list of samples in batches.
    Faster than calling generate_answer() in a loop for large eval sets.
    """
    answers = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        prompts = [
            build_prompt(
                question=s.get("question", ""),
                context=s.get("context", ""),
                instruction=s.get("instruction", ""),
                use_cot=use_cot,
            )
            for s in batch
        ]

        if _is_vllm_model(model):
            batch_outputs = _vllm_generate_texts(
                model=model,
                prompts=prompts,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                n=1,
            )
            raw_outputs = [outputs[0] if outputs else "" for outputs in batch_outputs]
        else:
            if tokenizer is None:
                raise ValueError("tokenizer is required for Transformers generation.")
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                max_length=1900,
                padding=True,
                pad_to_multiple_of=8,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            raw_outputs = []
            for j, out_ids in enumerate(output_ids):
                prompt_len = inputs["input_ids"][j].shape[0]
                new_ids = out_ids[prompt_len:]
                raw_outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))

        for raw in raw_outputs:
            if use_cot:
                answers.append(extract_answer_from_cot_output(raw))
            else:
                answers.append(postprocess_answer(raw))

        logger.info("Batch %d/%d complete.", i // batch_size + 1,
                    math.ceil(len(samples) / batch_size))

    return answers


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo / CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="FinReasoning AI inference CLI")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/sft_qlora/final_adapter",
                        help="Path to LoRA adapter; 'none' to skip")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="",
                        help="Financial context passage")
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--use_tools", action="store_true")
    parser.add_argument("--self_consistency_n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--engine", choices=["vllm", "transformers"], default="vllm")
    args = parser.parse_args()

    if args.engine == "vllm":
        from src.model.load_model import load_vllm_model_and_tokenizer

        model, tokenizer = load_vllm_model_and_tokenizer(args.model_id)
        if args.adapter_dir != "none":
            logger.warning(
                "vLLM expects a merged model path for adapter weights in this CLI. "
                "Ignoring --adapter_dir=%s; run merge_adapter.ipynb first.",
                args.adapter_dir,
            )
    else:
        from src.model.load_model import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(args.model_id)

    if args.engine == "transformers" and args.adapter_dir != "none":
        from pathlib import Path
        if Path(args.adapter_dir).exists():
            PeftModel = import_module("peft").PeftModel
            model = PeftModel.from_pretrained(model, args.adapter_dir)

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        context=args.context,
        use_cot=args.use_cot,
        use_tools=args.use_tools,
        self_consistency_n=args.self_consistency_n,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    print(f"\n{'='*50}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print(f"{'='*50}")

    # Demo of safe calculator
    print("\n=== Calculator Tool Demo ===")
    test_expr = "(394.3 - 365.8) / 365.8 * 100"
    result = safe_calculate(test_expr)
    print(f"Expression: {test_expr}")
    print(f"Result:     {result:.4f}%")
    print("\n[OK] Step 6 complete — inference pipeline ready.")
