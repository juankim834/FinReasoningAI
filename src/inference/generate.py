"""Inference helpers for direct and tool-augmented generation."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import math
import operator
import re
from typing import Any, Optional

import torch
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.tools.tool_router import dispatch_tool_call, parse_tool_call_from_output
from tools.financial_tools import FINANCIAL_TOOLS

logger = logging.getLogger(__name__)

DIRECT_SYSTEM_PROMPT = (
    "You are FinReasoningAI, a financial reasoning assistant. "
    "Answer financial questions clearly and end with a line starting with 'Answer:'."
)

COT_SYSTEM_PROMPT = (
    "You are FinReasoningAI, a financial reasoning assistant. "
    "Reason step by step before you answer, and end with a line starting with 'Answer:'."
)

_TOOL_PROMPT_INTRO = (
    "You are FinReasoningAI, a financial reasoning assistant that computes every number with tools.\n\n"
    "=== STRICT RULES ===\n"
    "1. NEVER compute a number in your head. For every arithmetic step — add, subtract,\n"
    "   multiply, divide, percentage, ratio, growth rate, or CAGR — call a tool.\n"
    "2. When a computation is needed, emit ONLY a single complete tool call. Nothing else.\n"
    "   No preamble, no explanation, no reasoning text. Just the tag:\n"
    '   <tool_call>{"name":"arithmetic","arguments":{"a":394.3,"b":365.8,"operation":"subtract"}}</tool_call>\n'
    "3. After receiving a tool result, you may either:\n"
    "   a) Emit the next single tool call (if more computation is needed), OR\n"
    "   b) Emit the final answer on one line: Final Answer: <bare number>\n"
    "4. NEVER emit multiple tool calls in one turn.\n"
    "5. ALWAYS use the number from the tool result as input to the next step.\n"
    "   Never restate or substitute a different number.\n\n"
    "=== FINAL ANSWER FORMAT ===\n"
    "Final Answer: <number only>\n"
    "No units, no explanation, no commas, no currency symbols, no percentage signs.\n\n"
    "=== AVAILABLE TOOLS ===\n\n"
    "1. arithmetic(a, b, operation)\n"
    "   operation: add | subtract | multiply | divide | percent_change\n\n"
    "   add        — a + b\n"
    "   subtract   — a - b\n"
    "   multiply   — a * b\n"
    "     ↳ Use this to apply a percentage rate to a base value.\n"
    "       14% of 200000  →  multiply(200000, 0.14)  =  28000\n"
    "        5% of 120000  →  multiply(120000, 0.05)  =   6000\n"
    "   divide     — a / b\n"
    "   percent_change — (a - b) / |b|\n"
    "     ↳ Use ONLY for period-over-period relative change.\n"
    "       Revenue 100→120  →  percent_change(120, 100)  =  0.20\n"
    "       Do NOT use percent_change to apply a rate; use multiply instead.\n\n"
    "2. compound_growth_rate(start_value, end_value, n_periods)\n"
    "   CAGR = (end_value / start_value)^(1/n_periods) − 1\n"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBER_RE = re.compile(
    r"-?\$?\s*(\d[\d,]*\.?\d*)\s*(%|billion|million|thousand|bps|x|M|B|K)?",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"(insufficient information|cannot determine|not enough information|unable to answer)",
    re.IGNORECASE,
)
# Strict parser: matches the last "Final Answer: <bare number>" line.
_FINAL_ANSWER_RE = re.compile(
    r"^\s*Final\s+Answer:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
INSUFFICIENT_INFO_RESPONSE = "Insufficient information."

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class _StopSequenceCriteria(StoppingCriteria):
    """Halt Transformers generation as soon as any stop string appears in the output."""

    def __init__(
        self,
        stop_sequences: list[str],
        tokenizer: PreTrainedTokenizerBase,
        prompt_len: int,
    ) -> None:
        self._stop_sequences = stop_sequences
        self._tokenizer = tokenizer
        self._prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        generated = self._tokenizer.decode(
            input_ids[0, self._prompt_len:], skip_special_tokens=False
        )
        return any(seq in generated for seq in self._stop_sequences)


def _tool_result(
    answer: Optional[str],
    tool_calls: list[dict[str, Any]],
    full_outputs: list[str],
    *,
    max_rounds_exceeded: bool,
) -> dict[str, Any]:
    """Assemble the standard return dict for generate_with_tools."""
    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "n_rounds": len(tool_calls),
        "full_output": "\n\n---\n\n".join(full_outputs),
        "tool_violation": len(tool_calls) == 0,
        "max_rounds_exceeded": max_rounds_exceeded,
    }


class NumericBiasLogitsProcessor(LogitsProcessor):
    """Bias decoding toward numeric tokens for expected numerical answers."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, bias_strength: float = 3.0) -> None:
        self.bias_strength = bias_strength
        numeric_chars = set("0123456789.,%-$")
        self.numeric_token_ids = []
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id])
            if token_str and any(char in numeric_chars for char in token_str):
                self.numeric_token_ids.append(token_id)
        self._active = False

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self._active and self.numeric_token_ids:
            scores[:, self.numeric_token_ids] += self.bias_strength
        return scores



def _is_vllm_model(model: Any) -> bool:
    """Return True when the model object is backed by vLLM."""
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
    n: int = 1,
    extra_stop: list[str] | None = None,
) -> Any:
    """Create vLLM sampling parameters while tolerating version drift.

    ``extra_stop`` allows callers to add custom stop strings (e.g. ``["</tool_call>"]``)
    on top of the standard EOS tokens.  ``include_stop_str_in_output=True`` is
    requested when supported so the closing tag is preserved in the output text and
    the parser can find the complete ``<tool_call>…</tool_call>`` block without a
    post-hoc repair step.
    """
    from importlib import import_module

    SamplingParams = import_module("vllm").SamplingParams
    supported = set(inspect.signature(SamplingParams).parameters)
    stop_strs = ["<|im_end|>", "<|endoftext|>"] + (extra_stop or [])
    params = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p if temperature > 0 else 1.0,
        "max_tokens": max_new_tokens,
        "stop": stop_strs,
        "include_stop_str_in_output": True,
    }
    return SamplingParams(**{key: value for key, value in params.items() if key in supported})



def _vllm_generate_texts(
    model: Any,
    prompts: list[str],
    *,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
    n: int = 1,
    extra_stop: list[str] | None = None,
) -> list[list[str]]:
    """Generate one or more completions for each prompt with vLLM."""
    outputs = model.generate(
        prompts,
        _build_vllm_sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n,
            extra_stop=extra_stop,
        ),
    )
    return [[completion.text for completion in request_output.outputs] for request_output in outputs]



def _safe_eval(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST for calculator-style prompts."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        allowed = {"sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "exp": math.exp}
        if node.func.id in allowed:
            return float(allowed[node.func.id](*[_safe_eval(arg) for arg in node.args]))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")



def safe_calculate(expression: str) -> float | str:
    """Safely evaluate a math expression and return either the result or an error string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval(tree.body)
    except Exception as exc:  # pragma: no cover - defensive helper
        logger.warning("Calculator error for %r: %s", expression, exc)
        return f"Error: {exc}"



def _tool_system_prompt(tools: Optional[list[dict[str, Any]]]) -> str:
    """Build the system prompt used for tool-augmented inference.

    Intentionally does NOT prepend COT_SYSTEM_PROMPT: chain-of-thought preamble
    encourages the model to write long reasoning before tool calls, which inflates
    token usage and triggers multi-call emissions in a single turn.
    """
    del tools
    return _TOOL_PROMPT_INTRO



def _build_messages(
    question: str,
    context: str = "",
    instruction: str = "",
    use_cot: bool = False,
    use_tools: bool = False,
    tools: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, str]]:
    """Construct a chat history for the current request."""
    system_prompt = DIRECT_SYSTEM_PROMPT
    if use_tools:
        system_prompt = _tool_system_prompt(tools)
    elif use_cot:
        system_prompt = COT_SYSTEM_PROMPT

    user_content = question.strip()
    if context.strip():
        user_content = f"{context.strip()}\n\n{question.strip()}"
    if instruction.strip():
        user_content = f"{instruction.strip()}\n\n{user_content}"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]



def _apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat messages with the tokenizer, using tool support when available."""
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    signature = inspect.signature(tokenizer.apply_chat_template)
    if tools and "tools" in signature.parameters:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kwargs)



def build_prompt(
    question: str,
    context: str = "",
    instruction: str = "",
    use_cot: bool = False,
    use_tools: bool = False,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    tools: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Build a prompt string for direct or tool-augmented generation."""
    messages = _build_messages(
        question=question,
        context=context,
        instruction=instruction,
        use_cot=use_cot,
        use_tools=use_tools,
        tools=tools,
    )
    if tokenizer is None:
        return "\n".join(f"[{message['role']}] {message['content']}" for message in messages)
    return _apply_chat_template(tokenizer, messages, tools=tools if use_tools else None)



def postprocess_answer(raw_output: str) -> str:
    """Strip scratchpad artifacts and common stop tokens from a raw completion."""
    cleaned = _THINK_RE.sub("", raw_output)
    cleaned = cleaned.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()



def extract_answer_from_cot_output(raw_output: str) -> str:
    """Extract the final answer span after a think block when present."""
    tag_pos = raw_output.rfind("</think>")
    if tag_pos == -1:
        return postprocess_answer(raw_output)
    return postprocess_answer(raw_output[tag_pos + len("</think>"):])



def extract_final_answer(text: str) -> tuple[Optional[str], bool]:
    """Extract a strict 'Final Answer: <number>' line from model output.

    Scans for the *last* line matching the pattern so that intermediate
    tool-call turns don't shadow the real answer.

    Returns:
        ``(answer_str, False)``  — strict match found; ``answer_str`` is the
            raw numeric string (e.g. ``"20.0"``).
        ``(fallback_str, True)`` — no strict match; ``fallback_str`` is the
            result of the legacy CoT extractor (may be ``None``).
            ``parse_failed=True`` is recorded for benchmark diagnostics.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1), False
    fallback = extract_answer_from_cot_output(text) or None
    return fallback, True



def check_grounding(answer: str, context: str, tol: float = 0.02) -> bool:
    """Lightweight wrapper around evaluation grounding checks."""
    from src.eval.evaluate import compute_grounding_rate

    if not context.strip():
        return True
    return compute_grounding_rate(answer, context, tol=tol) >= 0.5



def _generate_once_transformers(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_numeric_bias: bool,
    question: str,
    stop_sequences: list[str] | None = None,
) -> str:
    """Run a single Hugging Face generation and decode the new tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900, padding=False).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    do_sample = temperature > 0

    processors = LogitsProcessorList()
    if use_numeric_bias:
        numeric_processor = NumericBiasLogitsProcessor(tokenizer)
        if re.search(r"(how much|how many|calculate|compute|ratio|margin|growth|rate)", question, re.IGNORECASE):
            numeric_processor.activate()
            processors.append(numeric_processor)

    criteria = StoppingCriteriaList()
    if stop_sequences:
        criteria.append(_StopSequenceCriteria(stop_sequences, tokenizer, prompt_len))

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            # Explicitly neutralise Qwen's generation_config defaults (temperature=0.7,
            # top_p=0.8, top_k=20) when doing greedy decoding to suppress the
            # "do_sample=False but temperature/top_p/top_k are set" UserWarning.
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=None if do_sample else 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=processors if processors else None,
            stopping_criteria=criteria if criteria else None,
        )
    new_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def _generate_once(
    model: Any,
    tokenizer: Optional[PreTrainedTokenizerBase],
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    use_numeric_bias: bool = False,
    question: str = "",
    stop: list[str] | None = None,
) -> str:
    """Backend-agnostic single generation helper."""
    if _is_vllm_model(model):
        return _vllm_generate_texts(
            model=model,
            prompts=[prompt],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=1,
            extra_stop=stop,
        )[0][0]
    if tokenizer is None:
        raise ValueError("tokenizer is required for Transformers generation.")
    return _generate_once_transformers(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_numeric_bias=use_numeric_bias,
        question=question,
        stop_sequences=stop,
    )



_TOOL_CALL_STOP = ["</tool_call>"]
_CORRECTION_MSG = (
    "Your last response was not a valid tool call. "
    "Emit exactly one complete tool call and nothing else:\n"
    '<tool_call>{"name":"arithmetic","arguments":{"a":...,"b":...,"operation":"..."}}</tool_call>'
)


def generate_with_tools(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: str = "",
    tools: list[dict[str, Any]] = FINANCIAL_TOOLS,
    max_new_tokens: int = 384,
    max_tool_rounds: int = 12,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Run a strict one-tool-call-per-round loop and return the final answer.

    Design decisions vs. the old implementation
    -------------------------------------------
    * use_cot=False: CoT preamble encourages long explanations before tool calls,
      wasting the per-round token budget and triggering multi-call emissions.
    * max_new_tokens=384 per round: enough for one compact tool-call JSON.
    * stop=["</tool_call>"]: generation halts on the closing tag, so the loop
      never receives a truncated or multi-call turn.
    * Closing-tag repair: if include_stop_str_in_output is unsupported, the tag
      is restored so the parser always sees a complete block.
    * Normalized assistant message: only a clean <tool_call>{...}</tool_call> is
      fed back; raw_output preamble stays in full_outputs for debugging only.
    * role="user" for tool results: avoids Qwen native tool-call schema conflict.
    * Correction retries: malformed calls get a repair prompt instead of early exit.
    """
    del tools  # described in the system prompt; not passed to the chat template

    messages = _build_messages(question=question, context=context, use_cot=False, use_tools=True)
    tool_calls: list[dict[str, Any]] = []
    full_outputs: list[str] = []
    last_call_signature: Optional[str] = None
    retries_remaining = max_retries

    for _ in range(max_tool_rounds + max_retries + 1):
        prompt = _apply_chat_template(tokenizer, messages)
        raw_output = _generate_once(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            question=question,
            stop=_TOOL_CALL_STOP,
        )
        full_outputs.append(raw_output)

        # Restore closing tag when the backend strips the stop string.
        if "<tool_call>" in raw_output and "</tool_call>" not in raw_output:
            raw_output = raw_output.rstrip() + "</tool_call>"
            full_outputs[-1] = raw_output

        parsed = parse_tool_call_from_output(raw_output)

        # ── No tool call block → treat as final answer turn ───────────────
        if parsed is None:
            answer, _ = extract_final_answer(raw_output)
            return _tool_result(answer, tool_calls, full_outputs, max_rounds_exceeded=False)

        # ── Malformed call → correction retry ────────────────────────────
        if not parsed["ok"]:
            logger.warning(
                "Malformed tool call (retries_remaining=%d): %s",
                retries_remaining,
                parsed["error"],
            )
            full_outputs.append(f"[PARSE_ERROR: {parsed['error']}]")
            if retries_remaining <= 0:
                answer, _ = extract_final_answer(raw_output)
                return _tool_result(answer, tool_calls, full_outputs, max_rounds_exceeded=False)
            retries_remaining -= 1
            messages.append({"role": "user", "content": _CORRECTION_MSG})
            continue

        call = parsed["call"]

        # ── Repetition guard ──────────────────────────────────────────────
        call_signature = json.dumps(call, sort_keys=True)
        if call_signature == last_call_signature:
            logger.warning(
                "Repeated identical tool call (%s %s); breaking loop.",
                call["name"],
                call["arguments"],
            )
            answer, _ = extract_final_answer(raw_output)
            return _tool_result(answer, tool_calls, full_outputs, max_rounds_exceeded=False)
        last_call_signature = call_signature
        retries_remaining = max_retries  # reset on each valid call

        # ── Dispatch ──────────────────────────────────────────────────────
        tool_result_data = dispatch_tool_call(call["name"], call["arguments"])
        tool_calls.append({
            "name": call["name"],
            "arguments": call["arguments"],
            "result": tool_result_data,
        })

        # Feed back only the normalized call — not raw_output, which may
        # contain preamble text that pollutes the next turn.
        normalized_call = (
            "<tool_call>"
            + json.dumps({"name": call["name"], "arguments": call["arguments"]}, ensure_ascii=False)
            + "</tool_call>"
        )
        messages.append({"role": "assistant", "content": normalized_call})

        # role="user" (not role="tool") avoids Qwen native tool-call schema.
        result_text = json.dumps(tool_result_data, ensure_ascii=False)
        messages.append({
            "role": "user",
            "content": (
                f"Tool result: {result_text}\n\n"
                "Now either emit the next single tool call, or — if all "
                "computations are done — output exactly:\n"
                "Final Answer: <bare number only>"
            ),
        })

    answer, _ = extract_final_answer(full_outputs[-1])
    return _tool_result(answer, tool_calls, full_outputs, max_rounds_exceeded=True)



def generate_answer(
    model: Any,
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
    """Generate a single answer with optional CoT or tool support."""
    del num_beams
    if use_tools:
        if tokenizer is None:
            raise ValueError("tokenizer is required for tool-augmented generation.")
        tool_result = generate_with_tools(
            model,
            tokenizer,
            question=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )
        answer = postprocess_answer(tool_result["answer"] or "")
    elif self_consistency_n > 1:
        if tokenizer is None:
            raise ValueError("tokenizer is required for self-consistency generation.")
        from src.inference.self_consistency import sample_with_self_consistency

        prompt = build_prompt(
            question=question,
            context=context,
            instruction=instruction,
            use_cot=use_cot,
            tokenizer=tokenizer,
        )
        answer, confidence, _ = sample_with_self_consistency(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            n=self_consistency_n,
            temperature=temperature or 0.7,
            max_new_tokens=max_new_tokens,
            min_confidence=min_confidence,
        )
        if confidence < min_confidence:
            return INSUFFICIENT_INFO_RESPONSE
        answer = postprocess_answer(answer)
    else:
        prompt = build_prompt(
            question=question,
            context=context,
            instruction=instruction,
            use_cot=use_cot,
            tokenizer=tokenizer,
        )
        raw_output = _generate_once(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_numeric_bias=use_numeric_bias,
            question=question,
        )
        answer = extract_answer_from_cot_output(raw_output) if use_cot else postprocess_answer(raw_output)

    if grounding_check and context.strip() and not check_grounding(answer, context):
        return INSUFFICIENT_INFO_RESPONSE
    if not answer or _REFUSAL_RE.search(answer):
        return INSUFFICIENT_INFO_RESPONSE
    return answer



def batch_generate(
    model: Any,
    tokenizer: Optional[PreTrainedTokenizerBase],
    samples: list[dict[str, Any]],
    use_cot: bool = False,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[str]:
    """Generate answers for a list of samples."""
    del batch_size
    answers: list[str] = []
    for sample in samples:
        answers.append(
            generate_answer(
                model=model,
                tokenizer=tokenizer,
                question=str(sample.get("question", "")),
                context=str(sample.get("context", "")),
                instruction=str(sample.get("instruction", "")),
                use_cot=use_cot,
                max_new_tokens=max_new_tokens,
                grounding_check=False,
            )
        )
    return answers


if __name__ == "__main__":
    import argparse
    from importlib import import_module
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="FinReasoningAI inference CLI")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/sft_qlora/final_adapter")
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument("--use_tools", action="store_true")
    parser.add_argument("--self_consistency_n", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--engine", choices=["vllm", "transformers"], default="transformers")
    args = parser.parse_args()

    if args.engine == "vllm":
        from src.model.load_model import load_vllm_model_and_tokenizer

        model, tokenizer = load_vllm_model_and_tokenizer(args.model_id, enable_lora=False)
    else:
        from src.model.load_model import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(args.model_id)
        if args.adapter_dir != "none" and Path(args.adapter_dir).exists():
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
    )
    print(f"Answer: {answer}")
