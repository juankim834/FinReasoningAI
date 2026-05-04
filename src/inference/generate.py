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
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

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
    "You may call available tools when a computation would improve accuracy. "
    "When you want to use one, emit exactly one JSON payload inside <tool_call>...</tool_call> "
    'using the shape {"name": <tool_name>, "arguments": {...}}.'
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



def _build_vllm_sampling_params(*, temperature: float, top_p: float, max_new_tokens: int, n: int = 1) -> Any:
    """Create vLLM sampling parameters while tolerating version drift."""
    from importlib import import_module

    SamplingParams = import_module("vllm").SamplingParams
    supported = set(inspect.signature(SamplingParams).parameters)
    params = {
        "n": n,
        "temperature": temperature,
        "top_p": top_p if temperature > 0 else 1.0,
        "max_tokens": max_new_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
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
) -> list[list[str]]:
    """Generate one or more completions for each prompt with vLLM."""
    outputs = model.generate(
        prompts,
        _build_vllm_sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            n=n,
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
    """Build the system prompt used for tool-augmented inference."""
    tool_block = json.dumps(tools or FINANCIAL_TOOLS, indent=2, ensure_ascii=False)
    return f"{COT_SYSTEM_PROMPT}\n\n{_TOOL_PROMPT_INTRO}\n\n## Available Tools\n{tool_block}"



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
) -> str:
    """Run a single Hugging Face generation and decode the new tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900, padding=False).to(model.device)
    do_sample = temperature > 0
    processors = LogitsProcessorList()
    if use_numeric_bias:
        numeric_processor = NumericBiasLogitsProcessor(tokenizer)
        if re.search(r"(how much|how many|calculate|compute|ratio|margin|growth|rate)", question, re.IGNORECASE):
            numeric_processor.activate()
            processors.append(numeric_processor)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=processors if processors else None,
        )
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
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
    )



def generate_with_tools(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: str = "",
    tools: list[dict[str, Any]] = FINANCIAL_TOOLS,
    max_new_tokens: int = 512,
    max_tool_rounds: int = 3,
) -> dict[str, Any]:
    """Run a simple tool-augmented reasoning loop and return the final answer."""
    messages = _build_messages(question=question, context=context, use_cot=True, use_tools=True, tools=tools)
    tool_calls: list[dict[str, Any]] = []
    full_output = ""

    for _ in range(max_tool_rounds + 1):
        prompt = _apply_chat_template(tokenizer, messages, tools=tools)
        raw_output = _generate_once(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=0.9,
            question=question,
        )
        full_output = raw_output
        parsed_call = parse_tool_call_from_output(raw_output)
        if parsed_call is None:
            answer = extract_answer_from_cot_output(raw_output)
            return {"answer": answer, "tool_calls": tool_calls, "full_output": raw_output}

        tool_result = dispatch_tool_call(parsed_call["name"], parsed_call["arguments"])
        tool_calls.append({
            "name": parsed_call["name"],
            "arguments": parsed_call["arguments"],
            "result": tool_result,
        })
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": (
                f"Tool result for {parsed_call['name']}: "
                f"{json.dumps(tool_result, ensure_ascii=False)}\n"
                "Use this result to produce your final answer."
            ),
        })

    answer = extract_answer_from_cot_output(full_output)
    return {"answer": answer, "tool_calls": tool_calls, "full_output": full_output}



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
            tools=FINANCIAL_TOOLS,
            max_new_tokens=max_new_tokens,
        )
        answer = postprocess_answer(tool_result["answer"])
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
