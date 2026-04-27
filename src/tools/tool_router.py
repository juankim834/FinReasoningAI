"""
File: src/tools/tool_router.py

Step 7b & 7c — Tool-Augmented Reasoning & ReAct Agent Loop

Tools available:
  1. calculator:    Safe arithmetic evaluation (Python AST, no builtins).
  2. table_parser:  Parse markdown/CSV financial tables into structured data.
  3. rag_retrieve:  Query the FAISS index for relevant document passages.

ReAct agent loop (Step 7c):
  - Follows: Thought → Action → Observation → (repeat) → Answer
  - Max 5 reasoning steps to bound latency.
  - Each step uses the model to generate the next Thought+Action.
  - Tool outputs (Observations) are injected back into the context.
  - If the model produces a final "Answer:" tag, the loop terminates.

Agent prompt template:
  The model is given a system prompt explaining the tool interface and a
  growing transcript of Thought/Action/Observation turns.

[WARN] TRADE-OFF: ReAct adds latency proportional to N steps × generation time.
For a 14B model at 4-bit, each step ≈ 1–2 seconds on A100.
5 steps → ~5–10 seconds total. For sub-second latency, disable ReAct and
use direct generation with pre-retrieved context instead.
"""

from __future__ import annotations

import io
import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ──────────────────────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
Available tools:

1. calculate(expression: str) → float
   Evaluate a mathematical expression. Use standard Python math operators.
   Variables: only numeric literals (no external variables).
   Example: calculate((394.3 - 365.8) / 365.8 * 100)

2. parse_table(text: str, format: "markdown"|"csv") → dict
   Parse a financial table from markdown or CSV text into a structured dict.
   Returns {"columns": [...], "rows": [[...], ...], "summary": str}
   Example: parse_table("| Year | Revenue |\\n|------|---------|\\n| 2022 | $394B |", "markdown")

3. retrieve(query: str, top_k: int = 2) → str
   Search the financial document index for relevant passages.
   Returns up to top_k most relevant text passages.
   Example: retrieve("Apple 2022 annual revenue guidance")

To use a tool, output EXACTLY:
Action: tool_name(arguments)

To give a final answer, output EXACTLY:
Answer: <your final answer here>
"""

REACT_SYSTEM_PROMPT = f"""You are FinReasoning AI, an expert financial analyst with access to tools.
{TOOL_DESCRIPTIONS}
Think step by step. Use tools when you need to calculate numbers or look up information.
Limit yourself to 5 reasoning steps maximum. Be concise in your Thoughts.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ──────────────────────────────────────────────────────────────────────────────

def tool_calculate(expression: str) -> str:
    """Calculator tool — delegates to safe_calculate from generate.py."""
    from src.inference.generate import safe_calculate
    result = safe_calculate(expression)
    if isinstance(result, float):
        if abs(result) >= 1e9:
            return f"{result:.4e}"
        elif abs(result) >= 1e6:
            return f"{result:,.2f}"
        else:
            return f"{result:.6g}"
    return str(result)


def tool_parse_table(text: str, format: str = "markdown") -> str:
    """
    Table parser — converts markdown or CSV tables to JSON-serializable dict.

    For financial tables, this handles:
      - Multi-row headers
      - Currency formatting ($1.2B, $394,321M)
      - Percentage values (12.3%)
      - Mixed text/numeric cells
    """
    import pandas as pd

    try:
        if format == "csv":
            df = pd.read_csv(io.StringIO(text))
        elif format == "markdown":
            # Parse markdown table format
            lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
            # Remove separator row (---|---|...)
            lines = [l for l in lines if not re.match(r"^\|[-: |]+\|$", l)]
            # Parse header + rows
            if not lines:
                return json.dumps({"error": "Empty table"})
            header = [h.strip() for h in lines[0].strip("|").split("|")]
            rows = []
            for line in lines[1:]:
                cells = [c.strip() for c in line.strip("|").split("|")]
                rows.append(cells)
            df = pd.DataFrame(rows, columns=header)
        else:
            return json.dumps({"error": f"Unknown format: {format}"})

        # Clean numeric cells
        for col in df.columns:
            df[col] = df[col].apply(_clean_financial_cell)

        summary = f"Table with {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}"
        return json.dumps({
            "columns": list(df.columns),
            "rows": df.values.tolist(),
            "summary": summary,
        }, default=str)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _clean_financial_cell(cell: Any) -> Any:
    """
    Parse financial cell values:
      "$1.2B" → 1.2e9
      "12.3%" → 12.3 (keep as percentage string)
      "1,234,567" → 1234567.0
    """
    if not isinstance(cell, str):
        return cell
    cell = cell.strip().replace(",", "")

    # Strip currency symbol
    cell = re.sub(r"^\$", "", cell)

    # Check for unit suffix
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    match = re.match(r"^(-?\d+\.?\d*)\s*([TBMK%]?)$", cell, re.IGNORECASE)
    if match:
        num_str = match.group(1)
        unit = match.group(2).upper()
        try:
            val = float(num_str)
            if unit in multipliers:
                return val * multipliers[unit]
            if unit == "%":
                return f"{val}%"  # Keep as string to preserve percentage semantics
            return val
        except ValueError:
            pass

    # Try plain float
    try:
        return float(cell)
    except ValueError:
        return cell  # Return as-is for non-numeric cells


def tool_retrieve(
    query: str,
    retriever=None,
    top_k: int = 2,
) -> str:
    """RAG retrieval tool — returns formatted passage text."""
    if retriever is None:
        return (
            "[RAG not configured. Initialize FinancialRetriever and pass it to ToolRouter.]"
        )
    try:
        chunks = retriever.retrieve_and_rerank(query, top_k=top_k)
        context = retriever.format_context(chunks, max_chars=2000)
        return context if context else "[No relevant passages found.]"
    except Exception as exc:
        return f"[Retrieval error: {exc}]"


# ──────────────────────────────────────────────────────────────────────────────
# Tool router
# ──────────────────────────────────────────────────────────────────────────────

class ToolRouter:
    """
    Routes tool calls from the model's output to the appropriate tool function.

    Parses the pattern: Action: tool_name(arguments)
    """

    def __init__(self, retriever=None) -> None:
        self.retriever = retriever
        self._tool_map = {
            "calculate": tool_calculate,
            "parse_table": tool_parse_table,
            "retrieve": self._retrieve_wrapper,
        }

    def _retrieve_wrapper(self, query: str, top_k: int = 2) -> str:
        return tool_retrieve(query, retriever=self.retriever, top_k=top_k)

    def parse_action(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract (tool_name, arguments_str) from a model-generated action line.
        Returns None if no valid action found.
        """
        # Match: Action: tool_name(args)
        match = re.search(r"Action:\s*(\w+)\((.+?)\)\s*$", text, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        # Also match without arguments
        match = re.search(r"Action:\s*(\w+)\(\)\s*$", text, re.MULTILINE)
        if match:
            return match.group(1).strip(), ""
        return None

    def execute(self, tool_name: str, args_str: str) -> str:
        """
        Execute the named tool with the given arguments string.
        Safely parses the argument string using ast.literal_eval where possible.
        """
        import ast

        tool_fn = self._tool_map.get(tool_name)
        if tool_fn is None:
            return f"[Unknown tool: {tool_name}. Available: {list(self._tool_map.keys())}]"

        # Parse arguments
        try:
            # Try to parse as a single string argument first
            if tool_name == "calculate":
                # Special case: expression may contain operators, don't literal_eval
                result = tool_fn(args_str.strip("'\""))
            elif tool_name == "retrieve":
                # May have optional top_k kwarg
                if "," in args_str:
                    parts = [p.strip() for p in args_str.split(",", 1)]
                    query = parts[0].strip("'\" ")
                    try:
                        top_k = int(re.search(r"\d+", parts[1]).group())
                    except (AttributeError, ValueError):
                        top_k = 2
                    result = tool_fn(query, top_k=top_k)
                else:
                    result = tool_fn(args_str.strip("'\" "))
            elif tool_name == "parse_table":
                # The format token ('markdown' or 'csv') is always the last
                # comma-separated value. Everything before it is the table text,
                # which may itself contain commas and real/literal newlines.
                fmt = "markdown"
                text_part = args_str.strip()
                fmt_match = re.search(r",\s*['\"]?(markdown|csv)['\"]?\s*$", text_part)
                if fmt_match:
                    fmt = fmt_match.group(1)
                    text_part = text_part[: fmt_match.start()]
                text_part = text_part.strip().strip("'\"")
                # Normalise literal two-char \n sequences to real newlines so
                # the markdown line-splitter works regardless of how the caller
                # built the args string.
                text_part = text_part.replace("\\n", "\n")
                result = tool_fn(text_part, fmt)
            else:
                # Generic: try to parse as keyword args
                try:
                    parsed = ast.literal_eval(f"dict({args_str})")
                    result = tool_fn(**parsed)
                except (ValueError, SyntaxError):
                    result = tool_fn(args_str.strip("'\" "))
        except Exception as exc:
            result = f"[Tool error in {tool_name}: {exc}]"

        return str(result)


# ──────────────────────────────────────────────────────────────────────────────
# ReAct Agent Loop
# ──────────────────────────────────────────────────────────────────────────────

REACT_AGENT_PROMPT_TEMPLATE = """\
<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{instruction}

Context:
{context}

Question: {question}
<|im_end|>
<|im_start|>assistant
{transcript}"""

ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?:\n|$)")


def react_agent(
    model,
    tokenizer,
    question: str,
    context: str = "",
    tool_router: Optional[ToolRouter] = None,
    instruction: str = "Answer the following financial question using the available tools.",
    max_steps: int = 5,
    max_new_tokens_per_step: int = 200,
    temperature: float = 0.2,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    ReAct-style agent loop for complex financial reasoning.

    Loop:
      1. Generate next Thought + Action from the current transcript.
      2. If model outputs "Answer:", terminate and return.
      3. Execute the action and append Observation to transcript.
      4. Repeat up to max_steps times.
      5. If max_steps exceeded, generate a final direct answer.

    Args:
        model:       Fine-tuned model.
        tokenizer:   Matching tokenizer.
        question:    The question to answer.
        context:     Optional pre-retrieved context.
        tool_router: ToolRouter instance (if None, tools are unavailable).
        max_steps:   Maximum reasoning steps (default 5 to bound latency).
        temperature: Low temperature for more deterministic tool selection.

    Returns:
        (final_answer, reasoning_transcript)

    Example transcript:
        Thought: I need to calculate the revenue growth rate.
        Action: calculate((394.3 - 365.8) / 365.8 * 100)
        Observation: 7.791...
        Thought: The growth rate is 7.79%.
        Answer: Apple's revenue grew by 7.79% from 2021 to 2022.
    """
    if tool_router is None:
        tool_router = ToolRouter()

    transcript_parts: List[str] = []
    step_log: List[Dict[str, str]] = []

    for step in range(max_steps):
        # Build current transcript text
        transcript = "\n".join(transcript_parts)

        # Build full prompt
        prompt = REACT_AGENT_PROMPT_TEMPLATE.format(
            system_prompt=REACT_SYSTEM_PROMPT,
            instruction=instruction,
            context=context[:2000] if context else "(No context provided)",
            question=question,
            transcript=transcript + ("\n" if transcript else ""),
        )

        # Generate next step
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3800,
        ).to(model.device)

        import torch
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens_per_step,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Stop at newline after Answer: to avoid over-generation
            )

        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        step_output = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        logger.debug("Step %d output: %s", step + 1, step_output[:200])

        # Check for final answer
        answer_match = ANSWER_RE.search(step_output)
        if answer_match:
            final_answer = answer_match.group(1).strip()
            step_log.append({
                "step": step + 1,
                "type": "answer",
                "content": final_answer,
            })
            transcript_parts.append(step_output)
            return final_answer, step_log

        # Check for action
        action_result = tool_router.parse_action(step_output)
        if action_result:
            tool_name, args_str = action_result
            observation = tool_router.execute(tool_name, args_str)

            step_log.append({
                "step": step + 1,
                "type": "action",
                "tool": tool_name,
                "args": args_str,
                "observation": observation,
                "thought": step_output,
            })

            # Append to transcript
            transcript_parts.append(step_output)
            transcript_parts.append(f"Observation: {observation}")

            logger.info(
                "Step %d: %s(%s) → %s",
                step + 1, tool_name, args_str[:50], str(observation)[:80]
            )
        else:
            # Model produced a thought but no action — continue
            transcript_parts.append(step_output)
            step_log.append({
                "step": step + 1,
                "type": "thought",
                "content": step_output,
            })

    # Max steps exceeded — generate a direct final answer
    logger.warning("Max steps (%d) exceeded. Generating direct final answer.", max_steps)

    from src.inference.generate import generate_answer
    final_answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=question,
        context=context + "\n\n" + "\n".join(transcript_parts),
    )
    step_log.append({
        "step": max_steps + 1,
        "type": "fallback_answer",
        "content": final_answer,
    })

    return final_answer, step_log


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="ToolRouter and ReAct agent demo")
    parser.add_argument("--demo_tool", choices=["calculate", "parse_table", "react"],
                        default="calculate")
    args = parser.parse_args()

    router = ToolRouter()

    if args.demo_tool == "calculate":
        expr = "(394.3 - 365.8) / 365.8 * 100"
        result = router.execute("calculate", expr)
        print(f"calculate({expr}) = {result}")

    elif args.demo_tool == "parse_table":
        md_table = textwrap.dedent("""
            | Year | Revenue | Net Income | EPS   |
            |------|---------|------------|-------|
            | 2022 | $394.3B | $99.8B     | $6.11 |
            | 2021 | $365.8B | $94.7B     | $5.61 |
            | 2020 | $274.5B | $57.4B     | $3.28 |
        """).strip()
        result = router.execute("parse_table", f"'{md_table}', 'markdown'")
        print(f"parse_table result:\n{result}")

    elif args.demo_tool == "react":
        print("ReAct agent demo requires a loaded model.")
        print("Run with FINREASONING_MODEL_ID set to use the full agent.")

    print("\n[OK] Step 7b/c complete — ToolRouter and ReAct agent ready.")
