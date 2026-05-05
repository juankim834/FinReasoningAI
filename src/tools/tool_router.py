"""Route tool calls emitted by the model to Python implementations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from tools.financial_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Detects any <tool_call> opening tag so we can catch unclosed / multi-call emissions.
_TOOL_CALL_OPEN_PATTERN = re.compile(r"<tool_call>", re.DOTALL)

# Allowed tool names and valid arithmetic operations for schema validation.
_ALLOWED_TOOLS = {"arithmetic", "compound_growth_rate"}
_ARITHMETIC_OPS = {"add", "subtract", "multiply", "divide", "percent_change"}


def _strip_code_fences(text: str) -> str:
    """Remove optional markdown code fences around JSON payloads."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Extract the first balanced JSON object from arbitrary tool-call text.

    This handles cases where the model adds commentary before or after the JSON.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def dispatch_tool_call(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Route a parsed tool call to its implementation and return a JSON-safe payload."""
    tool_fn = TOOL_REGISTRY.get(tool_name)
    if tool_fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = tool_fn(**tool_args)
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        logger.exception("Tool execution failed for %s", tool_name)
        return {"error": f"Tool execution failed: {exc}"}
    return result if isinstance(result, dict) else {"result": result}



def parse_tool_call_from_output(model_output: str) -> Optional[dict[str, Any]]:
    """Parse a custom <tool_call>...</tool_call> block from model output.

    Returns:
        ``None``
            No ``<tool_call>`` block was found.  The caller should treat this
            turn as the model's final answer.
        ``{"ok": True, "call": {"name": ..., "arguments": {...}}, "error": None}``
            A single, well-formed tool call was parsed and validated.
        ``{"ok": False, "call": None, "error": "<reason>"}``
            A block was found but is malformed (multiple blocks, bad JSON,
            missing fields, or invalid enum value).  The caller should log the
            error and may treat this turn as a failed generation.
    """
    matches = _TOOL_CALL_PATTERN.findall(model_output)

    if not matches:
        # Check whether the model emitted <tool_call> without a closing tag.
        # This happens when the model concatenates multiple raw calls or generation
        # is truncated mid-tag (e.g. by max_new_tokens).
        open_count = len(_TOOL_CALL_OPEN_PATTERN.findall(model_output))
        if open_count > 0:
            msg = (
                f"Found {open_count} <tool_call> opening tag(s) but no matching "
                "</tool_call> closing tag. The model emitted multiple tool calls in "
                "one turn or generation was truncated. Treating as malformed."
            )
            logger.warning(msg)
            return {"ok": False, "call": None, "error": msg}
        return None

    if len(matches) > 1:
        msg = f"Multiple <tool_call> blocks found ({len(matches)}); emit exactly one per turn."
        logger.warning(msg)
        return {"ok": False, "call": None, "error": msg}

    raw_payload = _strip_code_fences(matches[0])
    json_payload = _extract_balanced_json(raw_payload) or raw_payload

    try:
        payload = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        msg = f"JSON decode error: {exc} | raw: {raw_payload[:200]}"
        logger.warning("Tool call JSON parse failed: %s", msg)
        return {"ok": False, "call": None, "error": msg}

    name = payload.get("name")
    arguments = payload.get("arguments", {})

    if not isinstance(name, str):
        msg = f"'name' missing or not a string: {payload}"
        return {"ok": False, "call": None, "error": msg}

    if not isinstance(arguments, dict):
        msg = f"'arguments' missing or not a dict: {payload}"
        return {"ok": False, "call": None, "error": msg}

    if name not in _ALLOWED_TOOLS:
        msg = f"Unknown tool '{name}'. Allowed: {sorted(_ALLOWED_TOOLS)}"
        logger.warning(msg)
        return {"ok": False, "call": None, "error": msg}

    if name == "arithmetic":
        op = str(arguments.get("operation", ""))
        if op not in _ARITHMETIC_OPS:
            msg = (
                f"arithmetic.operation '{op}' not in allowed set "
                f"{sorted(_ARITHMETIC_OPS)}"
            )
            return {"ok": False, "call": None, "error": msg}

    return {"ok": True, "call": {"name": name, "arguments": arguments}, "error": None}


class ToolRouter:
    """Backward-compatible wrapper around the module-level dispatch helpers."""

    def parse_action(self, text: str) -> Optional[tuple[str, str]]:
        """Return a legacy action tuple when a tool call is present."""
        parsed = parse_tool_call_from_output(text)
        if parsed is None or not parsed.get("ok"):
            return None
        call = parsed["call"]
        return call["name"], json.dumps(call["arguments"])

    def execute(self, tool_name: str, args_str: str) -> str:
        """Execute a tool using a JSON argument string."""
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {}
        return json.dumps(dispatch_tool_call(tool_name, arguments), ensure_ascii=False)


__all__ = ["dispatch_tool_call", "parse_tool_call_from_output", "ToolRouter"]
