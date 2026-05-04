"""Route tool calls emitted by the model to Python implementations."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from tools.financial_tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


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
    """Parse Qwen-style tool-call JSON from model output."""
    match = _TOOL_CALL_PATTERN.search(model_output)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        logger.warning("Unable to decode tool call payload: %s", match.group(1))
        return None

    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


class ToolRouter:
    """Backward-compatible wrapper around the module-level dispatch helpers."""

    def parse_action(self, text: str) -> Optional[tuple[str, str]]:
        """Return a legacy action tuple when a tool call is present."""
        parsed = parse_tool_call_from_output(text)
        if parsed is None:
            return None
        return parsed["name"], json.dumps(parsed["arguments"])

    def execute(self, tool_name: str, args_str: str) -> str:
        """Execute a tool using a JSON argument string."""
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {}
        return json.dumps(dispatch_tool_call(tool_name, arguments), ensure_ascii=False)


__all__ = ["dispatch_tool_call", "parse_tool_call_from_output", "ToolRouter"]
