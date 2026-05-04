"""Financial tool definitions and implementations for tool-augmented inference."""

from __future__ import annotations

from typing import Any

from tools.number_parser import extract_number

# Benchmark tool set: arithmetic + compound_growth_rate only.
# calculate_financial_ratio and parse_percentage are removed to reduce
# ambiguity (ratio = divide, parse_percentage adds unnecessary extra calls).
FINANCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "arithmetic",
            "description": (
                "Perform basic arithmetic on two numbers. "
                "Use this for any addition, subtraction, multiplication, division, "
                "or percentage-change computation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First operand (new/later value for percent_change)"},
                    "b": {"type": "number", "description": "Second operand (old/earlier value for percent_change)"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide", "percent_change"],
                        "description": (
                            "Operation to perform. "
                            "add: a+b. subtract: a-b. multiply: a*b. divide: a/b. "
                            "percent_change: (a-b)/|b| where a is the new/later value "
                            "and b is the old/earlier value."
                        ),
                    },
                },
                "required": ["a", "b", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compound_growth_rate",
            "description": "Calculate CAGR given start value, end value, and number of periods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_value": {"type": "number", "description": "Starting value (must be > 0)"},
                    "end_value": {"type": "number", "description": "Ending value (must be >= 0)"},
                    "n_periods": {"type": "number", "description": "Number of periods (must be > 0)"},
                },
                "required": ["start_value", "end_value", "n_periods"],
            },
        },
    },
]


def arithmetic(a: float, b: float, operation: str) -> dict[str, Any]:
    """Perform basic arithmetic: add, subtract, multiply, divide, or percent_change."""
    a, b = float(a), float(b)
    op = operation.lower().strip()
    if op in ("add", "+"):
        result = a + b
        expr = f"{a} + {b}"
    elif op in ("subtract", "sub", "-"):
        result = a - b
        expr = f"{a} - {b}"
    elif op in ("multiply", "mul", "*"):
        result = a * b
        expr = f"{a} * {b}"
    elif op in ("divide", "div", "/"):
        if b == 0:
            return {"error": "Division by zero."}
        result = a / b
        expr = f"{a} / {b}"
    elif op in ("percent_change", "pct_change", "pct_chg", "%_change"):
        if b == 0:
            return {"error": "Base value b cannot be zero for percent_change."}
        result = (a - b) / abs(b)
        expr = f"({a} - {b}) / |{b}|"
    else:
        return {
            "error": (
                f"Unknown operation '{operation}'. "
                "Use: add, subtract, multiply, divide, or percent_change."
            )
        }
    return {"result": result, "explanation": f"Computed {expr} = {result:.6f}."}


def calculate_financial_ratio(numerator: float, denominator: float, ratio_name: str) -> dict[str, Any]:
    """Calculate a named financial ratio and explain the result."""
    if denominator == 0:
        return {"error": "Denominator cannot be zero."}
    result = float(numerator) / float(denominator)
    explanation = f"Computed {ratio_name} as {numerator} / {denominator} = {result:.6f}."
    return {"result": result, "explanation": explanation}



def parse_percentage(value_str: str) -> dict[str, Any]:
    """Parse a percentage-like string into a normalized decimal float."""
    raw = value_str.strip()
    if not raw:
        return {"error": "value_str cannot be empty."}

    parsed = extract_number(raw)
    if parsed is None:
        return {"error": f"Unable to parse percentage value: {value_str}"}

    if "%" in raw or "percent" in raw.lower():
        normalized = parsed if abs(parsed) <= 1 else parsed / 100.0
    else:
        normalized = parsed
    explanation = f"Normalized '{value_str}' to decimal value {normalized:.6f}."
    return {"result": normalized, "explanation": explanation}



def compound_growth_rate(start_value: float, end_value: float, n_periods: float) -> dict[str, Any]:
    """Calculate CAGR and explain the intermediate formula."""
    if start_value <= 0:
        return {"error": "start_value must be greater than zero."}
    if end_value < 0:
        return {"error": "end_value must be non-negative."}
    if n_periods <= 0:
        return {"error": "n_periods must be greater than zero."}

    result = (float(end_value) / float(start_value)) ** (1.0 / float(n_periods)) - 1.0
    explanation = (
        "Computed CAGR as "
        f"(({end_value} / {start_value}) ** (1 / {n_periods})) - 1 = {result:.6f}."
    )
    return {"result": result, "explanation": explanation}


# Only the two benchmark tools are in the active registry.
# The full implementations are kept below for potential future use.
TOOL_REGISTRY = {
    "arithmetic": arithmetic,
    "compound_growth_rate": compound_growth_rate,
}

__all__ = [
    "FINANCIAL_TOOLS",
    "TOOL_REGISTRY",
    "arithmetic",
    "calculate_financial_ratio",
    "parse_percentage",
    "compound_growth_rate",
]
