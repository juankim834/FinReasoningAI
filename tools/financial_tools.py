"""Financial tool definitions and implementations for tool-augmented inference."""

from __future__ import annotations

from typing import Any

from tools.number_parser import extract_number

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
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"},
                    "operation": {
                        "type": "string",
                        "description": (
                            "One of: add | subtract | multiply | divide | percent_change. "
                            "percent_change computes (a - b) / |b|."
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
            "name": "calculate_financial_ratio",
            "description": "Compute a standard financial ratio from two numeric values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {"type": "number", "description": "The numerator value"},
                    "denominator": {"type": "number", "description": "The denominator value"},
                    "ratio_name": {"type": "string", "description": "E.g. 'P/E ratio', 'gross margin'"},
                },
                "required": ["numerator", "denominator", "ratio_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_percentage",
            "description": "Parse and normalize a percentage string to a float.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value_str": {"type": "string", "description": "E.g. '12.5%' or '0.125'"},
                },
                "required": ["value_str"],
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
                    "start_value": {"type": "number"},
                    "end_value": {"type": "number"},
                    "n_periods": {"type": "number"},
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


TOOL_REGISTRY = {
    "arithmetic": arithmetic,
    "calculate_financial_ratio": calculate_financial_ratio,
    "parse_percentage": parse_percentage,
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
