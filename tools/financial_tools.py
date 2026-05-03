"""Utility financial tools for safe expression evaluation, ratio computation, and table parsing."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd
from sympy import SympifyError, sympify


def calculator(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression safely using SymPy.

    The function supports basic arithmetic, percentages, exponents, and nested
    expressions. Percentage syntax is normalized by converting `N%` into
    `(N/100)` before evaluation.

    Args:
        expression: A string mathematical expression (for example,
            ``"(1000 * 0.05) / (1 - 0.03**2)"``).

    Returns:
        A dictionary with the following shape:
        - On success: ``{"result": float, "expression": str, "error": None}``
        - On failure: ``{"result": None, "expression": str, "error": str}``
    """
    try:
        normalized = expression.replace("%", "/100")
        value = sympify(normalized, evaluate=True)

        if not value.is_real:
            raise ValueError("Expression did not evaluate to a real number.")

        result = float(value)
        return {"result": result, "expression": expression, "error": None}
    except (SympifyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {"result": None, "expression": expression, "error": str(exc)}


def financial_ratio(
    numerator: float,
    denominator: float,
    as_percent: bool = True,
) -> dict[str, Any]:
    """Compute a financial ratio safely with optional percentage formatting.

    Args:
        numerator: Ratio numerator value.
        denominator: Ratio denominator value.
        as_percent: When ``True``, format as a percent string using four decimal
            places (e.g., ``"12.5000%"``). When ``False``, format as a decimal
            string using six decimal places (e.g., ``"0.125000"``).

    Returns:
        A dictionary with the following shape:
        - On success: ``{"result": float, "formatted": str, "error": None}``
        - On failure: ``{"result": None, "formatted": "", "error": str}``
    """
    try:
        if denominator == 0:
            raise ZeroDivisionError("Denominator cannot be zero.")

        result = float(numerator) / float(denominator)
        formatted = f"{result * 100:.4f}%" if as_percent else f"{result:.6f}"
        return {"result": result, "formatted": formatted, "error": None}
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return {"result": None, "formatted": "", "error": str(exc)}


def parse_table(raw: str) -> dict[str, Any]:
    """Parse tabular text as CSV first, then markdown, then fail with raw echo.

    Parsing strategy:
    1. Attempt CSV parsing with pandas.
    2. If CSV fails or yields no usable columns, attempt markdown table parsing.
    3. If both parsing strategies fail, return a structured error payload and
       include the original raw string.

    Args:
        raw: Raw text containing tabular content.

    Returns:
        A dictionary with the following shape:
        - On success: ``{"records": list[dict], "columns": list[str], "error": None}``
        - On failure: ``{"records": [], "columns": [], "error": str, "raw": str}``
    """

    def _frame_to_output(df: pd.DataFrame) -> dict[str, Any]:
        columns = [str(col) for col in df.columns]
        records = df.to_dict(orient="records")
        return {"records": records, "columns": columns, "error": None}

    # 1) CSV attempt
    try:
        csv_df = pd.read_csv(io.StringIO(raw))
        if not csv_df.empty or len(csv_df.columns) > 0:
            return _frame_to_output(csv_df)
    except Exception:
        pass

    # 2) Markdown table attempt
    try:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) >= 2:
            header_cells = [c.strip() for c in table_lines[0].strip("|").split("|")]
            separator_cells = [c.strip() for c in table_lines[1].strip("|").split("|")]

            if (
                len(header_cells) == len(separator_cells)
                and all(cell and set(cell) <= {":", "-"} for cell in separator_cells)
            ):
                data_rows = []
                for row in table_lines[2:]:
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    if len(cells) == len(header_cells):
                        data_rows.append(cells)

                md_df = pd.DataFrame(data_rows, columns=header_cells)
                return _frame_to_output(md_df)
    except Exception:
        pass

    return {
        "records": [],
        "columns": [],
        "error": "Unable to parse input as CSV or markdown table.",
        "raw": raw,
    }


TOOL_REGISTRY = {
    "calculator": calculator,
    "financial_ratio": financial_ratio,
    "parse_table": parse_table,
}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_ratio",
            "description": "Compute numerator/denominator safely with optional percent formatting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator": {
                        "type": "number",
                        "description": "Numerator value.",
                    },
                    "denominator": {
                        "type": "number",
                        "description": "Denominator value.",
                    },
                    "as_percent": {
                        "type": "boolean",
                        "description": "Whether to format as percentage.",
                        "default": True,
                    },
                },
                "required": ["numerator", "denominator"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_table",
            "description": "Parse raw table text as CSV, then markdown table.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw": {
                        "type": "string",
                        "description": "Raw table text input.",
                    }
                },
                "required": ["raw"],
                "additionalProperties": False,
            },
        },
    },
]
