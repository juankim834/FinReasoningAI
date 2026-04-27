"""
File: src/data/schemas.py

Pydantic data schemas for the three task types used in FinReasoning training.

Data mix rationale (from Agent.md Step 2b):
  - 60% Type A (Financial QA): Direct question answering from financial text is
    the most common real-world use case; dominates the training signal.
  - 30% Type B (Numerical Reasoning): Numerical accuracy is the core differentiator
    of FinReasoning vs generic LLMs. Dedicated tasks build robust arithmetic paths.
  - 10% Type C (Structured Analysis with CoT): Kept sparse (<15% per spec) to avoid
    CoT overfitting. CoT traces are invaluable for multi-step chains but degrade
    performance on simpler tasks if overrepresented.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator
import uuid


class FinancialQA(BaseModel):
    """
    Type A — Financial QA
    Sources: FinQA, ConvFinQA, custom scraped 10-K/earnings data.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Literal["financial_qa"] = "financial_qa"
    instruction: str = Field(
        default="Answer the following financial question based on the provided context.",
        description="Task instruction prepended to the prompt."
    )
    context: str = Field(description="10-K excerpt, financial table, or macro data passage.")
    question: str
    answer: str = Field(description="Concise final answer only (no reasoning steps).")
    reasoning: Optional[str] = Field(
        default=None,
        description="CoT scratchpad. Null for ~90% of samples to keep CoT sparse."
    )

    @field_validator("reasoning")
    @classmethod
    def wrap_reasoning_in_think_tags(cls, v: Optional[str]) -> Optional[str]:
        """Ensure reasoning traces use <think>...</think> tags for scratchpad suppression."""
        if v is not None and not v.strip().startswith("<think>"):
            v = f"<think>\n{v.strip()}\n</think>"
        return v


class NumericalReasoning(BaseModel):
    """
    Type B — Numerical Reasoning
    Sources: TAT-QA style, synthetic template generation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Literal["numerical_reasoning"] = "numerical_reasoning"
    instruction: str = Field(
        default="Evaluate the following financial expression and provide the numerical result."
    )
    expression: str = Field(
        description="Symbolic expression, e.g. '(revenue_2023 - revenue_2022) / revenue_2022'"
    )
    variables: Dict[str, float] = Field(
        description="Variable bindings, e.g. {'revenue_2023': 4.2e9, 'revenue_2022': 3.9e9}"
    )
    answer: Union[float, str] = Field(
        description="Numeric result or 'N/A' if expression is undefined."
    )
    unit: Optional[str] = Field(
        default=None,
        description="Unit of the answer, e.g. '%', '$M', 'bps'."
    )


class StructuredAnalysis(BaseModel):
    """
    Type C — Structured Analysis (always includes CoT reasoning trace).
    Sources: Custom + LLM-synthesized analysis tasks.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Literal["structured_analysis"] = "structured_analysis"
    instruction: str = Field(
        default="Analyze the following financial data and answer the question."
    )
    financial_data: Dict[str, Any] = Field(
        description="Parsed table or KPI dict, e.g. {'revenue': [...], 'year': [...]}"
    )
    question: str
    answer: str
    format: Literal["bullet", "paragraph"] = "paragraph"
    reasoning: Optional[str] = Field(
        default=None,
        description="Required for structured analysis tasks."
    )

    @field_validator("reasoning")
    @classmethod
    def wrap_reasoning_in_think_tags(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip().startswith("<think>"):
            v = f"<think>\n{v.strip()}\n</think>"
        return v


# Union type for all samples
AnyFinancialSample = Union[FinancialQA, NumericalReasoning, StructuredAnalysis]


TASK_TYPE_MAP = {
    "financial_qa": FinancialQA,
    "numerical_reasoning": NumericalReasoning,
    "structured_analysis": StructuredAnalysis,
}


def parse_sample(record: Dict[str, Any]) -> AnyFinancialSample:
    """Deserialize a raw dict into the appropriate schema class."""
    task = record.get("task")
    cls = TASK_TYPE_MAP.get(task)
    if cls is None:
        raise ValueError(f"Unknown task type: {task!r}. Expected one of {list(TASK_TYPE_MAP)}")
    return cls(**record)
