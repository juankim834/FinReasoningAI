# FinReasoningAI Development Summary

## What This Repository Does

FinReasoningAI is a financial reasoning system built around `Qwen/Qwen2.5-14B-Instruct` with a QLoRA-first training path, robust inference controls, and evaluation tooling focused on numerical reliability.

Primary objectives:

- Improve numerical correctness on financial QA and calculation-heavy prompts.
- Reduce hallucinated numeric outputs via grounding checks, tool calls, and self-consistency.
- Keep the workflow practical for Colab/GPU experimentation and local CLI usage.

## Current Codebase Map

- `src/model/load_model.py`
  - Loads tokenizer/model in 4-bit NF4 (`bitsandbytes`), handles VRAM checks, and includes bf16 causal-mask safety patching for runtime compatibility.
  - Supports adapter wiring and optional vLLM-serving compatibility helpers.
- `src/model/apply_lora.py`
  - Defines and applies LoRA/QLoRA adapter configuration on top of the base causal LM.
- `src/data/fincot_loader.py`
  - Loads `TheFinAI/FinCoT`, normalizes heterogeneous field names, resolves SFT split names, and classifies samples as numerical vs non-numerical reasoning.
- `src/data/preprocess.py`
  - Builds prompt/completion pairs from canonical fields, removes negative targets, balances class mix, creates train/test splits, and writes processed datasets.
- `src/data/synthetic_gen.py`
  - Generates synthetic financial reasoning samples used as additional training data.
- `src/train/sft_train.py`
  - Main SFT entrypoint with TRL compatibility for both newer and older versions.
  - Includes fixes for legacy `DataCollatorForCompletionOnlyLM` response-template failures and long-sequence filtering behavior.
- `src/train/dpo_train.py`
  - Optional DPO stage and automatic preference-pair construction from sampled SFT outputs.
- `src/inference/generate.py`
  - Core inference API and CLI (`python -m src.inference.generate`).
  - Handles prompt construction, direct vs CoT prompts, optional tool-augmented turns, numeric-bias logits processing, grounding/refusal logic, and self-consistency delegation.
- `src/inference/self_consistency.py`
  - Implements N-sample self-consistency aggregation:
    - numerical answers -> median-based selection;
    - textual answers -> majority vote;
    - returns confidence scores and optional verifier hooks.
- `src/tools/tool_router.py`
  - Parses `<tool_call>...</tool_call>` JSON payloads, validates schema/tool names, dispatches calls, and exposes a backward-compatible router class.
- `tools/financial_tools.py`, `tools/number_parser.py`
  - Active benchmark tools are `arithmetic` and `compound_growth_rate`; includes robust numeric parsing helpers.
- `src/eval/evaluate.py`
  - Evaluation metrics: EM with numeric tolerance, task-aware F1, parsability, grounding.
  - Supports greedy, CoT, and self-consistency-style inference evaluation paths.
- `src/rag/retriever.py`
  - Retrieval pipeline for document chunking/indexing/querying (RAG extension path).
- `tests/test_robustness.py`
  - Robustness, aggregation, and integration-style checks (CPU-safe subsets plus GPU/full-path tests).

## End-to-End Workflow

1. Build data:
   - Load FinCoT and/or generate synthetic samples.
   - Preprocess into prompt/completion datasets (`data/processed_*`).
2. Train:
   - Run SFT (`src.train.sft_train`), save adapter in `outputs/sft_qlora/...`.
   - Optionally run DPO (`src.train.dpo_train`) for preference alignment.
3. Evaluate:
   - Run notebook or CLI evaluation (`src.eval.evaluate`) for aggregate and per-task metrics.
4. Inference:
   - Use direct, CoT, or tool-augmented generation.
   - Enable `self_consistency_n > 1` when accuracy is preferred over latency.

## Inference Reliability Design

- **Grounding checks:** numeric answers are checked against provided context where applicable.
- **Self-consistency:** confidence-weighted aggregation across N stochastic samples.
- **Tool routing:** strict `<tool_call>` parsing/validation to reduce malformed tool invocations.
- **Refusal behavior:** low-confidence or insufficient-evidence paths return explicit fallback responses.

## Notebooks and Operations

- `FinReasoningAI_Colab.ipynb` is the main training/orchestration notebook.
- `FinReasoningAI_Eval.ipynb` handles baseline vs fine-tuned evaluation comparisons.
- Additional notebooks (`gradio_demo.ipynb`, `merge_adapter.ipynb`, archived Colab copy) support demo and deployment operations.

## Practical Notes

- The repository currently includes active `src/data` modules (not just docs references).
- Batch size/sequence length and dtype choices should be tuned by GPU memory (A100 40GB vs 80GB behavior differs materially).
- TRL version differences are explicitly handled in `sft_train.py`; keep that path intact when upgrading dependencies.
