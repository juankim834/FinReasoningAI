# FinReasoningAI Development Summary

## What This Codebase Is
FinReasoningAI is a financial reasoning stack built around Qwen2.5-14B with QLoRA fine-tuning, evaluation tooling, and inference-time safety features (grounding checks, optional tool use, self-consistency).

Core goals:
- strong numerical reasoning on finance tasks
- low hallucination rate for numeric outputs
- practical training/eval workflow on Colab + Drive

## Repository Layout
- `src/model/`
  - `load_model.py`: loads base model/tokenizer (Transformers 4-bit or vLLM path), VRAM checks, flash-attn fallback.
  - `apply_lora.py`: applies QLoRA adapters to the base model.
- `src/data/`
  - `synthetic_gen.py`: generates synthetic mixed-task financial dataset.
  - `preprocess.py`: canonical formatting/splitting pipeline (Step 2 logic), including ChatML prompt/completion conversion and stratified split.
  - `schemas.py`: data schemas/helpers.
- `src/train/`
  - `sft_train.py`: supervised fine-tuning pipeline with TRL version compatibility.
  - `dpo_train.py`: optional DPO stage and preference data workflow.
- `src/eval/`
  - `evaluate.py`: metrics and evaluation loop (EM, F1, parsability, grounding; optional baseline with adapters disabled).
- `src/inference/`
  - `generate.py`: single/batch inference, prompt building, postprocessing, optional tool call execution and grounding/refusal logic.
  - `self_consistency.py`: sampling aggregation for self-consistency.
- `src/rag/`
  - `retriever.py`: retrieval pipeline (index/query helpers).
- `src/tools/`
  - `tool_router.py`: tool dispatch for agentic/tool-augmented inference.
- `tools/`
  - `financial_tools.py`, `number_parser.py`: utility tools and numeric parsing/scoring helpers.
- notebooks:
  - `FinReasoningAI_Colab.ipynb`: end-to-end primary notebook (generation, preprocess, training, optional DPO).
  - `FinReasoningAI_Eval.ipynb`: evaluation notebook (now includes baseline vs fine-tuned side-by-side).

## Canonical Data Pipeline (Current)
Defined by `src/data/preprocess.py` and used by notebook workflows:
- Input: raw JSONL (`data/raw/synthetic.jsonl` or directory of JSONL files).
- Task-aware formatting into ChatML-style prompt/completion.
- Stratified split by `task`: default 90% train / 5% val / 5% test.
- `load_eval_test_samples(...)` recreates the test split from raw data and preserves fields needed for eval (`question`, `answer`, `context`, etc.).

Important implication:
- Evaluation should use `load_eval_test_samples` from raw data (not only processed prompt-completion disk dataset), because metrics require original fields.

## Training Flow (Current)
Main path:
1. Load base model/tokenizer via `src/model/load_model.py`
2. Apply QLoRA via `src/model/apply_lora.py`
3. Load processed dataset
4. Train with `src/train/sft_train.py`
5. Save adapter to `outputs/sft_qlora/final_adapter`

Notes:
- `sft_train.py` handles old/new TRL APIs.
- Completion-only loss behavior aligns with prompt/completion dataset format.
- Optional DPO second stage available in `src/train/dpo_train.py`.

## Evaluation Flow (Current Notebook State)
`FinReasoningAI_Eval.ipynb` is now structured as:
1. Setup & imports
2. Drive mount + repo sync
   - if repo exists at `/content/drive/MyDrive/FinReasoningAI/FinReasoningAI`: `git pull`
   - else: `git clone`
3. Load test set from `/content/drive/MyDrive/FinReasoningAI/data` using `load_eval_test_samples`
4. Build prompts with `format_as_prompt_completion` (same format family as Step 2)
5. Baseline evaluation (base model, no LoRA adapter)
6. Fine-tuned evaluation (same base model + LoRA adapter)
7. Results comparison (overall + by-task deltas)
8. Export full row-level CSV

Current eval outputs:
- JSON summary/results:
  - `/content/drive/MyDrive/FinReasoningAI/outputs/eval_baseline_vs_finetuned.json`
- Full CSV with test fields + both predictions + metrics:
  - `/content/drive/MyDrive/FinReasoningAI/outputs/eval_baseline_vs_finetuned_full.csv`

## Metrics and Comparison
Primary eval metrics in `src/eval/evaluate.py`:
- exact match (`compute_exact_match`, numeric tolerance aware)
- F1 (`compute_f1_for_task`)
- parsability (`is_answer_parsable`)
- grounding (`compute_grounding_rate`)

Comparison logic in eval notebook computes:
- baseline aggregate metrics
- fine-tuned aggregate metrics
- delta (fine-tuned minus baseline)
- by-task EM/F1 deltas

## Tests
- `tests/test_robustness.py` contains robustness checks and integration-style evaluation tests.
- Some tests can run CPU-only; full integration/model tests require model+GPU environment variables.

## Recommended Run Order
1. Data generation (if needed)
2. Preprocessing
3. SFT training (optional DPO)
4. Adapter output verification
5. Eval notebook baseline vs fine-tuned
6. Inspect exported CSV for failure analysis

## Current Practical Notes
- Notebook workflows are Colab/Drive oriented.
- vLLM eval path in the notebook uses LoRA request for fine-tuned pass and no LoRA request for baseline pass.
- No `src/` files were modified for the latest eval-notebook updates; notebook orchestration changed instead.
