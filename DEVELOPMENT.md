# FinReasoningAI Development Summary

## What This Codebase Is

FinReasoningAI is a financial reasoning stack built around **Qwen2.5-14B** with **QLoRA** fine-tuning, evaluation tooling, and inference-time controls (chain-of-thought stripping, grounding checks, optional tool use, self-consistency).

Core goals:

- Strong numerical reasoning on finance-style tasks.
- Lower hallucination rate on numeric outputs (median aggregation, grounding, refusal paths).
- A practical training and evaluation workflow (notebooks, CLI entry points, Colab + Drive oriented docs).

## Code Summary (Checked-In Tree)

| Area | Role |
|------|------|
| `src/model/load_model.py` | Loads the base model and tokenizer (Transformers 4-bit NF4 path, optional vLLM + LoRA wiring). VRAM / flash-attn style concerns live here. |
| `src/model/apply_lora.py` | Attaches QLoRA adapters to a loaded base model. |
| `src/train/sft_train.py` | Supervised fine-tuning via TRL, with compatibility shims for TRL API differences. |
| `src/train/dpo_train.py` | Optional DPO stage and preference-style training workflow. |
| `src/inference/generate.py` | End-user inference: prompt building (direct vs CoT system prompts), single-shot `generate`, optional **tool-augmented** generation (`tool_router` + `financial_tools`), **numeric token bias** logits processor, **grounding** and refusal heuristics, and delegation to self-consistency when `self_consistency_n > 1`. Exposes a small **CLI** (`python -m src.inference.generate`). |
| `src/inference/self_consistency.py` | **Step 5c — self-consistency:** N stochastic samples (HF `generate` loop or vLLM `n` sampling), aggregation, confidence, optional **AnswerVerifier** scaffold (sequence-classification reranker). See [Self-consistency](#self-consistency-inference) below. |
| `src/eval/evaluate.py` | Metrics (EM with numeric tolerance, F1 by task, parsability, grounding) and eval loops that can run **greedy**, **CoT**, or **self_consistency** inference modes via shared sampling helpers. |
| `src/tools/tool_router.py` | Parses `<tool_call>...</tool_call>` JSON from model text and dispatches registered tools. |
| `src/rag/retriever.py` | Retrieval helpers (embedding / FAISS style pipeline as implemented in-repo). |
| `tools/financial_tools.py`, `tools/number_parser.py` | Declarative tool definitions and numeric parsing / scoring utilities used from inference and tests. |
| `tests/test_robustness.py` | Robustness and integration-style tests (including self-consistency aggregation). |
| `configs/training_config.yaml` | Example hyperparameters; includes `self_consistency_n` (default 1; raise for accuracy vs latency trade-off). |
| Notebooks | `FinReasoningAI_Colab.ipynb` (train / orchestration), `FinReasoningAI_Eval.ipynb` (eval + baseline vs fine-tuned), `gradio_demo.ipynb`, `merge_adapter.ipynb`, plus an archived Colab copy. |

Data generation and preprocessing are **documented in README** as `src.data` modules and `data/raw` → `data/processed` paths; those directories may be absent in a minimal clone—use the README commands when the full layout is present.

## Repository Layout (As Present in This Repo)

- `src/model/` — `load_model.py`, `apply_lora.py`
- `src/train/` — `sft_train.py`, `dpo_train.py`
- `src/eval/` — `evaluate.py`
- `src/inference/` — `generate.py`, `self_consistency.py`
- `src/rag/` — `retriever.py`
- `src/tools/` — `tool_router.py`
- `tools/` — `financial_tools.py`, `number_parser.py`
- `tests/` — `test_robustness.py`
- `configs/` — `training_config.yaml`
- Root notebooks and `requirements.txt`

## Self-Consistency Inference

Implemented in `src/inference/self_consistency.py`:

- **Sampling:** `sample_with_self_consistency(model, tokenizer, prompt, n=8, temperature=0.7, ...)` runs N completions. **vLLM** models use batched `n` sampling in one call; **Hugging Face** runs N sequential `generate` calls with `do_sample=True`, `top_p=0.9`.
- **Think stripping:** `<think>...</think>` blocks are removed before aggregation so votes apply to the visible answer text.
- **Routing:** `self_consistent_answer` treats a run as **numerical** if more than half of cleaned strings match a currency/scale number regex, or if `force_numerical=True`.
- **Numerical aggregate:** Parses each answer with `_normalize_to_float` (currency, commas, `%`, K/M/B/T and word scales), takes the **median** of parsed values, then returns the **original candidate string** whose numeric value is closest to that median (outlier-resistant vs mean).
- **Fallback:** If nothing parses as a number, falls back to **majority vote** on raw strings (same as categorical path).
- **Categorical aggregate:** Whitespace-normalized, case-folded **majority vote**; returns an original-casing instance of the winning string.
- **Confidence:** For numerical answers, fraction of samples within **10% relative** of the chosen final numeric value; for categorical, exact normalized-string match rate. `sample_with_self_consistency` logs a warning when confidence is below `min_confidence` (default 0.4).
- **Optional verifier (Step 5d):** `AnswerVerifier` loads `AutoModelForSequenceClassification` (1 logit, sigmoid to 0–1) and implements `score_candidates` / `rerank_with_verifier` for future reranking; training this head is described in the module docstring, not automated here.

**Integration:**

- `generate_answer` in `generate.py`: when `self_consistency_n > 1`, builds the prompt then calls `sample_with_self_consistency`. If aggregate **confidence** is below `min_confidence` (default 0.30 in `generate_answer`), the API returns **`Insufficient information.`** before the usual grounding pass on that string.
- `evaluate.py` uses the same helper when eval config sets `self_consistency_n > 1`.
- CLI: `--self_consistency_n` on `python -m src.inference.generate`.

**Trade-off:** Latency and cost scale ~linearly with N; README suggests N=3 for interactive use and N=8–10 for batch accuracy.

## Canonical Data Pipeline (README / Full Layout)

When `src.data` and `data/` are available (see README):

- Input: raw JSONL under `data/raw/`.
- Preprocess: ChatML-style prompt/completion formatting, splits, tokenization to `data/processed/`.
- Eval notebooks that need original fields (question, answer, context) should reconstruct the **test split from raw** where applicable, not only tokenized prompt-completion rows.

## Training Flow

1. Load base model/tokenizer via `src/model/load_model.py`.
2. Apply QLoRA via `src/model/apply_lora.py`.
3. Load processed dataset (when present).
4. Train with `src/train/sft_train.py`; save adapter (e.g. `outputs/sft_qlora/final_adapter`).
5. Optional DPO: `src/train/dpo_train.py`.

Notes: `sft_train.py` tolerates TRL version differences; completion-only loss matches prompt/completion datasets.

## Evaluation Flow (`FinReasoningAI_Eval.ipynb`)

Typical structure:

1. Setup, Drive mount, repo sync (`git pull` / `git clone`).
2. Load test samples (from Drive `data/` when used with Colab).
3. Build prompts consistent with training format.
4. **Baseline** (base model, no adapter) vs **fine-tuned** (base + LoRA), including optional vLLM LoRA request patterns.
5. Compare aggregate and by-task metrics; export JSON + CSV under `outputs/`.

Primary metrics live in `src/eval/evaluate.py`: exact match (numeric-aware), F1, parsability, grounding.

## Tests

- `tests/test_robustness.py` — robustness checks, self-consistency aggregation tests, integration-style eval tests.
- Some tests are CPU-friendly; full model tests need GPU + env configuration.

## Recommended Run Order

1. Data generation / preprocessing (when `src.data` is available), or use existing JSONL.
2. SFT training (optional DPO).
3. Verify adapter output path.
4. Eval notebook or `python -m src.eval.evaluate` (as configured).
5. Inference CLI or notebook with optional `--self_consistency_n`.

## Known Issue: `Could not find response key Answer:` (TRL 0.8.x)

**Symptom:** `trl/trainer/utils.py: UserWarning: Could not find response key Answer: in the following instance: You are an expert with extensive financial knowledge…`

**Root causes (both must be understood together):**

1. `DataCollatorForCompletionOnlyLM` receives pre-tokenized sequences and scans their token IDs for the `response_template` subsequence. The old code used `response_template="Answer:"`. `Answer:` is placed at the **start of the completion**, i.e. at the END of the full `text` string. When the prompt is long (tool-JSON schema block + large table context), right-side truncation at `max_length=2048` cuts `Answer:` off before the collator sees it. The collator warns and masks all labels to `-100` → zero gradient for that sample.

2. Some samples from external datasets loaded by `fincot_loader` use a different system prompt ("You are an expert with extensive financial knowledge and strong programming skills…") that does not follow the `Answer:` convention. Even after `add_text` prepends `Answer:`, the combined text may exceed 2048 tokens and hit the truncation boundary.

**Fix applied in `src/train/sft_train.py`:**

- `_build_old_trl_collator` now uses the **ChatML assistant-turn token IDs** (`<|im_start|>assistant\n`) as `response_template` instead of the string `"Answer:"`. These tokens are structurally part of the *prompt* (produced by `apply_chat_template`) and appear **before** the completion, so they are never affected by completion-side truncation. All tokens after the assistant marker receive loss — including any CoT and the `Answer:` line.
- `_prepare_old_trl_dataset` (`add_text`) always normalizes completions to start cleanly with `Answer:` and concatenates directly onto the prompt without an extra newline (the prompt from `apply_chat_template` already ends with `\n`).
- `_tokenize_old_trl_dataset` now filters out samples whose full tokenized length exceeds `max_seq_length`, logging a warning and dropping them to prevent wasting training iterations on nearly-empty completions.

**A100 40 GB batch size:** Default notebook config updated from `BATCH_SIZE=4` to `BATCH_SIZE=2` with `GRAD_ACCUM=16` (same effective batch of 32). The 40 GB A100 peaks at ~30–38 GB during training; `BATCH_SIZE=4` at `max_seq_length=2048` risks OOM.

## Practical Notes

- Notebook paths target Google Colab and Drive by default.
- For vLLM evaluation, fine-tuned runs typically enable LoRA on the engine; baseline runs omit LoRA.
- `generate_answer` combines self-consistency confidence gating with separate **context grounding** checks when `grounding_check=True` and context is non-empty.
