# FinReasoning AI

A production-grade financial reasoning system built on **Qwen2.5-14B** with **QLoRA** fine-tuning.
Designed for numerical accuracy, hallucination resistance, and self-hostable deployment on a single A100 80GB GPU.

---

## Architecture Overview

```
Qwen2.5-14B-Instruct  (4-bit NF4, double quantization)
        │
        ├── QLoRA Adapters  (r=64, alpha=128, 7 target modules)
        │
        ├── SFT Training    (primary stage, TRL SFTTrainer)
        │   └── Optional DPO  (preference alignment, beta=0.1)
        │
        ├── Inference
        │   ├── Direct answer (default — no CoT in output)
        │   ├── CoT mode     (<think> scratchpad, stripped post-generation)
        │   ├── Self-consistency (N=5–10 samples, median/majority vote)
        │   └── Tool use     (calculator, table parser, RAG retrieve)
        │
        └── RAG Extension   (bge-m3 + FAISS + cross-encoder reranking)
```

### VRAM Budget (Single A100 80GB)

| Phase | Estimated VRAM | Notes |
|-------|---------------|-------|
| Inference (4-bit) | ~12 GB | 2048-token context |
| SFT Training | ~31–38 GB | bs=4, grad_ckpt=True, bf16 |
| DPO Training | ~28 GB | Two 4-bit model copies |
| + RAG embedder (bge-m3) | +1.5 GB | CPU inference option |
| **Peak (SFT + overhead)** | **~42 GB** | Well within 80 GB ✅ |

---

## Project Structure

```
FinReasoningAI/
├── requirements.txt
├── README.md
├── configs/               # YAML configs (optional override)
├── data/
│   ├── raw/               # Raw / synthetic JSONL files
│   └── processed/         # Tokenized HuggingFace DatasetDict
├── outputs/               # Training checkpoints and adapters
├── src/
│   ├── model/
│   │   ├── load_model.py      # Load Qwen2.5-14B in 4-bit NF4
│   │   └── apply_lora.py      # Attach QLoRA adapters
│   ├── data/
│   │   ├── schemas.py         # Pydantic schemas (Type A/B/C)
│   │   ├── synthetic_gen.py   # Generate 5,000 synthetic FinQA pairs
│   │   └── preprocess.py      # ChatML formatting + tokenization + split
│   ├── train/
│   │   ├── sft_train.py       # SFT with TRL SFTTrainer
│   │   └── dpo_train.py       # Optional DPO second stage
│   ├── eval/
│   │   └── evaluate.py        # EM, F1, parsability, grounding metrics
│   ├── inference/
│   │   ├── generate.py        # Core inference + tool use + constrained decoding
│   │   └── self_consistency.py # N-sample aggregation (median/majority)
│   ├── rag/
│   │   └── retriever.py       # bge-m3 + FAISS + cross-encoder reranking
│   └── tools/
│       └── tool_router.py     # Calculator, table parser, ReAct agent
└── tests/
    └── test_robustness.py     # Numerical perturbation, entity swap, unit tests
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Flash Attention 2 (recommended for A100, optional):
pip install flash-attn --no-build-isolation
```

### 2. Generate synthetic training data

```bash
# Fast: template-only generation (no GPU required)
python -m src.data.synthetic_gen --output data/raw/synthetic.jsonl --total 5000 --no_llm

# Full: use Qwen2.5-7B as generator (~16GB VRAM)
python -m src.data.synthetic_gen --output data/raw/synthetic.jsonl --total 5000
```

### 3. Preprocess and tokenize

```bash
python -m src.data.preprocess \
    --data_path data/raw/synthetic.jsonl \
    --model_id Qwen/Qwen2.5-14B-Instruct \
    --max_length 2048 \
    --output_dir data/processed
```

### 4. Run SFT training

```bash
python -m src.train.sft_train \
    --model_id Qwen/Qwen2.5-14B-Instruct \
    --data_dir data/processed \
    --output_dir outputs/sft_qlora \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 8 \
    --lr 2e-4 \
    --max_seq_length 2048
```

### 5. (Optional) DPO second stage

```bash
# First, construct preference pairs from the SFT model
# (see src/train/dpo_train.py construct_preference_pairs_from_sft)

python -m src.train.dpo_train \
    --sft_adapter_dir outputs/sft_qlora/final_adapter \
    --pref_data_path data/raw/dpo_preferences.jsonl \
    --output_dir outputs/dpo_qlora \
    --beta 0.1
```

### 6. Evaluate

```bash
python -m src.eval.evaluate \
    --model_id Qwen/Qwen2.5-14B-Instruct \
    --adapter_dir outputs/sft_qlora/final_adapter \
    --test_data data/processed \
    --output_csv outputs/eval_results.csv
```

### 7. Inference

```bash
# Direct answer (default)
python -m src.inference.generate \
    --model_id Qwen/Qwen2.5-14B-Instruct \
    --adapter_dir outputs/sft_qlora/final_adapter \
    --question "What was Apple's revenue growth rate from 2021 to 2022?" \
    --context "Apple Inc. reported revenues of \$394.3B in 2022, up from \$365.8B in 2021."

# Self-consistency (more accurate, 8× slower)
python -m src.inference.generate \
    --self_consistency_n 8 \
    --temperature 0.7 \
    --question "..."

# CoT mode
python -m src.inference.generate --use_cot --question "..."

# Tool-augmented
python -m src.inference.generate --use_tools --question "..."
```

### 8. Run tests

```bash
# Metric unit tests (no GPU required)
pytest tests/test_robustness.py -v -k "not model_and_tokenizer"

# Full integration tests (requires GPU + FINREASONING_MODEL_ID)
FINREASONING_MODEL_ID=Qwen/Qwen2.5-14B-Instruct \
FINREASONING_ADAPTER_DIR=outputs/sft_qlora/final_adapter \
pytest tests/test_robustness.py -v
```

### 9. Build RAG index

```bash
# Build
python -m src.rag.retriever build \
    --docs_dir data/raw/filings \
    --index_dir outputs/rag_index

# Query
python -m src.rag.retriever query \
    --index_dir outputs/rag_index \
    --question "What was Apple's R&D spending in 2022?"
```

---

## Data Design

### Task Type Mix

| Type | Task | Source | Fraction |
|------|------|--------|----------|
| A | Financial QA | FinQA, ConvFinQA, custom | 60% |
| B | Numerical Reasoning | TAT-QA, synthetic | 30% |
| C | Structured Analysis (+ CoT) | Custom, LLM-synthesized | 10% |

### CoT Strategy

Training includes `<think>...</think>` blocks for ~10% of samples (all Type C + selected Type A/B). At inference, the **default mode** strips all think blocks. The model retains latent reasoning capacity that improves numerical accuracy even without visible CoT.

To expose the scratchpad: use `--use_cot` flag, or set `use_cot=True` in `generate_answer()`.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base model | Qwen2.5-14B-Instruct | 128k context, strong math, financial tokenization |
| PEFT | QLoRA r=64, alpha=128 | ~0.5% trainable params, ~38GB peak VRAM |
| Optimizer | paged_adamw_32bit | Offloads optimizer pages to CPU, saves ~10GB VRAM |
| Precision | bf16 (not fp16) | Wider dynamic range for financial number magnitudes |
| CoT density | <10% of training mix | Prevents CoT overfitting on direct-answer tasks |
| Self-consistency | Median (numeric), majority (text) | Robust to hallucinated outliers |
| Grounding check | ±2% tolerance on context numbers | Rejects hallucinated numbers not in source |

---

## Benchmark Targets

| Benchmark | Metric | Target |
|-----------|--------|--------|
| FinQA | Exact Match | ≥55% |
| ConvFinQA | Exact Match | ≥60% |
| TAT-QA | Exact Match (numeric) | ≥65% |
| Internal eval | Parsability Rate | ≥95% |
| Internal eval | Grounding Rate | ≥85% |

---

## Extending the System

- **Add new document types**: extend `chunk_document()` in `retriever.py`
- **Add new tools**: register in `ToolRouter._tool_map` in `tool_router.py`
- **Add DPO data**: implement custom preference scoring in `dpo_train.py`
- **Serve with vLLM**: merge LoRA adapter (`model.merge_and_unload()`) then load with vLLM

---

## License

All code is MIT-licensed. Model weights are subject to the [Qwen2.5 model license](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct).
No proprietary APIs are used.
