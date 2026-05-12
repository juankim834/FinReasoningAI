# FinReasoningAI

FinReasoningAI is a financial reasoning stack built around `Qwen/Qwen2.5-14B-Instruct` with a QLoRA-first training path, reliability-oriented inference, evaluation helpers, and an optional RAG layer.

The current repository is centered on:

- 4-bit model loading and QLoRA adapter application
- SFT and optional DPO training entry points
- direct, CoT, self-consistency, and tool-augmented inference
- evaluation focused on numerical accuracy, parsability, and grounding
- FAISS-based retrieval for financial documents
- notebooks for Colab training, evaluation, and demos

## What Is In This Checkout

```text
FinReasoningAI/
|- configs/
|  `- training_config.yaml
|- src/
|  |- eval/
|  |  `- evaluate.py
|  |- inference/
|  |  |- generate.py
|  |  `- self_consistency.py
|  |- model/
|  |  |- apply_lora.py
|  |  `- load_model.py
|  |- rag/
|  |  `- retriever.py
|  |- tools/
|  |  `- tool_router.py
|  `- train/
|     |- dpo_train.py
|     `- sft_train.py
|- tests/
|  `- test_robustness.py
|- tools/
|  |- financial_tools.py
|  `- number_parser.py
|- FinReasoningAI_Colab.ipynb
|- FinReasoningAI_Eval.ipynb
|- gradio_demo.ipynb
`- requirements.txt
```

## Important Repo Note

Some training and evaluation code still imports modules under `src.data` such as `src.data.preprocess`. Those files are not present in this checkout, so the data-preparation pipeline documented in older versions of the README is not currently runnable from this repository alone.

What does work in the current codebase:

- model loading and adapter wiring
- inference CLI and library usage
- evaluation utilities, assuming you already have prepared data
- SFT and DPO training entry points, assuming you already have prepared datasets
- RAG indexing and retrieval
- robustness and metric tests

## Installation

```bash
pip install -r requirements.txt
```

Optional for A100-style environments:

```bash
pip install flash-attn --no-build-isolation
```

## Core Components

### Model loading

`src/model/load_model.py` loads Qwen in 4-bit NF4 with bitsandbytes, performs VRAM checks, and supports both Transformers and vLLM inference backends.

### QLoRA application

`src/model/apply_lora.py` prepares the quantized model for k-bit training and applies LoRA adapters targeting:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

### Training

- `src/train/sft_train.py`: supervised fine-tuning entry point
- `src/train/dpo_train.py`: optional DPO stage on top of an SFT adapter

### Inference

- `src/inference/generate.py`: direct generation, CoT mode, self-consistency, and tool use
- `src/inference/self_consistency.py`: aggregation for multi-sample decoding
- `src/tools/tool_router.py`: parses and dispatches `<tool_call>...</tool_call>` payloads
- `tools/financial_tools.py`: active benchmark tools are `arithmetic` and `compound_growth_rate`

### Evaluation

`src/eval/evaluate.py` provides:

- exact match with numeric tolerance
- task-aware F1
- answer parsability
- grounding checks against context or expressions

### Retrieval / RAG

`src/rag/retriever.py` provides:

- document chunking
- embedding with `BAAI/bge-m3`
- FAISS index build/load/query
- optional reranking with `BAAI/bge-reranker-v2-m3`

## Example Usage / Quick Start

### 1. Run inference

Transformers backend:

```bash
python -m src.inference.generate ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --adapter_dir outputs/sft_qlora/final_adapter ^
  --question "What was Apple's revenue growth rate from 2021 to 2022?" ^
  --context "Apple reported revenues of $394.3B in 2022 and $365.8B in 2021."
```

Tool-augmented inference:

```bash
python -m src.inference.generate ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --adapter_dir outputs/sft_qlora/final_adapter ^
  --use_tools ^
  --question "What was the percent change from 365.8 to 394.3?" ^
  --context "2021 revenue was 365.8 and 2022 revenue was 394.3."
```

Self-consistency:

```bash
python -m src.inference.generate ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --adapter_dir outputs/sft_qlora/final_adapter ^
  --self_consistency_n 8 ^
  --temperature 0.7 ^
  --question "What was Apple's free cash flow in 2022?" ^
  --context "Apple's free cash flow in 2022 was $111.4 billion."
```

vLLM backend:

```bash
python -m src.inference.generate ^
  --engine vllm ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --question "What was Apple's net income in 2022?" ^
  --context "Apple's net income in 2022 was $99.8 billion."
```

### 2. Run SFT training

This requires an already prepared Hugging Face `DatasetDict` on disk. In the current checkout, the default expected path is `data/processed_fincot_sft`.

```bash
python -m src.train.sft_train ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --data_dir data/processed_fincot_sft ^
  --output_dir outputs/sft_qlora ^
  --num_train_epochs 1 ^
  --per_device_train_batch_size 4 ^
  --gradient_accumulation_steps 8 ^
  --learning_rate 2e-4 ^
  --max_seq_length 2048
```

### 3. Run optional DPO training

This requires an existing SFT adapter and a JSONL preference dataset.

```bash
python -m src.train.dpo_train ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --sft_adapter_dir outputs/sft_qlora/final_adapter ^
  --pref_data_path data/raw/dpo_preferences.jsonl ^
  --output_dir outputs/dpo_qlora ^
  --beta 0.1
```

### 4. Evaluate a model

If `--test_data` points to a directory, evaluation loads a saved dataset from disk. If it points to a file, it expects JSONL.

Transformers evaluation with adapter:

```bash
python -m src.eval.evaluate ^
  --engine transformers ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --adapter_dir outputs/sft_qlora/final_adapter ^
  --test_data data/processed_fincot_sft ^
  --output_csv outputs/eval_results.csv
```

vLLM evaluation:

```bash
python -m src.eval.evaluate ^
  --engine vllm ^
  --model_id Qwen/Qwen2.5-14B-Instruct ^
  --test_data data/processed_fincot_sft ^
  --output_csv outputs/eval_results.csv
```

### 5. Build and query a RAG index

Build:

```bash
python -m src.rag.retriever build ^
  --docs_dir data/raw/filings ^
  --index_dir outputs/rag_index
```

Query:

```bash
python -m src.rag.retriever query ^
  --index_dir outputs/rag_index ^
  --question "What was Apple's R&D spending in 2022?"
```

### 6. Run the demo notebook

Open `gradio_demo.ipynb` on Google Colab and run the cells to interact with a simple Gradio interface for asking financial questions.

## Configuration

`configs/training_config.yaml` contains project defaults for:

- model and quantization settings
- LoRA hyperparameters
- SFT and DPO training values
- inference defaults
- RAG settings
- evaluation tolerances

Treat it as reference configuration. The current training scripts primarily take CLI arguments directly.

## Tests

Run the CPU-safe metric and aggregation tests:

```bash
pytest tests/test_robustness.py -v -k "not model_and_tokenizer"
```

Run integration tests with a real GPU-backed model:

```bash
set FINREASONING_MODEL_ID=Qwen/Qwen2.5-14B-Instruct
set FINREASONING_ADAPTER_DIR=outputs/sft_qlora/final_adapter
pytest tests/test_robustness.py -v
```

## Notebooks

- `FinReasoningAI_Colab.ipynb`: primary Colab workflow
- `FinReasoningAI_Eval.ipynb`: evaluation workflow
- `gradio_demo.ipynb`: demo notebook
- `outputs/demo_notebook/`: saved notebook outputs

## Datasets

The intended datasets are:

- training: [TheFinAI/FinCoT](https://huggingface.co/datasets/TheFinAI/FinCoT)
- evaluation: [FinQA](https://github.com/czyssrs/FinQA)

## License

Code in this repository is MIT-licensed. Base model weights remain subject to the upstream Qwen license:

[Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
