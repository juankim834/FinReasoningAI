"""
File: src/rag/retriever.py

Step 7a — RAG with Financial Documents

Architecture:
  - Embedding model: bge-m3 (BAAI/bge-m3) via FlagEmbedding.
    Chosen over OpenAI embeddings because: open-source, strong on financial text
    (trained on diverse multilingual corpora including legal/financial), and
    produces 1024-dim dense embeddings with good semantic coverage of financial terms.
  - Vector store: FAISS (flat L2 or IVF-PQ for large corpora).
    For <100K chunks: use IndexFlatIP (exact inner-product search, deterministic).
    For >100K chunks: use IndexIVFPQ (approximate, 10× faster, ~5% accuracy trade-off).
  - Reranking: cross-encoder reranking from top-k=20 to top-2 for the final context.
    Cross-encoders are more accurate than bi-encoders for short passage reranking
    because they jointly encode query + passage.

Document pipeline:
  1. Ingest: parse 10-K filings / earnings transcripts as text.
  2. Chunk: split into 512-token chunks with 64-token overlap (preserves context).
  3. Embed: encode chunks with bge-m3.
  4. Index: store in FAISS with metadata (company, year, section, chunk_id).

[WARN] TRADE-OFF: bge-m3 produces 1024-dim embeddings. Storing 500K embeddings
requires 500K × 1024 × 4 bytes ≈ 2 GB memory. Use float16 or PQ compression
for very large corpora (>1M chunks).

Memory: bge-m3 model ≈ 1.5 GB VRAM (or CPU inference). FAISS index RAM: see above.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Document chunking
# ──────────────────────────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    tokenizer=None,
) -> List[Dict[str, Any]]:
    """
    Split a document into overlapping chunks.

    Args:
        text:          Raw document text.
        metadata:      Metadata dict (company, year, filing_type, section).
        chunk_size:    Target chunk size in words (approximate; use tokenizer for precision).
        chunk_overlap: Word overlap between consecutive chunks.
        tokenizer:     Optional tokenizer for token-based splitting (more accurate).

    Returns:
        List of {"text": str, "metadata": dict, "chunk_id": str} dicts.
    """
    import uuid

    if tokenizer is not None:
        # Token-based chunking (more accurate)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunk_id = f"{metadata.get('company', 'doc')}_{metadata.get('year', '0')}_{start}"
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": {**metadata, "start_token": start, "end_token": end},
                "chunk_id": chunk_id,
            })
            start += chunk_size - chunk_overlap
    else:
        # Word-based chunking (fallback)
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_id = f"{metadata.get('company', 'doc')}_{metadata.get('year', '0')}_{start}"
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "start_word": start, "end_word": end},
                "chunk_id": chunk_id,
            })
            start += chunk_size - chunk_overlap

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Embedding model
# ──────────────────────────────────────────────────────────────────────────────

class BGEEmbedder:
    """
    Wrapper around BAAI/bge-m3 for dense embedding of financial text.

    bge-m3 advantages for financial text:
      - Handles long passages (up to 8192 tokens) natively
      - Strong on domain-specific terminology without fine-tuning
      - Supports multiple retrieval paradigms (dense, sparse, multi-vector)
      - Open-source and self-hostable (no API calls)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=True,
                device=device if device != "auto" else None,
            )
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. Falling back to SentenceTransformer."
            )
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._is_flag = False
        else:
            self._is_flag = True

        self.batch_size = batch_size
        self.normalize = normalize
        self.dim = 1024  # bge-m3 output dimension

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into dense embeddings. Returns (n, dim) float32 array."""
        if self._is_flag:
            output = self.model.encode(
                texts,
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            embeddings = np.array(output["dense_vecs"], dtype=np.float32)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            ).astype(np.float32)

        if self.normalize and self._is_flag:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

        return embeddings


# ──────────────────────────────────────────────────────────────────────────────
# FAISS index
# ──────────────────────────────────────────────────────────────────────────────

class FinancialRetriever:
    """
    FAISS-backed retriever for financial documents with bge-m3 embeddings.

    Supports:
      - Building an index from a list of document chunks
      - Saving/loading the index and metadata to disk
      - Top-k retrieval by query
      - Cross-encoder reranking to top-r results
    """

    def __init__(
        self,
        embedder: Optional[BGEEmbedder] = None,
        index_path: Optional[str] = None,
        use_gpu_faiss: bool = False,
    ) -> None:
        self.embedder = embedder or BGEEmbedder()
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self._use_gpu = use_gpu_faiss

        if index_path and Path(index_path).exists():
            self.load(index_path)

    def build_index(
        self,
        chunks: List[Dict[str, Any]],
        index_type: str = "flat",
        n_list: int = 100,
    ) -> None:
        """
        Embed all chunks and build a FAISS index.

        Args:
            chunks:     List of chunk dicts (must have "text" key).
            index_type: "flat" for exact search (≤100K chunks) or "ivf" for
                        approximate search (>100K chunks, ~10× faster, ~5% quality loss).
            n_list:     Number of IVF cells (for index_type="ivf").
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        logger.info("Embedding %d chunks with bge-m3...", len(chunks))
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts)

        dim = embeddings.shape[1]
        logger.info("Embedding dim: %d, n_chunks: %d", dim, len(chunks))

        if index_type == "flat":
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_list, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training IVF index (n_list=%d)...", n_list)
            index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {index_type!r}")

        if self._use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU.")
            except Exception:
                logger.warning("Failed to move FAISS to GPU. Using CPU.")

        index.add(embeddings)
        self.index = index
        self.chunks = chunks
        logger.info("FAISS index built: %d vectors, index_type=%s.", index.ntotal, index_type)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve top-k chunks most relevant to the query.

        Returns:
            List of (chunk_dict, similarity_score) tuples, sorted descending.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_emb = self.embedder.encode([query])  # (1, dim)
        distances, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append((self.chunks[idx], float(dist)))

        return results

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_r: int = 2,
        reranker_model_id: str = "BAAI/bge-reranker-v2-m3",
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Cross-encoder reranking: re-score top-k candidates using a cross-encoder,
        then return the top-r highest-scored.

        Args:
            query:       The user query.
            candidates:  Output from retrieve() — list of (chunk, score) pairs.
            top_r:       Number of chunks to return after reranking.
            reranker_model_id: HuggingFace cross-encoder model ID.

        [WARN] TRADE-OFF: Cross-encoder reranking is 5–10× slower than bi-encoder
        retrieval. For latency-sensitive applications, skip reranking and use
        top-2 directly from FAISS. Reranking is recommended for batch/async.
        """
        if not candidates:
            return []

        try:
            from FlagEmbedding import FlagReranker
            reranker = FlagReranker(reranker_model_id, use_fp16=True)
            pairs = [(query, c["text"]) for c, _ in candidates]
            scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(scores, float):
                scores = [scores]
        except (ImportError, Exception) as exc:
            logger.warning("Cross-encoder reranking failed: %s. Using bi-encoder scores.", exc)
            return candidates[:top_r]

        reranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, score) for (chunk, _orig_score), score in reranked[:top_r]]

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 20,
        top_r: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Full pipeline: retrieve top-k candidates, rerank to top-r, return chunk dicts.

        This is the entry point for RAG context assembly.
        """
        candidates = self.retrieve(query, top_k=top_k)
        reranked = self.rerank(query, candidates, top_r=top_r)
        return [chunk for chunk, _ in reranked]

    def format_context(self, chunks: List[Dict[str, Any]], max_chars: int = 4000) -> str:
        """
        Combine retrieved chunks into a single context string for the model prompt.
        Annotates each chunk with its source metadata.
        """
        parts = []
        total_chars = 0
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = (
                f"[Source: {meta.get('company', 'Unknown')} "
                f"{meta.get('filing_type', '')} "
                f"{meta.get('year', '')} — {meta.get('section', '')}]"
            )
            text = chunk.get("text", "")
            part = f"{source}\n{text}"
            if total_chars + len(part) > max_chars:
                break
            parts.append(part)
            total_chars += len(part)

        return "\n\n".join(parts)

    def save(self, directory: str) -> None:
        """Save FAISS index and chunk metadata to disk."""
        import faiss

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        cpu_index = faiss.index_gpu_to_cpu(self.index) if self._use_gpu else self.index
        faiss.write_index(cpu_index, str(out / "index.faiss"))

        # Save chunk metadata
        with open(out / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Retriever saved to %s (%d chunks).", directory, len(self.chunks))

    def load(self, directory: str) -> None:
        """Load a previously saved FAISS index and chunk metadata."""
        import faiss

        index_file = Path(directory) / "index.faiss"
        chunks_file = Path(directory) / "chunks.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info("Retriever loaded from %s (%d chunks).", directory, len(self.chunks))


# ──────────────────────────────────────────────────────────────────────────────
# RAG pipeline (retriever + generation)
# ──────────────────────────────────────────────────────────────────────────────

def rag_answer(
    retriever: FinancialRetriever,
    model,
    tokenizer,
    question: str,
    top_k: int = 20,
    top_r: int = 2,
    **generate_kwargs,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full RAG pipeline: retrieve relevant context, then generate an answer.

    Returns:
        (answer_str, retrieved_chunks)
    """
    from src.inference.generate import generate_answer

    chunks = retriever.retrieve_and_rerank(question, top_k=top_k, top_r=top_r)
    context = retriever.format_context(chunks)

    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=question,
        context=context,
        **generate_kwargs,
    )

    return answer, chunks


# ──────────────────────────────────────────────────────────────────────────────
# Standalone demo / CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Build or query a FinReasoning RAG index")
    subparsers = parser.add_subparsers(dest="command")

    # Build index command
    build_parser = subparsers.add_parser("build", help="Build FAISS index from documents")
    build_parser.add_argument("--docs_dir", required=True,
                              help="Directory of .txt documents to index")
    build_parser.add_argument("--index_dir", default="outputs/rag_index")
    build_parser.add_argument("--chunk_size", type=int, default=512)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("--index_dir", default="outputs/rag_index")
    query_parser.add_argument("--question", required=True)
    query_parser.add_argument("--top_k", type=int, default=20)
    query_parser.add_argument("--top_r", type=int, default=2)

    args = parser.parse_args()

    if args.command == "build":
        docs_dir = Path(args.docs_dir)
        all_chunks = []
        for doc_file in sorted(docs_dir.glob("*.txt")):
            text = doc_file.read_text(encoding="utf-8", errors="ignore")
            metadata = {"source": doc_file.name, "section": "full"}
            chunks = chunk_document(text, metadata, chunk_size=args.chunk_size)
            all_chunks.extend(chunks)
            logger.info("Chunked %s → %d chunks", doc_file.name, len(chunks))

        embedder = BGEEmbedder()
        retriever = FinancialRetriever(embedder=embedder)
        retriever.build_index(all_chunks, index_type="flat" if len(all_chunks) < 100000 else "ivf")
        retriever.save(args.index_dir)
        print(f"\n[OK] Index built: {len(all_chunks)} chunks → {args.index_dir}")

    elif args.command == "query":
        embedder = BGEEmbedder()
        retriever = FinancialRetriever(embedder=embedder, index_path=args.index_dir)
        results = retriever.retrieve_and_rerank(args.question, top_k=args.top_k, top_r=args.top_r)
        context = retriever.format_context(results)
        print(f"\nQuery: {args.question}")
        print(f"\nRetrieved context:\n{context[:2000]}")
        print(f"\n[OK] Step 7a complete — RAG retrieval ready.")
    else:
        parser.print_help()
