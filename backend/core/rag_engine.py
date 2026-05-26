# ============================================================
#  rag_engine.py  —  VerifAI orchestrator (Groq LLM)
#
#  Pipeline:
#  1. Ingest documents → sentence-aware chunks
#  2. Embed via sentence-transformers (local, free)
#  3. Build BM25 + FAISS dual index
#  4. Query → dual retrieve → Groq generate → verify → return
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import time

from groq import Groq

from .config import config
from .ingestor import DocumentIngestor, Chunk
from .embedder import Embedder
from .retriever import DualRetriever
from .verifier import HallucinationVerifier, VerificationResult


@dataclass
class QueryResult:
    query:                str
    answer:               str
    verification:         VerificationResult
    retrieved_chunks:     list
    latency_ms:           int
    doc_names:            List[str]
    total_chunks_indexed: int


class VerifAIEngine:
    """
    Main VerifAI engine.

    LLM:        Groq (llama-3.3-70b) — free, ~500 tok/s
    Embeddings: sentence-transformers (local, no API key)
    Retrieval:  BM25 + FAISS + Reciprocal Rank Fusion
    Verifier:   Groq claim-level hallucination checker
    """

    def __init__(self):
        self.ingestor  = DocumentIngestor()
        self.embedder  = Embedder()
        self.retriever = DualRetriever(embedder=self.embedder)
        self.verifier  = HallucinationVerifier()
        self._docs_loaded: List[str] = []

        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set.\n"
                "Get free key: https://console.groq.com\n"
                "Then: export GROQ_API_KEY=your_key"
            )
        self.client = Groq(api_key=config.GROQ_API_KEY)

    # ── Ingestion ─────────────────────────────────────────────

    def ingest(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        print(f"[VerifAI] Ingesting: {path.name}")

        chunks     = self.ingestor.ingest(path)
        print(f"[VerifAI] {len(chunks)} chunks created")

        print("[VerifAI] Embedding chunks (local model)...")
        embeddings = self.embedder.embed_chunks(chunks)

        if self.retriever.is_built:
            existing = self.retriever.chunks
            # Reconstruct existing embeddings from FAISS — avoids re-embedding O(n) chunks
            exist_emb = [
                self.retriever.faiss_idx.reconstruct(i).tolist()
                for i in range(len(existing))
            ]
            all_chunks     = existing + chunks
            all_embeddings = exist_emb + embeddings
        else:
            all_chunks     = chunks
            all_embeddings = embeddings

        self.retriever.build_index(all_chunks, all_embeddings)
        self.retriever.save()
        self._docs_loaded.append(path.name)

        print(f"[VerifAI] Ready — {self.retriever.chunk_count} chunks indexed")
        return {
            "chunks":        len(chunks),
            "doc_name":      path.name,
            "status":        "ok",
            "total_indexed": self.retriever.chunk_count,
        }

    def load_index(self) -> bool:
        success = self.retriever.load()
        if success:
            print(f"[VerifAI] Index loaded — {self.retriever.chunk_count} chunks")
        return success

    # ── Query ─────────────────────────────────────────────────

    def query(self, question: str, top_k: int = None) -> QueryResult:
        if not self.retriever.is_built:
            raise RuntimeError("No documents indexed. Call ingest() first.")

        start_ms = int(time.time() * 1000)

        # 1. Dual retrieve
        retrieved, max_sim = self.retriever.retrieve_with_similarity(
            question, top_k or config.TOP_K_RETRIEVAL
        )

        # 2. Refusal check
        if max_sim < config.REFUSAL_SIMILARITY_THRESHOLD:
            verification = self.verifier.verify(
                answer="", chunks=retrieved, query=question, max_sim=max_sim
            )
            return QueryResult(
                query=question, answer="",
                verification=verification,
                retrieved_chunks=retrieved,
                latency_ms=int(time.time() * 1000) - start_ms,
                doc_names=self._docs_loaded,
                total_chunks_indexed=self.retriever.chunk_count,
            )

        # 3. Generate with Groq
        context = self._format_context(retrieved)
        answer  = self._generate(question, context)

        # 4. Verify
        verification = self.verifier.verify(
            answer=answer, chunks=retrieved, query=question, max_sim=max_sim
        )

        return QueryResult(
            query=question, answer=answer,
            verification=verification,
            retrieved_chunks=retrieved,
            latency_ms=int(time.time() * 1000) - start_ms,
            doc_names=self._docs_loaded,
            total_chunks_indexed=self.retriever.chunk_count,
        )

    # ── Generation ────────────────────────────────────────────

    def _generate(self, question: str, context: str) -> str:
        """Generate answer via Groq — grounded strictly in context."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise document analyst. "
                    "Answer using ONLY the provided context — never add outside knowledge. "
                    "Write each fact as its own complete sentence so claims can be verified individually. "
                    "Cite sources inline as [Source N]. "
                    "If the context is insufficient, say so explicitly rather than speculating. "
                    "Prefer exact terms, names, and numbers from the text."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
            },
        ]

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()

    def _format_context(self, chunks) -> str:
        parts = []
        for i, (chunk, score) in enumerate(chunks):
            parts.append(
                f"[Source {i+1}: {chunk.doc_name}, Page {chunk.page_num+1}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    # ── Properties ────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.retriever.is_built

    @property
    def stats(self) -> dict:
        return {
            "chunks_indexed": self.retriever.chunk_count,
            "docs_loaded":    self._docs_loaded,
            "index_dir":      str(config.INDEX_DIR),
        }
