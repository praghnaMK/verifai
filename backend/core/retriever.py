# ============================================================
#  retriever.py  —  Dual retrieval with Reciprocal Rank Fusion
#
#  BM25 (keyword-based)  +  FAISS (semantic/dense)
#  Merged via Reciprocal Rank Fusion (RRF)
#
#  Why dual retrieval?
#  - BM25 excels at exact keyword matches (technical terms, names)
#  - FAISS excels at semantic similarity (paraphrases, synonyms)
#  - RRF fusion consistently outperforms either alone
#  - Research shows this reduces hallucination by ~40% vs single retrieval
# ============================================================

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

import faiss
from rank_bm25 import BM25Okapi

from .config import config
from .ingestor import Chunk

_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'this', 'that', 'these', 'those', 'it', 'its',
    'as', 'if', 'not', 'so', 'than', 'then', 'there', 'their', 'they',
    'them', 'he', 'she', 'we', 'you', 'i', 'my', 'your', 'our', 'his', 'her',
})


class DualRetriever:
    """
    Hybrid retrieval engine: BM25 + FAISS merged via RRF.

    Usage:
        retriever = DualRetriever(embedder=embedder)
        retriever.build_index(chunks, embeddings)
        results = retriever.retrieve("What is the return policy?", top_k=5)
    """

    def __init__(self, embedder=None):
        self.chunks:    List[Chunk] = []
        self.bm25:      BM25Okapi | None = None
        self.faiss_idx: faiss.IndexFlatIP | None = None
        self.index_dir  = Path(config.INDEX_DIR)
        self._built     = False
        self._embedder  = embedder

    def _get_embedder(self):
        if self._embedder is None:
            from .embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    # ── Build ────────────────────────────────────────────────

    def build_index(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """
        Build BM25 and FAISS indices from chunks + pre-computed embeddings.

        Args:
            chunks:     List of Chunk objects (text + metadata)
            embeddings: Parallel list of embedding vectors (one per chunk)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) ≠ embedding count ({len(embeddings)})"
            )

        self.chunks = chunks

        # ── BM25 ─────────────────────────────────────────────
        tokenized = [self._tokenize(c.text) for c in chunks]
        self.bm25  = BM25Okapi(tokenized)

        # ── FAISS ────────────────────────────────────────────
        dim    = len(embeddings[0])
        matrix = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(matrix)
        self.faiss_idx = faiss.IndexFlatIP(dim)
        self.faiss_idx.add(matrix)

        self._built = True

    # ── Retrieve ─────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks using RRF-fused dual retrieval.

        Returns:
            List of (Chunk, rrf_score) sorted by relevance descending.
            rrf_score is in (0, 1] — higher = more relevant.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build_index() first.")

        top_k = top_k or config.TOP_K_RETRIEVAL

        bm25_ranks   = self._bm25_rank(query, top_k * 2)
        faiss_ranks  = self._faiss_rank(query, top_k * 2)
        fused        = self._reciprocal_rank_fusion(bm25_ranks, faiss_ranks)

        return fused[:top_k]

    def retrieve_with_similarity(
        self, query: str, top_k: int = None
    ) -> Tuple[List[Tuple[Chunk, float]], float]:
        """
        Retrieve + return the maximum cosine similarity (used for refusal decision).

        RRF scores are normalized to max=1.0, making them useless for refusal.
        We use raw FAISS cosine similarity instead — a real semantic distance in [0,1].

        Returns:
            (results, max_cosine_similarity)
        """
        results = self.retrieve(query, top_k)
        # Raw FAISS cosine sim is the reliable relevance signal
        raw_faiss = self._faiss_rank(query, 3)
        max_sim = raw_faiss[0][1] if raw_faiss else 0.0
        return results, max_sim

    # ── BM25 ─────────────────────────────────────────────────

    def _bm25_rank(
        self, query: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """Return (chunk_index, bm25_score) for top_k chunks."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Normalise to [0,1]
        max_score = scores.max() if scores.max() > 0 else 1.0
        scores    = scores / max_score

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    # ── FAISS ────────────────────────────────────────────────

    def _faiss_rank(
        self, query: str, top_k: int
    ) -> List[Tuple[int, float]]:
        """Return (chunk_index, cosine_similarity) for top_k chunks."""
        q_vec = np.array([self._get_embedder().embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        scores, indices = self.faiss_idx.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                # Cosine similarity is already in [-1, 1]; clamp to [0, 1]
                results.append((int(idx), float(max(0.0, score))))
        return results

    # ── Reciprocal Rank Fusion ───────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        bm25_results:  List[Tuple[int, float]],
        faiss_results: List[Tuple[int, float]],
    ) -> List[Tuple[Chunk, float]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ 1 / (k + rank_i)
        where k=60 (standard constant that smooths rank differences)

        Weighted variant: BM25_WEIGHT and FAISS_WEIGHT control contribution.
        """
        k      = config.RRF_K
        scores = {}

        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] = scores.get(idx, 0.0) + \
                          config.BM25_WEIGHT * (1.0 / (k + rank + 1))

        for rank, (idx, _) in enumerate(faiss_results):
            scores[idx] = scores.get(idx, 0.0) + \
                          config.FAISS_WEIGHT * (1.0 / (k + rank + 1))

        # Normalise RRF scores to [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in ranked
                if idx < len(self.chunks)]

    # ── Persistence ──────────────────────────────────────────

    def save(self):
        """Persist index to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(
            self.faiss_idx,
            str(self.index_dir / "faiss.index")
        )
        # Save chunks and BM25 via pickle
        with open(self.index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""
        try:
            faiss_path = self.index_dir / "faiss.index"
            if not faiss_path.exists():
                return False
            self.faiss_idx = faiss.read_index(str(faiss_path))
            with open(self.index_dir / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
            with open(self.index_dir / "bm25.pkl", "rb") as f:
                self.bm25 = pickle.load(f)
            self._built = True
            return True
        except Exception:
            return False

    # ── Utility ──────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase tokenizer for BM25 with stop word filtering."""
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if t not in _STOP_WORDS and len(t) > 1]

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
