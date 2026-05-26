# ============================================================
#  config.py  —  VerifAI global configuration
# ============================================================

import os
from dataclasses import dataclass

@dataclass
class Config:
    # ── LLM (Groq) ──────────────────────────────────────────
    # Groq is FREE and extremely fast (~500 tok/s)
    # Get key at: https://console.groq.com
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Model options (all free on Groq):
    #   llama-3.3-70b-versatile  ← best quality
    #   llama-3.1-8b-instant     ← fastest
    #   mixtral-8x7b-32768       ← large context window
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_TEMPERATURE: float = 0.1       # Low temp = more factual
    LLM_MAX_TOKENS: int = 1500

    # ── Retrieval ────────────────────────────────────────────
    CHUNK_SIZE: int = 800              # Characters per chunk
    CHUNK_OVERLAP: int = 150          # Overlap between chunks
    TOP_K_RETRIEVAL: int = 8          # Chunks to retrieve per query
    BM25_WEIGHT: float = 0.4          # Weight in RRF fusion
    FAISS_WEIGHT: float = 0.6         # Weight in RRF fusion
    RRF_K: int = 60                   # RRF constant (standard = 60)

    # ── Hallucination detection ──────────────────────────────
    GROUNDING_THRESHOLD: float = 0.60  # Below this → flag as low-confidence
    REFUSAL_SIMILARITY_THRESHOLD: float = 0.30  # Below this → refuse (cosine sim)
    VERIFICATION_BATCH_SIZE: int = 5   # Claims verified per LLM call

    # ── Embeddings ───────────────────────────────────────────
    # Groq doesn't provide embeddings — use sentence-transformers (free, local)
    # No API key needed. Runs on CPU. ~80MB model download on first run.
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # 384-dim, fast, accurate

    # ── Storage ──────────────────────────────────────────────
    INDEX_DIR: str = "verifai_index"
    UPLOAD_DIR: str = "uploads"

    # ── API ──────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_VERSION: str = "v1"

config = Config()
