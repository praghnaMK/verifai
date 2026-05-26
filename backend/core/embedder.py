# ============================================================
#  embedder.py  —  Local embeddings via sentence-transformers
#
#  Uses all-MiniLM-L6-v2 — free, runs on CPU, no API key.
#  384-dim vectors. ~80MB download on first run (cached).
#  Disk cache: same text = zero recomputation.
# ============================================================

import hashlib
import json
import numpy as np
from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from .config import config


class Embedder:
    """
    Local embedding engine using sentence-transformers.
    No API key. No cost. Runs entirely on CPU.

    Cache: SHA256(text) → vector saved to disk.
    Re-indexing the same document = instant (cache hit).
    """

    _model = None  # Singleton — load model once

    def __init__(self):
        if Embedder._model is None:
            print(f"[Embedder] Loading model '{config.EMBEDDING_MODEL}' (first run downloads ~80MB)...")
            Embedder._model = SentenceTransformer(config.EMBEDDING_MODEL)
            print(f"[Embedder] Model ready — {Embedder._model.get_embedding_dimension()}d vectors")

        self._model    = Embedder._model
        self._cache_dir = Path(config.INDEX_DIR) / "embed_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache = {}

    # ── Public API ───────────────────────────────────────────

    def embed_chunks(self, chunks) -> List[List[float]]:
        """Embed a list of Chunk objects. Returns parallel list of vectors."""
        texts = [c.text for c in chunks]
        return self.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embed_with_cache(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings — batch for efficiency."""
        # Separate cached vs uncached
        results    = [None] * len(texts)
        uncached   = []
        uncached_i = []

        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in self._mem_cache:
                results[i] = self._mem_cache[key]
                continue
            cache_file = self._cache_dir / f"{key}.json"
            if cache_file.exists():
                vec = json.loads(cache_file.read_text())
                self._mem_cache[key] = vec
                results[i] = vec
                continue
            uncached.append(text)
            uncached_i.append(i)

        # Batch encode uncached texts
        if uncached:
            print(f"[Embedder] Encoding {len(uncached)} new texts...")
            vecs = self._model.encode(
                uncached,
                batch_size=32,
                show_progress_bar=len(uncached) > 20,
                convert_to_numpy=True,
            )
            for text, vec, i in zip(uncached, vecs, uncached_i):
                key  = hashlib.sha256(text.encode()).hexdigest()
                vec_list = vec.tolist()
                # Persist
                (self._cache_dir / f"{key}.json").write_text(json.dumps(vec_list))
                self._mem_cache[key] = vec_list
                results[i] = vec_list

        return results

    # ── Internal ─────────────────────────────────────────────

    def _embed_with_cache(self, text: str) -> List[float]:
        key        = hashlib.sha256(text.encode()).hexdigest()
        if key in self._mem_cache:
            return self._mem_cache[key]
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            vec = json.loads(cache_file.read_text())
            self._mem_cache[key] = vec
            return vec
        vec = self._model.encode([text], convert_to_numpy=True)[0].tolist()
        cache_file.write_text(json.dumps(vec))
        self._mem_cache[key] = vec
        return vec

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()
