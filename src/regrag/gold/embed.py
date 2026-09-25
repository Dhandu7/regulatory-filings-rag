"""Local ONNX embeddings (fastembed) with a persistent content-addressed cache.

The cache key is (model, sha256(text)), so re-running gold, re-chunking with a config
that reproduces some identical chunks, or rebuilding an index never re-embeds the same text.
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache

import duckdb
import numpy as np

# BGE models are trained with an instruction prefix on the query side only.
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
                "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: "}


def model_slug(model: str) -> str:
    return model.split("/")[-1].replace(".", "_").replace("-", "_").lower()


class Embedder:
    def __init__(self, model: str, cache_path: str | None = None, batch_size: int = 64):
        from fastembed import TextEmbedding
        self.model_name = model
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model)
        self._lock = threading.Lock()
        self._db = None
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            self._db = duckdb.connect(cache_path)
            self._db.execute("CREATE TABLE IF NOT EXISTS embed_cache ("
                             "model VARCHAR, text_hash VARCHAR, vec FLOAT[], PRIMARY KEY (model, text_hash))")
        self.hits = 0
        self.misses = 0
        self._query_cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _h(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        hashes = [self._h(t) for t in texts]
        cached: dict[str, list[float]] = {}
        if self._db is not None and hashes:
            with self._lock:
                self._db.execute("CREATE TEMP TABLE IF NOT EXISTS _q (h VARCHAR)")
                self._db.execute("DELETE FROM _q")
                self._db.executemany("INSERT INTO _q VALUES (?)", [(h,) for h in set(hashes)])
                rows = self._db.execute("SELECT text_hash, vec FROM embed_cache JOIN _q ON text_hash = h "
                                        "WHERE model = ?", [self.model_name]).fetchall()
            cached = {h: v for h, v in rows}
        missing = [(h, t) for h, t in dict(zip(hashes, texts)).items() if h not in cached]
        self.hits += len(texts) - sum(1 for h in hashes if h not in cached)
        self.misses += len(missing)
        if missing:
            # Length-sorted batches: each batch is padded to its longest member, so mixing short
            # prose chunks with long table chunks wastes most of the compute.
            missing.sort(key=lambda ht: len(ht[1]))
            vecs = list(self._model.embed([t for _, t in missing], batch_size=self.batch_size))
            new = {h: v.astype(np.float32).tolist() for (h, _), v in zip(missing, vecs)}
            cached.update(new)
            if self._db is not None:
                with self._lock:
                    self._db.executemany("INSERT OR IGNORE INTO embed_cache VALUES (?, ?, ?)",
                                         [(self.model_name, h, v) for h, v in new.items()])
        return np.asarray([cached[h] for h in hashes], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Query embeddings are cached in-process (repeat and eval questions are common)."""
        key = text.strip()
        if key not in self._query_cache:
            if len(self._query_cache) > 4096:
                self._query_cache.clear()
            prefix = QUERY_PREFIX.get(self.model_name, "")
            self._query_cache[key] = next(iter(self._model.embed([prefix + key]))).astype(np.float32)
        return self._query_cache[key]


@lru_cache(maxsize=2)
def get_embedder(model: str, cache_path: str | None, batch_size: int = 64) -> Embedder:
    return Embedder(model, cache_path, batch_size)
