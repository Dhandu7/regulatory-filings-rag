"""Gold: embed silver chunks of one chunk-config version into a versioned vector index.

index_version = <chunk_config_version>__<embed_model_slug>. A new chunking config or
embedding model creates a new index alongside the old one; `activate` flips which one
the API serves, so rollbacks are one command and eval runs are reproducible.
"""
from __future__ import annotations

import logging
import os

from ..config import ChunkingConfig
from ..storage import Lake
from .embed import get_embedder, model_slug
from .store import get_store

log = logging.getLogger(__name__)


def index_version_for(chunk_config_version: str, embed_model: str) -> str:
    return f"{chunk_config_version}__{model_slug(embed_model)}"


def embed_text(row: dict) -> str:
    return f"{row['context']}\n{row['text']}" if row.get("context") else row["text"]


def run_gold(cfg: dict, chunk_config_path: str | None = None, activate: bool = True,
             batch: int = 256) -> dict:
    g = cfg["gold"]
    ccfg = ChunkingConfig.load(chunk_config_path or cfg["silver"]["chunking_config"])
    version = index_version_for(ccfg.version, g["embed_model"])
    chunks = load_chunks(cfg, ccfg.version)
    store = get_store(cfg)
    store.ensure_version({"index_version": version, "chunk_config_version": ccfg.version,
                          "embed_model": g["embed_model"], "config": ccfg.__dict__})
    have = store.existing_ids(version)
    # Latest silver row per chunk_id wins (re-runs append; ids are deterministic).
    todo = list({c["chunk_id"]: c for c in chunks if c["chunk_id"] not in have}.values())
    todo.sort(key=lambda r: len(embed_text(r)))   # similar lengths per batch -> far less padding
    embedder = get_embedder(g["embed_model"], _cache_path(cfg), g["embed_batch_size"])
    loaded = 0
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = embedder.embed_documents([embed_text(r) for r in part])
        loaded += store.upsert(version, part, vecs)
        log.info("gold %s: %d/%d", version, min(i + batch, len(todo)), len(todo))
    # Rows loaded before the ordinal column existed get it filled in place (used for neighbor expansion).
    backfilled = store.backfill_ordinals(version, {c["chunk_id"]: c["ordinal"] for c in chunks})
    if activate:
        store.activate(version)
    summary = {"index_version": version, "silver_chunks": len(chunks), "already_indexed": len(have),
               "loaded": loaded, "ordinals_backfilled": backfilled, "embed_cache_hits": embedder.hits,
               "embed_cache_misses": embedder.misses, "active": store.active_version()}
    log.info("gold summary: %s", summary)
    return summary


def load_chunks(cfg: dict, version: str) -> list[dict]:
    """Silver chunks for one chunk version.

    Locally they are Parquet parts in the lake. On Databricks the Spark silver job writes a Delta
    table instead; set `gold.chunks_table` (e.g. workspace.regrag.silver_chunks) to read it."""
    table = cfg["gold"].get("chunks_table")
    if not table:
        return Lake(cfg["lake_root"]).read_rows(f"silver/chunks/chunk_config={version}")
    from pyspark.sql import SparkSession  # available on Databricks compute
    spark = SparkSession.builder.getOrCreate()
    df = spark.table(table).where(f"chunk_config_version = '{version}'")
    return [row.asDict() for row in df.collect()]


def _cache_path(cfg: dict) -> str:
    return os.path.join(os.path.dirname(cfg["catalog_path"]), "cache", "embeddings.duckdb")
