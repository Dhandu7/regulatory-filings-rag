"""Silver: bronze manifest -> parsed documents/pages -> versioned chunks, with quarantine.

Tables written (Parquet parts, one per run):
  silver/documents/        one row per unique document (doc_id = sha256 of the raw bytes)
  silver/pages/            parsed page text + tables (lets us re-chunk without re-parsing)
  silver/chunks/chunk_config=<version>/
  silver/quarantine/       records that failed bronze validation or parsing, with reason

Incremental: a bronze record is processed once (keyed by source:source_id). Re-chunking
with a new config reads silver/pages only.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime

import pyarrow as pa

from ..bronze.validate import load_quarantine_ids, rkey
from ..config import ChunkingConfig
from ..sources import find_docket
from ..storage import Lake
from .chunk import chunk_id, chunk_pages, context_header
from .parse import Page, ParseError, parse_document

log = logging.getLogger(__name__)

DOC_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("record_key", pa.string()), ("source", pa.string()),
    ("source_id", pa.string()), ("title", pa.string()), ("docket", pa.string()),
    ("record_type", pa.string()), ("published_at", pa.string()), ("url", pa.string()),
    ("landing_url", pa.string()), ("object_uri", pa.string()), ("n_pages", pa.int32()),
    ("n_pages_total", pa.int32()), ("n_chars", pa.int64()), ("n_tables", pa.int32()),
    ("parser", pa.string()), ("text_hash", pa.string()), ("duplicate_of", pa.string()),
    ("status", pa.string()), ("run_id", pa.string()), ("parsed_at", pa.string()),
])
PAGE_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("page_no", pa.int32()), ("text", pa.string()),
    ("tables_json", pa.string()), ("headings_json", pa.string()), ("run_id", pa.string()),
])
CHUNK_SCHEMA = pa.schema([
    ("chunk_id", pa.string()), ("doc_id", pa.string()), ("chunk_config_version", pa.string()),
    ("ordinal", pa.int32()), ("chunk_type", pa.string()), ("section", pa.string()),
    ("page_start", pa.int32()), ("page_end", pa.int32()), ("n_tokens", pa.int32()),
    ("text", pa.string()), ("context", pa.string()), ("source", pa.string()),
    ("source_id", pa.string()), ("docket", pa.string()), ("title", pa.string()),
    ("published_at", pa.string()), ("url", pa.string()), ("run_id", pa.string()),
    ("chunked_at", pa.string()),
])
QUARANTINE_SCHEMA = pa.schema([
    ("record_key", pa.string()), ("source", pa.string()), ("source_id", pa.string()),
    ("sha256", pa.string()), ("object_uri", pa.string()), ("stage", pa.string()),
    ("reason", pa.string()), ("run_id", pa.string()), ("quarantined_at", pa.string()),
])


def _now() -> str:
    return datetime.now(UTC).isoformat()


def normalized_text_hash(text: str) -> str:
    """Hash of whitespace/case/digit-normalized text: catches re-issued copies of the
    same document (e.g. re-signed PDFs) whose bytes differ."""
    norm = re.sub(r"\s+", " ", re.sub(r"\d", "0", text.lower())).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def run_silver(cfg: dict, run_id: str | None = None, limit: int | None = None) -> dict:
    lake = Lake(cfg["lake_root"])
    scfg = cfg["silver"]
    ccfg = ChunkingConfig.load(scfg["chunking_config"])
    run_id = run_id or f"silver-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    manifest = lake.read_rows("bronze/manifest")
    done = {r["record_key"] for r in lake.read_rows("silver/documents", columns=["record_key"])}
    done |= {r["record_key"] for r in lake.read_rows("silver/quarantine", columns=["record_key"])}
    known_docs = {r["doc_id"]: r for r in lake.read_rows("silver/documents",
                                                          columns=["doc_id", "text_hash", "status"])}
    known_text = {r["text_hash"]: d for d, r in known_docs.items() if r["status"] == "ok"}
    bronze_quarantine = load_quarantine_ids(lake)

    todo = [r for r in manifest if rkey(r) not in done]
    if limit:
        todo = todo[:limit]
    docs, pages_out, chunks_out, quarantine = [], [], [], []

    def quarantine_rec(r: dict, stage: str, reason: str) -> None:
        quarantine.append({"record_key": rkey(r), "source": r["source"], "source_id": r["source_id"],
                           "sha256": r["sha256"], "object_uri": r["object_uri"], "stage": stage,
                           "reason": reason[:1000], "run_id": run_id, "quarantined_at": _now()})

    for r in todo:
        key = rkey(r)
        if key in bronze_quarantine:
            quarantine_rec(r, "bronze_validation", "failed a bronze validation check")
            continue
        extra = json.loads(r.get("extra_json") or "{}")
        base = {"record_key": key, "source": r["source"], "source_id": r["source_id"],
                "title": r["title"], "record_type": r["record_type"], "published_at": r["published_at"],
                "url": r["url"], "landing_url": extra.get("landing_url", r["url"]),
                "object_uri": r["object_uri"], "run_id": run_id, "parsed_at": _now()}

        # Exact-duplicate bytes: same doc_id as an existing document -> record, don't re-chunk.
        if r["sha256"] in known_docs:
            docs.append({**base, "doc_id": r["sha256"], "docket": r["docket"], "status": "duplicate",
                         "duplicate_of": r["sha256"], "text_hash": known_docs[r["sha256"]]["text_hash"]})
            continue
        try:
            parsed = parse_document(lake.get_bytes(r["object_uri"]), r["extension"],
                                    max_pages=scfg["max_pages"], extract_tables=scfg["extract_tables"])
        except ParseError as exc:
            quarantine_rec(r, "parse", str(exc))
            continue
        full_text = parsed.full_text
        if len(full_text) < scfg["min_text_chars"]:
            quarantine_rec(r, "parse", f"only {len(full_text)} chars extracted (scanned image PDF? needs OCR)")
            continue

        doc_id = r["sha256"]
        text_hash = normalized_text_hash(full_text)
        docket = r["docket"] or find_docket(full_text[:5000])
        doc = {**base, "doc_id": doc_id, "docket": docket, "n_pages": len(parsed.pages),
               "n_pages_total": parsed.n_pages_total, "n_chars": len(full_text),
               "n_tables": sum(len(p.tables) for p in parsed.pages), "parser": parsed.parser,
               "text_hash": text_hash, "duplicate_of": None, "status": "ok"}
        if text_hash in known_text:
            doc.update(status="duplicate", duplicate_of=known_text[text_hash])
            docs.append(doc)
            known_docs[doc_id] = doc
            continue
        known_text[text_hash] = doc_id
        known_docs[doc_id] = doc
        docs.append(doc)
        pages_out += [{"doc_id": doc_id, "page_no": p.page_no, "text": p.text,
                       "tables_json": json.dumps(p.tables), "headings_json": json.dumps(p.headings),
                       "run_id": run_id} for p in parsed.pages]
        chunks_out += build_chunks(doc, parsed.pages, ccfg, run_id)

    lake.append_rows("silver/documents", docs, schema=DOC_SCHEMA, part_name=run_id)
    lake.append_rows("silver/pages", pages_out, schema=PAGE_SCHEMA, part_name=run_id)
    lake.append_rows(f"silver/chunks/chunk_config={ccfg.version}", chunks_out, schema=CHUNK_SCHEMA,
                     part_name=run_id)
    lake.append_rows("silver/quarantine", quarantine, schema=QUARANTINE_SCHEMA, part_name=run_id)
    summary = {"run_id": run_id, "chunk_config_version": ccfg.version, "considered": len(todo),
               "documents_ok": sum(d["status"] == "ok" for d in docs),
               "duplicates": sum(d["status"] == "duplicate" for d in docs),
               "quarantined": len(quarantine), "chunks": len(chunks_out)}
    log.info("silver summary: %s", summary)
    return summary


def build_chunks(doc: dict, pages: list[Page], ccfg: ChunkingConfig, run_id: str) -> list[dict]:
    out = []
    for c in chunk_pages(pages, ccfg):
        out.append({
            "chunk_id": chunk_id(doc["doc_id"], ccfg.version, c.ordinal), "doc_id": doc["doc_id"],
            "chunk_config_version": ccfg.version, "ordinal": c.ordinal, "chunk_type": c.chunk_type,
            "section": c.section, "page_start": c.page_start, "page_end": c.page_end,
            "n_tokens": c.n_tokens, "text": c.text,
            "context": context_header(doc["title"], doc["docket"], c.section) if ccfg.prepend_context else "",
            "source": doc["source"], "source_id": doc["source_id"], "docket": doc["docket"],
            "title": doc["title"], "published_at": doc["published_at"], "url": doc["landing_url"],
            "run_id": run_id, "chunked_at": _now(),
        })
    return out


def rechunk(cfg: dict, chunking_config_path: str | None = None) -> dict:
    """Bring a chunk version up to date from stored silver pages (no re-download, no re-parse).

    Only documents that have no chunks yet in the target version are chunked, so this is both
    the way to build a brand-new version and a cheap no-op backfill after every silver run."""
    lake = Lake(cfg["lake_root"])
    ccfg = ChunkingConfig.load(chunking_config_path or cfg["silver"]["chunking_config"])
    target = f"silver/chunks/chunk_config={ccfg.version}"
    docs = {d["doc_id"]: d for d in lake.read_rows("silver/documents") if d["status"] == "ok"}
    have = {r["doc_id"] for r in lake.read_rows(target, columns=["doc_id"])}
    missing = set(docs) - have
    if not missing:
        return {"chunk_config_version": ccfg.version, "documents": 0, "chunks": 0}
    run_id = f"rechunk-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    by_doc: dict[str, list[Page]] = {}
    for p in lake.read_rows("silver/pages"):
        if p["doc_id"] in missing:
            by_doc.setdefault(p["doc_id"], []).append(
                Page(p["page_no"], p["text"], json.loads(p["tables_json"]),
                     [tuple(h) for h in json.loads(p["headings_json"])]))
    rows = []
    for doc_id, pages in by_doc.items():
        rows += build_chunks(docs[doc_id], sorted(pages, key=lambda p: p.page_no), ccfg, run_id)
    lake.append_rows(target, rows, schema=CHUNK_SCHEMA, part_name=run_id)
    return {"chunk_config_version": ccfg.version, "documents": len(by_doc), "chunks": len(rows)}
