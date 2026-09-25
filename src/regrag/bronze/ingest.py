"""Bronze: land raw regulator files + metadata in object storage, then validate.

Layout (all under lake_root):
  bronze/objects/source=<src>/ingest_date=<d>/<source_id>_<sha12>.<ext>   raw bytes, never modified
  bronze/manifest/<run_id>.parquet                                         one row per landed record
  bronze/fetch_errors/<run_id>.parquet                                     records we could not download
  bronze/validation/<run_id>.json                                          validation report
  _state/<src>.json                                                        incremental watermark
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta

import pyarrow as pa

from ..sources import SourceRecord, get_source
from ..storage import Lake, sha256_bytes

log = logging.getLogger(__name__)

MANIFEST_SCHEMA = pa.schema([
    ("run_id", pa.string()), ("source", pa.string()), ("source_id", pa.string()),
    ("title", pa.string()), ("url", pa.string()), ("published_at", pa.string()),
    ("registered_at", pa.string()), ("extension", pa.string()), ("record_number", pa.string()),
    ("record_type", pa.string()), ("docket", pa.string()), ("size_hint", pa.int64()),
    ("content_type", pa.string()), ("object_uri", pa.string()), ("sha256", pa.string()),
    ("bytes", pa.int64()), ("fetched_at", pa.string()), ("duplicate_of", pa.string()),
    ("extra_json", pa.string()),
])


def new_run_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:80]


def read_state(lake: Lake, source: str) -> dict:
    raw = lake.read_text(f"_state/{source}.json")
    return json.loads(raw) if raw else {}


def write_state(lake: Lake, source: str, state: dict) -> None:
    lake.overwrite_json(f"_state/{source}.json", json.dumps(state, indent=2, sort_keys=True))


def resolve_window(lake: Lake, source: str, since: date | None, until: date | None,
                   default_lookback_days: int = 7, overlap_days: int = 1) -> tuple[date, date]:
    """Explicit dates win; otherwise resume from the watermark with a small overlap
    (late-registered records), falling back to a default lookback on first run."""
    until = until or datetime.now(UTC).date()
    if since:
        return since, until
    wm = read_state(lake, source).get("watermark_registered_at")
    if wm:
        return date.fromisoformat(wm[:10]) - timedelta(days=overlap_days), until
    return until - timedelta(days=default_lookback_days), until


def ingest(cfg: dict, source_name: str, since: date | None = None, until: date | None = None,
           max_docs: int | None = None, run_id: str | None = None) -> dict:
    lake = Lake(cfg["lake_root"])
    src = get_source(source_name, cfg)
    since, until = resolve_window(lake, source_name, since, until)
    run_id = run_id or new_run_id(f"bronze-{source_name}")
    log.info("bronze %s: window %s..%s run=%s", source_name, since, until, run_id)

    existing = lake.read_rows("bronze/manifest",
                              columns=["source", "source_id", "sha256", "object_uri", "duplicate_of"])
    seen_ids = {(r["source"], r["source_id"]) for r in existing}
    by_sha = {r["sha256"]: r["object_uri"] for r in existing if r["duplicate_of"] is None}

    rows, errors, listed, skipped = [], [], 0, 0
    today = datetime.now(UTC).date().isoformat()
    max_registered = read_state(lake, source_name).get("watermark_registered_at")

    for rec in src.list_records(since, until, max_docs=max_docs):
        listed += 1
        if (rec.source, rec.source_id) in seen_ids:
            skipped += 1
            continue
        seen_ids.add((rec.source, rec.source_id))
        try:
            data, content_type = src.fetch(rec)
        except Exception as exc:  # noqa: BLE001 - we record and continue; validation flags the rate
            log.warning("fetch failed %s: %s", rec.source_id, exc)
            errors.append({"run_id": run_id, "source": rec.source, "source_id": rec.source_id,
                           "url": rec.url, "error": repr(exc)[:500],
                           "failed_at": datetime.now(UTC).isoformat()})
            continue
        rows.append(_land(lake, run_id, rec, data, content_type, today, by_sha))
        if rec.registered_at and (not max_registered or rec.registered_at > max_registered):
            max_registered = rec.registered_at

    lake.append_rows("bronze/manifest", rows, schema=MANIFEST_SCHEMA, part_name=run_id)
    lake.append_rows("bronze/fetch_errors", errors, part_name=run_id)

    from .validate import validate_run  # local import: validate depends on manifest helpers
    report = validate_run(cfg, lake, run_id, rows, errors, listed=listed)

    # Only advance the watermark when the batch passed validation, so a bad pull is re-tried.
    if report["status"] != "failed" and max_registered:
        write_state(lake, source_name, {"watermark_registered_at": max_registered, "last_run_id": run_id})

    summary = {"run_id": run_id, "source": source_name, "window": [since.isoformat(), until.isoformat()],
               "listed": listed, "landed": len(rows), "skipped_existing": skipped,
               "fetch_errors": len(errors), "validation": report["status"]}
    log.info("bronze summary: %s", summary)
    return summary


def _land(lake: Lake, run_id: str, rec: SourceRecord, data: bytes, content_type: str,
          today: str, by_sha: dict[str, str]) -> dict:
    sha = sha256_bytes(data)
    duplicate_of = None
    if sha in by_sha:
        # Identical bytes already landed under another id (re-filed / cross-posted doc):
        # keep the metadata row, point at the existing object, don't store twice.
        object_uri, duplicate_of = by_sha[sha], by_sha[sha]
    else:
        ext = rec.extension or "bin"
        rel = (f"bronze/objects/source={rec.source}/ingest_date={today}/"
               f"{_safe(rec.source_id)}_{sha[:12]}.{ext}")
        object_uri = lake.put_bytes(rel, data)
        by_sha[sha] = object_uri
    return {
        "run_id": run_id, "source": rec.source, "source_id": rec.source_id, "title": rec.title,
        "url": rec.url, "published_at": rec.published_at, "registered_at": rec.registered_at,
        "extension": rec.extension, "record_number": rec.record_number, "record_type": rec.record_type,
        "docket": rec.docket, "size_hint": rec.size_hint, "content_type": content_type,
        "object_uri": object_uri, "sha256": sha, "bytes": len(data),
        "fetched_at": datetime.now(UTC).isoformat(), "duplicate_of": duplicate_of,
        "extra_json": json.dumps(rec.extra, sort_keys=True),
    }
