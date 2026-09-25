"""Ontario Energy Board — Regulatory Document Search (RDS).

RDS is an OpenText/HPE Content Manager "WebDrawer" instance. Its search endpoint
returns JSON with `format=json`, supports Content Manager search clauses
(`title:`, `extension:`, `createdOn:`/`registeredOn:` with `YYYY-MM-DD to YYYY-MM-DD`),
and serves the file at /Record/{uri}/File/document.

Note: requesting `properties=all` fails (the container property is access-controlled),
so we always ask for an explicit property list. We deliberately do not collect
the author field (a staff member's name) — it is not needed downstream.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date

from .base import PoliteSession, SourceRecord, find_docket

PROPERTIES = ",".join([
    "RecordTitle", "RecordNumber", "RecordDateCreated", "RecordDateRegistered",
    "RecordRecordType", "RecordExtension", "RecordDocumentSize", "RecordMimeType",
])


def _val(rec: dict, key: str):
    v = rec.get(key)
    if not isinstance(v, dict):
        return v
    if "DateTime" in v:
        return None if v.get("IsClear") else v["DateTime"].replace(".0000000Z", "Z")
    if "RecordTypeName" in v:
        return v["RecordTypeName"].get("Value")
    return v.get("Value")


class OEBSource:
    name = "oeb"

    def __init__(self, cfg: dict, http_cfg: dict):
        self.base = cfg["base_url"].rstrip("/")
        self.extensions = [e.lower() for e in cfg.get("extensions", ["pdf"])]
        self.title_filter = cfg.get("title_filter")
        self.page_size = int(cfg.get("page_size", 100))
        self.http = PoliteSession(http_cfg["user_agent"], http_cfg.get("timeout_s", 60),
                                  cfg.get("request_delay_s", 0.5))

    def build_query(self, since: date, until: date) -> str:
        clauses = [f"registeredOn:{since.isoformat()} to {until.isoformat()}"]
        if self.extensions:
            clauses.append("(" + " or ".join(f"extension:{e}" for e in self.extensions) + ")")
        if self.title_filter:
            clauses.append(f"title:{self.title_filter}")
        return " and ".join(clauses)

    def parse_result(self, r: dict) -> SourceRecord:
        uri = str(r["Uri"])
        title = _val(r, "RecordTitle") or ""
        return SourceRecord(
            source=self.name,
            source_id=uri,
            title=title,
            url=f"{self.base}/Record/{uri}/File/document",
            published_at=_val(r, "RecordDateCreated"),
            registered_at=_val(r, "RecordDateRegistered"),
            extension=(_val(r, "RecordExtension") or "").lower(),
            record_number=_val(r, "RecordNumber"),
            record_type=_val(r, "RecordRecordType"),
            docket=find_docket(title),
            size_hint=_val(r, "RecordDocumentSize"),
            extra={"mime_type": _val(r, "RecordMimeType"),
                   "landing_url": f"{self.base}/Record/{uri}"},
        )

    def list_records(self, since: date, until: date, max_docs: int | None = None) -> Iterator[SourceRecord]:
        q = self.build_query(since, until)
        start, yielded = 1, 0
        while True:
            resp = self.http.get(f"{self.base}/Record", params={
                "q": q, "format": "json", "pageSize": self.page_size, "start": start,
                "sortBy": "registeredOn-", "properties": PROPERTIES,
            })
            payload = resp.json()
            status = payload.get("ResponseStatus") or {}
            if status.get("ErrorCode"):
                raise RuntimeError(f"OEB search error: {status}")
            results = payload.get("Results", [])
            for r in results:
                yield self.parse_result(r)
                yielded += 1
                if max_docs and yielded >= max_docs:
                    return
            if not payload.get("HasMoreItems") or not results:
                return
            start += len(results)

    def fetch(self, rec: SourceRecord) -> tuple[bytes, str]:
        resp = self.http.get(rec.url)
        return resp.content, resp.headers.get("Content-Type", "application/octet-stream").split(";")[0]
