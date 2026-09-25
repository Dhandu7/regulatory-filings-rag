"""SEC EDGAR fallback source (utility-company filings via the submissions API).

EDGAR primary documents are usually HTML, not PDF; the silver parser handles both.
SEC's fair-access policy requires a descriptive User-Agent and <= 10 req/s.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date

from .base import PoliteSession, SourceRecord


class EdgarSource:
    name = "edgar"

    def __init__(self, cfg: dict, http_cfg: dict):
        self.base = cfg["base_url"].rstrip("/")
        self.archive = cfg["archive_url"].rstrip("/")
        self.ciks = [c.zfill(10) for c in cfg["ciks"]]
        self.forms = set(cfg.get("forms", []))
        self.http = PoliteSession(http_cfg["user_agent"], http_cfg.get("timeout_s", 60),
                                  cfg.get("request_delay_s", 0.2))

    def list_records(self, since: date, until: date, max_docs: int | None = None) -> Iterator[SourceRecord]:
        yielded = 0
        for cik in self.ciks:
            sub = self.http.get(f"{self.base}/submissions/CIK{cik}.json").json()
            company = sub.get("name", cik)
            recent = sub["filings"]["recent"]
            for i, acc in enumerate(recent["accessionNumber"]):
                form, filed = recent["form"][i], recent["filingDate"][i]
                if self.forms and form not in self.forms:
                    continue
                if not (since.isoformat() <= filed <= until.isoformat()):
                    continue
                doc = recent["primaryDocument"][i]
                acc_nodash = acc.replace("-", "")
                yield SourceRecord(
                    source=self.name,
                    source_id=f"{acc}/{doc}",
                    title=f"{company} {form} {recent.get('primaryDocDescription', [''] * (i + 1))[i] or ''}".strip(),
                    url=f"{self.archive}/{int(cik)}/{acc_nodash}/{doc}",
                    published_at=f"{filed}T00:00:00Z",
                    registered_at=recent.get("acceptanceDateTime", [None] * (i + 1))[i],
                    extension=doc.rsplit(".", 1)[-1].lower(),
                    record_number=acc,
                    record_type=form,
                    docket=acc,
                    extra={"cik": cik, "company": company},
                )
                yielded += 1
                if max_docs and yielded >= max_docs:
                    return

    def fetch(self, rec: SourceRecord) -> tuple[bytes, str]:
        resp = self.http.get(rec.url)
        return resp.content, resp.headers.get("Content-Type", "application/octet-stream").split(";")[0]
