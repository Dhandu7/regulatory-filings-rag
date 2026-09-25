from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Protocol

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

DOCKET_PATTERNS = [
    # OEB case numbers; lookarounds instead of \b because titles use '_' as a separator
    re.compile(r"(?<![A-Za-z0-9])EB-\d{4}-\d{4}(?!\d)"),
    re.compile(r"(?<![A-Za-z0-9])RP-\d{4}-\d{4}(?!\d)"),     # legacy OEB
    re.compile(r"\bProceeding\s+(?:ID\s+)?(\d{5})\b", re.IGNORECASE),  # AUC
]


def find_docket(*texts: str | None) -> str | None:
    for text in texts:
        if not text:
            continue
        for pat in DOCKET_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1) if pat.groups else m.group(0)
    return None


@dataclass
class SourceRecord:
    source: str
    source_id: str              # stable id at the regulator (OEB record Uri, EDGAR accession/doc)
    title: str
    url: str
    published_at: str           # ISO-8601 UTC
    registered_at: str | None   # when the regulator registered it (incremental watermark)
    extension: str
    record_number: str | None = None
    record_type: str | None = None
    docket: str | None = None
    size_hint: int | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class Source(Protocol):
    name: str

    def list_records(self, since: date, until: date, max_docs: int | None = None) -> Iterator[SourceRecord]: ...

    def fetch(self, rec: SourceRecord) -> tuple[bytes, str]: ...


class TransientHTTPError(Exception):
    pass


class PoliteSession:
    """requests.Session with a descriptive UA, rate limiting, and retry on 429/5xx."""

    def __init__(self, user_agent: str, timeout_s: int = 60, delay_s: float = 0.5):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.timeout = timeout_s
        self.delay = delay_s
        self._last = 0.0

    @retry(retry=retry_if_exception_type((TransientHTTPError, requests.ConnectionError, requests.Timeout)),
           wait=wait_exponential(multiplier=1, max=30), stop=stop_after_attempt(5), reraise=True)
    def get(self, url: str, **kw) -> requests.Response:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        resp = self.s.get(url, timeout=self.timeout, **kw)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise TransientHTTPError(f"{resp.status_code} for {url}")
        resp.raise_for_status()
        return resp
