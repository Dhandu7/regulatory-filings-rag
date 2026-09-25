"""Offline fixtures: synthetic regulator PDFs, a fake Source, and a deterministic stub embedder,
so the suite runs in CI with no network and no model download."""
from __future__ import annotations

import copy
import hashlib
from datetime import date

import numpy as np
import pytest
from fpdf import FPDF

from regrag.config import load_config
from regrag.sources import SourceRecord


def make_pdf(pages: list[list[str]], header: str | None = None, footer: str | None = None,
             table: list[list[str]] | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(False)
    for i, lines in enumerate(pages, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        if header:
            pdf.set_xy(10, 8)
            pdf.cell(0, 5, header)
        y = 20
        for line in lines:
            pdf.set_xy(10, y)
            pdf.cell(0, 5, line)
            y += 6
        if table and i == 1:
            y += 4
            for row in table:
                x = 10
                for cell in row:
                    pdf.set_xy(x, y)
                    pdf.cell(45, 7, cell, border=1)
                    x += 45
                y += 7
        if footer:
            pdf.set_xy(10, 285)
            pdf.cell(0, 5, f"{footer} {i}")
    return bytes(pdf.output())


DECISION_PAGES = [
    ["DECISION AND ORDER", "EB-2026-0015", "Northwind Hydro Inc.", "Application for 2027 rates"],
    ["1 INTRODUCTION", "Northwind Hydro Inc. filed an application on March 3, 2026 for new rates.",
     "The application seeks a revenue requirement of $42.7 million for the 2027 test year.",
     "Intervenors in the proceeding were VECC and SEC."],
    ["2 OEB FINDINGS", "The OEB approves a revenue requirement of $41.9 million.",
     "The OEB finds that the load forecast is reasonable and approves it as filed.",
     "The effective date of the new rates is January 1, 2027."],
]


@pytest.fixture
def decision_pdf() -> bytes:
    return make_pdf(DECISION_PAGES, header="Ontario Energy Board EB-2026-0015",
                    footer="Decision and Order page",
                    table=[["Account", "Balance"], ["1588", "$120,000"], ["1589", "$(45,000)"]])


class FakeSource:
    name = "oeb"

    def __init__(self, docs: dict[str, tuple[bytes, str, str]]):
        # source_id -> (bytes, title, extension)
        self.docs = docs
        self.fetch_calls = 0

    def list_records(self, since: date, until: date, max_docs=None):
        for i, (sid, (_, title, ext)) in enumerate(self.docs.items()):
            yield SourceRecord(source="oeb", source_id=sid, title=title, url=f"https://example.test/{sid}",
                               published_at=f"{until.isoformat()}T12:00:00Z",
                               registered_at=f"{until.isoformat()}T12:00:{i:02d}Z", extension=ext,
                               record_number=f"D26-{sid}", record_type="APPLICATION DOCUMENTS",
                               docket=None, size_hint=None,
                               extra={"landing_url": f"https://example.test/Record/{sid}"})

    def fetch(self, rec):
        self.fetch_calls += 1
        data, _, ext = self.docs[rec.source_id]
        return data, "application/pdf" if ext == "pdf" else "text/html"


class StubEmbedder:
    """Hashing bag-of-words embedder: deterministic, dependency-free, semantically crude."""
    dim = 384

    def __init__(self):
        self.hits = self.misses = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_documents(self, texts):
        self.misses += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = copy.deepcopy(load_config())
    c["lake_root"] = str(tmp_path / "lake")
    c["catalog_path"] = str(tmp_path / "catalog.duckdb")
    c["gold"]["store"] = "local"
    c["validation"]["min_bytes"] = 100
    monkeypatch.setattr("regrag.gold.index.get_embedder", lambda *a, **k: StubEmbedder())
    monkeypatch.setattr("regrag.serve.chain.get_embedder", lambda *a, **k: StubEmbedder())
    monkeypatch.setattr("regrag.serve.chain.has_credentials", lambda: False)
    return c


@pytest.fixture
def fake_source(monkeypatch, decision_pdf):
    other = make_pdf([["PROCEDURAL ORDER NO. 2", "EB-2025-0295", "Enbridge Gas Inc. DSM extension request.",
                       "Enbridge Gas Inc. asked the OEB to extend its 2023-2025 DSM framework by one year.",
                       "Intervenors shall file submissions on the extension request by October 10, 2026.",
                       "Enbridge Gas Inc. may file a reply submission by October 17, 2026."]])
    src = FakeSource({
        "1001": (decision_pdf, "Decision and Order_Northwind Hydro_EB-2026-0015", "pdf"),
        "1002": (other, "PO2_EGI DSM_EB-2025-0295", "pdf"),
        "1003": (decision_pdf, "Decision and Order_Northwind Hydro (re-posted)", "pdf"),  # identical bytes
        "1004": (b"<html><body>Service unavailable</body></html>" + b" " * 200, "Broken download", "pdf"),
    })
    monkeypatch.setattr("regrag.bronze.ingest.get_source", lambda name, cfg: src)
    return src
