from datetime import date

import pytest

from regrag.config import load_config
from regrag.serve.cache import cache_key
from regrag.serve.llm import parse_citations
from regrag.sources import OEBSource

WEBDRAWER_RESULT = {
    "RecordDateCreated": {"IsClear": False, "DateTime": "2026-09-14T16:35:05.0000000Z"},
    "RecordDateRegistered": {"IsClear": False, "DateTime": "2026-09-14T16:40:00.0000000Z"},
    "RecordDocumentSize": {"Value": 238904}, "RecordExtension": {"Value": "PDF"},
    "RecordNumber": {"Value": "D26-12300"}, "RecordMimeType": {"Value": "application/pdf"},
    "RecordRecordType": {"RecordTypeName": {"Value": "APPLICATION DOCUMENTS"}},
    "RecordTitle": {"Value": "Decision on Confidentiality_PO2_EB-2026-0123_20260914"},
    "TrimType": "Record", "Uri": 955586,
}


def _oeb():
    cfg = load_config()
    return OEBSource(cfg["sources"]["oeb"], cfg["http"])


def test_oeb_result_parsing():
    rec = _oeb().parse_result(WEBDRAWER_RESULT)
    assert rec.source_id == "955586" and rec.extension == "pdf" and rec.size_hint == 238904
    assert rec.published_at == "2026-09-14T16:35:05Z" and rec.docket == "EB-2026-0123"
    assert rec.url.endswith("/Record/955586/File/document")


def test_oeb_query_building():
    q = _oeb().build_query(date(2026, 9, 1), date(2026, 9, 24))
    assert q == "registeredOn:2026-09-01 to 2026-09-24 and (extension:pdf)"


def test_citation_parsing():
    assert parse_citations("Approved [1]. Also [2, 3] and [9].", n_sources=4) == [1, 2, 3]


def test_cache_key_scoped_by_index_version():
    a = cache_key("What was approved?", v="v1-aaa", k=6)
    assert a == cache_key("  what was APPROVED ", v="v1-aaa", k=6)
    assert a != cache_key("What was approved?", v="v2-bbb", k=6)


@pytest.mark.network
def test_live_oeb_search_returns_pdfs():
    recs = list(_oeb().list_records(date(2026, 9, 1), date(2026, 9, 24), max_docs=3))
    assert recs and all(r.extension == "pdf" for r in recs)
