"""Integration test for the pgvector store. Skips unless DATABASE_URL points at a Postgres
with the vector extension available (CI service container, or `make pg-up` locally)."""
import os
import uuid

import numpy as np
import pytest

psycopg = pytest.importorskip("psycopg")


@pytest.fixture
def store():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    from regrag.gold.store import PgVectorStore
    schema = f"test_{uuid.uuid4().hex[:8]}"   # isolated: never touches the real 384-d tables
    try:
        s = PgVectorStore(url, dim=4, schema=schema)
    except psycopg.OperationalError:
        pytest.skip("postgres not reachable")
    except psycopg.errors.FeatureNotSupported:
        pytest.skip("pgvector not installed")
    yield s
    s.conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


def test_hybrid_search_versioning_and_filters(store):
    v = "v1-000000000000__stub"
    store.ensure_version({"index_version": v, "chunk_config_version": "v1-000000000000", "embed_model": "stub"})
    rows = [
        {"chunk_id": "a", "doc_id": "d1", "source": "oeb", "source_id": "1", "docket": "EB-2026-0015",
         "title": "Decision", "text": "The OEB approves a revenue requirement of $41.9 million.",
         "context": "Decision | EB-2026-0015", "page_start": 1, "page_end": 1, "chunk_type": "text",
         "published_at": "2026-09-01T00:00:00Z"},
        {"chunk_id": "b", "doc_id": "d2", "source": "oeb", "source_id": "2", "docket": "EB-2025-0295",
         "title": "PO2", "text": "Intervenors shall file submissions by October 10, 2026.",
         "context": "PO2 | EB-2025-0295", "page_start": 1, "page_end": 1, "chunk_type": "text",
         "published_at": "2026-09-02T00:00:00Z"},
    ]
    vecs = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    assert store.upsert(v, rows, vecs) == 2
    store.upsert(v, rows, vecs)                                   # idempotent
    assert store.existing_ids(v) == {"a", "b"}
    store.activate(v)
    assert store.active_version() == v

    hits = store.search(v, np.array([0, 1, 0, 0], np.float32), "revenue requirement", k=2)
    assert {h.chunk_id for h in hits} == {"a", "b"}             # dense favours b, lexical favours a
    only = store.search(v, np.array([1, 0, 0, 0], np.float32), "submissions", k=5,
                        filters={"docket": "EB-2025-0295"})
    assert [h.chunk_id for h in only] == ["b"]


def test_ordinals_neighbors_and_rare_terms(store):
    v = "v1-000000000001__stub"
    store.ensure_version({"index_version": v, "chunk_config_version": "v1-000000000001", "embed_model": "stub"})
    base = {"source": "oeb", "docket": None, "title": "t", "context": "", "chunk_type": "text",
            "published_at": "2026-09-01T00:00:00Z"}
    rows = [{**base, "chunk_id": f"c{i}", "doc_id": f"d{i}", "source_id": str(i), "page_start": 1, "page_end": 1,
             "text": "generation licence decision granted"} for i in range(30)]
    rows += [{**base, "chunk_id": "s0", "doc_id": "ds", "source_id": "s", "page_start": 1, "page_end": 1,
              "text": "Stelco Inc. owns a generation facility"},
             {**base, "chunk_id": "s1", "doc_id": "ds", "source_id": "s", "page_start": 2, "page_end": 2,
              "text": "located in Nanticoke, Ontario"}]
    vecs = np.array([[1, 0, 0, 0]] * 30 + [[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    store.upsert(v, rows, vecs)                                   # loaded without ordinals, like old rows
    n = store.backfill_ordinals(v, {"s0": 0, "s1": 1, **{f"c{i}": 0 for i in range(30)}})
    assert n == 32 and store.backfill_ordinals(v, {"s0": 0}) == 0
    assert store.neighbors(v, [("ds", 1)])[("ds", 1)]["text"] == "located in Nanticoke, Ontario"
    # "generation" and "licence" appear in 31 of 32 chunks; only "stelco" is rare enough to query on.
    assert store._rare_terms(v, ["stelco", "generation", "licence"]) == ["stelco"]
    top = [h.chunk_id for h in store.search(v, np.array([1, 0, 0, 0], np.float32),
                                            "Where is Stelco's generation facility?", k=5, lexical="rare")]
    assert "s0" in top
