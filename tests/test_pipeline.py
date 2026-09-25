"""Bronze -> silver -> gold -> QA on synthetic filings, all offline."""
import json
from datetime import date

from fastapi.testclient import TestClient

from regrag.bronze.ingest import ingest, read_state
from regrag.gold.index import run_gold
from regrag.gold.store import get_store
from regrag.silver.run import run_silver
from regrag.storage import Lake

D = date(2026, 9, 20)


def test_bronze_lands_validates_and_is_idempotent(cfg, fake_source):
    res = ingest(cfg, "oeb", since=D, until=D)
    assert res["landed"] == 4 and res["fetch_errors"] == 0
    assert res["validation"] == "warn"                        # the HTML-as-PDF file trips a warn check
    lake = Lake(cfg["lake_root"])
    rows = lake.read_rows("bronze/manifest")
    dup = next(r for r in rows if r["source_id"] == "1003")
    assert dup["duplicate_of"] is not None                    # identical bytes stored once
    report = json.loads(lake.read_text(f"bronze/validation/{res['run_id']}.json"))
    assert report["quarantine_source_ids"] == ["oeb:1004"]
    assert read_state(lake, "oeb")["watermark_registered_at"].startswith("2026-09-20")

    calls = fake_source.fetch_calls
    again = ingest(cfg, "oeb", since=D, until=D)
    assert again["landed"] == 0 and again["skipped_existing"] == 4
    assert fake_source.fetch_calls == calls                   # nothing re-downloaded


def test_failed_validation_does_not_advance_watermark(cfg, fake_source, monkeypatch):
    monkeypatch.setattr("regrag.bronze.validate.check_hash_integrity",
                        lambda lake, rows: __import__("regrag.bronze.validate", fromlist=["x"]).CheckResult(
                            "object_hash_matches_manifest", "error", False, "simulated"))
    res = ingest(cfg, "oeb", since=D, until=D)
    assert res["validation"] == "failed"
    assert read_state(Lake(cfg["lake_root"]), "oeb") == {}


def test_silver_dedups_and_quarantines(cfg, fake_source):
    ingest(cfg, "oeb", since=D, until=D)
    s = run_silver(cfg)
    assert s["documents_ok"] == 2 and s["duplicates"] == 1 and s["quarantined"] == 1
    lake = Lake(cfg["lake_root"])
    q = lake.read_rows("silver/quarantine")
    assert q[0]["record_key"] == "oeb:1004" and q[0]["stage"] == "bronze_validation"
    docs = {d["source_id"]: d for d in lake.read_rows("silver/documents")}
    assert docs["1001"]["docket"] == "EB-2026-0015"           # docket recovered from text
    chunks = lake.read_rows(f"silver/chunks/chunk_config={s['chunk_config_version']}")
    assert {c["chunk_type"] for c in chunks} == {"text", "table"}
    assert all(c["chunk_config_version"] == s["chunk_config_version"] for c in chunks)
    assert run_silver(cfg)["considered"] == 0                  # incremental: nothing left to do


def test_gold_is_versioned_and_incremental(cfg, fake_source):
    ingest(cfg, "oeb", since=D, until=D)
    run_silver(cfg)
    g1 = run_gold(cfg)
    assert g1["loaded"] > 0 and g1["active"] == g1["index_version"]
    g2 = run_gold(cfg)
    assert g2["loaded"] == 0 and g2["already_indexed"] == g1["loaded"]
    assert get_store(cfg).active_version() == g1["index_version"]


def test_api_answers_with_citations_and_caches(cfg, fake_source, monkeypatch):
    ingest(cfg, "oeb", since=D, until=D)
    run_silver(cfg)
    run_gold(cfg)
    monkeypatch.setattr("regrag.serve.api.load_config", lambda: cfg)
    from regrag.serve import chain
    chain._QA.clear()
    client = TestClient(__import__("regrag.serve.api", fromlist=["app"]).app)

    r = client.post("/ask", json={"question": "What revenue requirement did the OEB approve for Northwind Hydro?"})
    assert r.status_code == 200
    body = r.json()
    assert body["cache"] == "miss" and body["model"] == "extractive"
    assert body["sources"][0]["docket"] == "EB-2026-0015"
    assert any(s["cited"] for s in body["sources"])
    assert "41.9" in body["answer"]
    assert client.post("/ask", json={"question": "what revenue requirement did the OEB approve for "
                                                 "Northwind Hydro"}).json()["cache"] == "hit"

    filtered = client.post("/ask", json={"question": "submissions deadline", "docket": "EB-2025-0295",
                                         "retrieval": "hybrid"}).json()
    assert filtered["retrieval"] == "hybrid"
    assert {s["docket"] for s in filtered["sources"]} == {"EB-2025-0295"}
    assert client.get("/health").json()["index_version"] == body["index_version"]


def test_rechunk_backfills_new_chunk_version(cfg, fake_source, tmp_path):
    import yaml
    ingest(cfg, "oeb", since=D, until=D)
    first = run_silver(cfg)
    raw = yaml.safe_load(open("configs/chunking/v1.yaml"))
    raw.update(name="v2", max_tokens=120, overlap_tokens=20)
    p = tmp_path / "v2.yaml"
    p.write_text(yaml.safe_dump(raw))
    from regrag.silver.run import rechunk
    res = rechunk(cfg, str(p))
    assert res["chunk_config_version"].startswith("v2-") and res["documents"] == first["documents_ok"]
    assert rechunk(cfg, str(p))["documents"] == 0            # idempotent backfill
    old = Lake(cfg["lake_root"]).read_rows(f"silver/chunks/chunk_config={first['chunk_config_version']}")
    assert old, "previous version is left untouched"


def test_dbt_passes_when_nothing_was_quarantined(cfg, monkeypatch, decision_pdf, tmp_path):
    """A fresh lake with no quarantined files must still give dbt a (zero-row) quarantine table."""
    from datetime import UTC, datetime

    from conftest import FakeSource

    from regrag.ops.dbt import run_dbt
    src = FakeSource({"2001": (decision_pdf, "Decision and Order_Northwind Hydro_EB-2026-0015", "pdf")})
    monkeypatch.setattr("regrag.bronze.ingest.get_source", lambda name, cfg: src)
    today = datetime.now(UTC).date()
    ingest(cfg, "oeb", since=today, until=today)
    assert run_silver(cfg)["quarantined"] == 0
    assert Lake(cfg["lake_root"]).glob("silver/quarantine/*.parquet")        # zero-row schema part
    monkeypatch.setenv("REGRAG_LAKE_ROOT", cfg["lake_root"])
    monkeypatch.setenv("REGRAG_DBT_DUCKDB", str(tmp_path / "dbt.duckdb"))
    assert run_dbt(cfg) == 0
