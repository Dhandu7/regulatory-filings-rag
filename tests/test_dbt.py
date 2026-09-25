"""dbt schema/relationship/freshness tests run against a lake built from synthetic filings."""
from datetime import UTC, datetime

from regrag.bronze.ingest import ingest
from regrag.ops.dbt import run_dbt
from regrag.silver.run import run_silver


def test_dbt_build_and_freshness_pass_on_fixture_lake(cfg, fake_source, monkeypatch, tmp_path):
    today = datetime.now(UTC).date()
    ingest(cfg, "oeb", since=today, until=today)
    run_silver(cfg)
    monkeypatch.setenv("REGRAG_LAKE_ROOT", cfg["lake_root"])
    monkeypatch.setenv("REGRAG_DBT_DUCKDB", str(tmp_path / "dbt.duckdb"))
    assert run_dbt(cfg) == 0
