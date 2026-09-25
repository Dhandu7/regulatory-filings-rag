"""Demo console: job allowlist/validation and the read endpoints."""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from regrag.bronze.ingest import ingest
from regrag.gold.index import run_gold
from regrag.serve.jobs import JobError, build_argv
from regrag.silver.run import run_silver


def test_build_argv_maps_params_to_cli_flags():
    assert build_argv("ingest", {"source": "oeb", "since": "2026-09-01", "max_docs": "25"}) == \
        ["ingest", "--source", "oeb", "--since", "2026-09-01", "--max-docs", "25"]
    assert build_argv("run", {"skip_dbt": True}) == ["run", "--source", "oeb", "--skip-dbt"]
    assert build_argv("watch", {}) == ["watch", "--once"]
    assert build_argv("eval", {"retrieval": "hybrid", "k": 4}) == ["eval", "-k", "4", "--retrieval", "hybrid"]


@pytest.mark.parametrize("command,params", [
    ("rm", {}),                                          # not an allowlisted command
    ("ingest", {"source": "oeb; rm -rf /"}),              # injection attempt in a choice field
    ("ingest", {"since": "yesterday"}),                  # not an ISO date
    ("ingest", {"max_docs": 999999}),                    # out of range
    ("rechunk", {"chunking_config": "../../etc/passwd"}),  # not a known config file
])
def test_build_argv_rejects_bad_input(command, params):
    with pytest.raises(JobError):
        build_argv(command, params)


def test_console_endpoints(cfg, fake_source, monkeypatch):
    d = date(2026, 9, 20)
    ingest(cfg, "oeb", since=d, until=d)
    run_silver(cfg)
    run_gold(cfg)
    import regrag.serve.api as api
    import regrag.serve.console as console
    from regrag.serve import chain
    monkeypatch.setattr(api, "load_config", lambda: cfg)
    monkeypatch.setattr(console, "load_config", lambda: cfg)
    chain._QA.clear()
    client = TestClient(api.app)

    assert "Regulatory Filings RAG" in client.get("/").text
    s = client.get("/api/status").json()
    assert s["counts"]["documents_ok"] == 2 and s["counts"]["quarantined"] == 1   # the HTML-as-PDF fixture
    assert s["counts"]["chunks"] > 0 and s["active_index"]
    assert client.get("/api/validation").json()["status"] == "warn"
    assert [d["source_id"] for d in client.get("/api/documents?q=northwind").json()][:1] in (["1001"], ["1003"])
    assert client.get("/api/versions").json()[0]["is_active"]
    assert client.post("/api/jobs", json={"command": "nope"}).status_code == 400
    assert client.post("/api/activate", json={"index_version": "missing"}).status_code == 404


def test_jobs_can_be_disabled_for_databricks_apps(monkeypatch):
    import regrag.serve.api as api
    monkeypatch.setenv("REGRAG_CONSOLE_JOBS", "off")
    client = TestClient(api.app)
    r = client.post("/api/jobs", json={"command": "validation"})
    assert r.status_code == 403 and "Databricks job" in r.json()["detail"]
