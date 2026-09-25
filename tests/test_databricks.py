"""Databricks deployment pieces that can be checked without a workspace."""
import shutil
import sys
import types

import pytest
import yaml

import regrag.config as config
from regrag.config import PROJECT_ROOT, SecretUnavailable, load_config


@pytest.fixture(autouse=True)
def _fresh_config_cache():
    load_config.cache_clear()
    yield
    load_config.cache_clear()


def test_secret_interpolation(monkeypatch):
    monkeypatch.setattr(config, "read_databricks_secret", lambda scope, key: f"{scope}:{key}")
    assert config._interpolate("${secret:regrag/database-url}") == "regrag:database-url"


def test_secret_default_used_only_when_unavailable(monkeypatch):
    def unavailable(scope, key):
        raise SecretUnavailable("no workspace")
    monkeypatch.setattr(config, "read_databricks_secret", unavailable)
    assert config._interpolate("${secret:regrag/user-agent:-fallback agent}") == "fallback agent"
    with pytest.raises(SecretUnavailable):
        config._interpolate("${secret:regrag/database-url}")


def test_databricks_config_resolves_paths_and_secrets(monkeypatch):
    monkeypatch.setattr(config, "read_databricks_secret", lambda scope, key: f"<{key}>")
    cfg = load_config(str(PROJECT_ROOT / "configs" / "pipeline.databricks.yaml"))
    assert cfg["lake_root"].startswith("/Volumes/")
    assert cfg["gold"]["database_url"] == "<database-url>"
    assert cfg["gold"]["chunks_table"].endswith(".regrag.silver_chunks")
    assert cfg["silver"]["chunking_config"] == str(PROJECT_ROOT / "configs" / "chunking" / "v1.yaml")


def test_relative_paths_resolve_against_the_configs_repo(tmp_path):
    """On Databricks the configs live in workspace files, not next to the installed wheel."""
    shutil.copytree(PROJECT_ROOT / "configs", tmp_path / "configs")
    cfg = load_config(str(tmp_path / "configs" / "pipeline.yaml"))
    assert cfg["silver"]["chunking_config"] == str(tmp_path / "configs" / "chunking" / "v1.yaml")
    assert cfg["project_root"] == str(tmp_path)


def test_gold_reads_chunks_from_delta_table(monkeypatch):
    seen = {}

    class Row(dict):
        def asDict(self):
            return dict(self)

    class DF:
        def where(self, cond):
            seen["where"] = cond
            return self

        def collect(self):
            return [Row(chunk_id="c1", text="t")]

    class Builder:
        def getOrCreate(self):
            return types.SimpleNamespace(table=lambda name: seen.update(table=name) or DF())

    fake = types.ModuleType("pyspark.sql")
    fake.SparkSession = types.SimpleNamespace(builder=Builder())
    monkeypatch.setitem(sys.modules, "pyspark", types.ModuleType("pyspark"))
    monkeypatch.setitem(sys.modules, "pyspark.sql", fake)

    from regrag.gold.index import load_chunks
    rows = load_chunks({"gold": {"chunks_table": "workspace.regrag.silver_chunks"}, "lake_root": "/nope"}, "v1-abc")
    assert rows == [{"chunk_id": "c1", "text": "t"}]
    assert seen == {"table": "workspace.regrag.silver_chunks", "where": "chunk_config_version = 'v1-abc'"}


def test_job_entry_point_fails_the_task_on_failed_validation(monkeypatch):
    from regrag import cli
    monkeypatch.setattr("regrag.bronze.ingest.ingest", lambda *a, **k: {"validation": "failed"})
    monkeypatch.setattr(sys, "argv", ["regrag-job", "ingest"])
    with pytest.raises(SystemExit) as exc:
        cli.run()
    assert exc.value.code == 1


def test_bundle_references_exist():
    bundle = yaml.safe_load((PROJECT_ROOT / "databricks.yml").read_text())
    tasks = {t["task_key"]: t for t in bundle["resources"]["jobs"]["regrag_daily"]["tasks"]}
    assert (PROJECT_ROOT / tasks["silver_spark"]["spark_python_task"]["python_file"]).exists()
    assert (PROJECT_ROOT / tasks["dbt_tests"]["dbt_task"]["project_directory"] / "dbt_project.yml").exists()
    for key in ("bronze_ingest", "gold_index"):
        wheel = tasks[key]["python_wheel_task"]
        assert wheel["entry_point"] == "regrag-job"
        config_arg = wheel["parameters"][wheel["parameters"].index("--config") + 1]
        assert (PROJECT_ROOT / config_arg.replace("${workspace.file_path}/", "")).exists()
    assert 'regrag-job = "regrag.cli:run"' in (PROJECT_ROOT / "pyproject.toml").read_text()
    app = yaml.safe_load((PROJECT_ROOT / "app.yaml").read_text())
    secrets = {r["name"] for r in bundle["resources"]["apps"]["regrag_console"]["resources"]}
    assert {e["valueFrom"] for e in app["env"] if "valueFrom" in e} <= secrets
