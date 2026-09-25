"""End-to-end incremental run: bronze -> (validation gate) -> silver -> dbt tests -> gold.

Plain functions so Prefect, Databricks Workflows, the watcher, and the CLI all share one
code path. Each stage is idempotent, so a failed run can simply be re-run.
"""
from __future__ import annotations

import logging
from datetime import date

from ..bronze.ingest import ingest
from ..gold.index import run_gold
from ..silver.run import rechunk, run_silver
from .dbt import run_dbt

log = logging.getLogger(__name__)


class ValidationFailed(RuntimeError):
    pass


def run_pipeline(cfg: dict, source: str = "oeb", since: date | None = None, max_docs: int | None = None,
                 run_dbt_step: bool = True) -> dict:
    result: dict = {"status": "ok"}
    result["bronze"] = bronze = ingest(cfg, source, since=since, max_docs=max_docs)
    if bronze["validation"] == "failed":
        result["status"] = "bronze_validation_failed"
        return result          # gate: nothing downstream sees an unvalidated batch
    result["silver"] = silver = run_silver(cfg)
    # If the chunking config/algorithm changed since earlier runs, backfill older documents into
    # the current chunk version from stored pages (no-op when already complete).
    result["rechunk"] = backfill = rechunk(cfg)
    if run_dbt_step:
        rc = run_dbt(cfg)
        result["dbt"] = "passed" if rc == 0 else "failed"
        if rc != 0:
            result["status"] = "dbt_tests_failed"
            return result      # gate: don't publish an index built on data that failed tests
    if silver["chunks"] or backfill["chunks"] or bronze["landed"] == 0:
        result["gold"] = run_gold(cfg)
    return result
