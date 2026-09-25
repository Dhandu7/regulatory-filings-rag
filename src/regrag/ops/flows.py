"""Prefect orchestration: a daily incremental flow with one task per medallion stage.

    prefect server start                          # or use Prefect Cloud free tier
    python -m regrag.ops.flows serve              # registers the daily 06:00 America/Toronto schedule
    python -m regrag.ops.flows run                # one-off run now
"""
from __future__ import annotations

import sys
from datetime import date

from prefect import flow, get_run_logger, task

from ..bronze.ingest import ingest
from ..config import load_config
from ..gold.index import run_gold
from ..silver.run import rechunk, run_silver
from .dbt import run_dbt


@task(retries=2, retry_delay_seconds=300)
def bronze_task(source: str, since: str | None, max_docs: int | None) -> dict:
    res = ingest(load_config(), source, since=date.fromisoformat(since) if since else None, max_docs=max_docs)
    if res["validation"] == "failed":
        raise RuntimeError(f"bronze validation failed for {res['run_id']}; see bronze/validation/")
    return res


@task
def silver_task() -> dict:
    cfg = load_config()
    return {"silver": run_silver(cfg), "rechunk_backfill": rechunk(cfg)}


@task
def dbt_task() -> None:
    if run_dbt(load_config()) != 0:
        raise RuntimeError("dbt build / source freshness failed")


@task
def gold_task() -> dict:
    return run_gold(load_config())


@flow(name="regulatory-filings-daily", log_prints=True)
def daily_incremental(source: str = "oeb", since: str | None = None, max_docs: int | None = None) -> dict:
    log = get_run_logger()
    bronze = bronze_task(source, since, max_docs)
    silver = silver_task(wait_for=[bronze])
    dbt_done = dbt_task(wait_for=[silver])
    gold = gold_task(wait_for=[dbt_done])
    log.info("bronze=%s silver=%s gold=%s", bronze, silver, gold)
    return {"bronze": bronze, "silver": silver, "gold": gold}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        from prefect.schedules import Cron
        daily_incremental.serve(name="daily-oeb", schedules=[Cron("0 6 * * *", timezone="America/Toronto")],
                                parameters={"source": "oeb"}, tags=["regrag"])
    else:
        daily_incremental()
