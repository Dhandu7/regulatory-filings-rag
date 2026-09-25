"""Read/operate endpoints for the demo console (served at /). Mounted under /api."""
from __future__ import annotations

import glob
import json
import os
from typing import Any

import duckdb
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..bronze.validate import latest_report
from ..config import PROJECT_ROOT, ChunkingConfig, load_config
from ..storage import Lake
from .chain import get_qa
from .jobs import COMMANDS, RUNNER, JobBusy, JobError, chunking_configs

router = APIRouter(prefix="/api", tags=["console"])


def jobs_enabled() -> bool:
    """False when deployed as a Databricks App: there the pipeline runs as the Databricks job."""
    return os.environ.get("REGRAG_CONSOLE_JOBS", "on").lower() not in ("off", "0", "false")


def _lake_sql(sql: str, params: list | None = None) -> list[dict]:
    """Query lake Parquet tables with DuckDB. `{lake}` in the SQL is replaced by the lake root."""
    lake_root = load_config()["lake_root"]
    con = duckdb.connect()
    try:
        cur = con.execute(sql.replace("{lake}", lake_root), params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    except duckdb.IOException:
        return []           # table not written yet (fresh install)
    finally:
        con.close()


def _one(sql: str) -> Any:
    rows = _lake_sql(sql)
    return next(iter(rows[0].values())) if rows else 0


@router.get("/status")
def status() -> dict:
    cfg = load_config()
    qa = get_qa(cfg)
    lake = Lake(cfg["lake_root"])
    version = ChunkingConfig.load(cfg["silver"]["chunking_config"]).version
    docs = _lake_sql("select status, count(*) n from read_parquet('{lake}/silver/documents/*.parquet') group by 1")
    by_status = {r["status"]: r["n"] for r in docs}
    state = {}
    for p in lake.glob("_state/*.json"):
        state[os.path.basename(p)[:-5]] = json.loads(lake.read_fs_path(p))
    report = latest_report(lake)
    return {
        "llm": qa.model if qa.llm else None,
        "retrieval_default": qa.default_retrieval,
        "active_index": qa.store.active_version(),
        "chunk_config_version": version,
        "counts": {
            "bronze_records": _one("select count(*) from read_parquet('{lake}/bronze/manifest/*.parquet')"),
            "documents_ok": by_status.get("ok", 0),
            "duplicates": by_status.get("duplicate", 0),
            "quarantined": _one("select count(*) from read_parquet('{lake}/silver/quarantine/*.parquet')"),
            "chunks": _one(f"select count(distinct chunk_id) from "
                           f"read_parquet('{{lake}}/silver/chunks/chunk_config={version}/*.parquet')"),
        },
        "watermarks": state,
        "latest_validation": report and {k: report[k] for k in ("run_id", "status", "listed", "landed",
                                                                "fetch_errors", "validated_at")},
        "running_job": (j := RUNNER.running()) and j.public(),
        "jobs_enabled": jobs_enabled(),
    }


@router.get("/versions")
def versions() -> list[dict]:
    return get_qa(load_config()).store.versions()


class ActivateRequest(BaseModel):
    index_version: str


@router.post("/activate")
def activate(req: ActivateRequest) -> list[dict]:
    store = get_qa(load_config()).store
    if req.index_version not in {v["index_version"] for v in store.versions()}:
        raise HTTPException(404, f"unknown index version {req.index_version}")
    store.activate(req.index_version)
    return store.versions()


@router.get("/validation")
def validation() -> dict | None:
    return latest_report(Lake(load_config()["lake_root"]))


@router.get("/quarantine")
def quarantine() -> list[dict]:
    return _lake_sql("""
        select q.source_id, q.stage, q.reason, q.quarantined_at, m.title, m.url
        from read_parquet('{lake}/silver/quarantine/*.parquet') q
        left join read_parquet('{lake}/bronze/manifest/*.parquet', union_by_name=true) m
          on m.source = q.source and m.source_id = q.source_id
        order by q.quarantined_at desc""")


@router.get("/documents")
def documents(q: str = "", limit: int = 100) -> list[dict]:
    limit = max(1, min(limit, 500))
    like = f"%{q.lower()}%"
    return _lake_sql(f"""
        select source_id, title, docket, record_type, substr(published_at, 1, 10) as date, n_pages,
               n_tables, status, landing_url as url
        from read_parquet('{{lake}}/silver/documents/*.parquet')
        where lower(title) like ? or lower(coalesce(docket, '')) like ?
        order by published_at desc limit {limit}""", [like, like])


@router.get("/eval")
def eval_results() -> dict:
    out = {}
    for tag in ("dense", "hybrid", "rerank", "dense_holdout", "hybrid_holdout", "rerank_holdout"):
        files = sorted(glob.glob(str(PROJECT_ROOT / "eval" / "results" / f"*_{tag}.json")))
        if files:
            with open(files[-1]) as fh:
                data = json.load(fh)
            data["file"] = os.path.basename(files[-1])
            out[tag] = data
    questions = {}
    for qpath in sorted((PROJECT_ROOT / "eval").glob("questions*.jsonl")):
        for line in qpath.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                questions[row["id"]] = row
    out["questions"] = questions
    return out


@router.get("/chunking-configs")
def list_chunking_configs() -> list[str]:
    return chunking_configs()


class JobRequest(BaseModel):
    command: str
    params: dict = Field(default_factory=dict)


@router.get("/commands")
def commands() -> list[str]:
    return COMMANDS


@router.post("/jobs")
def start_job(req: JobRequest) -> dict:
    if not jobs_enabled():
        raise HTTPException(403, "Pipeline commands are disabled here; run the regrag-daily-incremental "
                                 "Databricks job instead.")
    try:
        return RUNNER.start(req.command, req.params).public()
    except JobError as exc:
        raise HTTPException(400, str(exc)) from exc
    except JobBusy as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/jobs")
def list_jobs() -> list[dict]:
    return RUNNER.recent()


@router.get("/jobs/{job_id}")
def job(job_id: str, offset: int = 0) -> dict:
    try:
        j = RUNNER.get(job_id)
        text, nxt = RUNNER.read_log(job_id, max(0, offset))
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc
    return {**j.public(), "log": text, "offset": nxt}


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str) -> dict:
    try:
        return RUNNER.cancel(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc
