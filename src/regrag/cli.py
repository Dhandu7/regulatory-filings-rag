"""regrag command line: one subcommand per pipeline stage, plus `run` for the whole thing."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date


def _date(s: str) -> date:
    return date.fromisoformat(s)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="regrag")
    p.add_argument("--config", help="pipeline yaml (default configs/pipeline.yaml or $REGRAG_CONFIG)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="bronze: pull + land + validate")
    ing.add_argument("--source", default="oeb", choices=["oeb", "edgar"])
    ing.add_argument("--since", type=_date)
    ing.add_argument("--until", type=_date)
    ing.add_argument("--max-docs", type=int)

    sub.add_parser("validation", help="print latest bronze validation report")

    sil = sub.add_parser("silver", help="parse, dedup, quarantine, chunk")
    sil.add_argument("--limit", type=int)
    rc = sub.add_parser("rechunk", help="build/backfill a chunk version from stored pages")
    rc.add_argument("chunking_config", nargs="?", help="default: silver.chunking_config")

    gold = sub.add_parser("gold", help="embed chunks into the vector index")
    gold.add_argument("--chunking-config")
    gold.add_argument("--no-activate", action="store_true")

    act = sub.add_parser("activate", help="switch the served index version")
    act.add_argument("index_version")
    sub.add_parser("versions", help="list index versions")

    dbt = sub.add_parser("dbt", help="run dbt build (models + schema/freshness tests) over the lake")
    dbt.add_argument("args", nargs="*", help="raw dbt args; default runs build (current chunk version) + freshness")

    ask = sub.add_parser("ask", help="ask a question from the terminal")
    ask.add_argument("question")
    ask.add_argument("-k", type=int)
    ask.add_argument("--docket")

    srv = sub.add_parser("serve", help="run the QA API")
    # Databricks Apps assigns the port via DATABRICKS_APP_PORT and needs 0.0.0.0 (REGRAG_HOST).
    srv.add_argument("--host", default=os.environ.get("REGRAG_HOST", "127.0.0.1"))
    srv.add_argument("--port", type=int,
                     default=int(os.environ.get("DATABRICKS_APP_PORT") or os.environ.get("PORT") or 8000))

    ev = sub.add_parser("eval", help="retrieval hit-rate + latency on the golden set")
    ev.add_argument("--questions", default="eval/questions.jsonl")
    ev.add_argument("-k", type=int, default=6)
    ev.add_argument("--with-llm", action="store_true", help="also generate answers (needs Claude credentials)")
    ev.add_argument("--retrieval", choices=["dense", "hybrid", "rerank"], help="default: config serve.retrieval")
    ev.add_argument("--expand", type=int, help="neighbor chunks per hit (default: config serve.expand_window)")
    ev.add_argument("--judge", action="store_true", help="also grade answers with a Claude judge (needs --with-llm)")

    run = sub.add_parser("run", help="incremental end-to-end: ingest -> silver -> dbt -> gold")
    run.add_argument("--source", default="oeb", choices=["oeb", "edgar"])
    run.add_argument("--since", type=_date)
    run.add_argument("--max-docs", type=int)
    run.add_argument("--skip-dbt", action="store_true")

    w = sub.add_parser("watch", help="poll the regulator for new filings and process them as they land")
    w.add_argument("--interval", type=int, default=900, help="seconds between polls")
    w.add_argument("--once", action="store_true")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .config import load_config
    cfg = load_config(a.config)

    def out(obj) -> None:
        print(json.dumps(obj, indent=2, default=str))

    if a.cmd == "ingest":
        from .bronze.ingest import ingest
        res = ingest(cfg, a.source, a.since, a.until, a.max_docs)
        out(res)
        return 1 if res["validation"] == "failed" else 0
    if a.cmd == "validation":
        from .bronze.validate import latest_report
        from .storage import Lake
        out(latest_report(Lake(cfg["lake_root"])))
        return 0
    if a.cmd == "silver":
        from .silver.run import run_silver
        out(run_silver(cfg, limit=a.limit))
        return 0
    if a.cmd == "rechunk":
        from .silver.run import rechunk
        out(rechunk(cfg, a.chunking_config))
        return 0
    if a.cmd == "gold":
        from .gold.index import run_gold
        out(run_gold(cfg, a.chunking_config, activate=not a.no_activate))
        return 0
    if a.cmd in ("activate", "versions"):
        from .gold.store import get_store
        store = get_store(cfg)
        if a.cmd == "activate":
            store.activate(a.index_version)
        out(store.versions())
        return 0
    if a.cmd == "dbt":
        from .ops.dbt import run_dbt
        return run_dbt(cfg, a.args or None)
    if a.cmd == "ask":
        from .serve.chain import get_qa
        out(get_qa(cfg).ask(a.question, k=a.k, filters={"docket": a.docket} if a.docket else None))
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("regrag.serve.api:app", host=a.host, port=a.port)
        return 0
    if a.cmd == "eval":
        from .evaluation.run_eval import run_eval
        try:
            res = run_eval(cfg, a.questions, k=a.k, with_llm=a.with_llm, retrieval=a.retrieval,
                           expand_window=a.expand, judge=a.judge)
        except RuntimeError as exc:
            print(f"eval failed: {exc}", file=sys.stderr)
            return 1
        out(res["summary"])
        if res["summary"]["n_errors"]:
            print(f"eval: {res['summary']['n_errors']} question(s) failed; first error: "
                  f"{res['summary']['errors'][0]['error']}", file=sys.stderr)
            return 1
        return 0
    if a.cmd == "run":
        from .ops.pipeline import run_pipeline
        res = run_pipeline(cfg, a.source, since=a.since, max_docs=a.max_docs, run_dbt_step=not a.skip_dbt)
        out(res)
        return 0 if res["status"] == "ok" else 1
    if a.cmd == "watch":
        from .ops.watcher import watch
        watch(cfg, interval_s=a.interval, once=a.once)
        return 0
    return 2


def run() -> None:
    """Entry point for Databricks python_wheel_task: the job runner ignores return values, so
    exit explicitly and a failed validation gate fails the task (and stops downstream tasks)."""
    sys.exit(main())


if __name__ == "__main__":
    run()
