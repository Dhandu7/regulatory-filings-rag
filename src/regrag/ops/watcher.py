"""Near-real-time watcher: poll the regulator for newly registered filings and push each
small batch through the same pipeline.

This is micro-batch "streaming": OEB exposes no push feed, so the event source is a
watermark poll. The daily batch flow and the watcher share the watermark in
_state/<source>.json, and every stage is idempotent, so running both is safe: whichever
sees a record first processes it and the other skips it.
"""
from __future__ import annotations

import logging
import signal
import time
from datetime import UTC, datetime

from .pipeline import run_pipeline

log = logging.getLogger(__name__)


def watch(cfg: dict, source: str = "oeb", interval_s: int = 900, once: bool = False) -> None:
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    while not stop["flag"]:
        started = datetime.now(UTC)
        try:
            # dbt runs in the daily flow; per-poll we keep latency low and rely on bronze validation.
            res = run_pipeline(cfg, source, run_dbt_step=False)
            landed = res["bronze"]["landed"]
            if landed:
                log.info("watcher: %d new filing(s) indexed -> %s", landed, res.get("gold", {}).get("index_version"))
            else:
                log.info("watcher: no new filings")
        except Exception:
            log.exception("watcher poll failed; will retry next interval")
        if once:
            return
        elapsed = (datetime.now(UTC) - started).total_seconds()
        time.sleep(max(5, interval_s - elapsed))
