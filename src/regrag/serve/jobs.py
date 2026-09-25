"""Background jobs for the demo console: run allowlisted `regrag` CLI commands as subprocesses.

Only commands in COMMANDS can run, and every parameter is validated and passed as a
separate argv element (no shell), so the web page can't be used to run arbitrary code.
One job runs at a time: the pipeline stages share single-writer files (the DuckDB
embedding cache, lake parts), and a demo doesn't need concurrent pipeline runs.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from ..config import PROJECT_ROOT

JOBS_DIR = PROJECT_ROOT / "data" / "jobs"
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class JobError(ValueError):
    """Invalid command or parameters (HTTP 400)."""


class JobBusy(RuntimeError):
    """Another job is still running (HTTP 409)."""


def _date(v) -> str | None:
    if v in (None, ""):
        return None
    try:
        return date.fromisoformat(str(v)).isoformat()
    except ValueError as exc:
        raise JobError(f"not an ISO date: {v!r}") from exc


def _int(v, lo: int, hi: int, name: str) -> int | None:
    if v in (None, ""):
        return None
    try:
        n = int(v)
    except (TypeError, ValueError) as exc:
        raise JobError(f"{name} must be an integer") from exc
    if not lo <= n <= hi:
        raise JobError(f"{name} must be between {lo} and {hi}")
    return n


def _choice(v, options: set[str], name: str, default: str | None = None) -> str | None:
    if v in (None, ""):
        return default
    if v not in options:
        raise JobError(f"{name} must be one of {sorted(options)}")
    return v


def chunking_configs() -> list[str]:
    return sorted(str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "configs" / "chunking").glob("*.yaml"))


def _chunk_cfg(v) -> str | None:
    if v in (None, ""):
        return None
    if v not in chunking_configs():
        raise JobError(f"unknown chunking config {v!r}")
    return v


def build_argv(command: str, p: dict) -> list[str]:
    """Translate a (command, params) request into regrag CLI arguments."""
    def opt(flag: str, value) -> list[str]:
        return [] if value is None else [flag, str(value)]

    if command == "ingest":
        return (["ingest", "--source", _choice(p.get("source"), {"oeb", "edgar"}, "source", "oeb")]
                + opt("--since", _date(p.get("since"))) + opt("--until", _date(p.get("until")))
                + opt("--max-docs", _int(p.get("max_docs"), 1, 5000, "max_docs")))
    if command == "validation":
        return ["validation"]
    if command == "silver":
        return ["silver"] + opt("--limit", _int(p.get("limit"), 1, 100000, "limit"))
    if command == "rechunk":
        cfg = _chunk_cfg(p.get("chunking_config"))
        return ["rechunk"] + ([cfg] if cfg else [])
    if command == "dbt":
        return ["dbt"]
    if command == "gold":
        return (["gold"] + opt("--chunking-config", _chunk_cfg(p.get("chunking_config")))
                + (["--no-activate"] if p.get("no_activate") else []))
    if command == "run":
        return (["run", "--source", _choice(p.get("source"), {"oeb", "edgar"}, "source", "oeb")]
                + opt("--since", _date(p.get("since")))
                + opt("--max-docs", _int(p.get("max_docs"), 1, 5000, "max_docs"))
                + (["--skip-dbt"] if p.get("skip_dbt") else []))
    if command == "watch":
        return ["watch", "--once"]            # one poll; a long-running loop doesn't belong in a web job
    if command == "eval":
        return (["eval"] + opt("-k", _int(p.get("k"), 1, 20, "k"))
                + opt("--retrieval", _choice(p.get("retrieval"), {"dense", "hybrid", "rerank"}, "retrieval"))
                + (["--with-llm"] if p.get("with_llm") else []))
    raise JobError(f"unknown command {command!r}")


COMMANDS = ["run", "ingest", "validation", "silver", "rechunk", "dbt", "gold", "watch", "eval"]


@dataclass
class Job:
    id: str
    command: str
    argv: list[str]
    started: float
    log_path: Path
    status: str = "running"            # running | succeeded | failed | cancelled
    exit_code: int | None = None
    finished: float | None = None
    proc: subprocess.Popen | None = field(default=None, repr=False)

    def public(self) -> dict:
        return {"id": self.id, "command": self.command, "args": self.argv, "status": self.status,
                "exit_code": self.exit_code, "started": self.started, "finished": self.finished,
                "duration_s": round((self.finished or time.time()) - self.started, 1)}


class JobRunner:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def running(self) -> Job | None:
        return next((j for j in self.jobs.values() if j.status == "running"), None)

    def start(self, command: str, params: dict) -> Job:
        argv = build_argv(command, params or {})
        with self.lock:
            if (busy := self.running()) is not None:
                raise JobBusy(f"job {busy.id} ({busy.command}) is still running")
            JOBS_DIR.mkdir(parents=True, exist_ok=True)
            job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            job = Job(job_id, command, argv, time.time(), JOBS_DIR / f"{job_id}.log")
            env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTHONUNBUFFERED": "1",
                   "NO_COLOR": "1"}
            log = open(job.log_path, "wb")
            log.write(f"$ regrag {' '.join(argv)}\n".encode())
            log.flush()
            job.proc = subprocess.Popen([sys.executable, "-m", "regrag.cli", *argv], cwd=PROJECT_ROOT,
                                        stdout=log, stderr=subprocess.STDOUT, env=env)
            self.jobs[job_id] = job
        threading.Thread(target=self._wait, args=(job, log), daemon=True).start()
        return job

    def _wait(self, job: Job, log) -> None:
        code = job.proc.wait()
        log.close()
        job.exit_code = code
        job.finished = time.time()
        if job.status == "running":
            job.status = "succeeded" if code == 0 else "failed"

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status == "running" and job.proc:
            job.status = "cancelled"
            job.proc.terminate()
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def read_log(self, job_id: str, offset: int = 0, limit: int = 200_000) -> tuple[str, int]:
        job = self.get(job_id)
        with open(job.log_path, "rb") as fh:
            fh.seek(offset)
            data = fh.read(limit)
        return _ANSI.sub("", data.decode("utf-8", errors="replace")), offset + len(data)

    def recent(self, n: int = 20) -> list[dict]:
        return [j.public() for j in sorted(self.jobs.values(), key=lambda j: -j.started)[:n]]


RUNNER = JobRunner()
