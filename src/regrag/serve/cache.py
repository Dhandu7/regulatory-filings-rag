"""Answer cache: exact-match on a normalized question, scoped by everything that can change
the answer (index version, prompt version, model, k, filters). Changing any of them is a
miss by construction, so the cache never serves an answer from a stale index or prompt.

SQLite (WAL) rather than DuckDB so the API, eval runs and the CLI can share it across processes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time


def normalize_question(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\-\.]", "", q.lower())).strip()


def cache_key(question: str, **scope) -> str:
    payload = json.dumps({"q": normalize_question(question), **scope}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path: str, ttl_s: int = 86400):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, created REAL, body TEXT)")
        self.ttl = ttl_s
        self.lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT created, body FROM answers WHERE key = ?", (key,)).fetchone()
        if not row or time.time() - row[0] > self.ttl:
            return None
        return json.loads(row[1])

    def put(self, key: str, body: dict) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?, ?)",
                            (key, time.time(), json.dumps(body, default=str)))
