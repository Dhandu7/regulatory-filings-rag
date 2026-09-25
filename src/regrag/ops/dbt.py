"""Run dbt (build + source freshness) against the lake, programmatically."""
from __future__ import annotations

import json
import logging
import os

from ..config import PROJECT_ROOT, ChunkingConfig

log = logging.getLogger(__name__)
DBT_DIR = PROJECT_ROOT / "dbt"


def run_dbt(cfg: dict, args: list[str] | None = None) -> int:
    """Returns a process-style exit code: 0 ok, 1 test/model failures."""
    from dbt.cli.main import dbtRunner

    os.environ.setdefault("REGRAG_LAKE_ROOT", cfg["lake_root"])
    os.environ.setdefault("REGRAG_DBT_DUCKDB", os.path.join(os.path.dirname(cfg["catalog_path"]), "dbt.duckdb"))
    common = ["--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)]
    version = ChunkingConfig.load(cfg["silver"]["chunking_config"]).version
    runner = dbtRunner()
    commands = [args] if args else [["build", "--vars", json.dumps({"chunk_config_version": version})],
                                    ["source", "freshness"]]
    rc = 0
    for cmd in commands:
        res = runner.invoke(cmd + common)
        if not res.success:
            log.error("dbt %s failed: %s", " ".join(cmd), res.exception or "see dbt output")
            rc = 1
    return rc
