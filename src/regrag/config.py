"""Config loading with interpolation:

  ${VAR} / ${VAR:-default}                 environment variable
  ${secret:scope/key} / ${secret:scope/key:-default}
                                           Databricks secret (read at runtime through the Databricks SDK,
                                           which is preinstalled on Databricks compute)

Relative paths in a config resolve against the repository that contains the config file
(the parent of its `configs/` directory), so the same code works from a source checkout and
from a wheel on Databricks whose configs were synced to the workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Bump whenever silver/chunk.py changes its output for an unchanged YAML config; it is folded into
# chunk_config_version so code changes can never silently alter an existing index version.
CHUNKER_ALGO_VERSION = 3
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
_SECRET_PATTERN = re.compile(r"\$\{secret:([\w.-]+)/([\w.-]+)(?::-([^}]*))?\}")


class SecretUnavailable(RuntimeError):
    pass


def read_databricks_secret(scope: str, key: str) -> str:
    """Read a secret from a Databricks secret scope (works in notebooks, wheel tasks, and Apps)."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise SecretUnavailable("databricks-sdk is not installed") from exc
    try:
        return WorkspaceClient().dbutils.secrets.get(scope=scope, key=key)
    except Exception as exc:  # noqa: BLE001 - auth/network/permission errors all mean "not available here"
        raise SecretUnavailable(f"could not read secret {scope}/{key}: {exc}") from exc


def _secret(m: re.Match) -> str:
    scope, key, default = m.group(1), m.group(2), m.group(3)
    try:
        return read_databricks_secret(scope, key)
    except SecretUnavailable:
        if default is not None:
            return default
        raise


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        value = _SECRET_PATTERN.sub(_secret, value)
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def resolve_path(p: str, root: Path = PROJECT_ROOT) -> str:
    """Resolve relative local paths against `root`; leave URLs and absolute paths alone."""
    if "://" in p or os.path.isabs(p):
        return p
    return str(root / p)


def _config_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from the project .env into os.environ.

    Real environment variables win (setdefault), and empty values are skipped so a blank
    `ANTHROPIC_API_KEY=` line can't mask credentials from elsewhere."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and value:
            os.environ.setdefault(key, value)


@cache
def load_config(path: str | None = None) -> dict:
    load_dotenv()
    path = path or os.environ.get("REGRAG_CONFIG", str(PROJECT_ROOT / "configs" / "pipeline.yaml"))
    root = _config_root(Path(path).resolve())
    with open(path) as fh:
        cfg = _interpolate(yaml.safe_load(fh))
    cfg["project_root"] = str(root)
    cfg["lake_root"] = resolve_path(cfg["lake_root"], root)
    cfg["catalog_path"] = resolve_path(cfg["catalog_path"], root)
    cfg["silver"]["chunking_config"] = resolve_path(cfg["silver"]["chunking_config"], root)
    return cfg


@dataclass(frozen=True)
class ChunkingConfig:
    name: str
    tokenizer: str
    max_tokens: int
    overlap_tokens: int
    min_tokens: int
    respect_sections: bool
    tables_as_separate_chunks: bool
    max_table_tokens: int
    prepend_context: bool
    version: str  # "<name>-<hash12>", derived, never hand-edited

    @classmethod
    def load(cls, path: str) -> ChunkingConfig:
        with open(resolve_path(path)) as fh:
            raw = yaml.safe_load(fh)
        canonical = json.dumps({**raw, "_algo": CHUNKER_ALGO_VERSION}, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        return cls(**raw, version=f"{raw['name']}-{digest}")
