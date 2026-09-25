"""Post-pull bronze validation (the Hydro One pattern: every pull is checked before
anything downstream trusts it).

Each check returns a CheckResult with a severity. Any failed `error` check fails the
run: the watermark does not advance and the orchestrator stops before silver.
`warn` checks are recorded but don't block. Per-record failures are listed so silver
can quarantine exactly those records.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from ..storage import Lake, sha256_bytes

REQUIRED_FIELDS = ["source", "source_id", "title", "url", "published_at", "object_uri", "sha256", "fetched_at"]
MAGIC = {"pdf": b"%PDF-", "htm": b"<", "html": b"<"}


def rkey(r: dict) -> str:
    """Record key used for quarantine lists: '<source>:<source_id>'."""
    return f"{r['source']}:{r['source_id']}"


@dataclass
class CheckResult:
    name: str
    severity: str        # error | warn
    passed: bool
    detail: str = ""
    failed_records: list[str] = field(default_factory=list)


def check_required_fields(rows: list[dict]) -> CheckResult:
    bad = [rkey(r) for r in rows if any(r.get(f) in (None, "") for f in REQUIRED_FIELDS)]
    return CheckResult("required_fields_not_null", "error", not bad,
                       f"{len(bad)} rows missing one of {REQUIRED_FIELDS}", bad)


def check_unique_ids(rows: list[dict]) -> CheckResult:
    seen, dupes = set(), []
    for r in rows:
        key = (r["source"], r["source_id"])
        if key in seen:
            dupes.append(rkey(r))
        seen.add(key)
    return CheckResult("unique_source_id", "error", not dupes, f"{len(dupes)} duplicate ids in batch", dupes)


def check_hash_integrity(lake: Lake, rows: list[dict]) -> CheckResult:
    """Re-read every landed object and confirm the stored bytes hash to the manifest sha256."""
    bad = []
    for r in rows:
        try:
            if sha256_bytes(lake.get_bytes(r["object_uri"])) != r["sha256"]:
                bad.append(rkey(r))
        except FileNotFoundError:
            bad.append(rkey(r))
    return CheckResult("object_hash_matches_manifest", "error", not bad, f"{len(bad)} objects failed", bad)


def check_file_signature(lake: Lake, rows: list[dict]) -> CheckResult:
    """Extension must agree with magic bytes (catches HTML error pages saved as .pdf)."""
    bad = []
    for r in rows:
        magic = MAGIC.get(r["extension"])
        if magic is None:
            continue
        head = lake.get_bytes(r["object_uri"])[:1024].lstrip()
        if not head.startswith(magic):
            bad.append(rkey(r))
    return CheckResult("file_signature_matches_extension", "warn", not bad,
                       f"{len(bad)} files whose bytes don't match their extension (quarantined in silver)", bad)


def check_size_bounds(rows: list[dict], min_bytes: int, max_bytes: int) -> CheckResult:
    bad = [rkey(r) for r in rows if not (min_bytes <= r["bytes"] <= max_bytes)]
    return CheckResult("size_within_bounds", "warn", not bad,
                       f"{len(bad)} files outside [{min_bytes}, {max_bytes}] bytes", bad)


def check_size_hint(rows: list[dict]) -> CheckResult:
    """Regulator-reported size vs downloaded size: detects truncated downloads."""
    bad = [rkey(r) for r in rows if r.get("size_hint") and r["size_hint"] != r["bytes"]]
    return CheckResult("download_size_matches_source", "warn", not bad, f"{len(bad)} size mismatches", bad)


def check_fetch_error_rate(n_ok: int, n_err: int, max_rate: float = 0.2) -> CheckResult:
    total = n_ok + n_err
    rate = n_err / total if total else 0.0
    return CheckResult("fetch_error_rate", "error", rate <= max_rate, f"{n_err}/{total} fetches failed ({rate:.0%})")


def check_freshness(rows: list[dict], warn_days: int) -> CheckResult:
    if not rows:
        return CheckResult("freshness", "warn", True, "no new rows this run")
    newest = max(r["published_at"] for r in rows)
    age = datetime.now(UTC) - datetime.fromisoformat(newest.replace("Z", "+00:00"))
    return CheckResult("freshness", "warn", age <= timedelta(days=warn_days),
                       f"newest published_at={newest} ({age.days}d old)")


def check_volume(lake: Lake, n_rows: int, listed: int, drop_ratio: float) -> CheckResult:
    """Compare to trailing runs; a sudden drop usually means the source API changed."""
    history = [json.loads(lake.read_fs_path(p)) for p in lake.glob("bronze/validation/*.json")[-10:]]
    counts = [h["listed"] for h in history if h.get("listed")]
    if len(counts) < 3:
        return CheckResult("volume_vs_trailing", "warn", True, f"listed={listed}; not enough history")
    avg = statistics.mean(counts)
    return CheckResult("volume_vs_trailing", "warn", listed >= avg * drop_ratio,
                       f"listed={listed} vs trailing avg {avg:.1f}")


def validate_run(cfg: dict, lake: Lake, run_id: str, rows: list[dict], errors: list[dict],
                 listed: int) -> dict:
    v = cfg["validation"]
    checks = [
        check_required_fields(rows),
        check_unique_ids(rows),
        check_hash_integrity(lake, rows),
        check_file_signature(lake, rows),
        check_size_bounds(rows, v["min_bytes"], v["max_bytes"]),
        check_size_hint(rows),
        check_fetch_error_rate(len(rows), len(errors)),
        check_freshness(rows, v["freshness_warn_days"]),
        check_volume(lake, len(rows), listed, v["volume_drop_warn_ratio"]),
    ]
    failed_errors = [c for c in checks if not c.passed and c.severity == "error"]
    failed_warns = [c for c in checks if not c.passed and c.severity == "warn"]
    status = "failed" if failed_errors else ("warn" if failed_warns else "passed")
    quarantine = sorted({sid for c in checks if not c.passed for sid in c.failed_records})
    report = {
        "run_id": run_id, "status": status, "listed": listed, "landed": len(rows),
        "fetch_errors": len(errors), "validated_at": datetime.now(UTC).isoformat(),
        "quarantine_source_ids": quarantine, "checks": [asdict(c) for c in checks],
    }
    lake.overwrite_json(f"bronze/validation/{run_id}.json", json.dumps(report, indent=2))
    return report


def load_quarantine_ids(lake: Lake) -> set[str]:
    ids: set[str] = set()
    for p in lake.glob("bronze/validation/*.json"):
        ids.update(json.loads(lake.read_fs_path(p)).get("quarantine_source_ids", []))
    return ids


def latest_report(lake: Lake) -> dict | None:
    paths = lake.glob("bronze/validation/*.json")
    return json.loads(lake.read_fs_path(paths[-1])) if paths else None
