"""Object-storage abstraction over fsspec (local disk by default, s3:// / gs:// in the cloud).

Tables in the lake are append-only Parquet "parts" under a prefix; readers glob the prefix.
"""
from __future__ import annotations

import hashlib
import io
import posixpath
import uuid
from collections.abc import Iterable

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Lake:
    def __init__(self, root: str):
        self.root = root.rstrip("/")
        self.fs, self._base = fsspec.core.url_to_fs(self.root)

    def path(self, *parts: str) -> str:
        return posixpath.join(self._base, *parts)

    def uri(self, *parts: str) -> str:
        """Fully qualified URI (what we persist in manifests)."""
        p = self.path(*parts)
        proto = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        return p if proto in ("file", "local") else f"{proto}://{p}"

    # -- blobs ---------------------------------------------------------------
    def put_bytes(self, rel: str, data: bytes) -> str:
        full = self.path(rel)
        self.fs.makedirs(posixpath.dirname(full), exist_ok=True)
        with self.fs.open(full, "wb") as fh:
            fh.write(data)
        return self.uri(rel)

    def get_bytes(self, uri_or_rel: str) -> bytes:
        """Accepts a URI persisted in a manifest (absolute) or a lake-relative path."""
        if "://" in uri_or_rel or uri_or_rel.startswith("/"):
            with fsspec.open(uri_or_rel, "rb") as fh:
                return fh.read()
        with self.fs.open(self.path(uri_or_rel), "rb") as fh:
            return fh.read()

    def read_fs_path(self, fs_path: str) -> bytes:
        """Read a path as returned by self.fs.glob (no protocol prefix)."""
        with self.fs.open(fs_path, "rb") as fh:
            return fh.read()

    def glob(self, pattern: str) -> list[str]:
        return sorted(self.fs.glob(self.path(pattern)))

    def exists(self, rel: str) -> bool:
        return self.fs.exists(self.path(rel))

    # -- tables --------------------------------------------------------------
    def append_rows(self, table: str, rows: list[dict], schema: pa.Schema | None = None,
                    part_name: str | None = None) -> str | None:
        if not rows:
            return None
        tbl = pa.Table.from_pylist(rows, schema=schema)
        rel = posixpath.join(table, f"{part_name or uuid.uuid4().hex}.parquet")
        buf = io.BytesIO()
        pq.write_table(tbl, buf, compression="zstd")
        return self.put_bytes(rel, buf.getvalue())

    def read_rows(self, table: str, columns: Iterable[str] | None = None) -> list[dict]:
        base = self.path(table)
        if not self.fs.exists(base):
            return []
        files = sorted(self.fs.glob(posixpath.join(base, "**", "*.parquet")))
        out: list[dict] = []
        for f in files:
            with self.fs.open(f, "rb") as fh:
                out.extend(pq.read_table(fh, columns=list(columns) if columns else None).to_pylist())
        return out

    def overwrite_json(self, rel: str, text: str) -> None:
        self.put_bytes(rel, text.encode())

    def read_text(self, rel: str) -> str | None:
        if not self.exists(rel):
            return None
        return self.get_bytes(rel).decode()
