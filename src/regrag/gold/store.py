"""Vector stores. Both implement the same interface:

  ensure_version(version_meta) / existing_ids(version) / upsert(version, rows, vectors)
  activate(version) / active_version() / search(version, query_vec, query_text, k, filters)

`PgVectorStore` searches HNSW cosine ANN and, optionally, a Postgres full-text leg fused with
reciprocal-rank fusion. Two lexical modes:
  exact  docket numbers, amounts, acronyms (the original hybrid mode)
  rare   any question term that appears in few chunks (e.g. a company name like "Stelco");
         used to widen the candidate pool that the cross-encoder reranker then orders
`neighbors()` returns the chunks around a hit so answers that straddle a chunk boundary are
still in the context the model sees.

`LocalStore` is a dependency-free numpy fallback for tests/CI with the same fusion logic.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

RRF_K = 60
CANDIDATES = 50
META_COLS = ["chunk_id", "doc_id", "source", "source_id", "docket", "title", "section", "page_start",
             "page_end", "chunk_type", "published_at", "url", "text", "context", "ordinal"]
# A term is "rare" if it appears in at most this share of chunks (with a floor for small indexes).
RARE_DF_SHARE = 0.005
RARE_DF_FLOOR = 25


@dataclass
class Hit:
    chunk_id: str
    score: float
    meta: dict


def _terms(text: str) -> list[str]:
    stop = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "is", "was", "what", "which", "did",
            "does", "how", "by", "with", "that", "this", "be", "are", "were", "its", "it", "as", "at", "from",
            "who", "when", "where", "why", "much", "many", "oeb", "board"}
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower()) if t not in stop]


_EXACT = re.compile(r"\b(?:[A-Z]{2,}-?\d[\w\-]*|[A-Z]{2,}\d*|\d[\d,\.\-/]*(?:\d|[A-Za-z]{1,3})|\d)\b")


def exact_terms(text: str) -> list[str]:
    """Tokens that dense embeddings blur but a lexical index nails: docket numbers (EB-2026-0015),
    account/section numbers, amounts, years, and acronyms (IESO, OESP, CCIM). Plain words are
    left to the dense leg; OR-ing every word of a question matched much of the corpus and hurt
    ranking in the eval."""
    stop = {"OEB", "THE"}
    out = []
    for m in _EXACT.findall(text):
        tok = m.strip(".,").lower()
        if m.upper() in stop or not tok:
            continue
        out.append(tok)
    return sorted(set(out))


def rare_threshold(n_chunks: int) -> int:
    return max(RARE_DF_FLOOR, int(n_chunks * RARE_DF_SHARE))


def rrf(*rankings: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return scores


# =========================================================================== pgvector
class PgVectorStore:
    def __init__(self, database_url: str, dim: int, schema: str | None = None):
        import psycopg
        from pgvector.psycopg import register_vector
        self.dim = dim
        self.conn = psycopg.connect(database_url, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if schema:
            self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            self.conn.execute(f'SET search_path TO "{schema}", public')
        register_vector(self.conn)
        self._migrate()

    def _migrate(self) -> None:
        self.conn.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_index_versions (
            index_version text PRIMARY KEY,
            chunk_config_version text NOT NULL,
            embed_model text NOT NULL,
            embed_dim int NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            activated_at timestamptz,
            is_active boolean NOT NULL DEFAULT false,
            n_chunks int NOT NULL DEFAULT 0,
            config_json jsonb
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_index ON rag_index_versions (is_active) WHERE is_active;
        CREATE TABLE IF NOT EXISTS rag_chunks (
            index_version text NOT NULL REFERENCES rag_index_versions(index_version) ON DELETE CASCADE,
            chunk_id text NOT NULL,
            doc_id text NOT NULL,
            source text, source_id text, docket text, title text, section text,
            page_start int, page_end int, chunk_type text, published_at timestamptz, url text,
            text text NOT NULL, context text,
            embedding vector({self.dim}) NOT NULL,
            tsv tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(context, '')), 'A') ||
                setweight(to_tsvector('english', text), 'B')) STORED,
            loaded_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (index_version, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS rag_chunks_hnsw ON rag_chunks USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS rag_chunks_tsv ON rag_chunks USING gin (tsv);
        CREATE INDEX IF NOT EXISTS rag_chunks_docket ON rag_chunks (index_version, docket);
        ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS ordinal int;
        CREATE INDEX IF NOT EXISTS rag_chunks_doc_ordinal ON rag_chunks (index_version, doc_id, ordinal);
        """)
        row = self.conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'").fetchone()
        if row and row[0] != f"vector({self.dim})":
            raise RuntimeError(f"rag_chunks.embedding is {row[0]} but the embed model has dim {self.dim}; "
                               "use a separate schema/database for a different embedding dimension")

    def ensure_version(self, meta: dict) -> None:
        self.conn.execute(
            "INSERT INTO rag_index_versions (index_version, chunk_config_version, embed_model, embed_dim, "
            "config_json) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (index_version) DO NOTHING",
            (meta["index_version"], meta["chunk_config_version"], meta["embed_model"], self.dim,
             json.dumps(meta.get("config", {}))))

    def existing_ids(self, version: str) -> set[str]:
        rows = self.conn.execute("SELECT chunk_id FROM rag_chunks WHERE index_version=%s", (version,)).fetchall()
        return {r[0] for r in rows}

    def upsert(self, version: str, rows: list[dict], vectors: np.ndarray) -> int:
        cols = ["index_version"] + [c for c in META_COLS] + ["embedding"]
        sql = (f"INSERT INTO rag_chunks ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
               "ON CONFLICT (index_version, chunk_id) DO NOTHING")
        with self.conn.cursor() as cur:
            cur.executemany(sql, [[version] + [r.get(c) for c in META_COLS] + [v] for r, v in zip(rows, vectors)])
        self.conn.execute("UPDATE rag_index_versions SET n_chunks = (SELECT count(*) FROM rag_chunks "
                          "WHERE index_version=%s) WHERE index_version=%s", (version, version))
        return len(rows)

    def backfill_ordinals(self, version: str, ordinals: dict[str, int]) -> int:
        """Fill `ordinal` for rows loaded before the column existed (no re-embedding needed)."""
        missing = {r[0] for r in self.conn.execute(
            "SELECT chunk_id FROM rag_chunks WHERE index_version=%s AND ordinal IS NULL", (version,))}
        pairs = [(ordinals[c], version, c) for c in missing if c in ordinals]
        if pairs:
            with self.conn.cursor() as cur:
                cur.executemany("UPDATE rag_chunks SET ordinal=%s WHERE index_version=%s AND chunk_id=%s", pairs)
        return len(pairs)

    def neighbors(self, version: str, keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        """Text chunks at the given (doc_id, ordinal) positions."""
        if not keys:
            return {}
        rows = self.conn.execute(
            "SELECT doc_id, ordinal, text, page_start, page_end FROM rag_chunks "
            "WHERE index_version=%s AND chunk_type='text' AND (doc_id, ordinal) IN "
            "(SELECT * FROM unnest(%s::text[], %s::int[]))",
            (version, [k[0] for k in keys], [k[1] for k in keys])).fetchall()
        return {(d, o): {"text": t, "page_start": ps, "page_end": pe} for d, o, t, ps, pe in rows}

    def _rare_terms(self, version: str, terms: list[str]) -> list[str]:
        if not terms:
            return []
        n = self.conn.execute("SELECT n_chunks FROM rag_index_versions WHERE index_version=%s",
                              (version,)).fetchone()
        limit = rare_threshold(n[0] if n else 0)
        rows = self.conn.execute(
            "SELECT t, (SELECT count(*) FROM rag_chunks WHERE index_version=%s "
            "           AND tsv @@ plainto_tsquery('english', t)) "
            "FROM unnest(%s::text[]) t WHERE plainto_tsquery('english', t)::text <> ''",
            (version, terms)).fetchall()
        return [t for t, df in rows if 0 < df <= limit]

    def activate(self, version: str) -> None:
        with self.conn.transaction():
            self.conn.execute("UPDATE rag_index_versions SET is_active=false WHERE is_active")
            self.conn.execute("UPDATE rag_index_versions SET is_active=true, activated_at=now() "
                              "WHERE index_version=%s", (version,))

    def active_version(self) -> str | None:
        row = self.conn.execute("SELECT index_version FROM rag_index_versions WHERE is_active").fetchone()
        return row[0] if row else None

    def versions(self) -> list[dict]:
        cur = self.conn.execute("SELECT index_version, chunk_config_version, embed_model, n_chunks, is_active, "
                                "created_at FROM rag_index_versions ORDER BY created_at")
        names = [d.name for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def search(self, version: str, qvec: np.ndarray, qtext: str, k: int = 6,
               filters: dict | None = None, hybrid: bool = True, lexical: str | None = None) -> list[Hit]:
        """`lexical`: None (dense only), "exact", or "rare"; defaults to "exact" when hybrid."""
        lexical = lexical if lexical is not None else ("exact" if hybrid else None)
        where, params = ["index_version = %(v)s"], {"v": version, "q": qvec, "n": max(CANDIDATES, k)}
        filters = filters or {}
        if filters.get("docket"):
            where.append("docket = %(docket)s")
            params["docket"] = filters["docket"]
        if filters.get("source"):
            where.append("source = %(source)s")
            params["source"] = filters["source"]
        if filters.get("date_from"):
            where.append("published_at >= %(date_from)s")
            params["date_from"] = filters["date_from"]
        if filters.get("date_to"):
            where.append("published_at <= %(date_to)s")
            params["date_to"] = filters["date_to"]
        w = " AND ".join(where)
        self.conn.execute("SET hnsw.ef_search = 100")
        self.conn.execute("SET hnsw.iterative_scan = relaxed_order")
        dense = [r[0] for r in self.conn.execute(
            f"SELECT chunk_id FROM rag_chunks WHERE {w} ORDER BY embedding <=> %(q)s LIMIT %(n)s", params)]
        lex_ids: list[str] = []
        terms = (exact_terms(qtext) if lexical == "exact"
                 else self._rare_terms(version, sorted(set(_terms(qtext)))) if lexical == "rare" else [])
        if terms:
            # plainto_tsquery per term keeps Postgres's own tokenization of "eb-2026-0015", "$1.5"
            params["terms"] = terms
            lex_ids = [r[0] for r in self.conn.execute(
                f"WITH q AS (SELECT string_agg(plainto_tsquery('english', t)::text, ' | ')::tsquery AS query "
                f"           FROM unnest(%(terms)s::text[]) t WHERE plainto_tsquery('english', t)::text <> '') "
                f"SELECT chunk_id FROM rag_chunks, q "
                f"WHERE {w} AND tsv @@ q.query ORDER BY ts_rank_cd(tsv, q.query) DESC LIMIT %(n)s", params)]
        fused = rrf(dense, lex_ids) if lexical else {c: 1.0 / (RRF_K + i) for i, c in enumerate(dense, 1)}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        if not top:
            return []
        cur = self.conn.execute(f"SELECT {', '.join(META_COLS)} FROM rag_chunks WHERE index_version=%s "
                                "AND chunk_id = ANY(%s)", (version, top))
        metas = {r[0]: dict(zip(META_COLS, r)) for r in cur.fetchall()}
        for m in metas.values():
            if isinstance(m.get("published_at"), datetime):
                m["published_at"] = m["published_at"].isoformat()
        return [Hit(c, fused[c], metas[c]) for c in top if c in metas]


# =========================================================================== local
class LocalStore:
    """Numpy brute-force cosine + BM25, persisted as .npz + .jsonl per index version."""

    def __init__(self, root: str, dim: int):
        self.root = root
        self.dim = dim
        os.makedirs(root, exist_ok=True)
        self._cache: dict[str, tuple[np.ndarray, list[dict]]] = {}

    def _paths(self, v: str) -> tuple[str, str]:
        return os.path.join(self.root, f"{v}.npz"), os.path.join(self.root, f"{v}.jsonl")

    def _load(self, v: str) -> tuple[np.ndarray, list[dict]]:
        if v not in self._cache:
            vp, mp = self._paths(v)
            if not os.path.exists(vp):
                self._cache[v] = (np.zeros((0, self.dim), np.float32), [])
            else:
                with open(mp) as fh:
                    metas = [json.loads(l) for l in fh]
                self._cache[v] = (np.load(vp)["v"], metas)
        return self._cache[v]

    def ensure_version(self, meta: dict) -> None:
        reg = self._registry()
        reg.setdefault(meta["index_version"], {**meta, "created_at": datetime.now(UTC).isoformat(),
                                               "is_active": False})
        self._save_registry(reg)

    def _registry(self) -> dict:
        p = os.path.join(self.root, "versions.json")
        return json.load(open(p)) if os.path.exists(p) else {}

    def _save_registry(self, reg: dict) -> None:
        with open(os.path.join(self.root, "versions.json"), "w") as fh:
            json.dump(reg, fh, indent=2, default=str)

    def existing_ids(self, version: str) -> set[str]:
        return {m["chunk_id"] for m in self._load(version)[1]}

    def upsert(self, version: str, rows: list[dict], vectors: np.ndarray) -> int:
        vecs, metas = self._load(version)
        have = {m["chunk_id"] for m in metas}
        keep = [i for i, r in enumerate(rows) if r["chunk_id"] not in have]
        new_metas = metas + [{c: rows[i].get(c) for c in META_COLS} for i in keep]
        new_vecs = np.vstack([vecs, vectors[keep]]) if keep else vecs
        vp, mp = self._paths(version)
        np.savez(vp, v=new_vecs)
        with open(mp, "w") as fh:
            fh.writelines(json.dumps(m, default=str) + "\n" for m in new_metas)
        self._cache[version] = (new_vecs, new_metas)
        reg = self._registry()
        if version in reg:
            reg[version]["n_chunks"] = len(new_metas)
            self._save_registry(reg)
        return len(keep)

    def backfill_ordinals(self, version: str, ordinals: dict[str, int]) -> int:
        vecs, metas = self._load(version)
        n = 0
        for m in metas:
            if m.get("ordinal") is None and m["chunk_id"] in ordinals:
                m["ordinal"] = ordinals[m["chunk_id"]]
                n += 1
        if n:
            with open(self._paths(version)[1], "w") as fh:
                fh.writelines(json.dumps(m, default=str) + "\n" for m in metas)
        return n

    def neighbors(self, version: str, keys: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
        want = set(keys)
        return {(m["doc_id"], m["ordinal"]): {"text": m["text"], "page_start": m["page_start"],
                                              "page_end": m["page_end"]}
                for m in self._load(version)[1]
                if m.get("chunk_type") == "text" and (m["doc_id"], m.get("ordinal")) in want}

    def activate(self, version: str) -> None:
        reg = self._registry()
        for k in reg:
            reg[k]["is_active"] = k == version
        self._save_registry(reg)

    def active_version(self) -> str | None:
        return next((k for k, v in self._registry().items() if v.get("is_active")), None)

    def versions(self) -> list[dict]:
        return [{"index_version": k, **v} for k, v in self._registry().items()]

    def search(self, version: str, qvec: np.ndarray, qtext: str, k: int = 6,
               filters: dict | None = None, hybrid: bool = True, lexical: str | None = None) -> list[Hit]:
        lexical = lexical if lexical is not None else ("exact" if hybrid else None)
        vecs, metas = self._load(version)
        if not metas:
            return []
        filters = filters or {}
        mask = np.ones(len(metas), bool)
        for i, m in enumerate(metas):
            if filters.get("docket") and m.get("docket") != filters["docket"]:
                mask[i] = False
            if filters.get("source") and m.get("source") != filters["source"]:
                mask[i] = False
            if filters.get("date_from") and (m.get("published_at") or "") < filters["date_from"]:
                mask[i] = False
            if filters.get("date_to") and (m.get("published_at") or "") > filters["date_to"]:
                mask[i] = False
        idx = np.flatnonzero(mask)
        sims = vecs[idx] @ qvec / (np.linalg.norm(vecs[idx], axis=1) * np.linalg.norm(qvec) + 1e-9)
        dense = [metas[idx[j]]["chunk_id"] for j in np.argsort(-sims)[:max(CANDIDATES, k)]]
        pool = [metas[i] for i in idx]
        if lexical == "exact":
            terms = exact_terms(qtext)
        elif lexical == "rare":
            limit = rare_threshold(len(metas))
            df = Counter(t for m in pool for t in set(_terms((m.get("context") or "") + " " + m["text"])))
            terms = [t for t in set(_terms(qtext)) if 0 < df[t] <= limit]
        else:
            terms = []
        lex_ids = self._bm25(pool, " ".join(terms)) if terms else []
        fused = rrf(dense, lex_ids)
        by_id = {m["chunk_id"]: m for m in metas}
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(c, fused[c], by_id[c]) for c in top]

    @staticmethod
    def _bm25(metas: list[dict], qtext: str, k1: float = 1.2, b: float = 0.75) -> list[str]:
        q = set(_terms(qtext))
        if not q:
            return []
        docs = [Counter(_terms((m.get("context") or "") + " " + m["text"])) for m in metas]
        avgdl = sum(sum(d.values()) for d in docs) / max(len(docs), 1)
        df = Counter(t for d in docs for t in q if t in d)
        n = len(docs)
        scores = []
        for m, d in zip(metas, docs):
            dl = sum(d.values())
            s = sum(math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * d[t] * (k1 + 1)
                    / (d[t] + k1 * (1 - b + b * dl / avgdl)) for t in q if t in d)
            if s > 0:
                scores.append((s, m["chunk_id"]))
        return [c for _, c in sorted(scores, reverse=True)[:CANDIDATES]]


def get_store(cfg: dict):
    g = cfg["gold"]
    if g["store"] == "pgvector":
        return PgVectorStore(g["database_url"], g["embed_dim"], g.get("schema"))
    if g["store"] == "local":
        return LocalStore(os.path.join(os.path.dirname(cfg["catalog_path"]), "vector_local"), g["embed_dim"])
    raise ValueError(f"unknown vector store {g['store']!r}")
