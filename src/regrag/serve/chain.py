"""LangChain QA chain: retriever (pgvector) -> Claude answer with numbered citations.

   question ─┬─> RegulatoryRetriever ──> format sources ─┐
             └──────────────────────────────────────────┴─> generate (Claude | extractive) ─> answer + citations

Retrieval modes (serve.retrieval):
  dense   HNSW cosine top-k
  hybrid  dense + full-text on exact tokens (dockets, amounts, acronyms), RRF-fused
  rerank  dense + rare-term full-text candidates, reordered by a cross-encoder, top-k kept
Each kept chunk can be widened with its neighboring chunks (serve.expand_window).

Without Claude credentials the chain degrades to an extractive answer (best-matching
sentences from the top sources, still cited) so retrieval can be exercised offline.
"""
from __future__ import annotations

import logging
import os
import re
import time

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import ConfigDict

from ..gold.embed import get_embedder
from ..gold.store import _terms, get_store
from .cache import AnswerCache, cache_key
from .llm import ClaudeAnswerer, has_credentials
from .prompts import PROMPT_VERSION, format_source
from .rerank import expand_with_neighbors, get_reranker

RETRIEVAL_MODES = ("dense", "hybrid", "rerank")

log = logging.getLogger(__name__)


class RegulatoryRetriever(BaseRetriever):
    """LangChain retriever over the versioned hybrid index."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: object
    embedder: object
    index_version: str
    k: int = 6
    mode: str = "dense"
    filters: dict | None = None
    reranker: object | None = None
    candidates: int = 40
    expand_window: int = 0

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        qvec = self.embedder.embed_query(query)
        if self.mode == "rerank":
            hits = self.store.search(self.index_version, qvec, query, k=self.candidates, filters=self.filters,
                                     lexical="rare")
            if hits:
                texts = [f"{h.meta.get('context') or ''}\n{h.meta['text']}" for h in hits]
                for h, score in zip(hits, self.reranker.scores(query, texts)):
                    h.score = score
                hits = sorted(hits, key=lambda h: -h.score)[:self.k]
        else:
            hits = self.store.search(self.index_version, qvec, query, k=self.k, filters=self.filters,
                                     hybrid=self.mode == "hybrid")
        metas = expand_with_neighbors(self.store, self.index_version,
                                      [{**h.meta, "score": h.score} for h in hits], self.expand_window)
        return [Document(page_content=m["context_text"], metadata=m) for m in metas]


def extractive_answer(question: str, docs: list[Document], max_sentences: int = 3) -> tuple[str, list[int]]:
    q = set(_terms(question))
    scored = []
    for i, d in enumerate(docs[:4], start=1):
        for sent in re.split(r"(?<=[.;:])\s+|\n(?=[A-Z0-9(])", d.page_content):
            s = " ".join(sent.split())
            if len(s) < 25:
                continue
            overlap = len(q & set(_terms(s)))
            if overlap:
                scored.append((overlap / (1 + 0.1 * i), i, s))
    if not scored:
        return "The retrieved filings do not appear to answer this question.", []
    best = sorted(scored, key=lambda x: -x[0])[:max_sentences]
    text = " ".join(f"{s} [{i}]" for _, i, s in best)
    return f"(Extractive answer - no LLM configured.) {text}", sorted({i for _, i, _ in best})


class RegulatoryQA:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        s, g = cfg["serve"], cfg["gold"]
        self.store = get_store(cfg)
        # No on-disk doc-embedding cache here: it's a single-writer DuckDB file owned by the gold job,
        # and serving only embeds queries (cached in memory by the embedder).
        self.embedder = get_embedder(g["embed_model"], None, g["embed_batch_size"])
        self.default_k = s["top_k"]
        self.default_retrieval = s.get("retrieval", "dense")
        if self.default_retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"serve.retrieval must be one of {RETRIEVAL_MODES}")
        self.default_hybrid = self.default_retrieval == "hybrid"   # kept for older callers
        rr = s.get("rerank", {})
        self.rerank_model = rr.get("model", "Xenova/ms-marco-MiniLM-L-12-v2")
        self.rerank_candidates = int(rr.get("candidates", 40))
        self.default_expand = int(s.get("expand_window", 0))
        self.model = s["model"]
        self.llm = ClaudeAnswerer(s["model"], s.get("effort", "medium")) if has_credentials() else None
        self.cache = AnswerCache(os.path.join(os.path.dirname(cfg["catalog_path"]), "cache", "answers.sqlite"),
                                 s.get("answer_cache_ttl_s", 86400))

    def index_version(self) -> str:
        v = self.store.active_version()
        if not v:
            raise RuntimeError("no active index version; run `regrag gold` first")
        return v

    def retriever(self, k: int, filters: dict | None, mode: str = "dense", index_version: str | None = None,
                  expand_window: int = 0) -> RegulatoryRetriever:
        return RegulatoryRetriever(
            store=self.store, embedder=self.embedder, k=k, filters=filters, mode=mode,
            index_version=index_version or self.index_version(), expand_window=expand_window,
            candidates=self.rerank_candidates,
            reranker=get_reranker(self.rerank_model) if mode == "rerank" else None)

    def build_chain(self, retriever: RegulatoryRetriever, use_llm: bool):
        def generate(inp: dict) -> dict:
            docs: list[Document] = inp["docs"]
            t0 = time.perf_counter()
            llm_error = None
            if use_llm and self.llm and docs:
                sources = "\n\n".join(format_source(i, d.metadata) for i, d in enumerate(docs, 1))
                try:
                    gen = self.llm.generate(inp["question"], sources, len(docs))
                except Exception as exc:  # noqa: BLE001 - API down / out of credits: degrade, don't 500
                    log.warning("Claude call failed, falling back to extractive answer: %s", exc)
                    llm_error = getattr(exc, "message", None) or str(exc)
                else:
                    answer, cited, usage, model = gen.text, gen.cited, gen.usage, gen.model
            if not (use_llm and self.llm and docs) or llm_error:
                answer, cited = extractive_answer(inp["question"], docs)
                usage, model = {}, "extractive"
                if llm_error:
                    answer = f"(Claude unavailable: {llm_error[:160]}) " + answer.removeprefix(
                        "(Extractive answer - no LLM configured.) ")
            return {**inp, "answer": answer, "cited": cited, "usage": usage, "model": model,
                    "llm_error": llm_error, "generation_ms": (time.perf_counter() - t0) * 1000}

        def timed_retrieve(q: str) -> dict:
            t0 = time.perf_counter()
            docs = retriever.invoke(q)
            return {"docs": docs, "retrieval_ms": (time.perf_counter() - t0) * 1000}

        return (RunnablePassthrough.assign(r=RunnableLambda(lambda x: timed_retrieve(x["question"])))
                | RunnableLambda(lambda x: {"question": x["question"], **x["r"]})
                | RunnableLambda(generate))

    def ask(self, question: str, k: int | None = None, filters: dict | None = None,
            use_llm: bool = True, retrieval: str | None = None, use_cache: bool = True,
            expand_window: int | None = None) -> dict:
        t0 = time.perf_counter()
        k = k or self.default_k
        mode = retrieval or self.default_retrieval
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval must be one of {RETRIEVAL_MODES}")
        expand = self.default_expand if expand_window is None else expand_window
        filters = {kk: v for kk, v in (filters or {}).items() if v} or None
        version = self.index_version()
        llm_on = bool(use_llm and self.llm)
        key = cache_key(question, v=version, p=PROMPT_VERSION, m=self.model if llm_on else "extractive",
                        k=k, f=filters, r=mode, x=expand,
                        rr=[self.rerank_model, self.rerank_candidates] if mode == "rerank" else None)
        if use_cache and (hit := self.cache.get(key)):
            hit["cache"] = "hit"
            hit["latency_ms"] = {"total": (time.perf_counter() - t0) * 1000}
            return hit
        res = self.build_chain(self.retriever(k, filters, mode, version, expand), llm_on).invoke({"question": question})
        sources = [{
            "n": i, "title": d.metadata.get("title"), "docket": d.metadata.get("docket"),
            "date": (d.metadata.get("published_at") or "")[:10], "section": d.metadata.get("section"),
            "pages": [d.metadata.get("page_start"), d.metadata.get("page_end")], "url": d.metadata.get("url"),
            "chunk_id": d.metadata.get("chunk_id"), "source_id": d.metadata.get("source_id"),
            "score": round(d.metadata.get("score", 0.0), 5), "cited": i in res["cited"],
            "snippet": d.metadata["text"][:300],
            "context": d.page_content,          # exactly what the model saw for this source
        } for i, d in enumerate(res["docs"], 1)]
        body = {"question": question, "answer": res["answer"], "sources": sources, "model": res["model"],
                "retrieval": mode, "expand_window": expand, "index_version": version,
                "prompt_version": PROMPT_VERSION, "usage": res["usage"],
                "latency_ms": {"retrieval": res["retrieval_ms"], "generation": res["generation_ms"],
                               "total": (time.perf_counter() - t0) * 1000}}
        if res.get("llm_error"):
            body["llm_error"] = res["llm_error"]
        elif use_cache:                        # never cache a degraded answer
            self.cache.put(key, body)
        body["cache"] = "miss"
        return body


_QA: dict[int, RegulatoryQA] = {}


def get_qa(cfg: dict) -> RegulatoryQA:
    """One QA instance per config object (load_config is itself cached)."""
    if id(cfg) not in _QA:
        _QA[id(cfg)] = RegulatoryQA(cfg)
    return _QA[id(cfg)]
