"""Cross-encoder reranking and neighbor expansion for retrieved chunks.

Dense retrieval embeds the question and each chunk separately, so a chunk that merely looks
like the question (another licence decision, say) can outrank the one that answers it. A
cross-encoder reads the question and chunk together and scores relevance directly; it is too
slow to run over the whole index, so it reorders a wider candidate pool (dense + rare-term
lexical) and keeps the top k.

Neighbor expansion attaches the chunks on either side of each kept hit, because answers often
sit just past a chunk boundary (e.g. the date on page 2 of a two-page letter).
"""
from __future__ import annotations

from functools import lru_cache


class Reranker:
    def __init__(self, model: str):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model_name = model
        self._model = TextCrossEncoder(model_name=model)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]


@lru_cache(maxsize=2)
def get_reranker(model: str) -> Reranker:
    return Reranker(model)


def expand_with_neighbors(store, version: str, metas: list[dict], window: int) -> list[dict]:
    """Return copies of `metas` with `context_text` = previous + hit + next chunk text (lines already
    in the hit, i.e. the chunker's overlap, are dropped) and page ranges widened to match."""
    if window <= 0:
        return [{**m, "context_text": m["text"]} for m in metas]
    hit_pos = {(m["doc_id"], m.get("ordinal")) for m in metas}
    wanted = {(m["doc_id"], m["ordinal"] + d) for m in metas if m.get("ordinal") is not None
              for d in range(-window, window + 1) if d and (m["doc_id"], m["ordinal"] + d) not in hit_pos}
    found = store.neighbors(version, sorted(wanted))
    out = []
    for m in metas:
        core_lines = set(m["text"].splitlines())
        before, after, pages = [], [], [m["page_start"], m["page_end"]]
        o = m.get("ordinal")
        for d in range(-window, window + 1):
            nb = found.get((m["doc_id"], o + d)) if d and o is not None else None
            if not nb:
                continue
            lines = [line for line in nb["text"].splitlines() if line not in core_lines]
            (before if d < 0 else after).extend(lines)
            pages += [nb["page_start"], nb["page_end"]]
        pages = [p for p in pages if p is not None]
        out.append({**m, "context_text": "\n".join(before + [m["text"]] + after),
                    "page_start": min(pages) if pages else m["page_start"],
                    "page_end": max(pages) if pages else m["page_end"]})
    return out
