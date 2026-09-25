"""Reranking, rare-term candidates, and neighbor expansion."""
from datetime import date

import numpy as np

from regrag.bronze.ingest import ingest
from regrag.gold.index import run_gold
from regrag.gold.store import LocalStore
from regrag.serve.rerank import expand_with_neighbors
from regrag.silver.run import run_silver


class FakeNeighborStore:
    def __init__(self, chunks):
        self.chunks = chunks            # {(doc_id, ordinal): {"text", "page_start", "page_end"}}

    def neighbors(self, version, keys):
        return {k: self.chunks[k] for k in keys if k in self.chunks}


def _meta(doc, ordinal, text, page):
    return {"doc_id": doc, "ordinal": ordinal, "text": text, "page_start": page, "page_end": page}


def test_expansion_adds_neighbors_drops_overlap_and_widens_pages():
    store = FakeNeighborStore({
        ("d", 0): {"text": "Intro line\nshared overlap line", "page_start": 1, "page_end": 1},
        ("d", 2): {"text": "The abeyance ends on November 5, 2026.", "page_start": 2, "page_end": 2},
    })
    hit = _meta("d", 1, "shared overlap line\nThe OEB extends the abeyance by 45 days.", 1)
    [out] = expand_with_neighbors(store, "v", [hit], window=1)
    assert out["context_text"].splitlines() == [
        "Intro line", "shared overlap line", "The OEB extends the abeyance by 45 days.",
        "The abeyance ends on November 5, 2026."]
    assert (out["page_start"], out["page_end"]) == (1, 2)
    assert out["text"] == hit["text"]                      # the hit itself is untouched


def test_expansion_skips_neighbors_that_are_hits_and_window_zero_is_noop():
    store = FakeNeighborStore({("d", 1): {"text": "B", "page_start": 1, "page_end": 1}})
    a, b = _meta("d", 0, "A", 1), _meta("d", 1, "B", 1)
    out = expand_with_neighbors(store, "v", [a, b], window=1)
    assert [o["context_text"] for o in out] == ["A", "B"]  # B is already a hit, not glued onto A
    assert expand_with_neighbors(store, "v", [a], window=0)[0]["context_text"] == "A"


def test_rare_term_lexical_finds_a_name_dense_search_misses(tmp_path):
    store = LocalStore(str(tmp_path), dim=4)
    v = "v1-x__stub"
    store.ensure_version({"index_version": v, "chunk_config_version": "v1-x", "embed_model": "stub"})
    rows, vecs = [], []
    for i in range(40):                                    # 40 look-alike licence decisions
        rows.append({"chunk_id": f"c{i}", "doc_id": f"d{i}", "text": "generation licence decision granted",
                     "ordinal": 0, "chunk_type": "text"})
        vecs.append([1, 0, 0, 0])
    rows.append({"chunk_id": "stelco", "doc_id": "ds", "text": "Stelco Inc. generation facility in Nanticoke",
                 "ordinal": 0, "chunk_type": "text"})
    vecs.append([0, 1, 0, 0])                              # far from the query vector
    store.upsert(v, rows, np.array(vecs, np.float32))
    q = np.array([1, 0, 0, 0], np.float32)
    dense_top = [h.chunk_id for h in store.search(v, q, "Where is Stelco's generation licence facility?", k=5,
                                                  hybrid=False)]
    rare_top = [h.chunk_id for h in store.search(v, q, "Where is Stelco's generation licence facility?", k=45,
                                                 lexical="rare")]
    assert "stelco" not in dense_top
    assert "stelco" in rare_top[:5]                        # "generation"/"licence" are too common to count


def test_api_rerank_mode_reorders_and_expands(cfg, fake_source, monkeypatch):
    ingest(cfg, "oeb", since=date(2026, 9, 20), until=date(2026, 9, 20))
    run_silver(cfg)
    run_gold(cfg)

    class KeywordReranker:                                 # stands in for the cross-encoder
        def scores(self, query, texts):
            return [float("41.9" in t) for t in texts]

    monkeypatch.setattr("regrag.serve.chain.get_reranker", lambda model: KeywordReranker())
    from regrag.serve import chain
    chain._QA.clear()
    qa = chain.get_qa(cfg)
    res = qa.ask("What revenue requirement was approved?", k=2, use_llm=False, retrieval="rerank",
                 expand_window=1, use_cache=False)
    assert res["retrieval"] == "rerank" and res["expand_window"] == 1
    assert "41.9" in res["sources"][0]["snippet"]          # the reranker's pick is ranked first
    assert all(len(s["context"]) >= len(s["snippet"]) for s in res["sources"])


def test_eval_survives_api_errors_and_stops_after_three_in_a_row(cfg, fake_source, monkeypatch, tmp_path):
    import json
    ingest(cfg, "oeb", since=date(2026, 9, 20), until=date(2026, 9, 20))
    run_silver(cfg)
    run_gold(cfg)
    from regrag.evaluation import run_eval as ev
    from regrag.serve import chain
    chain._QA.clear()
    qa = chain.get_qa(cfg)
    real_ask, calls = qa.ask, {"n": 0}

    def flaky(question, **kw):
        if question.startswith("warm-up"):
            return real_ask(question, **kw)
        calls["n"] += 1
        if calls["n"] == 1:
            return real_ask(question, **kw)
        raise RuntimeError("credit balance is too low")

    monkeypatch.setattr(qa, "ask", flaky)
    monkeypatch.setattr(ev, "PROJECT_ROOT", tmp_path)
    qfile = tmp_path / "questions_t.jsonl"
    qfile.write_text("".join(json.dumps({"id": f"t{i}", "question": "revenue requirement approved",
                                         "expected_source_ids": ["1001"], "answer_contains": ["41.9"]}) + "\n"
                             for i in range(6)))
    res = ev.run_eval(cfg, str(qfile), k=2, retrieval="dense")
    s = res["summary"]
    assert s["n_questions"] == 1 and s["n_errors"] == 3      # stopped after 3 consecutive failures
    assert list((tmp_path / "eval" / "results").glob("*_dense_t.json"))  # partial results saved


def test_llm_failure_degrades_to_extractive_and_is_not_cached(cfg, fake_source, monkeypatch):
    ingest(cfg, "oeb", since=date(2026, 9, 20), until=date(2026, 9, 20))
    run_silver(cfg)
    run_gold(cfg)
    from regrag.serve import chain
    chain._QA.clear()
    qa = chain.get_qa(cfg)

    class Broke:
        def generate(self, *a, **k):
            raise RuntimeError("Your credit balance is too low")
    qa.llm = Broke()
    res = qa.ask("What revenue requirement did the OEB approve?", k=2, retrieval="dense")
    assert res["model"] == "extractive" and "Claude unavailable" in res["answer"] and res["llm_error"]
    again = qa.ask("What revenue requirement did the OEB approve?", k=2, retrieval="dense")
    assert again["cache"] == "miss"                         # degraded answers are not cached


def test_llm_eval_counts_degraded_answers_as_errors(cfg, fake_source, monkeypatch, tmp_path):
    import json
    ingest(cfg, "oeb", since=date(2026, 9, 20), until=date(2026, 9, 20))
    run_silver(cfg)
    run_gold(cfg)
    from regrag.evaluation import run_eval as ev
    from regrag.serve import chain
    chain._QA.clear()
    qa = chain.get_qa(cfg)

    class Broke:
        def generate(self, *a, **k):
            raise RuntimeError("Your credit balance is too low")
    qa.llm = Broke()
    monkeypatch.setattr(ev, "PROJECT_ROOT", tmp_path)
    qfile = tmp_path / "questions_t.jsonl"
    qfile.write_text("".join(json.dumps({"id": f"t{i}", "question": "revenue requirement approved",
                                         "expected_source_ids": ["1001"], "answer_contains": ["41.9"]}) + "\n"
                             for i in range(6)))
    try:
        ev.run_eval(cfg, str(qfile), k=2, retrieval="dense", with_llm=True)
    except RuntimeError as exc:                          # every question failed -> no results to save
        assert "credit balance" in str(exc)
    else:
        raise AssertionError("expected the LLM eval to fail loudly")
