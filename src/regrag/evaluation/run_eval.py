"""Golden-set evaluation: retrieval hit rate, MRR, and latency (optionally answer accuracy).

questions.jsonl rows:
  {"id": "q01", "question": "...", "expected_source_ids": ["955586"],
   "answer_contains": ["$1.2 million"], "expected_docket": "EB-2026-0015"}

Metrics
  doc_hit@k     an expected document appears among the top-k chunks
  answer_hit@k  the context given to the model for some top-k source (chunk plus any neighbor
                expansion) contains an expected answer string, i.e. the evidence was retrieved
  mrr           mean reciprocal rank of the first chunk from an expected document
  latency       retrieval-only and end-to-end, p50/p95, answer cache disabled, after one warm-up
  answer_acc    (with --with-llm) the generated answer contains an expected answer string (strict)
  answer_acc_judged (with --judge) a Claude judge says the answer states the expected fact

Question files: eval/questions.jsonl is the development set (used while tuning retrieval);
eval/questions_holdout.jsonl was written and frozen before tuning and is the unbiased check.
Results: eval/results/<stamp>_<mode>[_<set>].json
"""
from __future__ import annotations

import json
import os
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

from ..config import PROJECT_ROOT
from ..serve.chain import get_qa


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def load_questions(path: str) -> list[dict]:
    with open(path if os.path.isabs(path) else PROJECT_ROOT / path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def run_eval(cfg: dict, questions_path: str, k: int = 6, with_llm: bool = False,
             retrieval: str | None = None, expand_window: int | None = None, judge: bool = False) -> dict:
    qa = get_qa(cfg)
    mode = retrieval or qa.default_retrieval
    expand = qa.default_expand if expand_window is None else expand_window
    questions = load_questions(questions_path)
    qset = Path(questions_path).stem.removeprefix("questions").strip("_") or "dev"
    qa.ask("warm-up query to load the models", k=k, use_llm=False, use_cache=False, retrieval=mode,
           expand_window=expand)
    grader = None
    if judge and with_llm and qa.llm:
        from .judge import Judge
        grader = Judge(cfg["serve"]["model"])

    rows, errors, consecutive = [], [], 0
    for q in questions:
        try:
            res = qa.ask(q["question"], k=k, use_llm=with_llm, retrieval=mode, use_cache=False,
                         expand_window=expand)
            if with_llm and res.get("llm_error"):
                # The API degrades to an extractive answer when Claude fails; in an LLM eval that is an
                # error, not an answer to score.
                raise RuntimeError(f"Claude call failed: {res['llm_error']}")
            consecutive = 0
        except Exception as exc:  # noqa: BLE001 - one failed API call must not lose the whole run
            errors.append({"id": q["id"], "error": repr(exc)[:300]})
            consecutive += 1
            if consecutive >= 3:
                break                  # repeated failures (e.g. out of credits): stop, keep what we have
            continue
        expected = set(q.get("expected_source_ids", []))
        needles = [_norm(a) for a in q.get("answer_contains", [])]
        ranks = [s["n"] for s in res["sources"] if s["source_id"] in expected]
        first = min(ranks) if ranks else None
        answer_rank = next((s["n"] for s in res["sources"]
                            if any(n in _norm(s.get("context", "")) for n in needles)), None)
        row = {
            "id": q["id"], "question": q["question"], "first_relevant_rank": first,
            "doc_hit@1": first is not None and first <= 1, "doc_hit@3": first is not None and first <= 3,
            "doc_hit@k": first is not None, "answer_hit@k": answer_rank is not None,
            "retrieval_ms": res["latency_ms"]["retrieval"], "total_ms": res["latency_ms"]["total"],
            "retrieved": [(s["source_id"], s["docket"], s["pages"]) for s in res["sources"]],
        }
        if with_llm and res["model"] != "extractive":
            row["answer"] = res["answer"]
            row["answer_acc"] = any(n in _norm(res["answer"]) for n in needles) if needles else None
            row["cited_expected"] = any(s["cited"] and s["source_id"] in expected for s in res["sources"])
            row["usage"] = res["usage"]
            if grader:
                verdict = grader.grade(q["question"], q.get("answer_contains", []), res["answer"])
                row["judge_correct"], row["judge_reason"] = verdict.get("correct"), verdict.get("reason")
        rows.append(row)

    n = len(rows)
    if not n:
        raise RuntimeError(f"every question failed; first error: {errors[0]['error'] if errors else 'none'}")
    ret_ms = [r["retrieval_ms"] for r in rows]
    tot_ms = [r["total_ms"] for r in rows]
    summary = {
        "n_questions": n, "question_set": qset, "k": k, "retrieval": mode, "expand_window": expand,
        "hybrid": mode == "hybrid", "index_version": qa.index_version(),
        "doc_hit@1": sum(r["doc_hit@1"] for r in rows) / n,
        "doc_hit@3": sum(r["doc_hit@3"] for r in rows) / n,
        f"doc_hit@{k}": sum(r["doc_hit@k"] for r in rows) / n,
        f"answer_hit@{k}": sum(r["answer_hit@k"] for r in rows) / n,
        "mrr": sum(1 / r["first_relevant_rank"] for r in rows if r["first_relevant_rank"]) / n,
        "retrieval_ms_p50": _pct(ret_ms, 50), "retrieval_ms_p95": _pct(ret_ms, 95),
        "total_ms_p50": _pct(tot_ms, 50), "total_ms_p95": _pct(tot_ms, 95),
        "retrieval_ms_mean": statistics.mean(ret_ms) if ret_ms else 0.0,
        "n_errors": len(errors), "errors": errors,
    }
    llm_rows = [r for r in rows if "answer" in r]
    if llm_rows:
        scored = [r for r in llm_rows if r["answer_acc"] is not None]
        summary["answer_acc"] = sum(r["answer_acc"] for r in scored) / len(scored) if scored else None
        summary["cited_expected_rate"] = sum(r["cited_expected"] for r in llm_rows) / len(llm_rows)
        judged = [r for r in llm_rows if r.get("judge_correct") is not None]
        if judged:
            summary["answer_acc_judged"] = sum(r["judge_correct"] for r in judged) / len(judged)
            summary["judge_graded"] = len(judged)
        cache_read = sum(r["usage"].get("cache_read_input_tokens", 0) for r in llm_rows)
        cache_write = sum(r["usage"].get("cache_creation_input_tokens", 0) for r in llm_rows)
        uncached = sum(r["usage"].get("input_tokens", 0) for r in llm_rows)
        summary["prompt_cache"] = {"read_tokens": cache_read, "write_tokens": cache_write,
                                   "uncached_input_tokens": uncached,
                                   "hit_ratio": cache_read / max(1, cache_read + cache_write + uncached)}
    else:
        summary["answer_acc"] = None

    out_dir = PROJECT_ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = mode + ("" if qset == "dev" else f"_{qset}")
    with open(out_dir / f"{stamp}_{tag}.json", "w") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2, default=str)
    _write_markdown(out_dir / f"latest_{tag}.md", summary, rows)
    return {"summary": summary, "rows": rows}


def _write_markdown(path, summary: dict, rows: list[dict]) -> None:
    k = summary["k"]
    lines = [f"# Eval results: {summary['retrieval']} retrieval, {summary['question_set']} set", "",
             f"Index `{summary['index_version']}`, {summary['n_questions']} questions, k={k}, "
             f"neighbor expansion {summary['expand_window']}", "",
             "| metric | value |", "|---|---|"]
    for key in ["doc_hit@1", "doc_hit@3", f"doc_hit@{k}", f"answer_hit@{k}", "mrr", "answer_acc",
                "answer_acc_judged"]:
        v = summary.get(key)
        lines.append(f"| {key} | {'n/a' if v is None else f'{v:.3f}'} |")
    for key in ["retrieval_ms_p50", "retrieval_ms_p95", "total_ms_p50", "total_ms_p95"]:
        lines.append(f"| {key} | {summary[key]:.1f} |")
    lines += ["", "| id | first relevant rank | answer evidence retrieved | retrieval ms |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['first_relevant_rank'] or 'miss'} | {'yes' if r['answer_hit@k'] else 'no'} "
                     f"| {r['retrieval_ms']:.0f} |")
    path.write_text("\n".join(lines) + "\n")
