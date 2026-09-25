# regrag: regulatory filings → medallion pipeline → cited RAG

Pulls public utility-regulator filings (Ontario Energy Board by default, SEC EDGAR as a fallback), runs them
through a bronze → silver → gold pipeline, and serves a question-answering API that cites the filing, docket,
section and page behind every claim.

```
 OEB RDS (WebDrawer JSON) ─┐                                   ┌─ dbt tests (schema, relationships, freshness)
 SEC EDGAR (fallback) ─────┤                                   │
                           ▼                                   ▼
  BRONZE  raw PDF/HTML + manifest ──validate──▶ SILVER parse · dedup · quarantine · chunk ──▶ GOLD embed → pgvector
          (source, timestamp, sha256)  gate      pages, tables, sections, docket, date         (versioned index)
                                                                                                   │
                                          FastAPI /ask ◀── LangChain retriever (HNSW; optional full-text RRF) ◀┘
                                              │
                                              └─▶ Claude (cached system prompt) → answer with [n] citations
```

## Quick start

```bash
make install            # venv + package (Python 3.11+; developed on 3.14)
make pg-up              # project-local Postgres 18 + pgvector on :5433 (see "Postgres" below)
export REGRAG_USER_AGENT="regrag-research/0.1 (you@example.com)"
.venv/bin/regrag run --since 2026-08-20 --max-docs 300   # ingest → validate → silver → dbt → gold
make serve              # demo console at http://127.0.0.1:8000 (API docs at /docs)
make eval               # 30-question dev set: hit rate, MRR, latency (add --with-llm --judge for answers)
```

**Demo console.** `make serve` then open http://localhost:8000. It has four tabs:
* **Ask:** questions with filters and clickable citations.
* **Pipeline:** every command (ingest, validation, silver, rechunk, dbt, gold, watch, eval, full run), each
  with its parameters, the equivalent CLI line, and a live log console.
* **Data:** counts, index versions with one-click activate or rollback, validation checks, quarantine, and a
  document browser.
* **Evaluation:** dense vs hybrid metrics and per-question results.

Jobs run the same `regrag` CLI as subprocesses, one at a time. Only allowlisted commands run, with validated
parameters and no shell, and the server binds to 127.0.0.1. The raw API is documented at `/docs`.

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "When does the extended abeyance of the Lanark and Balderson project end?"}'
```

The response includes `answer`, numbered `sources` (title, docket, date, section, pages, RDS link, whether it
was cited), `index_version`, `prompt_version`, token `usage` (including prompt-cache reads), latency and
`cache: hit|miss`. Optional fields: `docket`, `source`, `date_from`, `date_to`, `k`, `retrieval`
(`dense`|`hybrid`).

Answer generation uses Claude (`claude-opus-5` by default, via the official Anthropic SDK). Set
`ANTHROPIC_API_KEY` to turn it on. Without credentials the API still works and returns an **extractive**
answer: the best-matching sentences from the retrieved chunks, still cited. That keeps retrieval, the eval and CI
fully offline.

## Stages

### 1. Bronze: ingest and validate (`src/regrag/bronze`, `src/regrag/sources`)

* **OEB source.** The Regulatory Document Search is an OpenText Content Manager *WebDrawer*, and
  `/CMWebDrawer/Record?q=...&format=json` is a real JSON search API. We query
  `registeredOn:<from> to <to> and extension:pdf`, page through the results, and download
  `/Record/{uri}/File/document`. Two quirks: `properties=all` fails with "access denied" on a restricted field,
  so we request an explicit property list. We also don't collect the author field, which holds staff names.
* **EDGAR fallback.** Uses the `data.sec.gov` submissions API for configured utility CIKs. It needs a
  descriptive User-Agent.
* **Landing.** Raw bytes are stored immutably at
  `bronze/objects/source=<src>/ingest_date=<d>/<id>_<sha12>.<ext>`, alongside a Parquet manifest row per
  record: source, ids, title, URL, dates, **sha256**, bytes, fetched_at. Storage goes through `fsspec`, so
  `REGRAG_LAKE_ROOT=s3://…` or `gs://…` works unchanged.
* **Incremental and idempotent.** Each source keeps a watermark on `registered_at` (`_state/<src>.json`) with a
  one-day overlap. Records already in the manifest are never re-downloaded. Identical bytes under a new id are
  recorded but stored once (`duplicate_of`).
* **Validation after every pull** (`bronze/validate.py`). The report is written to
  `bronze/validation/<run_id>.json`:

  | check | severity | catches |
  |---|---|---|
  | required fields not null | error | source API shape changes |
  | unique (source, source_id) in batch | error | pagination bugs |
  | re-read object, sha256 == manifest | error | partial or corrupt writes |
  | fetch error rate ≤ 20% | error | outages, blocking |
  | magic bytes match extension | warn → quarantine | HTML error pages saved as `.pdf` |
  | size within bounds, matches regulator's size | warn → quarantine | truncated downloads |
  | freshness of newest record | warn | stalled feeds, backfills |
  | volume vs trailing runs | warn | silent API changes |

  An `error` failure blocks the run: the watermark doesn't advance, so the next run retries, and the
  orchestrator stops before silver. Records that fail a `warn` check are quarantined in silver.

### 2. Silver: parse, clean, dedup, quarantine, chunk (`src/regrag/silver`)

* **Parsing.** `pdfplumber` extracts text, and tables become Markdown. Table regions are removed from the body
  text so numbers aren't indexed twice. Running headers and footers are dropped: these are lines at a page's
  edges that repeat on at least half the pages, with digits normalized so page numbers match. Soft hyphens and
  line-break hyphenation are repaired. HTML (EDGAR) is handled by a stdlib parser.
* **Metadata.** Each chunk carries `docket` (EB-YYYY-NNNN, taken from the title or else the first pages of
  text), `published_at`, `section` (the nearest detected heading), `page_start` and `page_end`, and the RDS
  URL. Heading detection accepts numbered headings (`3.2 Load Forecast`). It accepts unnumbered ALL-CAPS lines
  only if they contain a structural word, which keeps signature blocks, letterheads and "BY EMAIL" out of the
  section field.
* **Dedup.** Exact dedup on sha256 at bronze, plus near-duplicates by a hash of case-, whitespace- and
  digit-normalized text, which catches re-posted or re-signed copies.
* **Quarantine** (`silver/quarantine`, with stage and reason): files that failed bronze validation, unreadable
  or encrypted PDFs, and near-empty text (scanned images; OCR isn't implemented).
* **Chunking** (`configs/chunking/v1.yaml`). Token-bounded (380, with 60 overlap), never crossing a section
  boundary. Tables become their own chunks, split by rows with the header repeated. Oversized rows and
  flattened-spreadsheet headers are split. Each chunk gets a `title | docket | section` context prefix for
  embedding.
* **Reproducibility.** The config file *plus* a chunker-algorithm version (`CHUNKER_ALGO_VERSION` in
  `config.py`) is hashed into `chunk_config_version` (e.g. `v1-d4fd9d711f72`). Chunk ids are
  `sha256(doc | version | ordinal)`, so the same config and code over the same documents always reproduce the
  same chunks, and a code change can't silently alter an existing version. Parsed pages are stored, so
  `regrag rechunk [config]` builds or backfills a version without re-downloading or re-parsing. The pipeline
  runs it after every silver run: a no-op normally, a backfill after a config change.
* **Spark on Databricks.** `databricks/src/silver_spark.py` runs the same parse, chunk and dedup functions from
  the wheel inside `mapInPandas` on serverless compute, and MERGEs into Delta tables with the same names dbt
  reads.

### dbt (`dbt/`)

dbt-duckdb reads the Parquet lake directly through `external_location`. The `databricks` target reads the Delta
tables. Tests run as part of `regrag run`, the Prefect flow, the Databricks job and CI:

* schema: not_null, unique, accepted_values, a regex on sha256, dockets and config versions, and a composite
  uniqueness test on `(chunk_config_version, chunk_id)`
* relationships: chunk → document → bronze manifest
* singular tests: chunk token bounds, valid page ranges, every ok document has chunks in the current chunk version,
  at most one "ok" document per normalized-text hash, and a quarantine-rate threshold (warn)
* `dbt source freshness`: warn after 2 days, error after 7 days without new data
* marts: `dim_documents`, `fct_chunks`, `rpt_pipeline_health` (landed, parsed, duplicate, quarantined and
  pending per run), `rpt_docket_coverage`

### 3. Gold: versioned vector index (`src/regrag/gold`)

* **Embeddings.** Local ONNX `BAAI/bge-small-en-v1.5` (384-d) via fastembed, so there are no API costs. Queries
  get BGE's query instruction prefix.
* **Embedding cache.** A DuckDB table keyed on `(model, sha256(text))`, so rebuilds and re-chunks only embed
  text they haven't seen. Batches are length-sorted, which roughly doubled throughput.
* **pgvector.** Each row is `rag_chunks(index_version, chunk_id, …, embedding vector(384), tsv tsvector)`,
  with an HNSW cosine index, a GIN full-text index, and a `rag_index_versions` registry that has exactly one
  active version.
* **Versioning.** `index_version = <chunk_config_version>__<embed_model>`. A new chunking config or embedding
  model builds a new index next to the old one. `regrag activate <version>` flips what the API serves, and
  rolling back is the same command.
* **Retrieval** (`serve.retrieval`; the default `rerank` was chosen by the eval below):
  1. *Candidates:* 20 chunks from the dense top 50 plus a **rare-term** full-text leg, fused with
     reciprocal-rank fusion. "Rare" means a question term that appears in at most 0.5% of chunks, such as a
     company name like "Stelco". Common words ("generation", "licence", "2027") are dropped, because matching
     them promoted look-alike filings.
  2. *Rerank:* a cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`, local ONNX) scores question and chunk
     together and keeps the top 6.
  3. *Neighbour expansion:* each kept chunk is sent to the model with the chunk before and after it (overlap
     lines removed, page range widened), because answers often sit just past a chunk boundary.
  
  `dense` (HNSW only) and `hybrid` (exact-token full-text) remain available as modes.

### 4. Serve and evaluate (`src/regrag/serve`, `src/regrag/evaluation`)

* **LangChain chain.** A `BaseRetriever` subclass over the versioned hybrid index, feeding an LCEL runnable
  that formats numbered sources and generates the answer.
* **Claude.** Called through the Anthropic SDK with adaptive thinking, `effort: medium`, and server-side
  refusal fallbacks (`fallbacks: "default"`). The system prompt (grounding rules, citation format, OEB glossary)
  is frozen text with a `cache_control` breakpoint, so every request after the first reads it from the prompt
  cache. The eval reports `cache_read_input_tokens` to prove it.
* **Answer cache.** Keyed on the normalized question plus index version, prompt version, model, k and filters.
  Any change to the index or prompt is a miss by construction, so the cache never serves stale answers.
* **Eval.** Two question sets, each question written from a filing in the corpus, with every expected
  answer string verified verbatim in its source:
  * `eval/questions.jsonl`: 30 questions, the **dev** set used while tuning retrieval.
  * `eval/questions_holdout.jsonl`: 22 questions, written and frozen **before** tuning, as the unbiased check.
  
  Metrics: `doc_hit@1/3/k`, `answer_hit@k` (the context given to the model contains the answer), MRR, and
  retrieval and end-to-end p50/p95 latency with the answer cache off. With `--with-llm` the eval also reports:
  * strict answer accuracy (the answer contains an expected phrase)
  * how often the expected filing is cited
  * prompt-cache hit ratio

  With `--judge`, a Claude judge also grades whether the answer states the fact, allowing paraphrase. It is
  reported next to the strict score, never instead of it. A failed API call is recorded per question, and
  partial results are still saved. Results go to `eval/results/<stamp>_<mode>[_holdout].json`.

### 5. Operate (`src/regrag/ops`, `databricks/`, `.github/workflows`)

* **Batch.** `regrag run` (or `make pipeline`) runs ingest, then the validation gate, silver, dbt build and
  freshness (gate), then gold. Every stage is idempotent, so recovery means re-running.
  * Prefect: `python -m regrag.ops.flows serve` registers `regulatory-filings-daily` at 06:00 America/Toronto,
    with one task per stage and retries on ingest.
  * Databricks: `databricks.yml` is an asset bundle with the same four tasks on serverless compute: ingest to a
    Unity Catalog volume, Spark silver to Delta, a dbt-databricks gate, then gold into hosted pgvector
    (Supabase). It also deploys the console as a Databricks App. Setup, secrets and seeding from the local
    corpus are covered in [databricks/README.md](databricks/README.md).
* **Streaming angle.** `regrag watch --interval 900` polls RDS for newly registered filings and pushes each
  micro-batch through the same pipeline, skipping dbt for latency. OEB has no push feed, so a watermark poll is
  the event source. The watcher and the daily batch share the watermark, and stages are idempotent, so running
  both is safe.
* **CI** (`.github/workflows/ci.yml`):
  * ruff, plus the offline test suite on Python 3.12 and 3.14. The suite uses synthetic PDFs, a fake source and
    a stub embedder, and includes a full `dbt build` and `source freshness` against a fixture lake.
  * A pgvector integration job against a `pgvector/pgvector:pg17` service container.
  * A nightly live smoke test against the real OEB API, the most likely thing to break silently.

## Results (live OEB corpus, 2026-09-24)

**Corpus.** 319 OEB filings: 300 registered from 2026-08-20 to 2026-09-23, plus 19 decisions from
2025-10 to 2026-08.

| Stage | Result |
|---|---|
| Bronze | 319 landed, 0 fetch errors. Validation passed; the backfill run was correctly flagged with a freshness warning. |
| Silver | 299 parsed. 7 duplicates caught (sha256 or normalized text). 13 quarantined as scanned or image-only PDFs. |
| Chunks | 17,572 chunks (13,968 text, 3,590 table). |
| Parse time | About 6 minutes on a laptop CPU. |
| dbt | 50 of 50 tests pass, and 3 of 3 sources are fresh. |
| Gold | 17,572 vectors in pgvector (HNSW). |
| Embedding | About 20 minutes on a laptop CPU. A rebuild after a chunker fix reused 8,860 embeddings from cache. |

**Retrieval** (k=6, answer cache off, after warm-up; `eval/results/`). The dev set was used to choose the
configuration; the held-out set was scored once per configuration after that choice.

| | set | doc hit@1 | doc hit@6 | answer evidence in context | MRR | retrieval p50 / p95 |
|---|---|---|---|---|---|---|
| dense, no expansion (old default) | dev (30) | 0.733 | 0.967 | 0.833 | 0.831 | 8 / 11 ms |
| dense + neighbour expansion | dev (30) | 0.733 | 0.967 | 0.933 | 0.831 | 9 / 11 ms |
| **rerank + expansion (default)** | dev (30) | **0.900** | **1.000** | **1.000** | **0.933** | 473 / 625 ms |
| dense, no expansion (old default) | held-out (22) | 0.591 | 0.864 | 0.909 | 0.693 | 8 / 14 ms |
| **rerank + expansion (default)** | held-out (22) | **0.909** | **1.000** | **1.000** | **0.924** | 381 / 505 ms |

*doc hit* means a chunk from the expected filing is retrieved. *Answer evidence* means the context sent to the
model contains the verified answer string.

Reranker choice (dev set): MiniLM-L-12 over 40 candidates scored the same at 1.9 s; `bge-reranker-base` scored
the same at 4.5 s; jina-turbo scored lower. The small model over 20 candidates keeps the gain at about 0.4-0.5 s.
The earlier hybrid modes (full-text over all question terms, or exact tokens only) scored below dense.
Common tokens recur across sibling filings, which is what the rare-term filter fixes.

**Answers from Claude** (claude-opus-5, effort medium). *Strict* means the answer contains an expected phrase.
*Judged* means a Claude judge confirms the answer states the fact.

| | set | strict | judged | expected filing cited | end-to-end p50 / p95 |
|---|---|---|---|---|---|
| dense, no expansion | dev (30) | 0.700 | 0.833 | 0.967 | 3.8 / 7.0 s |
| **rerank + expansion** | dev (30) | **0.867** | **0.967** | **1.000** | 3.6 / 6.5 s |
| dense, no expansion | held-out (22) | 0.909 | 0.909 | 0.864 | 3.2 / 5.6 s |
| **rerank + expansion** | held-out (22) | **1.000** | **1.000** | **1.000** | 3.8 / 6.1 s |

Notes on these numbers:
* 30 and 22 questions are small samples: one question moves a score by 3.3 or 4.5 points.
* Question h03's label was corrected after the first held-out run, and both held-out rows were re-run
  afterwards. The source record is internally inconsistent: the acknowledgment letter is dated September 1,
  and a later letter calls it "dated September 2". The judge had marked a correct answer (September 1) wrong.
  The label now accepts either date, and the note is kept in `questions_holdout.jsonl`.
* The held-out set is 100% with the new setup, but 22 questions can't distinguish 100% from, say, 95%. Treat
  it as "no regressions and the gain generalizes", not as perfection.
* Neighbour expansion sends about 55% more input tokens per question (dev: 217K vs 139K over 30 questions).

**What the pipeline caught along the way:**
* Validation flagged a stale backfill.
* Silver quarantined 13 image-only PDFs.
* dbt failed the build on 61 table chunks over the token ceiling. The cause was single table rows, and
  flattened spreadsheet headers, larger than the budget. The fix shipped as a new chunk version rather than a
  silent in-place change.

## Known limitations

* **No OCR.** Scanned PDFs are quarantined with a reason, not indexed. About 4% of this corpus.
* **Page cap.** Documents longer than `silver.max_pages` (400) are truncated. `n_pages_total` records the true
  length, and 7 exhibits in this corpus hit the cap.
* **Docket coverage is 84%.** Some exhibits carry no EB number in the title or first pages. A few pick up a
  cross-referenced prior case number.
* **The Databricks path is written but not run on a workspace.** That covers `databricks.yml`, the Spark silver
  job, the dbt `databricks` target and the App. It's covered locally by `tests/test_databricks.py` and a
  `dbt parse` for that target. Everything else was run end to end locally against the live OEB API.
* **The API holds a single Postgres connection.** Use `psycopg_pool` before serving real concurrency.
* **Prefect's ephemeral mode** logs SQLite lock warnings at startup. Use `prefect server start` or Prefect
  Cloud for the schedule.

## Postgres

`scripts/local_pg.sh up` builds pgvector 0.8.1 from source into `.vendor/` and starts a **project-local**
Postgres 18 cluster (`.pgdata/`, port 5433). It uses PG18's `extension_control_path` and
`dynamic_library_path`, so nothing is installed into Homebrew or any system Postgres. Any Postgres with pgvector
works: set `DATABASE_URL`. `REGRAG_VECTOR_STORE=local` swaps in a numpy store for DB-free use.

## Layout

```
configs/            pipeline.yaml, chunking/v1.yaml (versioned)
src/regrag/
  sources/          oeb.py (RDS WebDrawer), edgar.py
  bronze/           ingest.py (land + watermark), validate.py (post-pull checks)
  silver/           parse.py (PDF/HTML, tables, headers, headings), chunk.py, run.py (dedup/quarantine)
  gold/             embed.py (fastembed + cache), store.py (pgvector hybrid / local), index.py (versions)
  serve/            chain.py (LangChain), llm.py (Claude), prompts.py, cache.py, api.py (FastAPI)
  evaluation/       run_eval.py
  ops/              pipeline.py, flows.py (Prefect), watcher.py, dbt.py
dbt/                models (staging, marts), schema + singular tests, freshness
databricks.yml      Databricks bundle: daily job + console app (setup: databricks/README.md)
app.yaml            Databricks App entry for the console
databricks/         src/silver_spark.py (Spark silver), README.md
configs/            (also) pipeline.databricks.yaml: volume lake, Delta chunks, secrets
eval/               questions.jsonl, results/
tests/              offline suite + pgvector integration + live-source smoke
```
