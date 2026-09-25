# Running regrag on Databricks

The bundle in `/databricks.yml` deploys two things:

1. **Job `regrag-daily-incremental`**, on serverless compute, daily at 06:00 America/Toronto:
   `bronze_ingest` → `silver_spark` → `dbt_tests` → `gold_index`. Each task only starts if the
   previous one succeeded, so a failed validation or dbt test stops the run before the index changes.
2. **App `regrag-console`**: the demo console and `/ask` API, served by Databricks Apps.

Where each piece runs:

| Piece | Where |
|---|---|
| Raw filings (bronze) | Unity Catalog volume `/Volumes/<catalog>/regrag/lake` |
| Silver tables | Delta: `<catalog>.regrag.bronze_manifest`, `silver_documents`, `silver_pages`, `silver_chunks`, `silver_quarantine` |
| Data tests | dbt task on a SQL warehouse |
| Vector index (gold) | Hosted Postgres with pgvector (Supabase recommended); Databricks has no pgvector |
| Secrets | Secret scope `regrag` |

Status: everything here is written and unit-tested locally (`tests/test_databricks.py`, and `dbt parse` for
the `databricks` target), but it has **not been run on a workspace yet**. Expect a round of first-run fixes.
The most likely ones are listed at the end.

## 1. Accounts

- **Databricks.** Free Edition works (serverless compute and a starter SQL warehouse). Sign up at databricks.com.
- **Postgres with pgvector.** On Supabase, create a project, then in *Database → Extensions* enable `vector`.
  Copy the **Session pooler** connection string (*Connect → Session pooler*) and add `?sslmode=require`.
  Use the session pooler, not the transaction pooler: the store sets session options for HNSW search.

## 2. CLI and login

```bash
brew tap databricks/tap && brew install databricks
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

## 3. Catalog objects and secrets (once)

Replace `workspace` if your catalog has another name. In that case, also change `catalog` in `databricks.yml`,
and the `/Volumes/workspace/...` and `workspace.regrag...` paths in `configs/pipeline.databricks.yaml`.

```bash
databricks schemas create regrag workspace
databricks volumes create workspace regrag lake MANAGED

databricks secrets create-scope regrag
databricks secrets put-secret regrag database-url        # paste the Supabase session-pooler URL
databricks secrets put-secret regrag anthropic-api-key   # optional; answers are extractive without it
databricks secrets put-secret regrag user-agent          # e.g. regrag-research/0.1 (you@example.com)
```

Find the SQL warehouse id (the `warehouse_id` variable):

```bash
databricks warehouses list
```

## 4. Seed with what you already have (optional, recommended)

The daily job only looks back 7 days on its first run. To start from your local corpus instead:

**a.** Copy bronze to the volume. The Spark job then builds silver from the existing raw files, with no
re-download from the OEB. Object paths in the manifest point at your laptop, so rewrite them first:

```bash
python - <<'EOF'
import glob, pyarrow.parquet as pq, pyarrow.compute as pc, os
src, dst = os.path.abspath("data/lake"), "/Volumes/workspace/regrag/lake"
os.makedirs("data/seed/bronze/manifest", exist_ok=True)
for f in glob.glob("data/lake/bronze/manifest/*.parquet"):
    t = pq.read_table(f)
    t = t.set_column(t.schema.get_field_index("object_uri"), "object_uri",
                     pc.replace_substring(t["object_uri"], src, dst))
    pq.write_table(t, "data/seed/bronze/manifest/" + os.path.basename(f))
EOF
databricks fs cp -r data/lake/bronze/objects    dbfs:/Volumes/workspace/regrag/lake/bronze/objects
databricks fs cp -r data/lake/bronze/validation dbfs:/Volumes/workspace/regrag/lake/bronze/validation
databricks fs cp -r data/seed/bronze/manifest   dbfs:/Volumes/workspace/regrag/lake/bronze/manifest
databricks fs cp -r data/lake/_state            dbfs:/Volumes/workspace/regrag/lake/_state
```

**b.** Load the vector index into Supabase from your laptop. Embeddings come from the local cache, so this
takes minutes, not the original 20:

```bash
DATABASE_URL='<supabase session pooler url>' make gold
```

## 5. Deploy and run

```bash
databricks bundle validate --var warehouse_id=<id>
databricks bundle deploy   --var warehouse_id=<id>          # builds the wheel, syncs the repo
databricks bundle run regrag_daily --var warehouse_id=<id>  # one pipeline run now
databricks bundle run regrag_console                        # start the app; prints its URL
```

The `dev` target pauses the schedule and prefixes names with your user. To make the daily schedule live, deploy
with `-t prod` (ideally from CI with a service principal).

## Parity with the local pipeline

| | Local | Databricks |
|---|---|---|
| Ingest + validation | `regrag ingest` | same code (`regrag-job ingest`) writing to the volume |
| Silver | `silver/run.py` (Python) | `databricks/src/silver_spark.py`: same parse, chunk and dedup functions, applied with `mapInPandas`; identical chunk ids for the same config |
| dbt | dbt-duckdb over Parquet | dbt-databricks over Delta (the regex test switches to `rlike`) |
| Gold | reads Parquet chunks | reads the `silver_chunks` Delta table (`gold.chunks_table`) |
| Rechunk backfill | runs after silver | not wired yet: after a chunking change, run `silver_spark` against a fresh `silver_chunks` or add a rechunk task |
| Console | pipeline tab runs jobs | pipeline tab disabled (`REGRAG_CONSOLE_JOBS=off`); Data tab counts read the local lake, so they show zeros in the app |

## Likely first-run issues

- **Outbound internet.** Ingest calls the OEB API; gold downloads the embedding model (Hugging Face) and the
  tokenizer file, and connects to Supabase. If Free Edition serverless blocks egress, run ingest and gold
  locally (steps 4a and 4b) and keep Spark silver and dbt on Databricks.
- **Volume access from Spark workers.** `silver_spark.py` opens `/Volumes/...` files inside `mapInPandas`.
  If workers can't read the FUSE path, switch to reading bytes with `spark.read.format("binaryFile")`.
- **dbt task profile.** The job lets Databricks generate `profiles.yml` from `warehouse_id`, `catalog` and
  `schema`. If it complains about the profile name, set `profiles_directory: .` and use the `databricks` target
  in `dbt/profiles.yml`.
- **App size.** The app installs the package (`requirements.txt` → `.`), including fastembed/onnxruntime.
