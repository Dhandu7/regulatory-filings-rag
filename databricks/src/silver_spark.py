"""Silver on Spark: parse, dedup, quarantine, and chunk the bronze objects landed by `regrag-job ingest`.

Run by the bundle's `silver_spark` task (spark_python_task, so the job environment can install the
regrag wheel):  silver_spark.py --catalog workspace --config <workspace files>/configs/pipeline.databricks.yaml
"""
# Runs on serverless compute (Spark Connect): DataFrame APIs only, no .rdd and no .cache(); the
# expensive parse step is materialized once into a staging Delta table instead of being cached.
# Parsing and chunking reuse the exact functions from the `regrag` wheel (regrag.silver.parse /
# chunk / run) inside mapInPandas, so for the same config a Spark run and a local run produce
# identical chunk ids. Output: Delta tables <catalog>.regrag.silver_{documents,pages,chunks,
# quarantine} plus bronze_manifest, which the dbt `databricks` target and the gold task read.

import argparse
import glob
import json

import pandas as pd
import pyarrow as pa
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from regrag.config import ChunkingConfig, load_config
from regrag.silver.run import CHUNK_SCHEMA, DOC_SCHEMA, PAGE_SCHEMA, QUARANTINE_SCHEMA

parser = argparse.ArgumentParser()
parser.add_argument("--catalog", default="workspace")
parser.add_argument("--config", required=True, help="path to configs/pipeline.databricks.yaml")
ARGS = parser.parse_args()
spark = SparkSession.builder.getOrCreate()
CATALOG = ARGS.catalog
CFG = load_config(ARGS.config)
LAKE = CFG["lake_root"]                                          # a /Volumes/... path
SCFG = CFG["silver"]
CCFG = ChunkingConfig.load(SCFG["chunking_config"])              # loaded on the driver, shipped to workers
DB = f"{CATALOG}.regrag"
RUN_ID = f"spark-{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {DB}")
print(f"chunk config {CCFG.version}; lake {LAKE}; run {RUN_ID}")


def spark_schema(schema: pa.Schema) -> T.StructType:
    types = {pa.string(): T.StringType(), pa.int32(): T.IntegerType(), pa.int64(): T.LongType()}
    return T.StructType([T.StructField(f.name, types[f.type], True) for f in schema])


def table_exists(name: str) -> bool:
    return spark.catalog.tableExists(f"{DB}.{name}")


def merge_into(df, name: str, keys: list[str]) -> None:
    """Insert-only MERGE (idempotent re-runs); creates the Delta table on first write."""
    if not table_exists(name):
        df.write.format("delta").saveAsTable(f"{DB}.{name}")
        return
    view = f"_src_{name}"
    df.createOrReplaceTempView(view)
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    spark.sql(f"MERGE INTO {DB}.{name} t USING {view} s ON {on} WHEN NOT MATCHED THEN INSERT *")


# Bronze manifest (Parquet parts on the volume) -> Delta, idempotent on record_key.
manifest = (spark.read.option("mergeSchema", "true").parquet(f"{LAKE}/bronze/manifest/")
            .withColumn("record_key", F.concat_ws(":", "source", "source_id")))
merge_into(manifest, "bronze_manifest", ["record_key"])

done = None
for name in ("silver_documents", "silver_quarantine"):
    if table_exists(name):
        keys = spark.table(f"{DB}.{name}").select("record_key")
        done = keys if done is None else done.unionByName(keys)
todo = manifest if done is None else manifest.join(done, "record_key", "left_anti")

# Records that failed a bronze validation check are quarantined, never parsed (same rule as local silver).
bronze_bad = set()
for path in glob.glob(f"{LAKE}/bronze/validation/*.json"):
    with open(path) as fh:
        bronze_bad.update(json.load(fh).get("quarantine_source_ids", []))
print(f"to process: {todo.count()} records; bronze-quarantined keys known: {len(bronze_bad)}")


def process(batches):
    from regrag.silver.parse import ParseError, parse_document
    from regrag.silver.run import build_chunks, normalized_text_hash
    from regrag.sources import find_docket

    now = pd.Timestamp.utcnow().isoformat()
    for pdf in batches:
        out = []
        for r in pdf.to_dict("records"):
            key = r["record_key"]
            quarantine = {"record_key": key, "source": r["source"], "source_id": r["source_id"],
                          "sha256": r["sha256"], "object_uri": r["object_uri"], "run_id": RUN_ID,
                          "quarantined_at": now}
            if key in bronze_bad:
                out.append((key, "quarantine", json.dumps({**quarantine, "stage": "bronze_validation",
                                                           "reason": "failed a bronze validation check"})))
                continue
            try:
                with open(r["object_uri"], "rb") as fh:
                    parsed = parse_document(fh.read(), r["extension"], max_pages=SCFG["max_pages"],
                                            extract_tables=SCFG["extract_tables"])
                text = parsed.full_text
                if len(text) < SCFG["min_text_chars"]:
                    raise ParseError(f"only {len(text)} chars extracted (scanned image PDF? needs OCR)")
            except ParseError as exc:
                out.append((key, "quarantine", json.dumps({**quarantine, "stage": "parse",
                                                           "reason": str(exc)[:1000]})))
                continue
            extra = json.loads(r.get("extra_json") or "{}")
            doc = {"doc_id": r["sha256"], "record_key": key, "source": r["source"], "source_id": r["source_id"],
                   "title": r["title"], "docket": r["docket"] or find_docket(text[:5000]),
                   "record_type": r["record_type"], "published_at": r["published_at"], "url": r["url"],
                   "landing_url": extra.get("landing_url", r["url"]), "object_uri": r["object_uri"],
                   "n_pages": len(parsed.pages), "n_pages_total": parsed.n_pages_total, "n_chars": len(text),
                   "n_tables": sum(len(p.tables) for p in parsed.pages), "parser": parsed.parser,
                   "text_hash": normalized_text_hash(text), "duplicate_of": None, "status": "ok",
                   "run_id": RUN_ID, "parsed_at": now}
            out.append((key, "doc", json.dumps(doc)))
            for p in parsed.pages:
                out.append((key, "page", json.dumps({"doc_id": doc["doc_id"], "page_no": p.page_no, "text": p.text,
                                                     "tables_json": json.dumps(p.tables),
                                                     "headings_json": json.dumps(p.headings), "run_id": RUN_ID})))
            for c in build_chunks(doc, parsed.pages, CCFG, RUN_ID):
                out.append((key, "chunk", json.dumps(c)))
        yield pd.DataFrame(out, columns=["record_key", "kind", "payload"])


RESULT = T.StructType([T.StructField("record_key", T.StringType()), T.StructField("kind", T.StringType()),
                       T.StructField("payload", T.StringType())])
STAGING = f"{DB}._silver_staging"
(todo.repartition(64).mapInPandas(process, RESULT)
     .write.format("delta").mode("overwrite").saveAsTable(STAGING))   # materialize the parse once
staged = spark.table(STAGING)


def payloads(kind: str, schema: pa.Schema, only_keys=None):
    rows = staged.where(F.col("kind") == kind)
    if only_keys is not None:
        rows = rows.join(only_keys, "record_key")
    return rows.select(F.from_json("payload", spark_schema(schema)).alias("r")).select("r.*")


# Near-duplicate detection across existing + new documents: the first document per normalized-text
# hash stays 'ok' (existing documents win); later ones become 'duplicate' and get no pages or chunks.
new_docs = payloads("doc", DOC_SCHEMA)
candidates = new_docs.select("text_hash", F.col("doc_id").alias("keeper"), F.lit(1).alias("prio"),
                             F.col("record_key").alias("order_key"))
if table_exists("silver_documents"):
    existing_ok = (spark.table(f"{DB}.silver_documents").where("status = 'ok'")
                   .select("text_hash", F.col("doc_id").alias("keeper"), F.lit(0).alias("prio"),
                           F.col("record_key").alias("order_key")))
    candidates = existing_ok.unionByName(candidates)
first = Window.partitionBy("text_hash").orderBy("prio", "order_key")
keepers = (candidates.withColumn("rn", F.row_number().over(first)).where("rn = 1")
           .select("text_hash", "keeper", F.col("order_key").alias("keeper_key")))
docs = (new_docs.join(keepers, "text_hash", "left")
        .withColumn("status", F.when(F.col("record_key") == F.col("keeper_key"), "ok").otherwise("duplicate"))
        .withColumn("duplicate_of", F.when(F.col("status") == "duplicate", F.col("keeper")))
        .select(*[f.name for f in DOC_SCHEMA]))
# Materialize before merging: `docs` reads silver_documents, which the merge below changes.
docs.write.format("delta").mode("overwrite").saveAsTable(f"{STAGING}_docs")
docs = spark.table(f"{STAGING}_docs")
merge_into(docs, "silver_documents", ["record_key"])

ok_keys = docs.where("status = 'ok'").select("record_key")
merge_into(payloads("page", PAGE_SCHEMA, ok_keys).dropDuplicates(["doc_id", "page_no"]),
           "silver_pages", ["doc_id", "page_no"])
merge_into(payloads("chunk", CHUNK_SCHEMA, ok_keys).dropDuplicates(["chunk_config_version", "chunk_id"]),
           "silver_chunks", ["chunk_config_version", "chunk_id"])
merge_into(payloads("quarantine", QUARANTINE_SCHEMA), "silver_quarantine", ["record_key"])

summary = staged.groupBy("kind").count().toPandas().set_index("kind")["count"].to_dict()
print(json.dumps({"run_id": RUN_ID, "chunk_config_version": CCFG.version, "staged": summary,
                  "duplicates": docs.where("status = 'duplicate'").count()}))
for t in (STAGING, f"{STAGING}_docs"):
    spark.sql(f"DROP TABLE IF EXISTS {t}")
