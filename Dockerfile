# syntax=docker/dockerfile:1
# regrag: pipeline CLI + QA API/console in one image.
#
#   docker compose up -d                     # Postgres+pgvector and the API/console on :8000
#   docker compose run --rm api regrag run --since 2026-09-01 --max-docs 50
#
# The embedding model, reranker and tokenizer are downloaded at build time, so containers
# start without network access to Hugging Face (the OEB API is still needed to ingest).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    TIKTOKEN_CACHE_DIR=/opt/models/tiktoken \
    REGRAG_HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# Dependencies first so code edits don't invalidate this layer.
COPY pyproject.toml README.md ./
RUN mkdir -p src/regrag && touch src/regrag/__init__.py \
 && pip install ".[dbt]" \
 && pip uninstall -y regrag

# Bake the models the pipeline and API load at runtime (names match configs/pipeline.yaml).
RUN python - <<'EOF'
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import tiktoken
TextEmbedding("BAAI/bge-small-en-v1.5")
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
tiktoken.get_encoding("cl100k_base")
EOF

COPY . .
# Editable install: configs/, dbt/ and eval/ are resolved relative to /app.
RUN pip install --no-deps -e . \
 && useradd --create-home --uid 1000 regrag \
 && mkdir -p /app/data \
 && chown -R regrag:regrag /app /opt/models

USER regrag
EXPOSE 8000
VOLUME ["/app/data"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

CMD ["regrag", "serve"]
