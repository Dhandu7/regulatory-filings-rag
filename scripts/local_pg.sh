#!/usr/bin/env bash
# Project-local Postgres 18 + pgvector, without touching any system Postgres install.
# Builds pgvector from source into .vendor/ and points PG18's extension_control_path /
# dynamic_library_path at it (PG18 feature), so no files are written into Homebrew's prefix.
#   scripts/local_pg.sh up | down | status
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PG_BIN="${PG_BIN:-/opt/homebrew/opt/postgresql@18/bin}"
DATA="$ROOT/.pgdata"
PORT="${PGPORT_LOCAL:-5433}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"   # PG refuses to start "multithreaded" without a valid locale

build_pgvector() {
  find "$ROOT/.vendor/pgstage" \( -name 'vector.dylib' -o -name 'vector.so' \) 2>/dev/null | grep -q . && return
  mkdir -p "$ROOT/.vendor"
  [[ -d "$ROOT/.vendor/pgvector" ]] || git clone -q --depth 1 --branch v0.8.1 https://github.com/pgvector/pgvector.git "$ROOT/.vendor/pgvector"
  make -s -C "$ROOT/.vendor/pgvector" PG_CONFIG="$PG_BIN/pg_config"
  make -s -C "$ROOT/.vendor/pgvector" PG_CONFIG="$PG_BIN/pg_config" install DESTDIR="$ROOT/.vendor/pgstage"
}

case "${1:-status}" in
  up)
    build_pgvector
    if [[ ! -d "$DATA" ]]; then
      "$PG_BIN/initdb" -D "$DATA" -U postgres --auth=trust -E UTF8 >/dev/null
      SHARE="$(dirname "$(find "$ROOT/.vendor/pgstage" -name vector.control | head -1)")"
      LIB="$(dirname "$(find "$ROOT/.vendor/pgstage" -name 'vector.dylib' -o -name 'vector.so' | head -1)")"
      cat >> "$DATA/postgresql.conf" <<CONF
port = $PORT
unix_socket_directories = '/tmp'
extension_control_path = '\$system:$(dirname "$SHARE")'
dynamic_library_path = '\$libdir:$LIB'
CONF
    fi
    "$PG_BIN/pg_ctl" -D "$DATA" -l "$DATA/server.log" status >/dev/null 2>&1 || \
      "$PG_BIN/pg_ctl" -D "$DATA" -l "$DATA/server.log" -w start >/dev/null
    "$PG_BIN/psql" -h /tmp -p "$PORT" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname='utility_rag'" | grep -q 1 || \
      "$PG_BIN/psql" -h /tmp -p "$PORT" -U postgres -qc "CREATE DATABASE utility_rag"
    "$PG_BIN/psql" -h /tmp -p "$PORT" -U postgres -d utility_rag -qc "CREATE EXTENSION IF NOT EXISTS vector"
    echo "pgvector ready: postgresql://postgres@localhost:$PORT/utility_rag"
    ;;
  down) "$PG_BIN/pg_ctl" -D "$DATA" stop ;;
  status) "$PG_BIN/pg_ctl" -D "$DATA" status ;;
  *) echo "usage: $0 up|down|status" >&2; exit 2 ;;
esac
