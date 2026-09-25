# PYTHONPATH=src is belt-and-braces: on macOS, Python 3.14 skips .pth files flagged "hidden",
# which some synced folders apply to the editable-install .pth.
PY := PYTHONPATH=src .venv/bin/python
REGRAG := $(PY) -m regrag.cli

.PHONY: install pg-up pg-down test test-all ingest silver dbt gold pipeline serve eval eval-holdout watch flow

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev,ops,dbt]"

pg-up:     ; scripts/local_pg.sh up
pg-down:   ; scripts/local_pg.sh down

test:      ; $(PY) -m pytest -m "not network"
test-all:  ; DATABASE_URL=postgresql://postgres@localhost:5433/utility_rag $(PY) -m pytest

ingest:    ; $(REGRAG) ingest --source oeb
silver:    ; $(REGRAG) silver
dbt:       ; $(REGRAG) dbt
gold:      ; $(REGRAG) gold
pipeline:  ; $(REGRAG) run --source oeb
serve:     ; $(REGRAG) serve
eval:      ; $(REGRAG) eval
eval-holdout:; $(REGRAG) eval --questions eval/questions_holdout.jsonl
watch:     ; $(REGRAG) watch --interval 900
flow:      ; $(PY) -m regrag.ops.flows serve
