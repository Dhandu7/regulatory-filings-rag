"""FastAPI service: POST /ask returns an answer with numbered, linked citations."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import load_config
from .chain import get_qa
from .console import router as console_router

app = FastAPI(title="Regulatory Filings QA", version="0.1.0")
app.include_router(console_router)
CONSOLE_HTML = Path(__file__).parent / "static" / "console.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    docket: str | None = Field(default=None, description="restrict to one case, e.g. EB-2025-0295")
    source: str | None = None
    date_from: str | None = Field(default=None, description="ISO date")
    date_to: str | None = None
    use_llm: bool = True
    retrieval: Literal["dense", "hybrid", "rerank"] | None = Field(default=None,
                                                                   description="default: config serve.retrieval")
    expand_window: int | None = Field(default=None, ge=0, le=3, description="neighbor chunks added per hit")


@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    """Demo console: ask questions and run every pipeline command from the browser."""
    return FileResponse(CONSOLE_HTML, headers={"Cache-Control": "no-store"})


@app.get("/health")
def health() -> dict:
    qa = get_qa(load_config())
    return {"status": "ok", "index_version": qa.store.active_version(), "llm": qa.model if qa.llm else None}


@app.get("/versions")
def versions() -> list[dict]:
    return get_qa(load_config()).store.versions()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    qa = get_qa(load_config())
    try:
        return qa.ask(req.question, k=req.k, use_llm=req.use_llm, retrieval=req.retrieval,
                      expand_window=req.expand_window,
                      filters={"docket": req.docket, "source": req.source,
                               "date_from": req.date_from, "date_to": req.date_to})
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
