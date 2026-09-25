"""Section-aware chunking of parsed pages, fully determined by a ChunkingConfig.

Chunk ids are sha256(doc_id | config version | ordinal), so re-running the same
config over the same documents reproduces identical ids and text.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from ..config import ChunkingConfig
from .parse import Page


@lru_cache(maxsize=4)
def _encoder(name: str):
    import tiktoken
    return tiktoken.get_encoding(name)


def count_tokens(text: str, tokenizer: str) -> int:
    return len(_encoder(tokenizer).encode(text, disallowed_special=()))


@dataclass
class Chunk:
    ordinal: int
    text: str
    section: str | None
    page_start: int
    page_end: int
    chunk_type: str      # text | table
    n_tokens: int


@dataclass
class _Line:
    text: str
    page: int
    tokens: int


def _split_long(text: str, max_tokens: int, tokenizer: str) -> list[str]:
    enc = _encoder(tokenizer)
    ids = enc.encode(text, disallowed_special=())
    return [enc.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def chunk_pages(pages: list[Page], cfg: ChunkingConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[_Line] = []
    section: str | None = None
    buf_section: str | None = None
    carry: list[_Line] = []   # tiny section fragments (e.g. a bare heading) carried into the next chunk
    n_overlap = 0             # leading lines of buf that were already emitted (overlap context only)

    def emit(lines: list[_Line], sec: str | None, ctype: str = "text") -> None:
        text = "\n".join(l.text for l in lines).strip()
        if not text:
            return
        chunks.append(Chunk(len(chunks), text, sec, lines[0].page, lines[-1].page, ctype,
                            count_tokens(text, cfg.tokenizer)))

    def flush(boundary: bool) -> None:
        nonlocal buf, carry, n_overlap
        if len(buf) == n_overlap and not carry:   # nothing new since last emit
            buf, n_overlap = [], 0
            return
        fresh = carry + buf[n_overlap:]
        if boundary and sum(l.tokens for l in fresh) < cfg.min_tokens:
            if n_overlap:
                # Tail of a section that already produced chunks: emit it with its overlap
                # context rather than leaking it into the next section.
                emit(buf, buf_section)
                buf, n_overlap = [], 0
            else:
                # A section with almost no body (e.g. a bare heading): carry it forward.
                carry, buf, n_overlap = fresh, [], 0
            return
        lines = carry + buf
        carry = []
        emit(lines, buf_section)
        # overlap: keep trailing lines up to overlap_tokens (only within the same section)
        keep, kept = [], 0
        if not boundary:
            for l in reversed(lines):
                if kept + l.tokens > cfg.overlap_tokens:
                    break
                keep.insert(0, l)
                kept += l.tokens
        buf, n_overlap = keep, len(keep)

    def add_line(text: str, page: int) -> None:
        nonlocal buf
        t = count_tokens(text, cfg.tokenizer) + 1   # +1 for the joining newline
        if t > cfg.max_tokens:
            for piece in _split_long(text, cfg.max_tokens - 1, cfg.tokenizer):
                add_line(piece, page)
            return
        if sum(l.tokens for l in carry + buf) + t > cfg.max_tokens:
            flush(boundary=False)
        buf.append(_Line(text, page, t))

    for page in pages:
        heading_at = dict(page.headings)
        for idx, line in enumerate(page.text.splitlines()):
            if not line.strip():
                continue
            if idx in heading_at:
                if cfg.respect_sections and heading_at[idx] != section:
                    flush(boundary=True)
                section = heading_at[idx]
                if not buf:
                    buf_section = section
            elif not buf and not carry:
                buf_section = section
            add_line(line, page.page_no)
        if cfg.tables_as_separate_chunks:
            for md in page.tables:
                _emit_table(md, page.page_no, section, cfg, chunks)
    flush(boundary=False)
    if carry:  # trailing fragment: attach to the last chunk rather than drop it
        text = "\n".join(l.text for l in carry)
        if chunks and chunks[-1].chunk_type == "text":
            last = chunks[-1]
            last.text += "\n" + text
            last.page_end = carry[-1].page
            last.n_tokens = count_tokens(last.text, cfg.tokenizer)
        else:
            emit(carry, section)
    # tables are appended interleaved; re-number in reading order
    for i, c in enumerate(chunks):
        c.ordinal = i
    return chunks


def _emit_table(md: str, page_no: int, section: str | None, cfg: ChunkingConfig, chunks: list[Chunk]) -> None:
    lines = md.splitlines()
    header, rows = lines[:2], lines[2:]
    head_tokens = count_tokens("\n".join(header), cfg.tokenizer)
    if head_tokens > cfg.max_table_tokens // 2:
        # Flattened spreadsheet: the "header" is itself huge, so repeating it is useless.
        for piece in _split_long(md, cfg.max_table_tokens, cfg.tokenizer):
            chunks.append(Chunk(len(chunks), piece, section, page_no, page_no, "table",
                                count_tokens(piece, cfg.tokenizer)))
        return
    part: list[str] = []
    part_tokens = head_tokens
    budget = max(cfg.max_table_tokens - head_tokens, 50)
    expanded: list[str] = []
    for row in rows:   # a single row with paragraph-sized cells can exceed the budget on its own
        if count_tokens(row, cfg.tokenizer) > budget:
            expanded += _split_long(row, budget, cfg.tokenizer)
        else:
            expanded.append(row)
    for row in expanded:
        rt = count_tokens(row, cfg.tokenizer)
        if part and part_tokens + rt > cfg.max_table_tokens:
            text = "\n".join(header + part)
            chunks.append(Chunk(len(chunks), text, section, page_no, page_no, "table", part_tokens))
            part, part_tokens = [], head_tokens
        part.append(row)
        part_tokens += rt
    if part:
        text = "\n".join(header + part)
        chunks.append(Chunk(len(chunks), text, section, page_no, page_no, "table", part_tokens))


def chunk_id(doc_id: str, config_version: str, ordinal: int) -> str:
    return hashlib.sha256(f"{doc_id}|{config_version}|{ordinal}".encode()).hexdigest()[:32]


def context_header(title: str | None, docket: str | None, section: str | None) -> str:
    return " | ".join(p for p in (title, docket, section) if p)
