"""Document parsing: PDF/HTML bytes -> pages of cleaned text + tables + section headings.

Pure functions over bytes so the same code runs locally, in Prefect, or inside a
Spark `mapInPandas` on Databricks.
"""
from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass, field
from html.parser import HTMLParser


class ParseError(Exception):
    """Raised for files we can't parse (encrypted, corrupt, unsupported). Caller quarantines."""


@dataclass
class Page:
    page_no: int
    text: str
    tables: list[str] = field(default_factory=list)    # markdown tables
    headings: list[tuple[int, str]] = field(default_factory=list)  # (line_idx, heading)


@dataclass
class ParsedDoc:
    pages: list[Page]
    parser: str
    n_pages_total: int

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)


# --------------------------------------------------------------------------- headings
_NUMBERED = re.compile(r"^(?P<num>(?:\d{1,2}|[A-Z]|[IVX]{1,4})(?:\.\d{1,2}){0,3})\.?\s+(?P<title>\S.{1,90})$")
_SMALL_WORDS = {"and", "or", "of", "the", "to", "for", "in", "on", "a", "an", "by", "with", "at", "from", "vs"}


def _is_titleish(text: str) -> bool:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)]
    if not words or len(words) > 12:
        return False
    if text.rstrip()[-1:] in ".,;" or text.count(",") > 2:
        return False
    caps = sum(1 for w in words if w.isupper() and len(w) > 1)
    if caps / len(words) >= 0.7:
        return True
    titled = sum(1 for w in words if w[0].isupper() or w.lower() in _SMALL_WORDS)
    return titled / len(words) >= 0.8 and words[0][0].isupper()


# Unnumbered ALL-CAPS lines are only headings if they contain a structural word; this keeps
# signature blocks ("JANE DOE"), letterhead and routing lines ("BY EMAIL") out of sections.
HEADING_WORDS = {
    "decision", "decisions", "order", "orders", "introduction", "summary", "background", "findings", "finding",
    "analysis", "process", "application", "applications", "submission", "submissions", "costs", "cost",
    "implementation", "schedule", "appendix", "reasons", "issues", "issue", "conclusion", "conclusions",
    "rate", "rates", "evidence", "overview", "disposition", "procedural", "load", "forecast", "revenue",
    "capital", "operating", "deferral", "variance", "accounts", "account", "settlement", "interrogatories",
    "questions", "question", "notice", "licence", "license", "purpose", "recommendations", "attachment",
    "exhibit", "part", "section", "chapter", "policy", "proposal", "relief", "request", "response",
    "consultation", "approval", "approvals", "requirement", "requirements", "plan", "effective", "date",
    "discussion", "matters", "conditions", "contents", "table", "glossary", "definitions", "scope",
    "determination", "determinations", "ruling", "hearing", "motion", "review", "framework", "report",
}
_NOISE = re.compile(r"[~@]|\bpage\s+\d|\b(?:tel|fax|t|f)[.:]?\s*\d{3}|\d{3}[.\-\s]\d{3}[.\-\s]\d{4}|"
                    r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b|www\.|\.com\b", re.IGNORECASE)


def detect_heading(line: str) -> str | None:
    s = line.strip()
    if len(s) < 4 or len(s) > 100 or _NOISE.search(s):
        return None
    m = _NUMBERED.match(s)
    if m:
        title = m.group("title")
        # Letter/roman numbering ("A.", "IV.") is ambiguous with initials; demand an all-caps title.
        strict = not m.group("num")[0].isdigit()
        letters = [c for c in title if c.isalpha()]
        if strict and not (letters and all(c.isupper() for c in letters)):
            return None
        return s if _is_titleish(title) else None
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 6 and all(c.isupper() for c in letters) and len(s.split()) <= 10 \
            and not re.search(r"\d{4}", s):
        words = {w.lower() for w in re.findall(r"[A-Za-z]+", s)}
        if words & HEADING_WORDS:
            return s
    return None


# --------------------------------------------------------------------------- cleaning
_WS = re.compile(r"[ \t ]+")
_DIGITS = re.compile(r"\d+")


def _norm_line(line: str) -> str:
    return _DIGITS.sub("#", _WS.sub(" ", line.strip().lower()))


def strip_running_lines(pages_lines: list[list[str]], edge: int = 3, min_ratio: float = 0.5) -> list[list[str]]:
    """Drop header/footer lines: lines in the first/last `edge` lines of a page that
    (after replacing digits, so page numbers match) recur on >= min_ratio of pages."""
    if len(pages_lines) < 3:
        return pages_lines
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        edges = {_norm_line(l) for l in lines[:edge] + lines[-edge:] if l.strip()}
        counts.update(edges)
    threshold = max(2, int(len(pages_lines) * min_ratio))
    running = {k for k, c in counts.items() if c >= threshold}
    out = []
    for lines in pages_lines:
        n = len(lines)
        out.append([l for i, l in enumerate(lines)
                    if not ((i < edge or i >= n - edge) and _norm_line(l) in running)])
    return out


def clean_text(text: str) -> str:
    text = text.replace("­", "")                       # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # de-hyphenate line breaks
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_markdown(rows: list[list[str | None]]) -> str | None:
    rows = [[_WS.sub(" ", (c or "").replace("\n", " ")).strip() for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# --------------------------------------------------------------------------- PDF
def parse_pdf(data: bytes, max_pages: int = 400, extract_tables: bool = True) -> ParsedDoc:
    import pdfplumber
    from pdfminer.pdfdocument import PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFSyntaxError

    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except (PDFSyntaxError, PDFPasswordIncorrect) as exc:
        raise ParseError(f"unreadable pdf: {exc!r}") from exc
    except Exception as exc:  # pdfminer raises a zoo of exception types on corrupt files
        raise ParseError(f"pdf open failed: {exc!r}") from exc

    with pdf:
        total = len(pdf.pages)
        raw_pages: list[tuple[list[str], list[str]]] = []
        for page in pdf.pages[:max_pages]:
            tables_md: list[str] = []
            text_page = page
            if extract_tables:
                try:
                    found = page.find_tables()
                except Exception:  # noqa: BLE001 - table detection is best-effort
                    found = []
                bboxes = []
                for t in found:
                    md = table_to_markdown(t.extract())
                    if md:
                        tables_md.append(md)
                        bboxes.append(t.bbox)
                if bboxes:
                    # Remove table regions from the running text so table content isn't duplicated.
                    def outside(obj, _bb=bboxes):
                        cx = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
                        cy = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                        return not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in _bb)
                    text_page = page.filter(outside)
            try:
                txt = text_page.extract_text() or ""
            except Exception as exc:
                raise ParseError(f"text extraction failed on page {page.page_number}: {exc!r}") from exc
            raw_pages.append((txt.splitlines(), tables_md))

    stripped = strip_running_lines([lines for lines, _ in raw_pages])
    pages = []
    for i, (lines, (_, tables_md)) in enumerate(zip(stripped, raw_pages), start=1):
        text = clean_text("\n".join(lines))
        headings = [(j, h) for j, l in enumerate(text.splitlines()) if (h := detect_heading(l))]
        pages.append(Page(page_no=i, text=text, tables=tables_md, headings=headings))
    return ParsedDoc(pages=pages, parser="pdfplumber", n_pages_total=total)


# --------------------------------------------------------------------------- HTML (EDGAR)
class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def parse_html(data: bytes, chars_per_page: int = 4000) -> ParsedDoc:
    p = _HTMLText()
    p.feed(data.decode("utf-8", errors="replace"))
    text = clean_text("".join(p.parts))
    # HTML has no pages; synthesize ~page-sized blocks on paragraph boundaries for citation.
    pages, buf, n = [], [], 0
    for para in text.split("\n"):
        buf.append(para)
        n += len(para)
        if n >= chars_per_page:
            pages.append("\n".join(buf))
            buf, n = [], 0
    if buf:
        pages.append("\n".join(buf))
    out = []
    for i, t in enumerate(pages, start=1):
        headings = [(j, h) for j, l in enumerate(t.splitlines()) if (h := detect_heading(l))]
        out.append(Page(page_no=i, text=t, headings=headings))
    return ParsedDoc(pages=out, parser="html", n_pages_total=len(out))


def parse_document(data: bytes, extension: str, max_pages: int = 400, extract_tables: bool = True) -> ParsedDoc:
    ext = (extension or "").lower()
    head = data[:1024].lstrip()
    if ext == "pdf":
        if not head.startswith(b"%PDF-"):
            raise ParseError("extension is pdf but bytes are not a PDF (likely an HTML error page)")
        return parse_pdf(data, max_pages=max_pages, extract_tables=extract_tables)
    if ext in ("htm", "html"):
        return parse_html(data)
    raise ParseError(f"unsupported extension {ext!r}")
