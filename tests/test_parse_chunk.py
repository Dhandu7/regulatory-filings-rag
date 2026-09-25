import dataclasses

import pytest

from regrag.config import ChunkingConfig
from regrag.silver.chunk import chunk_id, chunk_pages
from regrag.silver.parse import Page, ParseError, detect_heading, parse_document, strip_running_lines
from regrag.sources import find_docket


def test_parse_pdf_strips_running_header_and_finds_sections(decision_pdf):
    doc = parse_document(decision_pdf, "pdf")
    assert len(doc.pages) == 3
    text = doc.full_text
    assert "Ontario Energy Board EB-2026-0015" not in text          # running header removed
    assert "Decision and Order page" not in text                    # running footer removed
    headings = [h for p in doc.pages for _, h in p.headings]
    assert "1 INTRODUCTION" in headings and "2 OEB FINDINGS" in headings
    assert "$41.9 million" in text


def test_parse_pdf_extracts_tables_as_markdown(decision_pdf):
    doc = parse_document(decision_pdf, "pdf")
    tables = doc.pages[0].tables
    assert tables, "expected the account table on page 1"
    assert "| Account | Balance |" in tables[0] and "1588" in tables[0]
    assert "$120,000" not in doc.pages[0].text                      # table text not duplicated in body


def test_html_saved_as_pdf_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_document(b"<html>error</html>", "pdf")


@pytest.mark.parametrize("line,expected", [
    ("1 INTRODUCTION AND SUMMARY", True), ("3.2 Load Forecast", True), ("DECISION AND ORDER", True),
    ("GAIL REGAN", False), ("BY EMAIL", False), ("T 416.926.1907 F 416.926.1601", False),
    ("1. The application is granted on such conditions as are contained in the attached", False),
])
def test_heading_detection(line, expected):
    assert (detect_heading(line) is not None) is expected


def test_strip_running_lines_keeps_body():
    bodies = ["alpha findings", "beta evidence", "gamma submissions", "delta order"]
    pages = [["Header X", b, f"Page {i}"] for i, b in enumerate(bodies, 1)]
    assert strip_running_lines(pages) == [[b] for b in bodies]


def test_find_docket():
    assert find_docket("Re: EB-2025-0009 Sioux Lookout") == "EB-2025-0009"
    assert find_docket("no docket here") is None


@pytest.fixture
def ccfg():
    return ChunkingConfig.load("configs/chunking/v1.yaml")


def _pages(n_sections=3, lines_per=40):
    pages = []
    for s in range(n_sections):
        lines = [f"{s + 1} SECTION FINDINGS {s + 1}"] + [
            f"Sentence {j} of section {s + 1} discusses the revenue requirement and rate base in detail."
            for j in range(lines_per)]
        pages.append(Page(page_no=s + 1, text="\n".join(lines), headings=[(0, lines[0])]))
    return pages


def test_chunks_are_deterministic_and_bounded(ccfg):
    a, b = chunk_pages(_pages(), ccfg), chunk_pages(_pages(), ccfg)
    assert [c.text for c in a] == [c.text for c in b]
    assert all(c.n_tokens <= ccfg.max_tokens for c in a)
    assert chunk_id("doc", ccfg.version, 0) == chunk_id("doc", ccfg.version, 0)


def test_chunks_respect_sections_and_have_no_overlap_only_chunks(ccfg):
    chunks = chunk_pages(_pages(), ccfg)
    for c in chunks:
        # a chunk never spans two sections
        assert len({l.split(" of section ")[-1].split()[0] for l in c.text.splitlines() if " of section " in l}) <= 1
        assert c.n_tokens >= ccfg.min_tokens
    texts = [c.text for c in chunks]
    assert len(texts) == len(set(texts))


def test_config_version_changes_with_config(ccfg, tmp_path):
    raw = {f.name: getattr(ccfg, f.name) for f in dataclasses.fields(ccfg) if f.name != "version"}
    raw["max_tokens"] = 200
    p = tmp_path / "v2.yaml"
    import yaml
    p.write_text(yaml.safe_dump(raw))
    assert ChunkingConfig.load(str(p)).version != ccfg.version


def test_oversized_table_row_is_split(ccfg):
    long_cell = " ".join(["The program renews underground cable in residential subdivisions."] * 150)
    md = "| Field | Value |\n|---|---|\n| Description | " + long_cell + " |"
    chunks = chunk_pages([Page(page_no=1, text="Intro line", tables=[md])], ccfg)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert len(tables) > 1
    assert all(c.n_tokens <= ccfg.max_table_tokens for c in tables)
    assert all(c.text.startswith("| Field | Value |") for c in tables)


def test_table_with_huge_header_is_split(ccfg):
    header = "| " + " | ".join(f"Year {y} forecast" for y in range(2000, 2300)) + " |"
    md = header + "\n|" + "---|" * 300 + "\n| a | b |"
    chunks = chunk_pages([Page(page_no=1, text="x", tables=[md])], ccfg)
    assert all(c.n_tokens <= ccfg.max_table_tokens for c in chunks if c.chunk_type == "table")
