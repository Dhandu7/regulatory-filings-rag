from .base import Source, SourceRecord, find_docket
from .edgar import EdgarSource
from .oeb import OEBSource


def get_source(name: str, cfg: dict) -> Source:
    classes = {"oeb": OEBSource, "edgar": EdgarSource}
    if name not in classes:
        raise ValueError(f"unknown source {name!r}; expected one of {sorted(classes)}")
    return classes[name](cfg["sources"][name], cfg["http"])


__all__ = ["EdgarSource", "OEBSource", "Source", "SourceRecord", "find_docket", "get_source"]
