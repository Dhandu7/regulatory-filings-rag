"""Answer generation with Claude via the official Anthropic SDK.

Prompt caching: the frozen system prompt carries a cache_control breakpoint, so every
request after the first reads it from cache (visible as usage.cache_read_input_tokens).
The per-question SOURCES + QUESTION come after the breakpoint and are never cached.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

from .prompts import SYSTEM_PROMPT, USER_TEMPLATE

log = logging.getLogger(__name__)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Generation:
    text: str
    model: str
    cited: list[int]
    usage: dict = field(default_factory=dict)
    stop_reason: str | None = None


def has_credentials() -> bool:
    """Best-effort check so the API can degrade to extractive mode instead of erroring."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return os.path.exists(os.path.expanduser("~/.config/anthropic"))


def parse_citations(text: str, n_sources: int) -> list[int]:
    nums = {int(n) for grp in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for n in re.split(r"\s*,\s*", grp)}
    return sorted(n for n in nums if 1 <= n <= n_sources)


class ClaudeAnswerer:
    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 4000):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def generate(self, question: str, sources_block: str, n_sources: int) -> Generation:
        import anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user",
                           "content": USER_TEMPLATE.format(sources=sources_block, question=question)}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError:
            log.warning("Claude rate limited")
            raise
        except anthropic.APIStatusError as exc:
            log.error("Claude API error %s: %s", exc.status_code, exc.message)
            raise
        except anthropic.APIConnectionError:
            log.error("Claude connection error")
            raise

        if resp.stop_reason == "refusal":
            cat = resp.stop_details.category if resp.stop_details else None
            return Generation(f"The model declined to answer (category: {cat}).", resp.model, [],
                              _usage(resp), resp.stop_reason)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return Generation(text, resp.model, parse_citations(text, n_sources), _usage(resp), resp.stop_reason)


def _usage(resp) -> dict:
    u = resp.usage
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
