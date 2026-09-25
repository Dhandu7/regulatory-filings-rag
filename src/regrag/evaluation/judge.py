"""LLM judge for answer correctness.

The strict metric checks whether an answer contains one of the golden answer phrases, which
under-counts correct paraphrases ("two years" vs "two year term"). The judge reads the question,
the golden facts, and the answer, and decides whether the answer states the fact. Both numbers
are reported side by side; the judge never replaces the strict score.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

JUDGE_PROMPT = """You grade answers from a question-answering system over Ontario Energy Board filings.

Question: {question}

Reference answer (the key fact; the answer may word it differently): {reference}

Answer to grade:
<answer>
{answer}
</answer>

The answer is correct if it states the reference fact, allowing paraphrase, different number or date
formats, and extra correct detail. It is incorrect if it omits the fact, contradicts it, or says the
information is not available in the sources."""

SCHEMA = {
    "type": "object",
    "properties": {"correct": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["correct", "reason"],
    "additionalProperties": False,
}


class Judge:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def grade(self, question: str, reference: list[str], answer: str) -> dict:
        import anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=question, reference=" / ".join(reference), answer=answer)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except anthropic.APIStatusError as exc:
            log.error("judge API error %s: %s", exc.status_code, exc.message)
            return {"correct": None, "reason": f"judge error {exc.status_code}"}
        except anthropic.APIConnectionError:
            return {"correct": None, "reason": "judge connection error"}
        if resp.stop_reason == "refusal":
            return {"correct": None, "reason": "judge declined"}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"correct": None, "reason": f"unparseable judge output: {text[:200]}"}
