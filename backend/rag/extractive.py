"""Structured extractive summary when the LLM is unavailable."""

import re
from typing import List, Tuple

from rag.formatting import format_structured_answer
from rag.vector_store import RetrievedChunk


def _snippet_from_chunk(chunk: RetrievedChunk, max_len: int = 280) -> str:
    text = chunk.text
    if "Journal:" in text:
        parts = text.split("\n", 3)
        text = parts[-1] if len(parts) >= 4 else text
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rsplit(" ", 1)[0] + "…"
    return text


def _finding_from_snippet(snippet: str, index: int) -> str:
    """Prefer outcome-style phrasing as a single bullet."""
    return f"{snippet} [{index}]"


def build_extractive_answer(
    query: str, chunks: List[RetrievedChunk], max_excerpts: int = 4
) -> Tuple[str, List[int]]:
    if not chunks:
        return (
            "Current evidence is insufficient to provide a reliable answer.",
            [],
        )

    findings: List[str] = []
    cited: List[int] = []
    for i, chunk in enumerate(chunks[:max_excerpts], start=1):
        snippet = _snippet_from_chunk(chunk)
        if not snippet:
            continue
        findings.append(_finding_from_snippet(snippet, i))
        cited.append(i)

    if not findings:
        return (
            "Current evidence is insufficient to provide a reliable answer.",
            [],
        )

    summary = (
        f"Evidence retrieved from PubMed for: {query[:120]}"
        + ("…" if len(query) > 120 else "")
    )
    answer = format_structured_answer(
        summary=summary,
        key_findings=findings,
        clinical_notes=[
            "Synthesis used abstract excerpts only (OpenAI unavailable). Review full papers before clinical decisions."
        ],
    )
    return answer, cited
