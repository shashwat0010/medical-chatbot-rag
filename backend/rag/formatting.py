"""Structured answer layout (bullets/sections, not prose paragraphs)."""

from typing import List, Optional


def format_structured_answer(
    *,
    summary: Optional[str] = None,
    key_findings: Optional[List[str]] = None,
    clinical_notes: Optional[List[str]] = None,
) -> str:
    sections: List[str] = []

    if summary and summary.strip():
        sections.append(f"**Summary:** {summary.strip()}")

    findings = [f.strip() for f in (key_findings or []) if f and f.strip()]
    if findings:
        sections.append("**Key findings:**")
        sections.extend(f"- {item}" for item in findings)

    notes = [n.strip() for n in (clinical_notes or []) if n and n.strip()]
    if notes:
        sections.append("**Clinical notes:**")
        sections.extend(f"- {item}" for item in notes)

    return "\n".join(sections)


def paragraph_to_bullets(text: str, max_bullets: int = 5) -> str:
    """Fallback: split a paragraph into bullet lines for legacy/plain answers."""
    import re

    text = text.strip()
    if not text:
        return text
    if "\n- " in text or text.startswith("- ") or text.startswith("**"):
        return text

    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]
    if len(sentences) <= 1:
        return format_structured_answer(key_findings=[text])

    return format_structured_answer(key_findings=sentences[:max_bullets])
