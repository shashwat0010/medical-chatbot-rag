import logging
from typing import List, Optional

from models.schemas import Citation, QueryResponse

logger = logging.getLogger(__name__)

FALLBACK_PHRASE = "Limited direct evidence found; related literature suggests possible associations."


def generate_response(
    answer: str,
    citations: List[Citation],
    confidence_score: float,
    insufficient_evidence: bool,
    confidence_note: str,
    sources_searched: Optional[List[str]] = None,
) -> QueryResponse:
    """Consistently maps retrieval results and confidence into a QueryResponse with clean fallback and labels."""
    if sources_searched is None:
        sources_searched = ["PubMed"]

    # Normalize response to fallback if insufficient evidence is flagged or indicated in answer
    lower_ans = answer.lower()
    if (
        insufficient_evidence
        or "current evidence is insufficient" in lower_ans
        or "no relevant papers were retrieved" in lower_ans
        or "no papers found" in lower_ans
    ):
        answer = FALLBACK_PHRASE
        insufficient_evidence = True
        confidence_label = "Limited evidence"
    else:
        # Determine confidence label based on calibrated score
        if confidence_score >= 0.72:
            confidence_label = "Strong evidence"
        elif confidence_score >= 0.55:
            confidence_label = "Moderate evidence"
        else:
            confidence_label = "Limited evidence"

    # Remove any existing duplicate prefixes from confidence note
    clean_note = confidence_note
    prefixes_to_strip = [
        "Strong evidence:",
        "Moderate evidence:",
        "Limited evidence:",
        "[Strong Evidence]",
        "[Moderate Evidence]",
        "[Limited Evidence]",
    ]
    for prefix in prefixes_to_strip:
        if clean_note.startswith(prefix):
            clean_note = clean_note[len(prefix):].strip()
        clean_note = clean_note.replace(prefix, "").strip()

    # Prefix the confidence note with the human-readable label
    formatted_note = f"{confidence_label}: {clean_note}"

    logger.info(
        "Response generated: label=%s, score=%.3f, insufficient=%s",
        confidence_label,
        confidence_score,
        insufficient_evidence,
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        confidence_note=formatted_note,
        confidence_score=confidence_score,
        insufficient_evidence=insufficient_evidence,
        sources_searched=sources_searched,
        confidence_label=confidence_label,
    )
