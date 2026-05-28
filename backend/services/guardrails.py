import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

EMERGENCY_PATTERNS = [
    r"\b(chest pain|heart attack|stroke|suicid|overdose|can't breathe|cannot breathe)\b",
    r"\b(severe bleeding|unconscious|anaphylaxis|cardiac arrest)\b",
    r"\b(911|emergency room now|dying)\b",
]

HIGH_RISK_PATTERNS = [
    r"\b(instead of chemotherapy|stop insulin|stop medication|replace chemotherapy|avoid professional treatment|home remedies instead of)\b",
    r"\b(treat chest pain at home|self-diagnosis|can i stop)\b"
]

DISCLAIMER = (
    "This tool provides research summaries for licensed clinicians and does not "
    "replace clinical judgment, diagnosis, or emergency care."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Current evidence is insufficient to provide a reliable answer."
)


@dataclass
class GuardrailResult:
    allowed: bool
    message: Optional[str] = None
    is_emergency: bool = False
    risk_level: str = "LOW"


GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening|hii|hii+)(\s.*)?$",
    r"^(who are you|what can you do|how can you help|help)$",
]

def is_greeting_or_meta(query: str) -> bool:
    normalized = query.strip().lower()
    for pattern in GREETING_PATTERNS:
        if re.match(pattern, normalized):
            return True
    return False

def check_query_safety(query: str, block_emergency: bool = True) -> GuardrailResult:
    normalized = query.strip().lower()
    if len(normalized) < 3:
        return GuardrailResult(
            allowed=False,
            message="Please enter a medical research question (at least 3 characters).",
        )

    if block_emergency:
        for pattern in EMERGENCY_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                logger.warning("Emergency-related query blocked")
                return GuardrailResult(
                    allowed=False,
                    is_emergency=True,
                    risk_level="EMERGENCY",
                    message=(
                        "This appears to describe an acute medical emergency. "
                        "Do not use this chatbot for urgent care. Seek immediate "
                        "emergency medical attention or call your local emergency number."
                    ),
                )
                
    for pattern in HIGH_RISK_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            logger.warning("High-risk medical intent detected")
            return GuardrailResult(
                allowed=True,
                risk_level="HIGH",
                message="High-risk medical intent detected."
            )

    treatment_only = re.search(
        r"^(should i take|what dose should i|prescribe me|treat my)\b",
        normalized,
    )
    if treatment_only:
        return GuardrailResult(
            allowed=False,
            message=(
                "This assistant summarizes published medical literature for clinicians. "
                "It cannot provide personal treatment or dosing advice. "
                "Please rephrase as a research question (e.g., efficacy of X in condition Y)."
            ),
        )

    return GuardrailResult(allowed=True)


def validate_answer_grounding(
    answer: str,
    llm_insufficient: bool,
    confidence_score: float,
) -> Tuple[str, bool, str]:
    """Only reject answers when the model explicitly flags insufficient evidence."""
    if llm_insufficient:
        return INSUFFICIENT_EVIDENCE_MESSAGE, True, (
            "The retrieved abstracts did not contain enough detail for a specific conclusion. "
            "Review cited papers directly."
        )
    note = _confidence_note(confidence_score)
    return answer, False, note


def _confidence_note(score: float) -> str:
    if score >= 0.72:
        return (
            "Good alignment between retrieved abstracts and your question. "
            "Verify against full-text sources and current guidelines."
        )
    if score >= 0.55:
        return (
            "Moderate confidence based on retrieved literature. "
            "Interpret findings cautiously and consider study quality."
        )
    return (
        "Evidence was retrieved but semantic match scores were modest. "
        "Review cited abstracts and consider additional literature search."
    )
