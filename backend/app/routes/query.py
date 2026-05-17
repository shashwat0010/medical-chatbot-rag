import logging

from fastapi import APIRouter, HTTPException, Request
from groq import RateLimitError as GroqRateLimitError, AuthenticationError as GroqAuthenticationError
from openai import APIConnectionError, AuthenticationError as OpenAIAuthenticationError, RateLimitError as OpenAIRateLimitError

from app.config import get_settings
from app.limiter import limiter
from models.schemas import QueryRequest, QueryResponse
from rag.pipeline import RAGPipeline
from services.embeddings import EmbeddingServiceError, OpenAIQuotaError
from services.guardrails import DISCLAIMER, check_query_safety, is_greeting_or_meta

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Query"])
_rate_limit = f"{get_settings().rate_limit_per_minute}/minute"

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _llm_quota_message() -> str:
    return (
        "LLM API quota exceeded. Please check your Groq or OpenAI billing status. "
        "You can also set USE_LOCAL_EMBEDDINGS=true in backend/.env to use free local search."
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit(_rate_limit)
async def query_medical_research(
    request: Request,
    body: QueryRequest,
) -> QueryResponse:
    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="Groq API key is not configured. Set GROQ_API_KEY in the environment.",
        )

    safety = check_query_safety(body.query, settings.block_emergency_keywords)
    if not safety.allowed:
        # If it's too short, but it's a greeting, we might want to allow it
        if not is_greeting_or_meta(body.query):
            raise HTTPException(status_code=400, detail=safety.message)

    logger.info("Query from %s: %s", request.client.host if request.client else "unknown", body.query[:100])

    try:
        result = await get_pipeline().run(body.query, max_papers=body.max_papers)
        if not result.confidence_note.startswith("Low confidence"):
            result.confidence_note = f"{result.confidence_note} {DISCLAIMER}"
        return result
    except OpenAIQuotaError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except (OpenAIRateLimitError, GroqRateLimitError) as exc:
        if "quota" in str(exc).lower():
            raise HTTPException(status_code=402, detail=_llm_quota_message()) from exc
        raise HTTPException(
            status_code=429,
            detail="API rate limit reached. Please wait a moment and try again.",
        ) from exc
    except (OpenAIAuthenticationError, GroqAuthenticationError) as exc:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Check your configuration in backend/.env",
        ) from exc
    except (EmbeddingServiceError, APIConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.exception("Configuration error during query")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your research query. Please try again.",
        ) from exc
