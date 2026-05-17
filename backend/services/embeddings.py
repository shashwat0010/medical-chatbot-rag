import logging
from typing import List

from langchain_openai import OpenAIEmbeddings
from openai import APIConnectionError, AuthenticationError, RateLimitError

from app.config import get_settings
from services.local_embeddings import fit_tfidf, tfidf_embed, transform_tfidf

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embeddings cannot be computed."""


class OpenAIQuotaError(EmbeddingServiceError):
    pass


def get_embedding_model() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


async def embed_texts(texts: List[str], *, allow_local_fallback: bool = True) -> List[List[float]]:
    if not texts:
        return []

    settings = get_settings()
    if settings.use_local_embeddings:
        logger.info("Using local TF-IDF embeddings (USE_LOCAL_EMBEDDINGS=true)")
        return fit_tfidf(texts)

    try:
        model = get_embedding_model()
        return await model.aembed_documents(texts)
    except RateLimitError as exc:
        err_text = str(exc).lower()
        if "insufficient_quota" in err_text or "quota" in err_text:
            logger.error("OpenAI embedding quota exceeded")
            if allow_local_fallback:
                logger.warning("Falling back to local TF-IDF embeddings")
                return fit_tfidf(texts)
            raise OpenAIQuotaError(
                "OpenAI API quota exceeded. Add billing at https://platform.openai.com/account/billing "
                "or set USE_LOCAL_EMBEDDINGS=true in backend/.env"
            ) from exc
        raise EmbeddingServiceError(f"OpenAI rate limit: {exc}") from exc
    except AuthenticationError as exc:
        raise EmbeddingServiceError(
            "Invalid OpenAI API key. Check OPENAI_API_KEY in backend/.env"
        ) from exc
    except APIConnectionError as exc:
        if allow_local_fallback:
            logger.warning("OpenAI unreachable; using local TF-IDF embeddings: %s", exc)
            return fit_tfidf(texts)
        raise EmbeddingServiceError(f"Cannot reach OpenAI API: {exc}") from exc


async def embed_query(text: str) -> List[List[float]]:
    """Embed a query using the same vocabulary as the last fit_tfidf / document batch."""
    settings = get_settings()
    if settings.use_local_embeddings:
        return transform_tfidf([text])
    try:
        model = get_embedding_model()
        return await model.aembed_documents([text])
    except RateLimitError:
        return transform_tfidf([text])
    except APIConnectionError:
        return transform_tfidf([text])
