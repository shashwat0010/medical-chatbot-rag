import logging
from typing import List

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy loads and caches the SentenceTransformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using the extremely lightweight and highly effective all-MiniLM-L6-v2
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info("Loading local SentenceTransformer model: %s", model_name)
            _model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.error("Failed to load SentenceTransformer: %s", exc)
            raise ImportError("sentence-transformers package not available or failed to load") from exc
    return _model


async def embed_texts_local(texts: List[str]) -> List[List[float]]:
    """Generates dense semantic embeddings for a list of documents/chunks locally."""
    if not texts:
        return []
    try:
        model = _get_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
    except Exception as exc:
        logger.warning("Local SentenceTransformer embedding failed: %s. Falling back to TF-IDF.", exc)
        from services.local_embeddings import fit_tfidf
        return fit_tfidf(texts)


async def embed_query_local(text: str) -> List[float]:
    """Generates a dense semantic embedding for a search query locally."""
    try:
        model = _get_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as exc:
        logger.warning("Local SentenceTransformer query embedding failed: %s. Falling back to TF-IDF.", exc)
        from services.local_embeddings import transform_tfidf
        res = transform_tfidf([text])
        return res[0]
