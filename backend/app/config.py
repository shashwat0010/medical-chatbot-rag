from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

# Always load backend/.env regardless of shell cwd when starting uvicorn
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    pubmed_max_results: int = 15
    pubmed_retrieval_top_k: int = 8
    min_relevance_score: float = 0.35
    min_evidence_chunks: int = 3

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    rate_limit_per_minute: int = 20
    log_level: str = "INFO"

    block_emergency_keywords: bool = True
    min_confidence_threshold: float = 0.4

    # Use TF-IDF instead of OpenAI embeddings (no API cost; lower semantic quality)
    use_local_embeddings: bool = False

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
