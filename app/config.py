"""Application settings, loaded from environment variables / the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    llm_provider: str = "gemini"  # gemini | openai | fake
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-lite-latest"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 10
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt")

    # Retrieval
    top_k: int = 4

    # Storage
    chroma_dir: str = "data/chroma_db"
    collection_name: str = "documents"
    upload_dir: str = "data/uploads"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def chroma_path(self) -> Path:
        return _resolve(self.chroma_dir)

    @property
    def upload_path(self) -> Path:
        return _resolve(self.upload_dir)


def _resolve(path: str) -> Path:
    """Relative paths are resolved from the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
