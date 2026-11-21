"""Shared test fixtures.

Tests use a temporary ChromaDB folder, a fake LLM and a tiny fake embedding
function, so they run offline and never need an API key.
"""

import hashlib

import pytest
from langchain_core.embeddings import Embeddings

from backend.config import get_settings


class FakeEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings (no model download needed)."""

    size = 64

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in text.lower().split():
            index = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture(autouse=True)
def test_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("COLLECTION_NAME", f"test_{tmp_path.name}")
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "20")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()
