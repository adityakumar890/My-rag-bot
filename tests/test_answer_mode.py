import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.chatbot import GENERAL_NOTE, LLMConfigurationError, NoDocumentsError, RAGPipeline
from app.config import get_settings
from app.vector_store import VectorStore


def make_pipeline(fake_embeddings, reply):
    return RAGPipeline(
        vector_store=VectorStore(embeddings=fake_embeddings),
        llm=FakeListChatModel(responses=[reply]),
    )


def set_mode(monkeypatch, mode):
    monkeypatch.setenv("ANSWER_MODE", mode)
    get_settings.cache_clear()


def test_documents_mode_requires_documents(fake_embeddings):
    pipeline = make_pipeline(fake_embeddings, "unused")
    with pytest.raises(NoDocumentsError):
        pipeline.answer("What is the capital of France?")


def test_hybrid_mode_answers_without_documents(fake_embeddings, monkeypatch):
    set_mode(monkeypatch, "hybrid")
    pipeline = make_pipeline(fake_embeddings, "Paris.")
    answer, sources = pipeline.answer("What is the capital of France?")
    assert answer == GENERAL_NOTE + "Paris."
    assert sources == []


def test_hybrid_mode_uses_documents_when_relevant(fake_embeddings, monkeypatch):
    set_mode(monkeypatch, "hybrid")
    pipeline = make_pipeline(fake_embeddings, "Mars is fourth.")
    pipeline.ingest("space.txt", b"Mars is the fourth planet from the Sun.")
    answer, sources = pipeline.answer("Which planet is fourth?")
    assert answer == "Mars is fourth."
    assert sources[0].metadata["filename"] == "space.txt"


def test_hybrid_mode_falls_back_to_general_knowledge(fake_embeddings, monkeypatch):
    set_mode(monkeypatch, "hybrid")
    pipeline = make_pipeline(fake_embeddings, "[GENERAL] Paris.")
    pipeline.ingest("space.txt", b"Mars is the fourth planet from the Sun.")
    answer, sources = pipeline.answer("What is the capital of France?")
    assert answer == GENERAL_NOTE + "Paris."
    assert sources == []


def test_unknown_answer_mode(fake_embeddings, monkeypatch):
    set_mode(monkeypatch, "everything")
    pipeline = make_pipeline(fake_embeddings, "unused")
    with pytest.raises(LLMConfigurationError, match="ANSWER_MODE"):
        pipeline.answer("Hi?")
