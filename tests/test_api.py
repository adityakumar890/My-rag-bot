import pytest
from fastapi.testclient import TestClient

from app.main import app, get_pipeline
from app.chatbot import RAGPipeline
from app.vector_store import VectorStore


@pytest.fixture
def client(fake_embeddings):
    pipeline = RAGPipeline(vector_store=VectorStore(embeddings=fake_embeddings))
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name, content):
    return client.post("/upload", files={"file": (name, content, "text/plain")})


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_and_list_documents(client):
    response = upload(client, "cats.txt", b"Cats are small domesticated animals. " * 20)
    assert response.status_code == 200
    assert response.json()["chunks"] > 0

    docs = client.get("/documents").json()
    assert docs["documents"][0]["filename"] == "cats.txt"
    assert docs["total_chunks"] == response.json()["chunks"]


def test_upload_invalid_file_type(client):
    response = upload(client, "virus.exe", b"MZ...")
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_chat_without_documents(client):
    response = client.post("/chat", json={"question": "Hello?"})
    assert response.status_code == 400
    assert "No documents" in response.json()["detail"]


def test_chat_returns_answer_and_relevant_sources(client):
    upload(client, "cats.txt", b"Cats are small domesticated animals that purr.")
    upload(client, "space.txt", b"Mars is the fourth planet from the Sun.")

    response = client.post("/chat", json={"question": "Which planet is fourth from the Sun?", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"]
    assert data["sources"][0]["filename"] == "space.txt"


def test_chat_rejects_empty_question(client):
    assert client.post("/chat", json={"question": ""}).status_code == 422
    assert client.post("/chat", json={"question": "   "}).status_code == 400


def test_delete_document(client):
    upload(client, "cats.txt", b"Cats purr.")
    assert client.delete("/documents/cats.txt").status_code == 200
    assert client.delete("/documents/cats.txt").status_code == 404
    assert client.get("/documents").json()["documents"] == []


def test_delete_document_with_special_characters(client):
    upload(client, "a#b?c.txt", b"Special name.")
    assert client.delete("/documents/a%23b%3Fc.txt").status_code == 200
    assert client.get("/documents").json()["documents"] == []
