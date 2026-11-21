"""Vector storage and similarity search with ChromaDB."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_settings
from app.embeddings import get_embeddings


class VectorStore:
    """Small wrapper around a persistent Chroma collection."""

    def __init__(self, embeddings=None):
        settings = get_settings()
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self.db = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings or get_embeddings(),
            persist_directory=str(settings.chroma_path),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, filename: str, chunks: list[Document]) -> int:
        """Store chunks. Re-uploading a file replaces its old chunks.

        New chunks are written first (upsert), then leftover old chunks are
        removed, so a failed upload never erases the previous version.
        """
        old_ids = set(self.db.get(where={"filename": filename}, include=[])["ids"])
        ids = [f"{filename}::{c.metadata['chunk_index']}" for c in chunks]
        self.db.add_documents(chunks, ids=ids)
        stale = list(old_ids - set(ids))
        if stale:
            self.db.delete(ids=stale)
        return len(chunks)

    def search(self, query: str, k: int) -> list[Document]:
        return self.db.similarity_search(query, k=k)

    def list_documents(self) -> dict[str, int]:
        """Return {filename: number_of_chunks}."""
        data = self.db.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            name = (meta or {}).get("filename", "unknown")
            counts[name] = counts.get(name, 0) + 1
        return counts

    def count(self) -> int:
        return len(self.db.get(include=[])["ids"])

    def delete_document(self, filename: str) -> int:
        existing = self.db.get(where={"filename": filename}, include=[])["ids"]
        if existing:
            self.db.delete(ids=existing)
        return len(existing)

    def clear(self) -> None:
        ids = self.db.get(include=[])["ids"]
        if ids:
            self.db.delete(ids=ids)
