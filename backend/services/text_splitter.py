"""Split document text into smaller overlapping chunks."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import get_settings


def split_into_chunks(
    filename: str, sections: list[tuple[int | None, str]]
) -> list[Document]:
    """Turn (page, text) sections into LangChain Documents with metadata."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for page, text in sections:
        for piece in splitter.split_text(text):
            metadata = {"filename": filename, "chunk_index": len(chunks)}
            if page is not None:
                metadata["page"] = page
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks
