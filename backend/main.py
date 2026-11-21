"""FastAPI backend for the RAG-Based AI Chatbot.

Run with:  uvicorn backend.main:app --reload
"""

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from backend.config import get_settings
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    MessageResponse,
    Source,
    UploadResponse,
)
from backend.services.document_loader import DocumentValidationError
from backend.services.llm import LLMConfigurationError
from backend.services.rag_pipeline import NoDocumentsError, RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-chatbot")

app = FastAPI(
    title="RAG-Based AI Chatbot",
    description="Upload PDF/TXT documents and ask questions about them.",
    version="1.0.0",
)


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Create the RAG pipeline once and reuse it for every request."""
    return RAGPipeline()


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), pipeline: RAGPipeline = Depends(get_pipeline)
):
    settings = get_settings()
    # Keep only the base name to avoid path tricks like "../../etc/passwd".
    filename = Path(file.filename or "").name
    # Read at most one byte over the limit so huge uploads can't exhaust memory.
    content = await file.read(settings.max_file_size_bytes + 1)

    try:
        chunks, characters = pipeline.ingest(filename, content)
    except DocumentValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed to process %s", filename)
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}")

    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.upload_path / filename).write_bytes(content)

    return UploadResponse(
        filename=filename,
        chunks=chunks,
        characters=characters,
        message=f"'{filename}' processed into {chunks} chunks.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        answer, chunks = pipeline.answer(question, top_k=request.top_k)
    except NoDocumentsError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")

    sources = [
        Source(
            filename=c.metadata.get("filename", "unknown"),
            chunk_index=c.metadata.get("chunk_index", 0),
            page=c.metadata.get("page"),
            content=c.page_content,
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    counts = pipeline.vector_store.list_documents()
    docs = [DocumentInfo(filename=name, chunks=n) for name, n in sorted(counts.items())]
    return DocumentListResponse(documents=docs, total_chunks=sum(counts.values()))


@app.delete("/documents/{filename}", response_model=MessageResponse)
def delete_document(filename: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    removed = pipeline.vector_store.delete_document(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    (get_settings().upload_path / Path(filename).name).unlink(missing_ok=True)
    return MessageResponse(message=f"Deleted '{filename}' ({removed} chunks).")


@app.delete("/documents", response_model=MessageResponse)
def clear_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    pipeline.vector_store.clear()
    return MessageResponse(message="All documents deleted.")
