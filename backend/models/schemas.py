"""Request and response models used by the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    filename: str
    chunk_index: int
    page: int | None = None
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    characters: int
    message: str


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class MessageResponse(BaseModel):
    message: str
