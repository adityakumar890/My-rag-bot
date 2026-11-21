"""The RAG pipeline: ingest documents and answer questions about them."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.config import get_settings
from backend.services.document_loader import load_document
from backend.services.llm import get_llm
from backend.services.text_splitter import split_into_chunks
from backend.services.vector_store import VectorStore

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions using ONLY the "
            "context below, which comes from the user's uploaded documents.\n"
            "If the answer is not in the context, say you could not find it in "
            "the documents. Be concise and clear.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class NoDocumentsError(RuntimeError):
    """Raised when the user asks a question before uploading any documents."""


def format_context(chunks: list[Document]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        label = meta.get("filename", "unknown")
        if meta.get("page"):
            label += f", page {meta['page']}"
        parts.append(f"[{i}] ({label})\n{chunk.page_content}")
    return "\n\n".join(parts)


class RAGPipeline:
    def __init__(self, vector_store: VectorStore | None = None, llm=None):
        self.vector_store = vector_store or VectorStore()
        self._llm = llm

    @property
    def llm(self):
        # Created lazily so the app can start even if the API key is missing.
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def ingest(self, filename: str, content: bytes) -> tuple[int, int]:
        """Validate, extract, chunk, embed and store a file.

        Returns (number_of_chunks, number_of_characters).
        """
        sections = load_document(filename, content)
        chunks = split_into_chunks(filename, sections)
        self.vector_store.add_documents(filename, chunks)
        characters = sum(len(text) for _, text in sections)
        return len(chunks), characters

    def answer(self, question: str, top_k: int | None = None) -> tuple[str, list[Document]]:
        if self.vector_store.count() == 0:
            raise NoDocumentsError("No documents uploaded yet. Please upload a PDF or TXT file first.")

        k = top_k or get_settings().top_k
        chunks = self.vector_store.search(question, k=k)
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": format_context(chunks), "question": question})
        return answer.strip(), chunks
