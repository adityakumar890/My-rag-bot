"""Validate uploaded files, extract their text and split it into chunks."""

import codecs
import io
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.config import get_settings


class DocumentValidationError(ValueError):
    """Raised when an uploaded document is invalid (wrong type, too big, empty...)."""


def validate_file(filename: str, content: bytes) -> str:
    """Check the file name, size and contents. Returns the lower-case extension."""
    settings = get_settings()

    if not filename:
        raise DocumentValidationError("File name is missing.")

    extension = Path(filename).suffix.lower()
    if extension not in settings.allowed_extensions:
        allowed = ", ".join(settings.allowed_extensions)
        raise DocumentValidationError(
            f"Unsupported file type '{extension or 'none'}'. Allowed types: {allowed}."
        )

    if len(content) == 0:
        raise DocumentValidationError("The uploaded file is empty.")

    if len(content) > settings.max_file_size_bytes:
        raise DocumentValidationError(
            f"File is too large. Maximum size is {settings.max_file_size_mb} MB."
        )

    if extension == ".pdf" and not content.startswith(b"%PDF"):
        raise DocumentValidationError("The file does not look like a valid PDF.")

    return extension


def extract_pdf_pages(content: bytes) -> list[tuple[int, str]]:
    """Return a list of (page_number, text) for every page that has text."""
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise DocumentValidationError("Encrypted PDFs are not supported.")
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
    except DocumentValidationError:
        raise
    except (PdfReadError, Exception) as exc:  # pypdf can raise many error types
        raise DocumentValidationError(f"Could not read PDF: {exc}") from exc
    return pages


def extract_txt(content: bytes) -> str:
    """Decode a text file: UTF-16 only with a BOM, else UTF-8, else Latin-1."""
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    else:
        try:
            content.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            encoding = "latin-1"

    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DocumentValidationError("Could not decode the text file.") from exc

    if "\x00" in text:
        raise DocumentValidationError(
            "The text file looks binary or uses an unsupported encoding (save it as UTF-8)."
        )
    return text.strip()


def load_document(filename: str, content: bytes) -> list[tuple[int | None, str]]:
    """Validate a file and return its text as (page_number, text) sections.

    TXT files return a single section with page_number = None.
    """
    extension = validate_file(filename, content)

    if extension == ".pdf":
        sections: list[tuple[int | None, str]] = list(extract_pdf_pages(content))
    else:
        text = extract_txt(content)
        sections = [(None, text)] if text else []

    if not sections:
        raise DocumentValidationError(
            "No readable text found in the document (scanned PDFs are not supported)."
        )
    return sections


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
