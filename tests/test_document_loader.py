import io

import pytest
from pypdf import PdfWriter

from app.document_loader import (
    DocumentValidationError,
    load_document,
    validate_file,
)


def test_txt_file_is_loaded():
    sections = load_document("notes.txt", b"Hello RAG world")
    assert sections == [(None, "Hello RAG world")]


def test_unsupported_extension_rejected():
    with pytest.raises(DocumentValidationError, match="Unsupported file type"):
        validate_file("image.png", b"data")


def test_empty_file_rejected():
    with pytest.raises(DocumentValidationError, match="empty"):
        validate_file("notes.txt", b"")


def test_too_large_file_rejected(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    get_settings.cache_clear()
    with pytest.raises(DocumentValidationError, match="too large"):
        validate_file("big.txt", b"a" * (1024 * 1024 + 1))


def test_fake_pdf_rejected():
    with pytest.raises(DocumentValidationError, match="valid PDF"):
        validate_file("fake.pdf", b"this is not a pdf")


def test_pdf_without_text_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(DocumentValidationError, match="No readable text"):
        load_document("blank.pdf", buffer.getvalue())


def test_whitespace_only_txt_rejected():
    with pytest.raises(DocumentValidationError, match="No readable text"):
        load_document("blank.txt", b"   \n  ")


def test_txt_encodings():
    assert load_document("a.txt", "café".encode("utf-8")) == [(None, "café")]
    assert load_document("b.txt", "café!".encode("latin-1")) == [(None, "café!")]
    assert load_document("c.txt", "hello".encode("utf-16")) == [(None, "hello")]


def test_bomless_utf16_txt_rejected():
    with pytest.raises(DocumentValidationError, match="binary"):
        load_document("d.txt", "hello".encode("utf-16-le"))
