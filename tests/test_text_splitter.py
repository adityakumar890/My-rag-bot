from app.document_loader import split_into_chunks


def test_long_text_is_split_with_metadata():
    text = "Python is a programming language. " * 50  # ~1700 characters
    chunks = split_into_chunks("python.txt", [(None, text)])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["filename"] == "python.txt" for c in chunks)


def test_page_numbers_are_kept():
    chunks = split_into_chunks("doc.pdf", [(1, "Page one text"), (2, "Page two text")])
    assert [c.metadata["page"] for c in chunks] == [1, 2]
