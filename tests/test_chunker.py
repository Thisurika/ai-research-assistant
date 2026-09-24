import pytest

from src.chunker import create_chunks


def test_creates_chunks():
    pages = [
        {
            "page_number": 1,
            "text": "Machine learning is useful. " * 200
        }
    ]

    chunks = create_chunks(
        pages=pages,
        source_name="sample.pdf",
        chunk_size=50,
        chunk_overlap=10
    )

    assert len(chunks) > 1
    assert all(
        chunk["token_count"] <= 50
        for chunk in chunks
    )


def test_chunk_metadata():
    pages = [
        {
            "page_number": 3,
            "text": "Artificial intelligence research."
        }
    ]

    chunks = create_chunks(
        pages=pages,
        source_name="research.pdf",
        chunk_size=50,
        chunk_overlap=10
    )

    assert chunks[0]["chunk_id"] == 1
    assert chunks[0]["source"] == "research.pdf"
    assert chunks[0]["page_number"] == 3


def test_overlap_must_be_smaller_than_chunk_size():
    pages = [
        {
            "page_number": 1,
            "text": "Example text"
        }
    ]

    with pytest.raises(ValueError):
        create_chunks(
            pages=pages,
            source_name="sample.pdf",
            chunk_size=100,
            chunk_overlap=100
        )