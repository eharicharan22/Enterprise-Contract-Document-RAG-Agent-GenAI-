"""
Basic unit tests. Run with:  pytest tests/ -v
These don't require API keys — they test pure logic (chunking, metadata).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from src.ingestion import chunk_documents


def test_chunking_preserves_metadata():
    docs = [
        Document(
            page_content="This is a long clause about termination. " * 30,
            metadata={"source": "sample.pdf", "page": 3},
        )
    ]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1, "Long document should split into multiple chunks"
    for c in chunks:
        assert c.metadata["source"] == "sample.pdf"
        assert c.metadata["page"] == 3
        assert "chunk_id" in c.metadata


def test_chunking_respects_overlap_bounds():
    docs = [Document(page_content="word " * 500, metadata={"source": "x.pdf", "page": 1})]
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)

    for c in chunks:
        assert len(c.page_content) <= 300 + 50  # allow splitter boundary slack


def test_chunk_ids_are_unique_for_different_content():
    docs = [
        Document(page_content="Clause A: payment terms are net 45 days.", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Clause B: governing law is Delaware.", metadata={"source": "a.pdf", "page": 2}),
    ]
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=0)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs should be unique per distinct content"


if __name__ == "__main__":
    test_chunking_preserves_metadata()
    test_chunking_respects_overlap_bounds()
    test_chunk_ids_are_unique_for_different_content()
    print("All tests passed ✅")
