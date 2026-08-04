"""
Document ingestion pipeline:
  PDF / DOCX / TXT  -->  page-aware loading  -->  recursive chunking
Every chunk keeps metadata: {source, page, chunk_id} so the UI can show
exact source attribution later.
"""
from __future__ import annotations

import hashlib
import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _load_single_file(file_path: str) -> List[Document]:
    """Loads one file into page-level LangChain Documents."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    docs = loader.load()

    # Normalize metadata across loader types
    for i, doc in enumerate(docs):
        doc.metadata["source"] = os.path.basename(file_path)
        # PyPDFLoader already sets "page" (0-indexed); others don't
        doc.metadata.setdefault("page", i)
        doc.metadata["page"] = doc.metadata["page"] + 1  # human-friendly, 1-indexed

    return docs


def load_documents(file_paths: List[str]) -> List[Document]:
    """Loads multiple files, skipping ones that fail, and returns all page docs."""
    all_docs: List[Document] = []
    for path in file_paths:
        try:
            all_docs.extend(_load_single_file(path))
        except Exception as e:
            print(f"[ingestion] Failed to load {path}: {e}")
    return all_docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Splits page-level documents into overlapping chunks using a recursive
    character splitter (respects paragraph/sentence boundaries first).
    Assigns a stable chunk_id (hash) used later for citation dedup.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        raw_id = f"{chunk.metadata.get('source')}-{chunk.metadata.get('page')}-{chunk.page_content[:50]}"
        chunk.metadata["chunk_id"] = hashlib.md5(raw_id.encode()).hexdigest()[:12]

    return chunks


def ingest_pipeline(file_paths: List[str]) -> List[Document]:
    """Full pipeline: load -> chunk. Returns ready-to-embed chunks."""
    docs = load_documents(file_paths)
    if not docs:
        return []
    chunks = chunk_documents(docs)
    print(f"[ingestion] {len(file_paths)} file(s) -> {len(docs)} pages -> {len(chunks)} chunks")
    return chunks
