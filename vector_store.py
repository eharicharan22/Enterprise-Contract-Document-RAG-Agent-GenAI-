"""
Vector store abstraction over Chroma (default, zero-setup, local disk)
and Qdrant (production-grade, run via `docker run -p 6333:6333 qdrant/qdrant`).

Switch backends purely via VECTOR_STORE env var — no other code changes.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import settings
from src.embeddings import get_embedding_model


class VectorStoreManager:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.backend = settings.vector_store
        self._store = None

    # ---------- lifecycle ----------

    def load_or_create(self):
        if self.backend == "qdrant":
            self._store = self._load_qdrant()
        else:
            self._store = self._load_chroma()
        return self._store

    def _load_chroma(self):
        from langchain_chroma import Chroma

        return Chroma(
            collection_name="contract_docs",
            embedding_function=self.embedding_model,
            persist_directory=settings.chroma_persist_dir,
        )

    def _load_qdrant(self):
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, VectorParams

        client = QdrantClient(url=settings.qdrant_url)

        # Create collection if it doesn't exist yet
        existing = [c.name for c in client.get_collections().collections]
        if settings.qdrant_collection not in existing:
            # bge-small-en-v1.5 -> 384 dims; text-embedding-3-small -> 1536 dims
            dim = 1536 if settings.embedding_provider == "openai" else 384
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        return QdrantVectorStore(
            client=client,
            collection_name=settings.qdrant_collection,
            embedding=self.embedding_model,
        )

    # ---------- operations ----------

    def add_documents(self, chunks: List[Document]) -> None:
        if self._store is None:
            self.load_or_create()
        # Chroma/Qdrant both batch-embed internally
        self._store.add_documents(chunks)

    def as_retriever(self, k: int = None) -> VectorStoreRetriever:
        if self._store is None:
            self.load_or_create()
        k = k or settings.top_k
        return self._store.as_retriever(search_type="similarity", search_kwargs={"k": k})

    def similarity_search_with_score(self, query: str, k: int = None):
        """Returns [(Document, cosine_score), ...] — used for showing scores in the UI."""
        if self._store is None:
            self.load_or_create()
        k = k or settings.top_k
        return self._store.similarity_search_with_relevance_scores(query, k=k)

    def reset(self) -> None:
        """Wipes the collection — used by the Streamlit 'Clear Index' button."""
        if self.backend == "chroma" and self._store is not None:
            self._store.delete_collection()
        elif self.backend == "qdrant" and self._store is not None:
            self._store.client.delete_collection(settings.qdrant_collection)
        self._store = None
