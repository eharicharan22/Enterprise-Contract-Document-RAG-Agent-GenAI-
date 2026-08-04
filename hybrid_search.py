"""
Hybrid Search = Dense (vector / cosine similarity) + Sparse (BM25 keyword)

Why hybrid: dense embeddings catch semantic meaning ("termination clause"
~ "how the agreement can be ended") but miss exact tokens (contract
numbers, defined terms, section IDs). BM25 nails exact-keyword recall.
Combining both with a weighted rank-fusion gives the best of each.

Score fusion: for each doc, final_score = w_dense * dense_score
                                          + w_sparse * bm25_score_norm
(both normalized to [0,1] before combining, since Chroma similarity and
BM25 scores live on different scales).
"""
from __future__ import annotations

from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever

from src.config import settings
from src.vector_store import VectorStoreManager


class HybridRetriever:
    """
    Wraps LangChain's EnsembleRetriever to fuse:
      - a dense retriever backed by Chroma/Qdrant (cosine similarity)
      - a sparse BM25Retriever built in-memory from the same chunks

    EnsembleRetriever uses Reciprocal Rank Fusion (RRF) under the hood,
    weighted by `weights=[dense_weight, sparse_weight]`.
    """

    def __init__(self, store_manager: VectorStoreManager, documents: List[Document] | None = None):
        self.store_manager = store_manager
        self._bm25: BM25Retriever | None = None
        self._ensemble: EnsembleRetriever | None = None

        if documents:
            self.build_sparse_index(documents)

    def build_sparse_index(self, documents: List[Document]) -> None:
        """(Re)builds the in-memory BM25 index. Call after ingesting new docs."""
        self._bm25 = BM25Retriever.from_documents(documents)
        self._bm25.k = settings.top_k

        dense_retriever = self.store_manager.as_retriever(k=settings.top_k)

        self._ensemble = EnsembleRetriever(
            retrievers=[dense_retriever, self._bm25],
            weights=[settings.dense_weight, settings.sparse_weight],
        )

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        if self._ensemble is None:
            raise RuntimeError(
                "Sparse index not built yet. Call build_sparse_index() after ingesting documents."
            )
        k = k or settings.top_k
        results = self._ensemble.invoke(query)
        return results[:k]

    def retrieve_dense_only(self, query: str, k: int = None) -> List[Document]:
        """Useful for debugging / comparing hybrid vs pure-vector recall."""
        k = k or settings.top_k
        return self.store_manager.as_retriever(k=k).invoke(query)

    def retrieve_sparse_only(self, query: str, k: int = None) -> List[Document]:
        if self._bm25 is None:
            raise RuntimeError("Sparse index not built yet.")
        k = k or settings.top_k
        self._bm25.k = k
        return self._bm25.invoke(query)
