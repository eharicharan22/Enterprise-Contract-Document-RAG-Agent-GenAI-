"""
Orchestrates: query -> hybrid retrieval -> context assembly -> LLM answer
-> structured response with source attribution (filename + page + snippet).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document

from src.hybrid_search import HybridRetriever
from src.llm import build_prompt, get_llm


@dataclass
class SourceChunk:
    source: str
    page: int
    snippet: str
    chunk_id: str


@dataclass
class RAGResponse:
    answer: str
    sources: List[SourceChunk] = field(default_factory=list)
    retrieved_docs: List[Document] = field(default_factory=list)  # kept for Ragas eval


class RAGPipeline:
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever
        self.llm = get_llm()
        self.prompt = build_prompt()

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        blocks = []
        for d in docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", "?")
            blocks.append(f"[Source: {src}, p.{page}]\n{d.page_content}")
        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _make_snippet(text: str, max_len: int = 220) -> str:
        text = " ".join(text.split())
        return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "…"

    def query(self, question: str, k: int = None) -> RAGResponse:
        docs = self.retriever.retrieve(question, k=k)

        if not docs:
            return RAGResponse(
                answer="I could not find this information in the provided documents.",
                sources=[],
                retrieved_docs=[],
            )

        context = self._format_context(docs)
        chain = self.prompt | self.llm
        result = chain.invoke({"context": context, "question": question})
        answer_text = result.content if hasattr(result, "content") else str(result)

        sources = [
            SourceChunk(
                source=d.metadata.get("source", "unknown"),
                page=d.metadata.get("page", 0),
                snippet=self._make_snippet(d.page_content),
                chunk_id=d.metadata.get("chunk_id", ""),
            )
            for d in docs
        ]

        return RAGResponse(answer=answer_text, sources=sources, retrieved_docs=docs)
