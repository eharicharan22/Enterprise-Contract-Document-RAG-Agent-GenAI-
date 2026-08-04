"""
FastAPI backend — exposes the RAG pipeline as REST endpoints so it can be
integrated into other enterprise systems (e.g. a Slack bot or internal portal).

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from src.ingestion import ingest_pipeline
from src.vector_store import VectorStoreManager
from src.hybrid_search import HybridRetriever
from src.rag_pipeline import RAGPipeline

app = FastAPI(
    title="Enterprise Contract RAG API",
    description="Hybrid-search RAG over contracts and enterprise documents, with source attribution.",
    version="1.0.0",
)

# Single shared in-process index for this demo. In production, swap for a
# per-tenant collection name / namespace.
_store_manager = VectorStoreManager()
_store_manager.load_or_create()
_hybrid_retriever = HybridRetriever(_store_manager)
_pipeline: Optional[RAGPipeline] = None
_all_chunks = []


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceOut(BaseModel):
    source: str
    page: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceOut]


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(_all_chunks)}


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    global _pipeline, _all_chunks

    tmp_dir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for f in files:
            path = os.path.join(tmp_dir, f.filename)
            with open(path, "wb") as out:
                shutil.copyfileobj(f.file, out)
            saved_paths.append(path)

        chunks = ingest_pipeline(saved_paths)
        if not chunks:
            raise HTTPException(400, "No content could be extracted from uploaded files.")

        _store_manager.add_documents(chunks)
        _all_chunks.extend(chunks)
        _hybrid_retriever.build_sparse_index(_all_chunks)
        _pipeline = RAGPipeline(_hybrid_retriever)

        return {"indexed_files": [f.filename for f in files], "chunk_count": len(chunks)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(400, "No documents indexed yet. Call /ingest first.")

    result = _pipeline.query(req.question, k=req.top_k)
    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceOut(source=s.source, page=s.page, snippet=s.snippet) for s in result.sources
        ],
    )


@app.delete("/index")
def clear_index():
    global _pipeline, _all_chunks
    _store_manager.reset()
    _store_manager.load_or_create()
    _all_chunks = []
    _pipeline = None
    return {"status": "cleared"}
