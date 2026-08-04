"""
Streamlit UI for the Enterprise Contract & Document RAG Agent.

Run:  streamlit run app.py
"""
import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.config import settings
from src.ingestion import ingest_pipeline
from src.vector_store import VectorStoreManager
from src.hybrid_search import HybridRetriever
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Contract RAG Agent", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "store_manager" not in st.session_state:
    st.session_state.store_manager = VectorStoreManager()
    st.session_state.store_manager.load_or_create()
if "hybrid_retriever" not in st.session_state:
    st.session_state.hybrid_retriever = HybridRetriever(st.session_state.store_manager)
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ---------------------------------------------------------------------------
# Sidebar: ingestion + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Contract RAG Agent")
    st.caption("Hybrid search (BM25 + vector) · Ragas hallucination monitoring")

    st.subheader("1. Upload documents")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🔄 Index documents", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Chunking, embedding, and indexing..."):
            tmp_paths = []
            tmp_dir = tempfile.mkdtemp()
            for f in uploaded_files:
                path = os.path.join(tmp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                tmp_paths.append(path)

            new_chunks = ingest_pipeline(tmp_paths)
            if new_chunks:
                st.session_state.store_manager.add_documents(new_chunks)
                st.session_state.chunks.extend(new_chunks)
                st.session_state.hybrid_retriever.build_sparse_index(st.session_state.chunks)
                st.session_state.pipeline = RAGPipeline(st.session_state.hybrid_retriever)
                st.session_state.indexed_files.extend([f.name for f in uploaded_files])
                st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.error("No content could be extracted from the uploaded file(s).")

    if st.session_state.indexed_files:
        st.subheader("Indexed files")
        for name in st.session_state.indexed_files:
            st.text(f"✓ {name}")

    st.divider()
    st.subheader("2. Retrieval settings")
    top_k = st.slider("Chunks to retrieve (k)", 1, 10, settings.top_k)
    st.caption(
        f"Hybrid weights — dense: **{settings.dense_weight}**, "
        f"sparse (BM25): **{settings.sparse_weight}**"
    )
    st.caption(f"LLM provider: **{settings.llm_provider}** · Embeddings: **{settings.embedding_provider}**")

    st.divider()
    if st.button("🗑️ Clear index", use_container_width=True):
        st.session_state.store_manager.reset()
        st.session_state.store_manager.load_or_create()
        st.session_state.chunks = []
        st.session_state.indexed_files = []
        st.session_state.pipeline = None
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main: chat interface
# ---------------------------------------------------------------------------
st.header("Ask a question about your documents")

if not st.session_state.pipeline:
    st.info("👈 Upload and index at least one document to get started.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            with st.expander(f"📎 {len(turn['sources'])} source(s) used"):
                for s in turn["sources"]:
                    st.markdown(f"**{s.source}** — page {s.page}")
                    st.caption(s.snippet)
                    st.divider()

    question = st.chat_input("e.g. What is the termination clause notice period?")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = st.session_state.pipeline.query(question, k=top_k)
            st.write(response.answer)

            with st.expander(f"📎 {len(response.sources)} source(s) used", expanded=True):
                for s in response.sources:
                    st.markdown(f"**{s.source}** — page {s.page}")
                    st.caption(s.snippet)
                    st.divider()

        st.session_state.chat_history.append(
            {"question": question, "answer": response.answer, "sources": response.sources}
        )

st.divider()

# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------
with st.expander("🧪 Run Ragas evaluation (faithfulness / hallucination check)"):
    st.caption(
        "Runs the pipeline against a small labeled test set and scores "
        "faithfulness, answer relevancy, and context precision."
    )
    if st.button("Run evaluation", disabled=not st.session_state.pipeline):
        with st.spinner("Running evaluation — this calls the LLM multiple times..."):
            import json
            from evaluation.evaluate import run_evaluation

            with open("evaluation/testset.json") as f:
                testset = json.load(f)
            questions = [row["question"] for row in testset]
            ground_truths = [row["ground_truth"] for row in testset]

            try:
                results = run_evaluation(st.session_state.pipeline, questions, ground_truths)
                cols = st.columns(len(results["summary"]))
                for col, (metric, score) in zip(cols, results["summary"].items()):
                    col.metric(metric.replace("_", " ").title(), f"{score:.0%}")
                st.dataframe(results["detail"])
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
