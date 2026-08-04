# Enterprise Contract & Document RAG Agent

A retrieval-augmented generation system for legal contracts and enterprise
documents, built around **hybrid search** (dense vectors + BM25 keyword
search) and continuous **hallucination monitoring** with Ragas.

```
[ PDF / DOCX Input ] → [ Chunking ] → [ Hybrid Search (BM25 + Cosine Sim) ] → [ LLM Generation ] → [ Ragas Eval ] → [ Streamlit UI ]
```

## Why hybrid search

Dense embeddings are great at semantic matches ("termination clause" ≈
"how the agreement can be ended") but weak on exact tokens — contract
numbers, defined terms, section IDs. BM25 nails exact-keyword recall but
misses paraphrases. This project fuses both with weighted **Reciprocal
Rank Fusion** via LangChain's `EnsembleRetriever`, using:

```
final_rank = dense_weight * dense_rank_score + sparse_weight * bm25_rank_score
cosine_similarity(a, b) = (a · b) / (‖a‖ ‖b‖)
```

Default weights: `dense_weight=0.6`, `sparse_weight=0.4` (tune in `.env`).

## Project structure

```
rag-agent/
├── app.py                  # Streamlit UI (upload, chat, source attribution, eval panel)
├── api.py                  # FastAPI backend (REST: /ingest, /query, /health)
├── requirements.txt
├── .env.example             # copy to .env and fill in
├── src/
│   ├── config.py            # env-driven settings, single source of truth
│   ├── ingestion.py          # PDF/DOCX/TXT loading + recursive chunking
│   ├── embeddings.py         # local (sentence-transformers) or OpenAI embeddings
│   ├── vector_store.py       # Chroma / Qdrant backend abstraction
│   ├── hybrid_search.py      # BM25 + dense vector fusion (EnsembleRetriever)
│   ├── llm.py                # OpenAI / Groq / Ollama chat model factory + prompt
│   └── rag_pipeline.py       # orchestrates retrieval → generation → source attribution
├── evaluation/
│   ├── evaluate.py           # Ragas: faithfulness, answer_relevancy, context_precision/recall
│   └── testset.json          # sample Q&A pairs for regression testing
└── tests/
    └── test_ingestion.py     # unit tests for chunking logic
```

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set LLM_PROVIDER (openai/groq/ollama) and the matching API key.
# Groq is recommended for a fast, free-tier-friendly setup:
#   https://console.groq.com/keys
# Embeddings default to a local model (no API key needed).
```

### Optional: run Qdrant locally instead of Chroma

```bash
docker run -p 6333:6333 qdrant/qdrant
# then in .env: VECTOR_STORE=qdrant
```

## Run the app

```bash
# Streamlit UI
streamlit run app.py

# FastAPI backend (alternative / for integration)
uvicorn api:app --reload --port 8000
# Interactive docs at http://localhost:8000/docs
```

## Run evaluation (hallucination / faithfulness check)

```bash
python evaluation/evaluate.py --testset evaluation/testset.json
```

This scores the pipeline on:
| Metric | What it measures |
|---|---|
| `faithfulness` | Is every claim in the answer actually supported by retrieved context? (the core hallucination check) |
| `answer_relevancy` | Does the answer actually address the question asked? |
| `context_precision` | Are the top-ranked retrieved chunks the relevant ones? |
| `context_recall` | Did retrieval surface enough of the information needed? (needs ground truth) |

Results are also printed as CSV to `evaluation/last_run_detail.csv`, and can
be triggered on-demand from the Streamlit sidebar ("Run Ragas evaluation").

## Run tests

```bash
pytest tests/ -v
```

## Extending this project

- **Re-ranking**: add a cross-encoder (e.g. `BAAI/bge-reranker-base`) after
  hybrid retrieval to re-score the top ~20 candidates before passing the top
  5 to the LLM — typically the single biggest precision boost.
- **Multi-tenant isolation**: namespace Chroma/Qdrant collections per client
  in `vector_store.py`.
- **Structured extraction**: add a second chain that pulls key contract
  fields (parties, effective date, renewal terms) into a JSON schema using
  the LLM's structured output mode.
- **Guardrails**: wrap `rag_pipeline.query()` with a regex/LLM check that
  the "not found" fallback is used consistently instead of hedged answers.

## Resume-ready framing

> Built an enterprise document RAG system featuring hybrid vector-keyword
> search and continuous hallucination monitoring, achieving strong
> retrieval precision on complex legal contracts, measured via automated
> Ragas faithfulness scoring across a regression test suite.
