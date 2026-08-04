# Enterprise Contract & Document RAG Agent (GenAI)

An end-to-end **Retrieval-Augmented Generation (RAG)** solution designed to automate legal contract analysis, compliance auditing, and intelligent document Q&A for large-scale enterprise repositories. Built with state-of-the-art GenAI models, vector indexing, and metadata filtering to deliver high-precision answers with exact document citations.

---

## 🔑 Key Features

* **Multi-Format Document Parsing:** Ingests complex legal contracts, policies, and enterprise documents (PDF, DOCX) while preserving structural hierarchy and metadata.
* **Intelligent Chunking & Embeddings:** Implements recursive character and semantic chunking strategies paired with domain-optimized vector embeddings for maximum context retrieval accuracy.
* **Hybrid Retrieval System:** Combines dense vector search (semantic similarity) with sparse keyword search (BM25) to accurately surface contract clauses, definitions, and risk factors.
* **Source Attribution & Citations:** Generates answers backed by precise page-level and section-level references to prevent hallucination and ensure auditability.
* **Contract Analysis Engine:** Automates key term extraction (expiration dates, liabilities, renewal terms, governing law) and flags compliance risks in real time.
* **Interactive UI & API:** Features an intuitive Streamlit workspace for interactive document Q&A alongside a FastAPI framework for enterprise application integration.

---

## 🛠️ Tech Stack & Frameworks

* **Language:** Python
* **LLM & RAG Orchestration:** LangChain / LlamaIndex
* **Embeddings & LLMs:** HuggingFace / OpenAI / Local Transformers
* **Vector Database:** ChromaDB / FAISS
* **Web Framework / UI:** Streamlit / FastAPI
* **Document Processing:** PyPDF, Unstructured, pdfplumber

---

## ⚡ Quickstart

### 1. Clone the Repository
```bash
git clone [https://github.com/eharicharan22/enterprise-contract-rag-agent.git](https://github.com/eharicharan22/enterprise-contract-rag-agent.git)
cd enterprise-contract-rag-agent
