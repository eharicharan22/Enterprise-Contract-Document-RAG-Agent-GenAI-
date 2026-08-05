"""
Centralized configuration loaded from environment variables (.env).
Supports OpenAI, Google Gemini, and Ollama.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:

    # ---------- LLM ----------
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Google Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    google_model: str = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

    # ---------- Embeddings ----------
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )

    # ---------- Vector Store ----------
    vector_store: str = os.getenv("VECTOR_STORE", "chroma").lower()
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        "./data/chroma_db",
    )

    qdrant_url: str = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333",
    )

    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION",
        "contract_docs",
    )

    # ---------- Chunking ----------
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 150)

    # ---------- Hybrid Search ----------
    dense_weight: float = _get_float("DENSE_WEIGHT", 0.6)
    sparse_weight: float = _get_float("SPARSE_WEIGHT", 0.4)
    top_k: int = _get_int("TOP_K", 5)


settings = Settings()
