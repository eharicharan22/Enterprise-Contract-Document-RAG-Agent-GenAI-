"""
Embedding provider factory. Defaults to a local sentence-transformers
model (BAAI/bge-small-en-v1.5) so the project runs with zero API keys.
Swap to OpenAI embeddings by setting EMBEDDING_PROVIDER=openai in .env.
"""
from langchain_core.embeddings import Embeddings

from src.config import settings


def get_embedding_model() -> Embeddings:
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )

    # default: local, free, CPU-friendly
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.local_embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for cosine sim
    )
