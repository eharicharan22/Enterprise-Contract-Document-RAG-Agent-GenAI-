"""
LLM factory (OpenAI / Groq / Ollama) and prompt template.
The prompt forces the model to (a) answer only from provided context,
(b) cite [Source: filename, p.X] inline, and (c) say "not found in the
provided documents" rather than hallucinate — this is what the Ragas
faithfulness check in evaluation.py later verifies empirically.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.config import settings

SYSTEM_PROMPT = """You are an enterprise contract & document analysis assistant.
Answer the user's question using ONLY the context excerpts provided below.

Rules:
1. Every factual claim must be grounded in the given context — never use outside knowledge.
2. After each claim, cite the source like this: [Source: {{filename}}, p.{{page}}].
3. If the context does not contain the answer, respond exactly:
   "I could not find this information in the provided documents."
4. Be precise and concise. For contracts, quote exact clause language when relevant (short quotes only).
5. Do not speculate about parties' intentions beyond what is written.

Context:
----------------
{context}
----------------
"""

USER_PROMPT = "{question}"


def get_llm() -> BaseChatModel:
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0,
        )

    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Use openai | groq | ollama.")


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ]
    )
