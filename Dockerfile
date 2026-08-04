# --- Enterprise Contract RAG Agent: production image ---
FROM python:3.11-slim

WORKDIR /app

# System deps needed by unstructured/pypdf/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persisted vector store lives here — mount a volume to this path in prod
RUN mkdir -p /app/data/chroma_db

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
