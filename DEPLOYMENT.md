# Deployment Guide

Pick the path that matches what you need. For a resume/demo project,
**Option A (Streamlit Community Cloud)** is fastest and free.

---

## Option A — Streamlit Community Cloud (easiest, free)

Good for: public demo link to put on your resume/LinkedIn.

1. Push the project to a **public** GitHub repo.
2. Go to https://share.streamlit.io → "New app" → pick your repo, branch, and set
   **Main file path** to `app.py`.
3. Click "Advanced settings" → **Secrets** → paste your `.env` contents in TOML format:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_xxxxxxxx"
   GROQ_MODEL = "llama-3.1-70b-versatile"
   EMBEDDING_PROVIDER = "local"
   VECTOR_STORE = "chroma"
   CHROMA_PERSIST_DIR = "./data/chroma_db"
   ```
4. Deploy. Build takes ~3-5 min (installs `sentence-transformers`, `torch`, etc.).

**Caveats:**
- Free tier has ~1GB RAM — `local` embeddings work fine, but if you hit memory
  limits, switch `EMBEDDING_PROVIDER=openai` to offload embedding compute.
- The filesystem is **ephemeral** — your Chroma index resets on redeploy/restart.
  Fine for a demo; for persistence use Option B or C with a real Qdrant instance.

---

## Option B — Docker on any VPS (Render, Railway, Fly.io, EC2, DigitalOcean)

Good for: something that persists, and a proper backend API if you need one.

The repo already includes `Dockerfile` (Streamlit UI), `Dockerfile.api`
(FastAPI backend), and `docker-compose.yml` (both + Qdrant).

### Test locally first
```bash
cp .env.example .env   # fill in your keys
docker compose up --build
# UI:  http://localhost:8501
# API: http://localhost:8000/docs
```

### Render.com (free/low-cost tier, easiest managed Docker option)
1. Push to GitHub.
2. New → **Web Service** → connect repo → Render auto-detects the `Dockerfile`.
3. Set environment variables (from `.env.example`) in the Render dashboard —
   don't commit `.env`.
4. Set **Health check path** to `/_stcore/health` (Streamlit) or `/health` (API).
5. Instance type: 512MB is tight for `sentence-transformers`; 1GB+ recommended,
   or use `EMBEDDING_PROVIDER=openai` to avoid loading a local model.
6. For persistence, add a Render **Disk** mounted at `/app/data/chroma_db`.

### Railway.app
1. New Project → Deploy from GitHub repo → Railway detects the Dockerfile.
2. Add a **Volume** mounted at `/app/data/chroma_db` for persistent storage.
3. Add env vars in the Railway dashboard (Variables tab).
4. If you need Qdrant instead of Chroma, add it from Railway's template
   marketplace and point `QDRANT_URL` at its internal address.

### Raw VPS (EC2 / DigitalOcean Droplet)
```bash
# on the server
git clone <your-repo-url> && cd rag-agent
cp .env.example .env && nano .env   # fill in keys
docker compose up -d --build

# put nginx or Caddy in front for TLS, e.g. Caddy:
#   your-domain.com {
#     reverse_proxy localhost:8501
#   }
```

---

## Option C — Hugging Face Spaces (free, good for public ML demos)

1. Create a new Space → SDK: **Docker**.
2. Push this repo's contents to the Space's git remote (HF Spaces builds
   directly from your `Dockerfile`).
3. In Space **Settings → Repository secrets**, add `GROQ_API_KEY` etc.
4. HF Spaces expects the app on port `7860` by default — either add
   `-p 7860` mapping or edit the `CMD` in `Dockerfile`:
   ```dockerfile
   CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
   ```
5. Free tier CPU Spaces are enough for `local` embeddings + Groq inference.

---

## Deploying the API separately from the UI

If a frontend team or another service needs to call the RAG pipeline
directly, deploy `Dockerfile.api` on its own (Render/Railway/ECS/Cloud Run),
and point the Streamlit app or any client at its `/query` endpoint instead
of importing `src/` directly. This is the cleaner setup for a real
"enterprise" deployment — UI and retrieval/generation scale independently.

---

## Secrets checklist (any platform)

- Never commit `.env` — it's already covered by `.dockerignore`; add it to
  `.gitignore` too if you haven't.
- Required at minimum: `LLM_PROVIDER` + matching API key
  (`GROQ_API_KEY` / `OPENAI_API_KEY`).
- If `VECTOR_STORE=qdrant`, also set `QDRANT_URL` to the deployed Qdrant
  instance's address (not `localhost` once you're off your machine).

## Sizing notes

| Component | Memory need | Notes |
|---|---|---|
| Streamlit app | ~300-500MB base | + ~400MB if `EMBEDDING_PROVIDER=local` (loads a small transformer) |
| FastAPI backend | ~250-400MB | same embedding caveat applies |
| Qdrant | ~200MB idle | scales with corpus size |
| Chroma | in-process, no separate service | fine for demos; Qdrant is the better choice once you're running a real deployment with concurrent users |

If your host is memory-constrained, the single biggest lever is switching
`EMBEDDING_PROVIDER=openai` — it removes the local `sentence-transformers`
+ `torch` footprint entirely, at the cost of a per-embedding API call.
