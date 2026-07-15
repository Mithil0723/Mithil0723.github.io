# Mithil's Portfolio RAG Backend

This is the backend for Mithil's portfolio, powered by a FastAPI + LangGraph architecture. It answers user questions grounded in portfolio data.

## Architecture
- **LangGraph StateGraph**: The query orchestrator. It runs `condense -> retrieve -> rerank -> generate`. 
- **Embeddings**: Documents are chunked and embedded using `nvidia/nv-embed-v1` via the NVIDIA NIM API.
- **Vector Store**: SQLite (`vectors.db`) stores the documents and BLOB embeddings. Cosine similarity is computed in memory via numpy.
- **Reranker**: Retrieved documents are reranked using `nvidia/nv-rerankqa-mistral-4b-v3` via NIM to find the top most relevant chunks.
- **LLM**: Generation is powered by `nvidia/nemotron-3-nano-30b-a3b` via NIM.

> **Note on Licensing**: The `nvidia/nv-embed-v1` model is currently licensed for **Non-Commercial Use Only**.

## Setup & Running Locally

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy the example environment file and fill in your NVIDIA API key (from [build.nvidia.com](https://build.nvidia.com)):
   ```bash
   cp .env.example .env
   ```

3. **IMPORTANT**: You must re-run ingestion to create/update the `vectors.db` file with 4096-dimensional embeddings:
   ```bash
   python ingest.py
   ```

4. Run the development server:
   ```bash
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | (Required) | API key from build.nvidia.com (needs Public API Endpoints permission) |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | Base URL for NIM models |
| `EMBED_MODEL` | `nvidia/nv-embed-v1` | 4096-dim embedding model |
| `RERANK_MODEL` | `nvidia/nv-rerankqa-mistral-4b-v3` | Reranker model |
| `LLM_MODEL` | `nvidia/nemotron-3-nano-30b-a3b` | Chat model |
| `LLM_MAX_TOKENS` | `400` | Max tokens to generate |
| `LLM_ENABLE_THINKING` | `false` | Whether the reasoning model should "think" |
| `LLM_FORCE_NONEMPTY_CONTENT` | `true` | Prevent the model from returning empty content if reasoning budget exhausted |
| `MATCH_THRESHOLD` | `0.3` | Minimum cosine similarity score for a retrieved chunk |
| `MATCH_COUNT` | `8` | Number of chunks to retrieve before reranking |
| `VECTOR_DB_PATH` | `vectors.db` | Path to the SQLite database |
| `ALLOWED_ORIGINS` | `https://mithil0723.github.io` | CORS origins (comma-separated) |
