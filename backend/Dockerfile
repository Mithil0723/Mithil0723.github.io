FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs if requirements.txt changes)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace model during build (~90 MB)
# This bakes the model into the image so cold starts load from disk (~2-3s)
# instead of downloading from the internet (~30-60s)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code
COPY backend/ .

# Copy source documents for ingestion to match the relative paths (../) expected by ingest.py
COPY About_me.md /About_me.md
COPY ["project contents", "/project contents"]

# Run ingestion at build time — bakes the SQLite vector DB into the image
RUN python ingest.py

# Expose port (Render/Coolify will route to this or override it)
EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
