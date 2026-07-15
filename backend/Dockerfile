FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (cached layer — only re-runs if requirements.txt changes)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ .

# Copy source documents for ingestion to match the relative paths (../) expected by ingest.py
COPY About_me.md /About_me.md
COPY ["project contents", "/project contents"]

# Run ingestion at build time — bakes the SQLite vector DB into the image
# Requires NVIDIA_API_KEY to hit the NIM embed API
ARG NVIDIA_API_KEY
ENV NVIDIA_API_KEY=$NVIDIA_API_KEY
RUN python ingest.py

# Expose port (Coolify will route to this)
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
