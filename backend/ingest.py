import os
import re
import glob
import json
import sqlite3
import logging
import hashlib
import numpy as np
from nim_client import nim_embed
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# NIM Embedding Configuration
# ─────────────────────────────────────────────
EMBED_BATCH_SIZE = 32

# ─────────────────────────────────────────────
# SQLite Vector Store
# ─────────────────────────────────────────────
DB_PATH = os.getenv("VECTOR_DB_PATH", "vectors.db")


def init_db():
    """Create the documents and meta tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            embedding BLOB NOT NULL,
            content_hash TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"SQLite database initialized at {DB_PATH}")


# ─────────────────────────────────────────────
# Helper: Strip YAML Frontmatter
# ─────────────────────────────────────────────
def strip_yaml_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter (between --- markers) from markdown text.
    Also strips any leading title line before the frontmatter block
    (e.g. 'RAG Chatbot\n# MANDATORY YAML...').
    """
    # Pattern: optional leading line, then --- block ---, then content
    pattern = r"^(?:.*?\n)?#[^\n]*\n(?:.*?\n)?---\s*$"
    # More robust: match everything up to and including the closing ---
    stripped = re.sub(
        r"\A.*?^---\s*$",
        "",
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    return stripped.strip()


# ─────────────────────────────────────────────
# Helper: Markdown-Aware Chunking
# ─────────────────────────────────────────────
def chunk_markdown(text: str, source_name: str) -> list[dict]:
    """
    Two-stage splitting:
      1. MarkdownHeaderTextSplitter — splits on ## and ### headers,
         preserving section context as metadata.
      2. RecursiveCharacterTextSplitter — breaks oversized sections
         into 800-char chunks with 150-char overlap.

    Returns a list of dicts: {"content": str, "metadata": dict}
    """
    # Stage 1: Split by markdown headers
    headers_to_split = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,  # Keep headers in content for context
    )
    md_chunks = md_splitter.split_text(text)

    # Stage 2: Further split large chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks = []
    for i, doc in enumerate(md_chunks):
        # doc is a LangChain Document with page_content and metadata
        content = doc.page_content
        section_metadata = doc.metadata  # e.g. {"header2": "TECHNICAL_IMPLEMENTATION"}

        # Build a human-readable section label
        section_parts = []
        for key in ["header1", "header2", "header3"]:
            if key in section_metadata:
                section_parts.append(section_metadata[key])
        section_label = " > ".join(section_parts) if section_parts else ""

        # Sub-split if content is too long
        sub_chunks = text_splitter.split_text(content)

        for j, sub_chunk in enumerate(sub_chunks):
            final_chunks.append({
                "content": sub_chunk,
                "metadata": {
                    "source": source_name,
                    "section": section_label,
                    "chunk_index": len(final_chunks),
                },
            })

    return final_chunks


# ─────────────────────────────────────────────
# Helper: Simple Chunking (for About_me.md)
# ─────────────────────────────────────────────
def chunk_simple(text: str, source_name: str) -> list[dict]:
    """
    Fallback chunking for files without heavy markdown structure.
    Uses RecursiveCharacterTextSplitter only.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "metadata": {
                "source": source_name,
                "section": "",
                "chunk_index": i,
            },
        }
        for i, chunk in enumerate(chunks)
    ]


# ─────────────────────────────────────────────
# Main Ingestion Pipeline
# ─────────────────────────────────────────────
def ingest_file(file_path: str, source_name: str, use_markdown_splitting: bool = False):
    """
    Load a file, chunk it, embed it, and upload to SQLite.
    Resumable: skips chunks whose content_hash already exists in the DB.
    """
    # 1. Load file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Loaded {len(text)} characters from {source_name}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return 0, 0

    # 2. Strip YAML frontmatter for project content files
    if use_markdown_splitting:
        text = strip_yaml_frontmatter(text)
        logger.info(f"  Stripped frontmatter → {len(text)} characters remaining")

    # 3. Skip empty files
    if not text.strip():
        logger.warning(f"  Skipping {source_name} — empty after processing")
        return 0, 0

    # 4. Chunk the text
    if use_markdown_splitting:
        chunks = chunk_markdown(text, source_name)
    else:
        chunks = chunk_simple(text, source_name)
    logger.info(f"  Split into {len(chunks)} chunks")

    # 5. Compute content hashes for all chunks
    for chunk_data in chunks:
        chunk_data["content_hash"] = hashlib.md5(
            chunk_data["content"].encode()
        ).hexdigest()

    # 6. Load existing content_hashes and embeddings for this source from the DB
    conn = sqlite3.connect(DB_PATH)
    existing_rows = conn.execute(
        "SELECT content_hash, embedding FROM documents WHERE json_extract(metadata, '$.source') = ?",
        (source_name,)
    ).fetchall()
    
    # Map hash to embedding blob
    existing_cache = {row[0]: row[1] for row in existing_rows if row[0]}

    # 7. Delete old rows for this source (will re-insert current chunks)
    logger.info(f"  Clearing existing documents for source: {source_name}")
    conn.execute(
        "DELETE FROM documents WHERE json_extract(metadata, '$.source') = ?",
        (source_name,)
    )
    conn.commit()

    # 8. Determine which chunks need new embeddings
    new_chunks = [c for c in chunks if c["content_hash"] not in existing_cache]
    cached_chunks = [c for c in chunks if c["content_hash"] in existing_cache]

    if not new_chunks and cached_chunks:
        logger.info(f"  All {len(chunks)} chunks already cached — skipping embedding")
    elif new_chunks:
        logger.info(
            f"  {len(new_chunks)} new chunks to embed, "
            f"{len(cached_chunks)} cached"
        )

    # 9. Batch-embed new chunks via NIM API
    if new_chunks:
        new_texts = [c["content"] for c in new_chunks]
        total_batches = (len(new_texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        all_embeddings = []
        for i in range(total_batches):
            batch = new_texts[i * EMBED_BATCH_SIZE : (i + 1) * EMBED_BATCH_SIZE]
            logger.info(f"  Embedding batch {i+1}/{total_batches}...")
            batch_embeddings = nim_embed(batch, input_type="passage")
            all_embeddings.extend(batch_embeddings)

        # Attach embeddings to new chunks
        for chunk_data, emb in zip(new_chunks, all_embeddings):
            chunk_data["embedding"] = emb.tobytes()

    # 10. For cached chunks, reuse the existing embedding blob from the DB
    if cached_chunks:
        for chunk_data in cached_chunks:
            chunk_data["embedding"] = existing_cache[chunk_data["content_hash"]]

    # 11. Insert all chunks into the DB
    successful = 0
    failed = 0

    for i, chunk_data in enumerate(chunks):
        try:
            emb_blob = chunk_data["embedding"]
            conn.execute(
                "INSERT INTO documents (content, metadata, embedding, content_hash) VALUES (?, ?, ?, ?)",
                (
                    chunk_data["content"],
                    json.dumps(chunk_data["metadata"]),
                    emb_blob,
                    chunk_data["content_hash"],
                )
            )
            successful += 1
        except Exception as e:
            logger.exception(f"  Failed to insert chunk {i+1}: {e}")
            failed += 1
            continue

    conn.commit()
    conn.close()

    logger.info(f"  {source_name}: {successful} uploaded, {failed} failed")
    return successful, failed


def main():
    # Initialize the database
    init_db()

    total_successful = 0
    total_failed = 0

    # ── 1. Ingest About_me.md (simple chunking) ──
    logger.info("=" * 60)
    logger.info("INGESTING: About_me.md")
    logger.info("=" * 60)
    s, f = ingest_file("../About_me.md", "About_me.md", use_markdown_splitting=False)
    total_successful += s
    total_failed += f

    # ── 2. Ingest all project content files (markdown-aware chunking) ──
    project_dir = "../project contents"
    if os.path.isdir(project_dir):
        md_files = sorted(glob.glob(os.path.join(project_dir, "*.md")))
        logger.info(f"\nFound {len(md_files)} project files in '{project_dir}'")

        for md_file in md_files:
            source_name = os.path.basename(md_file)
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"INGESTING: {source_name}")
            logger.info("=" * 60)
            s, f = ingest_file(md_file, source_name, use_markdown_splitting=True)
            total_successful += s
            total_failed += f
    else:
        logger.warning(f"Project contents directory not found: {project_dir}")

    # ── 3. Populate meta table ──
    conn = sqlite3.connect(DB_PATH)

    # Schema version
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("schema_version", "2")
    )

    # Embed model
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("embed_model", os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1"))
    )

    # Embed dimension — read from first row in DB
    row = conn.execute("SELECT embedding FROM documents LIMIT 1").fetchone()
    if row and row[0]:
        embed_dim = len(np.frombuffer(row[0], dtype=np.float32))
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("embed_dim", str(embed_dim))
        )
        logger.info(f"Embed dimension: {embed_dim}")

    conn.commit()
    conn.close()

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE: {total_successful} total chunks uploaded, {total_failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
