import os
import re
import glob
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# HuggingFace Embeddings (same model as server.py — must match!)
# ─────────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


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
    Load a file, chunk it, embed it, and upload to Supabase.
    Clears existing documents from this source before inserting.
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

    # 4. Clear old data for this source
    logger.info(f"  Clearing existing documents for source: {source_name}")
    supabase.table("documents").delete().eq(
        "metadata->>source", source_name
    ).execute()

    # 5. Chunk the text
    if use_markdown_splitting:
        chunks = chunk_markdown(text, source_name)
    else:
        chunks = chunk_simple(text, source_name)
    logger.info(f"  Split into {len(chunks)} chunks")

    # 6. Batch-embed all chunks at once (HuggingFace runs locally — no rate limits)
    logger.info(f"  Generating embeddings for {len(chunks)} chunks...")
    chunk_texts = [c["content"] for c in chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    # 7. Upload to Supabase
    successful = 0
    failed = 0

    for i, (chunk_data, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        try:
            data = {
                "content": chunk_data["content"],
                "metadata": chunk_data["metadata"],
                "embedding": embedding,
            }
            supabase.table("documents").insert(data).execute()
            successful += 1
        except Exception as e:
            logger.exception(f"  Failed to upload chunk {i+1}: {e}")
            failed += 1
            continue

    logger.info(f"  {source_name}: {successful} uploaded, {failed} failed")
    return successful, failed


def main():
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

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"INGESTION COMPLETE: {total_successful} total chunks uploaded, {total_failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
