import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def main():
    # 1. Load File — path is relative to the backend/ folder
    file_path = "../About_me.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"Loaded {len(text)} characters from {file_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        logger.info("Make sure About_me.md exists in the parent directory of backend/")
        return

    # 2. Clear old data to prevent duplicates on re-runs
    logger.info("Clearing existing documents from About_me.md...")
    supabase.table("documents").delete().eq(
        "metadata->>source", "About_me.md"
    ).execute()

    # 3. Split into Chunks using LangChain's RecursiveCharacterTextSplitter
    #    This splitter tries paragraph → sentence → word boundaries in order,
    #    so chunks are always semantically coherent.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split into {len(chunks)} chunks")

    # 4. Batch-embed all chunks at once (HuggingFace runs locally — no rate limits)
    logger.info("Generating embeddings for all chunks...")
    chunk_embeddings = embeddings.embed_documents(chunks)
    logger.info(f"Generated {len(chunk_embeddings)} embeddings")

    # 5. Upload to Supabase
    successful = 0
    failed = 0

    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        try:
            logger.info(f"Uploading chunk {i+1}/{len(chunks)}...")
            data = {
                "content": chunk,
                "metadata": {"source": "About_me.md", "chunk_index": i},
                "embedding": embedding,
            }
            supabase.table("documents").insert(data).execute()
            successful += 1
        except Exception as e:
            logger.exception(f"Failed to upload chunk {i+1}: {e}")
            failed += 1
            continue

    logger.info(f"Ingestion complete! {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
