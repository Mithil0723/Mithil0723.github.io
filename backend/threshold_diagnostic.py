"""One-off diagnostic: check Supabase vector store health at a low threshold."""
import os
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dotenv import load_dotenv
from supabase import create_client
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

test_queries = [
    "What projects has Mithil worked on?",
    "Tell me about the RAG chatbot",
    "What skills does Mithil have?",
]

with open("diagnostic_results.txt", "w", encoding="utf-8") as out:
    for query in test_queries:
        out.write(f"\n{'='*60}\n")
        out.write(f"Query: {query}\n")
        out.write(f"{'='*60}\n")
        vector = embeddings.embed_query(query)
        result = supabase.rpc("match_documents", {
            "query_embedding": vector,
            "match_threshold": 0.30,
            "match_count": 5,
        }).execute()

        if not result.data:
            out.write("  NO RESULTS -- vector store may be empty or stale\n")
        else:
            for doc in result.data:
                sim = doc.get("similarity", 0)
                source = doc.get("metadata", {}).get("source", "unknown")
                section = doc.get("metadata", {}).get("section", "")
                # Strip emojis for clean output
                section_clean = section.encode('ascii', 'ignore').decode('ascii').strip()
                out.write(f"  Similarity: {sim:.4f} | Source: {source} | Section: {section_clean}\n")

    out.write("\nDONE\n")
print("Results written to diagnostic_results.txt")
