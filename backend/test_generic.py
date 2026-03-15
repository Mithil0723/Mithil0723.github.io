import os
import sys
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import create_client

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

for query in ["Hi", "Hello, who are you?", "Tell me a joke about data", "What is the capital of France?"]:
    vec = embeddings.embed_query(query)
    result = sb.rpc("match_documents", {
        "query_embedding": vec,
        "match_threshold": 0.45,
        "match_count": 5,
    }).execute()
    print(f"Query: '{query}' -> Found {len(result.data)} docs")
