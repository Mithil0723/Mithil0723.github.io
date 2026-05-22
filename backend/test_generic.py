import os
import sys
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

DB_PATH = os.getenv("VECTOR_DB_PATH", "vectors.db")
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT content, embedding FROM documents").fetchall()
conn.close()

docs = [r[0] for r in rows]
embeddings_matrix = np.array([json.loads(r[1]) for r in rows], dtype=np.float32)

for query in ["Hi", "Hello, who are you?", "Tell me a joke about data", "What is the capital of France?"]:
    vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    if len(docs) == 0:
        print(f"Query: '{query}' -> Database is empty")
        continue
    similarities = embeddings_matrix @ vec
    matches = [sim for sim in similarities if sim >= 0.25]
    print(f"Query: '{query}' -> Found {len(matches)} docs above 0.25 threshold (max similarity: {max(similarities) if len(similarities) > 0 else 0:.4f})")
