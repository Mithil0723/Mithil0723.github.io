import os
import sys
import sqlite3
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from nim_client import nim_embed_single
from server import load_vectors

try:
    docs, embeddings_matrix = load_vectors()
except Exception as e:
    print(f"Failed to load vectors: {e}")
    sys.exit(1)

for query in ["Hi", "Hello, who are you?", "Tell me a joke about data", "What is the capital of France?"]:
    try:
        vec = nim_embed_single(query, input_type="query")
    except Exception as e:
        print(f"Failed to embed '{query}': {e}")
        continue
        
    if len(docs) == 0:
        print(f"Query: '{query}' -> Database is empty")
        continue
        
    similarities = embeddings_matrix @ vec
    matches = [sim for sim in similarities if sim >= 0.16]
    print(f"Query: '{query}' -> Found {len(matches)} docs above 0.16 threshold (max similarity: {max(similarities) if len(similarities) > 0 else 0:.4f})")
