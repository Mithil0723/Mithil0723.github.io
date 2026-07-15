"""
calibrate.py — Recalibrate the vector retrieval threshold.

Runs a set of portfolio-specific questions and off-topic questions
against the rebuilt NV-Embed index. Prints the cosine score distribution
and reports the recommended MATCH_THRESHOLD.
"""

import os
import sqlite3
import numpy as np
from dotenv import load_dotenv

# Load env vars to get NIM API config
load_dotenv()

from nim_client import nim_embed_single
from server import load_vectors

QUESTIONS_ON_TOPIC = [
    "What is Mithil's email?",
    "Where did Mithil go to school?",
    "What degree does Mithil have?",
    "Tell me about the RAG chatbot project.",
    "What technologies does Mithil know?",
    "What programming languages is Mithil proficient in?",
    "Does Mithil know Python?",
    "Explain the MedAssist AI project.",
    "Did Mithil build an autonomous agent?",
    "How does the local LLM proxy work?",
]

QUESTIONS_OFF_TOPIC = [
    "What is the weather tomorrow?",
    "How do I bake a cake?",
    "What is the capital of France?",
    "Give me a recipe for chocolate chip cookies.",
    "Who won the Super Bowl last year?",
]


def score_query(query: str, embeddings_matrix: np.ndarray) -> float:
    """Returns the highest cosine similarity score for a given query."""
    try:
        q_vec = nim_embed_single(query, input_type="query")
        # Dot product with normalized matrix = cosine similarities
        similarities = embeddings_matrix @ q_vec
        return float(np.max(similarities)) if len(similarities) > 0 else 0.0
    except Exception as e:
        print(f"Error embedding query '{query}': {e}")
        return 0.0


def main():
    print("=" * 60)
    print("MATCH_THRESHOLD CALIBRATION (NV-Embed v1)")
    print("=" * 60)

    try:
        docs, embeddings_matrix = load_vectors()
    except Exception as e:
        print(f"Failed to load vectors: {e}")
        print("Please ensure vectors.db exists and is up to date (run ingest.py).")
        return

    if len(docs) == 0:
        print("Vector database is empty. Please run ingest.py first.")
        return

    print(f"\nLoaded {len(docs)} documents ({embeddings_matrix.shape[1]}-dim).")
    
    print("\nScoring ON-TOPIC questions (should have high scores):")
    on_topic_scores = []
    for q in QUESTIONS_ON_TOPIC:
        score = score_query(q, embeddings_matrix)
        on_topic_scores.append(score)
        print(f"  {score:.4f} | {q}")

    print("\nScoring OFF-TOPIC questions (should have lower scores):")
    off_topic_scores = []
    for q in QUESTIONS_OFF_TOPIC:
        score = score_query(q, embeddings_matrix)
        off_topic_scores.append(score)
        print(f"  {score:.4f} | {q}")

    if not on_topic_scores or not off_topic_scores:
        print("\nCould not complete calibration due to embedding errors.")
        return

    min_on_topic = min(on_topic_scores)
    max_off_topic = max(off_topic_scores)
    
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS:")
    print(f"  Lowest ON-TOPIC score:  {min_on_topic:.4f}")
    print(f"  Highest OFF-TOPIC score: {max_off_topic:.4f}")
    
    recommended = (min_on_topic + max_off_topic) / 2
    
    print(f"\n  Recommended MATCH_THRESHOLD: {recommended:.2f}")
    print("=" * 60)
    print("\nUpdate your .env file with:")
    print(f"MATCH_THRESHOLD={recommended:.2f}")


if __name__ == "__main__":
    main()
