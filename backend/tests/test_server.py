# backend/tests/test_server.py
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch env vars before importing server
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvapi")
os.environ.setdefault("VECTOR_DB_PATH", ":memory:")  # Prevent creating files

from unittest.mock import patch, MagicMock
from server import (
    classify_intent,
    rerank,
    AgentState,
    get_llm,
    post_process_response,
    condense,
    load_vectors,
)
import numpy as np

def test_llm_uses_nemotron_model():
    with patch("server.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value = MagicMock()
        import importlib
        import server as srv
        
        # Reset the singleton
        srv._llm = None
        
        os.environ["LLM_MODEL"] = "nvidia/nemotron-3-nano-30b-a3b"
        os.environ["LLM_MAX_TOKENS"] = "400"
        
        srv.get_llm()
        
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == "nvidia/nemotron-3-nano-30b-a3b"
        assert call_kwargs["max_tokens"] == 400
        assert "chat_template_kwargs" in call_kwargs["model_kwargs"]["extra_body"]


# Intent Classifier Tests
def test_classify_hi_is_greeting():
    assert classify_intent("Hi") == "greeting"

def test_classify_portfolio_question():
    assert classify_intent("what projects has Mithil built?") == "portfolio_question"

def test_classify_weather_is_out_of_scope():
    assert classify_intent("weather today") == "out_of_scope"


# Rerank Node Tests (Mocking nim_rerank)
@patch("server.nim_rerank")
def test_rerank_orders_by_score(mock_nim_rerank):
    # Second chunk scores higher
    mock_nim_rerank.return_value = [
        {"index": 1, "logit": 0.8},
        {"index": 0, "logit": -0.3},
    ]
    state: AgentState = {
        "question": "test?",
        "context": [
            "Chunk 0",
            "Chunk 1",
        ],
        "answer": "",
        "history": [],
        "condensed_question": "",
    }
    result = rerank(state)
    assert len(result["context"]) == 2
    assert result["context"][0] == "Chunk 1"


@patch("server.nim_rerank")
def test_rerank_degrades_gracefully(mock_nim_rerank):
    mock_nim_rerank.side_effect = RuntimeError("API down")
    state: AgentState = {
        "question": "test?",
        "context": ["C1", "C2", "C3", "C4"],
        "answer": "",
        "history": [],
        "condensed_question": "",
    }
    result = rerank(state)
    # Should fall back to taking the top 3 passed in (which are by cosine sim)
    assert len(result["context"]) == 3
    assert result["context"] == ["C1", "C2", "C3"]


# Post-processing Tests
def test_post_process_strips_thinking():
    raw = "<think>I should say hello.</think>Hello there!"
    processed = post_process_response(raw)
    assert processed == "Hello there!"

def test_post_process_strips_affirmations():
    raw = "Sure! Here is the info."
    assert post_process_response(raw) == "Here is the info."
    
    raw2 = "Absolutely, I can help. The info is here."
    assert post_process_response(raw2) == "I can help. The info is here."

def test_post_process_sentence_limits():
    raw = "One. Two. Three. Four. Five. Six."
    # 4 sentences for standard questions
    processed = post_process_response(raw, is_greeting=False)
    assert processed == "One. Two. Three. Four."

    # 2 sentences for greetings
    processed_greeting = post_process_response(raw, is_greeting=True)
    assert processed_greeting == "One. Two."

def test_post_process_empty_fallback():
    raw = "   "
    processed = post_process_response(raw)
    assert "Mithil's email is always open" in processed
    
    raw2 = "<think>thinking...</think>"
    processed2 = post_process_response(raw2)
    assert "Mithil's email is always open" in processed2


# Condense Node Tests
@patch("server.get_condense_chain")
def test_condense_skips_when_no_history(mock_get_chain):
    state: AgentState = {
        "question": "What is Python?",
        "context": [],
        "answer": "",
        "history": [],
        "condensed_question": "",
    }
    result = condense(state)
    assert result["condensed_question"] == "What is Python?"
    mock_get_chain.assert_not_called()

@patch("server.get_condense_chain")
def test_condense_with_history(mock_get_chain):
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Rewritten question"
    mock_get_chain.return_value = mock_chain
    
    state: AgentState = {
        "question": "How did he use it?",
        "context": [],
        "answer": "",
        "history": [{"user": "Did he use Python?", "assistant": "Yes."}],
        "condensed_question": "",
    }
    result = condense(state)
    assert result["condensed_question"] == "Rewritten question"
    mock_chain.invoke.assert_called_once()
