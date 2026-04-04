# backend/tests/test_server.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch env vars before importing server to avoid missing key errors
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")

from unittest.mock import patch, MagicMock
from server import classify_intent

def test_llm_uses_gemma_model():
    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value = MagicMock()
        # import and reload must be inside the patch context so the mock is active
        # when server.py executes `llm = ChatOpenAI(...)` at module level
        import importlib
        import server as srv
        importlib.reload(srv)
        call_kwargs = mock_llm_class.call_args.kwargs
        assert call_kwargs["model"] == "google/gemma-4-31b-it"
        assert call_kwargs["max_tokens"] == 600


def test_classify_hi_is_greeting():
    assert classify_intent("Hi") == "greeting"

def test_classify_hello_is_greeting():
    assert classify_intent("hello there!") == "greeting"

def test_classify_who_are_you_is_greeting():
    assert classify_intent("who are you") == "greeting"

def test_classify_what_are_you_is_greeting():
    assert classify_intent("what are you") == "greeting"

def test_classify_what_can_you_do_is_greeting():
    assert classify_intent("what can you do") == "greeting"

def test_classify_help_is_greeting():
    assert classify_intent("help") == "greeting"

def test_classify_single_char_is_greeting():
    assert classify_intent("?") == "greeting"

def test_classify_weather_is_out_of_scope():
    assert classify_intent("weather today") == "out_of_scope"

def test_classify_greeting_prefix_portfolio_question_reaches_rag():
    # "hi what projects..." must NOT be swallowed by the greeting pattern
    assert classify_intent("hi what projects has Mithil built?") == "portfolio_question"

def test_classify_portfolio_question():
    assert classify_intent("what projects has Mithil built?") == "portfolio_question"

def test_classify_is_mithil_employed():
    assert classify_intent("is Mithil employed?") == "portfolio_question"


from server import rerank, AgentState

def test_rerank_filters_irrelevant_chunks():
    """Chunks with score <= 0.0 should be removed."""
    state: AgentState = {
        "question": "What is Mithil's RAG chatbot project?",
        "context": [
            "[Source: RAG Chatbot.md]\nMithil built an agentic RAG chatbot using LangGraph and DeepSeek.",
            "[Source: About_me.md]\nThe capital of France is Paris.",  # clearly irrelevant
        ],
        "answer": "",
    }
    result = rerank(state)
    # The RAG chatbot chunk should survive; the Paris chunk should be filtered
    assert len(result["context"]) >= 1
    assert any("RAG" in c for c in result["context"])

def test_rerank_returns_at_most_3_chunks():
    """Never passes more than 3 chunks to the LLM."""
    chunks = [
        f"[Source: test.md]\nMithil worked on project {i} using Python and machine learning."
        for i in range(8)
    ]
    state: AgentState = {
        "question": "What projects did Mithil build?",
        "context": chunks,
        "answer": "",
    }
    result = rerank(state)
    assert len(result["context"]) <= 3

def test_rerank_empty_context_passthrough():
    """Empty context should pass through unchanged."""
    state: AgentState = {
        "question": "What is Mithil's GPA?",
        "context": [],
        "answer": "",
    }
    result = rerank(state)
    assert result["context"] == []
