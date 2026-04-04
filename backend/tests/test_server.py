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
