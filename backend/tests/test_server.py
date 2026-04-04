# backend/tests/test_server.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch env vars before importing server to avoid missing key errors
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-key")

from unittest.mock import patch, MagicMock

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
