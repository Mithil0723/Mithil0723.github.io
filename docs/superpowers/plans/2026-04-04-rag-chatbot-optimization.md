# RAG Chatbot Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate hallucination, reduce verbosity, and improve factual accuracy in the RAG chatbot by adding cross-encoder reranking, swapping the LLM to Gemma 4 31B, hardening the prompt, and fixing intent classifier edge cases.

**Architecture:** A new `rerank` LangGraph node is inserted between `retrieve` and `grade`. It uses a local CrossEncoder to score all retrieved chunks against the question and passes only the top-3 relevant ones to the LLM. The LLM is swapped to `google/gemma-4-31b-it` which better respects instruction-tuned grounding rules.

**Tech Stack:** Python, FastAPI, LangGraph, LangChain, `sentence-transformers` (CrossEncoder), OpenRouter, Supabase pgvector

---

## File Map

| File | Change |
|---|---|
| `backend/server.py` | All changes — model swap, prompt hardening, intent classifier, rerank node, pipeline rewiring, retrieval params |
| `backend/tests/test_server.py` | New — unit tests for `classify_intent` and `rerank` |

---

## Task 1: Swap LLM model and raise max_tokens

**Files:**
- Modify: `backend/server.py:67-73`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/__init__.py` (empty) and `backend/tests/test_server.py`:

```python
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
    with patch("server.ChatOpenAI") as mock_llm_class:
        mock_llm_class.return_value = MagicMock()
        import importlib
        import server as srv
        importlib.reload(srv)
        call_kwargs = mock_llm_class.call_args[1]
        assert call_kwargs["model"] == "google/gemma-4-31b-it"
        assert call_kwargs["max_tokens"] == 600
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend
python -m pytest tests/test_server.py::test_llm_uses_gemma_model -v
```

Expected: FAIL — `AssertionError` because model is still `deepseek/deepseek-v3.2`

- [ ] **Step 3: Update the LLM block in `backend/server.py`**

Replace lines 67–73:

```python
llm = ChatOpenAI(
    model="google/gemma-4-31b-it",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=600,
)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd backend
python -m pytest tests/test_server.py::test_llm_uses_gemma_model -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/server.py backend/tests/__init__.py backend/tests/test_server.py
git commit -m "feat: swap LLM to Gemma 4 31B, raise max_tokens to 600"
```

---

## Task 2: Harden the system prompt

**Files:**
- Modify: `backend/server.py:78-116`

- [ ] **Step 1: Replace the `SYSTEM_INSTRUCTION` constant**

Replace the entire `SYSTEM_INSTRUCTION` string (lines 78–107) with:

```python
SYSTEM_INSTRUCTION = (
    "You are the AI assistant for Mithil Ravulapalli's portfolio — sharp, friendly, and concise.\n\n"
    "Mithil is a BS Senior majoring in Data Science at UIC. He is NOT a Data Scientist, "
    "NOT employed, and NOT a graduate student.\n\n"
    "---\n\n"
    "CORE RULES (follow strictly):\n\n"
    "1. GROUND EVERYTHING. Only state facts that are explicitly present in the provided "
    "context chunks. If a fact is not in the context, do not state it — not even as a guess, "
    "inference, or extrapolation.\n\n"
    "2. WHEN CONTEXT IS EMPTY. If no context chunks are provided, say exactly: "
    "\"That one's outside my knowledge, but Mithil's email is always open!\"\n\n"
    "3. WHEN CONTEXT IS PARTIAL. If context chunks exist but do not fully answer the question, "
    "state only what the context confirms, then append: "
    "\"For more detail, Mithil's email is always open!\"\n\n"
    "4. BE BRIEF. Hard limits — no exceptions:\n"
    "   - Conversational or greeting question → Maximum 2 sentences. Stop.\n"
    "   - Factual question about a project or skill → Maximum 4 sentences. Stop.\n"
    "   - Never use bullet points unless the user explicitly asks for a list.\n"
    "   - Never use filler phrases: 'Great question!', 'Certainly!', 'Of course!', 'Sure!', "
    "'Absolutely!', or any similar opener.\n"
    "   - Never repeat or summarize the question back to the user.\n\n"
    "5. EXAMPLE OF GOOD BREVITY:\n"
    "   User: \"What's Mithil's main project?\"\n"
    "   Good: \"Mithil's flagship project is an agentic RAG chatbot — a LangGraph-orchestrated "
    "pipeline that answers visitor questions grounded in his portfolio data, powered by "
    "Gemma 4 31B via OpenRouter.\"\n"
    "   Bad: \"Great question! Mithil has worked on several exciting projects. Let me walk you "
    "through his main one...[4 paragraphs]\"\n\n"
    "6. NEVER hallucinate skills, job titles, employment status, or project features not "
    "explicitly stated in the context. Do not infer. Do not extrapolate.\n\n"
    "7. Context chunks are tagged [Source] and [Section] — mine them thoroughly before responding.\n\n"
)
```

- [ ] **Step 2: Verify the server still imports cleanly**

```bash
cd backend
python -c "import server; print('OK')"
```

Expected output: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/server.py
git commit -m "feat: harden system prompt with explicit sentence limits and forbidden phrases"
```

---

## Task 3: Improve intent classifier

**Files:**
- Modify: `backend/server.py:142-175`
- Test: `backend/tests/test_server.py`

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_server.py`:

```python
# Import classify_intent directly — it has no external dependencies
from server import classify_intent

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
    assert classify_intent("what's the weather today?") == "out_of_scope"

def test_classify_portfolio_question():
    assert classify_intent("what projects has Mithil built?") == "portfolio_question"

def test_classify_is_mithil_employed():
    assert classify_intent("is Mithil employed?") == "portfolio_question"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_server.py -k "classify" -v
```

Expected: Several FAILs for `who are you`, `what are you`, `what can you do`, `help`, single char

- [ ] **Step 3: Replace `classify_intent` in `backend/server.py`**

Replace the entire `classify_intent` function (lines 142–175):

```python
def classify_intent(message: str) -> str:
    """
    Classifies user input into one of three buckets:
    - 'greeting'           : hi, hello, bot-identity questions, single chars, etc.
    - 'out_of_scope'       : clearly unrelated to portfolio (weather, math, etc.)
    - 'portfolio_question' : anything else — run full RAG pipeline
    """
    msg = message.lower().strip()

    # Single character or punctuation-only input
    if len(msg) <= 2 and not msg.isalpha() or len(msg) == 1:
        return "greeting"

    greeting_patterns = [
        r"^(hi|hey|hello|howdy|sup|what'?s up|yo)(\s.*)?([\.!\?]*)$",
        r"^(good (morning|afternoon|evening|night))([\.!\?]*)$",
        r"^(thanks|thank you|thx|ty)([\.!\?\s]*)$",
        r"^(bye|goodbye|see you|cya|take care)([\.!\?]*)$",
        r"^(nice|cool|great|awesome|ok|okay|got it|sounds good)([\.!\?]*)$",
        r"^(who are you|what are you|what can you do|help|what do you do)([\.!\?]*)$",
        r"^(tell me about yourself)([\.!\?]*)$",
    ]

    out_of_scope_patterns = [
        r"\b(weather|forecast|temperature|rain|snow)\b",
        r"\b(stock|crypto|bitcoin|price of)\b",
        r"\b(recipe|cook|bake|food)\b",
        r"\b(translate|what does .+ mean in)\b",
        r"\b(capital of|population of|how far is)\b",
    ]

    for pattern in greeting_patterns:
        if re.match(pattern, msg):
            return "greeting"

    for pattern in out_of_scope_patterns:
        if re.search(pattern, msg):
            return "out_of_scope"

    return "portfolio_question"
```

- [ ] **Step 4: Update the greeting reply for bot-identity queries in the `/chat` endpoint**

In the `chat_endpoint` function, replace the greeting reply (around line 309):

```python
if intent == "greeting":
    return {"reply": "Hey! I'm Mithil's portfolio assistant — ask me about his projects, skills, or background!"}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
python -m pytest tests/test_server.py -k "classify" -v
```

Expected: All 10 classify tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/server.py backend/tests/test_server.py
git commit -m "feat: improve intent classifier with bot-identity and edge-case patterns"
```

---

## Task 4: Add cross-encoder rerank node and update retrieval params

**Files:**
- Modify: `backend/server.py` — add import, add reranker singleton, add `rerank` node, rewire pipeline, update retrieval params

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
python -m pytest tests/test_server.py -k "rerank" -v
```

Expected: FAIL — `ImportError: cannot import name 'rerank' from 'server'`

- [ ] **Step 3: Add CrossEncoder import to `backend/server.py`**

Add after the existing imports (around line 11, after `from langchain_huggingface`):

```python
from sentence_transformers import CrossEncoder
```

- [ ] **Step 4: Add the reranker singleton after `get_embeddings()` in `backend/server.py`**

Add after the `get_embeddings` function (around line 62):

```python
# ─────────────────────────────────────────────
# 1b. CrossEncoder Reranker (lazy-loaded)
# ─────────────────────────────────────────────
_reranker = None


def get_reranker():
    """Lazy singleton — loads the CrossEncoder reranker on first call."""
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder reranker model (first request)...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("Reranker model loaded successfully")
    return _reranker
```

- [ ] **Step 5: Update retrieval params in the `retrieve` node**

In the `retrieve` function, update the `supabase.rpc` call params:

```python
response = supabase.rpc("match_documents", {
    "query_embedding": query_vector,
    "match_threshold": 0.40,
    "match_count": 8,
}).execute()
```

- [ ] **Step 6: Add the `rerank` node function after `retrieve` in `backend/server.py`**

Add the `rerank` function between `retrieve` and `grade`:

```python
def rerank(state: AgentState) -> AgentState:
    """
    Node 2 — Cross-encoder reranking.
    Scores each retrieved chunk against the question using ms-marco-MiniLM-L-6-v2.
    Keeps only chunks with score > 0.0, capped at top-3.
    Eliminates noisy retrievals before they reach the LLM.
    """
    if not state["context"]:
        return state

    logger.info(f"Node: rerank — scoring {len(state['context'])} chunks")
    question = state["question"]
    pairs = [[question, chunk] for chunk in state["context"]]
    scores = get_reranker().predict(pairs)

    scored = sorted(zip(scores, state["context"]), key=lambda x: x[0], reverse=True)
    filtered = [chunk for score, chunk in scored if score > 0.0][:3]

    logger.info(f"Rerank: {len(state['context'])} → {len(filtered)} chunks after filtering")
    return {**state, "context": filtered}
```

- [ ] **Step 7: Rewire the LangGraph pipeline**

Replace the pipeline assembly block (around lines 244–263) with:

```python
# Compile the LangGraph StateGraph
# Nodes run in order: retrieve → rerank → grade → generate → END
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("grade", grade)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "grade")

def route_after_grade(state: AgentState) -> str:
    if state.get("answer"):
        return "__end__"
    return "generate"

builder.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "__end__": END})
builder.add_edge("generate", END)

rag_graph = builder.compile()
```

- [ ] **Step 8: Run the rerank tests**

```bash
cd backend
python -m pytest tests/test_server.py -k "rerank" -v
```

Expected: All 3 rerank tests PASS

- [ ] **Step 9: Run the full test suite**

```bash
cd backend
python -m pytest tests/test_server.py -v
```

Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add backend/server.py backend/tests/test_server.py
git commit -m "feat: add cross-encoder rerank node, widen retrieval to top-8 at threshold 0.40"
```

---

## Task 5: End-to-end smoke test

**Files:**
- Run: `backend/run_tests.py` (requires running server)

- [ ] **Step 1: Start the server locally**

```bash
cd backend
uvicorn server:app --reload --port 8000
```

Expected: Server starts, logs show `Uvicorn running on http://127.0.0.1:8000`

- [ ] **Step 2: Run the existing test suite against the live server**

In a second terminal:

```bash
cd backend
python run_tests.py
```

Expected results to verify manually in `test_results.txt`:
- Test 1 (`hi`) → Short greeting, no RAG triggered
- Test 2 (`hello there!`) → Short greeting
- Test 3 (`what's the weather today?`) → Out-of-scope redirect
- Test 4 (`what projects has Mithil built?`) → Accurate list, ≤4 sentences, no fabrication
- Test 5 (`is Mithil a Data Scientist?`) → Clearly says NO
- Test 6 (`what is Mithil's GPA?`) → Fallback message
- Test 7 (`tell me about the RAG chatbot`) → Accurate description from context, ≤4 sentences
- Test 8 (`does Mithil know React?`) → Fallback or "not mentioned"
- Test 9 (`thanks`) → Short acknowledgment
- Test 10 (`what companies has Mithil worked at?`) → Fallback

- [ ] **Step 3: Commit final state**

```bash
git add backend/test_results.txt backend/test_results.json
git commit -m "test: add smoke test results for RAG optimization"
```
