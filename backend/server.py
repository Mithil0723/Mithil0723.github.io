"""
server.py — FastAPI RAG Agent for Mithil's portfolio chatbot.

Architecture: LangGraph StateGraph (retrieve → rerank → generate → END)
All models served via NVIDIA NIM API at integrate.api.nvidia.com.

Tasks addressed:
  1  — NV-Embed v1 embeddings (asymmetric query/passage, BLOB storage, dim guard)
  2  — NIM reranker with graceful degradation
  3  — Nemotron 3 LLM (reasoning disabled, env-driven)
  5  — Lazy-loaded LLM client
  6  — Network resilience (timeouts, retries, clean error messages)
  8  — CORS lockdown
  9  — Rate limiting with slowapi
  10 — Conversation history (session_id, condense node, LRU store)
  11 — Concise response enforcement (sentence truncation, affirmation strip)
  12 — Correctness cleanup (stale log, grade node collapsed, intent classifier)
"""

import os
import re
import json
import time
import uuid
import sqlite3
import logging
from typing import TypedDict, List, Optional
from collections import OrderedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# LangChain / LangGraph / LangSmith
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langsmith import traceable

# Rate limiting (Task 9)
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

import numpy as np

# Local NIM client (Tasks 1, 2, 6)
from nim_client import nim_embed_single, nim_rerank

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# ─────────────────────────────────────────────
# Rate Limiting (Task 9)
# ─────────────────────────────────────────────
def _get_real_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For (Render is behind a proxy)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=_get_real_ip)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


# ─────────────────────────────────────────────
# CORS Middleware (Task 8)
# ─────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "https://mithil0723.github.io"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # No cookies/auth — True + wildcard is invalid per CORS spec
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────
# 1. LLM Client — Lazy-loaded (Task 3, 5)
# ─────────────────────────────────────────────
_llm = None
_rag_chain = None


def get_llm():
    """Lazy singleton — builds the ChatOpenAI client on first call.
    Uvicorn binds the port immediately; missing key won't crash at import."""
    global _llm
    if _llm is None:
        logger.info("Initializing LLM client (first request)...")

        enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
        force_nonempty = os.getenv("LLM_FORCE_NONEMPTY_CONTENT", "true").lower() == "true"

        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                "force_nonempty_content": force_nonempty,
            }
        }

        _llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
            openai_api_key=os.getenv("NVIDIA_API_KEY"),
            openai_api_base=os.getenv(
                "NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"
            ),
            temperature=0.2,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "400")),
            request_timeout=30,  # Task 6: explicit timeout for generate
            model_kwargs={"extra_body": extra_body},
        )
        logger.info(
            f"LLM client initialized: model={os.getenv('LLM_MODEL', 'nvidia/nemotron-3-nano-30b-a3b')}"
        )
    return _llm


# ─────────────────────────────────────────────
# 2. Prompt Template
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are the AI assistant for Mithil Ravulapalli's portfolio — sharp, friendly, and concise.\n\n"
    "Mithil holds a BS in Data Science from UIC (recent graduate). He is NOT a Data Scientist "
    "by job title and NOT currently employed — do not claim any job title or employer for him.\n\n"
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
    "4. BE BRIEF. Hard limits — no exceptions. These limits apply to the main answer body "
    "(the required fallback suffixes in Rules 2 and 3 do not count against the cap):\n"
    "   - Conversational or greeting question → Maximum 2 sentences. Stop.\n"
    "   - Factual question about a project or skill → Maximum 4 sentences. Stop.\n"
    "   - Never use bullet points unless the user explicitly asks for a list.\n"
    "   - Never begin a response with a one-word affirmation, compliment, or filler phrase "
    "of any kind (e.g. 'Sure!', 'Great!', 'Certainly!', 'Of course!', 'Absolutely!', "
    "'Happy to help!' — and any variation of these).\n"
    "   - Never repeat or summarize the question back to the user.\n\n"
    "5. EXAMPLE OF GOOD BREVITY:\n"
    "   User: \"What's Mithil's main project?\"\n"
    "   Good: \"Mithil's flagship project is an agentic RAG chatbot — a LangGraph-orchestrated "
    "pipeline that answers visitor questions grounded in his portfolio data.\"\n"
    "   Bad: \"Great question! Mithil has worked on several exciting projects. Let me walk you "
    "through his main one...[4 paragraphs]\"\n\n"
    "6. NEVER hallucinate skills, job titles, employment status, or project features not "
    "explicitly stated in the context. Do not infer. Do not extrapolate.\n\n"
    "7. Context chunks are tagged [Source] and [Section] — mine them thoroughly before responding.\n\n"
)

# Prompt template WITHOUT history (single-shot questions)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        (
            "human",
            "CONTEXT:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "Answer using only the facts in the context above. Follow all rules in the system prompt exactly.",
        ),
    ]
)

# Prompt template WITH history (Task 10)
prompt_template_with_history = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTION),
        (
            "human",
            "PRIOR CONVERSATION (for continuity only — do NOT treat as grounding facts):\n"
            "{history}\n\n"
            "---\n\n"
            "CONTEXT (grounding source — only these facts may be stated):\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "Answer using only the facts in the CONTEXT above. "
            "The prior conversation is for understanding follow-up intent only. "
            "Follow all rules in the system prompt exactly.",
        ),
    ]
)

# Condense prompt for follow-up questions (Task 10)
condense_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions into standalone questions. "
            "Given the conversation history, rewrite the latest user message "
            "as a self-contained question that can be understood without context. "
            "Output ONLY the rewritten question, nothing else.",
        ),
        (
            "human",
            "CONVERSATION:\n{history}\n\n"
            "LATEST MESSAGE: {question}\n\n"
            "Rewrite as a standalone question:",
        ),
    ]
)


def get_rag_chain(with_history: bool = False):
    """Lazy accessor for the RAG chain."""
    template = prompt_template_with_history if with_history else prompt_template
    return template | get_llm() | StrOutputParser()


def get_condense_chain():
    """Lazy accessor for the condense chain."""
    return condense_template | get_llm() | StrOutputParser()


# ─────────────────────────────────────────────
# 3. SQLite Vector Store (lazy-loaded into memory)
# ─────────────────────────────────────────────
DB_PATH = os.getenv("VECTOR_DB_PATH", "vectors.db")
_vectors_cache = None

# Expected embedding dimension for the configured model
EXPECTED_EMBED_DIM = 4096  # NV-Embed v1


def load_vectors():
    """Load all embeddings from SQLite into memory (one-time).
    Returns (docs_list, embeddings_matrix) where embeddings_matrix
    is a numpy array of shape (n_docs, embed_dim).

    Task 1c: dimension guard — raises RuntimeError if loaded dim doesn't match meta.
    Task 1d: reads BLOBs instead of JSON.
    """
    global _vectors_cache
    if _vectors_cache is not None:
        return _vectors_cache

    logger.info(f"Loading vectors from SQLite database: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        logger.warning(f"Vector database not found: {DB_PATH}")
        _vectors_cache = ([], np.array([], dtype=np.float32).reshape(0, 0))
        return _vectors_cache

    conn = sqlite3.connect(DB_PATH)

    # Task 1c: Check schema version and expected dimension from meta table
    expected_dim = None
    try:
        meta_rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {row[0]: row[1] for row in meta_rows}
        expected_dim = int(meta.get("embed_dim", 0))
        stored_model = meta.get("embed_model", "unknown")
        schema_version = meta.get("schema_version", "unknown")
        logger.info(
            f"Vector DB meta: schema_version={schema_version}, "
            f"model={stored_model}, dim={expected_dim}"
        )
    except sqlite3.OperationalError:
        # meta table doesn't exist — likely old schema
        logger.warning(
            "No 'meta' table found in vectors.db — this may be an old 384-dim database. "
            "Re-run ingest.py to rebuild."
        )

    rows = conn.execute(
        "SELECT id, content, metadata, embedding FROM documents"
    ).fetchall()
    conn.close()

    if not rows:
        logger.warning("No documents found in SQLite database")
        _vectors_cache = ([], np.array([], dtype=np.float32).reshape(0, 0))
        return _vectors_cache

    docs = []
    embeddings_list = []
    for row in rows:
        docs.append(
            {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]),
            }
        )
        # Task 1d: Read BLOB format
        if isinstance(row[3], bytes):
            embeddings_list.append(np.frombuffer(row[3], dtype=np.float32))
        else:
            # Fallback for old JSON format (shouldn't happen after re-ingest)
            embeddings_list.append(np.array(json.loads(row[3]), dtype=np.float32))

    embeddings_matrix = np.array(embeddings_list, dtype=np.float32)

    # Task 1c: Dimension guard
    actual_dim = embeddings_matrix.shape[1] if len(embeddings_matrix.shape) == 2 else 0
    if expected_dim and actual_dim != expected_dim:
        error_msg = (
            f"DIMENSION MISMATCH: vectors.db contains {actual_dim}-dim embeddings "
            f"but meta table says {expected_dim}-dim. Re-run ingest.py to rebuild."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    embed_model = os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1")
    if actual_dim > 0 and actual_dim != EXPECTED_EMBED_DIM:
        error_msg = (
            f"DIMENSION MISMATCH: vectors.db contains {actual_dim}-dim embeddings "
            f"but current EMBED_MODEL ({embed_model}) expects {EXPECTED_EMBED_DIM}-dim. "
            f"Re-run ingest.py to rebuild."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    _vectors_cache = (docs, embeddings_matrix)
    logger.info(
        f"Loaded {len(docs)} document vectors ({actual_dim}-dim) into memory"
    )
    return _vectors_cache


# ─────────────────────────────────────────────
# 4. Session History Store (Task 10)
# ─────────────────────────────────────────────
MAX_SESSIONS = 500
SESSION_TTL = 30 * 60  # 30 minutes in seconds
MAX_TURNS = 4  # last 4 turns per session


class SessionStore:
    """Bounded in-memory LRU session store with TTL.
    500 sessions max, 30-minute TTL, 4 turns per session."""

    def __init__(self):
        self._store: OrderedDict[str, dict] = OrderedDict()

    def get(self, session_id: str) -> list[dict] | None:
        """Get history for a session. Returns None if expired or not found."""
        if session_id not in self._store:
            return None
        entry = self._store[session_id]
        if time.time() - entry["last_access"] > SESSION_TTL:
            del self._store[session_id]
            return None
        # Move to end (LRU)
        self._store.move_to_end(session_id)
        entry["last_access"] = time.time()
        return entry["turns"]

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        """Add a turn to a session's history."""
        if session_id not in self._store:
            # Evict oldest if at capacity
            while len(self._store) >= MAX_SESSIONS:
                self._store.popitem(last=False)
            self._store[session_id] = {"turns": [], "last_access": time.time()}

        entry = self._store[session_id]
        entry["turns"].append({"user": user_msg, "assistant": assistant_msg})
        # Keep only last MAX_TURNS
        if len(entry["turns"]) > MAX_TURNS:
            entry["turns"] = entry["turns"][-MAX_TURNS:]
        entry["last_access"] = time.time()
        self._store.move_to_end(session_id)

    def cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid
            for sid, entry in self._store.items()
            if now - entry["last_access"] > SESSION_TTL
        ]
        for sid in expired:
            del self._store[sid]


session_store = SessionStore()


# ─────────────────────────────────────────────
# 5. Intent Classifier (unchanged from original)
# ─────────────────────────────────────────────
# Task 12: Confirm greeting short-circuit fires before any network call — yes,
# classify_intent is called in chat_endpoint before rag_graph.invoke.


def classify_intent(message: str) -> str:
    """
    Classifies user input into one of three buckets:
    - 'greeting'           : hi, hello, bot-identity questions, single chars, etc.
    - 'out_of_scope'       : clearly unrelated to portfolio (weather, math, etc.)
    - 'portfolio_question' : anything else — run full RAG pipeline
    """
    msg = message.lower().strip()

    # Single character, or two non-alpha characters (e.g. "?", "!!")
    if len(msg) == 1 or (len(msg) == 2 and not msg.isalpha()):
        return "greeting"

    greeting_patterns = [
        # Only allow short social suffixes after greeting word — prevents "hi what projects..." matching
        r"^(hi|hey|hello|howdy|sup|what'?s up|yo)(\s+(there|friend|bot|mate|everyone|all))?[.!?]*$",
        r"^(good (morning|afternoon|evening|night))[.!?]*$",
        r"^(thanks|thank you|thx|ty)[.!?\s]*$",
        r"^(bye|goodbye|see you|cya|take care)[.!?]*$",
        r"^(nice|cool|great|awesome|ok|okay|got it|sounds good)[.!?]*$",
        r"^(who are you|what are you|what can you do|help|what do you do)[.!?]*$",
        r"^(tell me about yourself)[.!?]*$",
    ]

    out_of_scope_patterns = [
        r"\b(weather forecast|rain tomorrow|will it snow|weather today)\b",
        r"\b(stock|crypto|bitcoin|price of)\b",
        r"\b(recipe for|how to bake|how to cook)\b",
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


# ─────────────────────────────────────────────
# 6. Response Post-Processing (Task 11)
# ─────────────────────────────────────────────

# Compiled regexes for performance
_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_AFFIRMATION_RE = re.compile(
    r"^(Sure|Great|Certainly|Of course|Absolutely|Happy to help)[!,.]?\s*",
    re.IGNORECASE,
)

# Fallback strings that must be preserved after truncation
_FALLBACK_SUFFIX_1 = "That one's outside my knowledge, but Mithil's email is always open!"
_FALLBACK_SUFFIX_2 = "For more detail, Mithil's email is always open!"
_FALLBACK_DEFAULT = "That one's outside my knowledge, but Mithil's email is always open!"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles common abbreviations."""
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def post_process_response(response: str, is_greeting: bool = False) -> str:
    """
    Post-process LLM response (Task 11):
    1. Strip <think>...</think> leaked reasoning markers
    2. Strip leading affirmations
    3. Truncate to sentence limit (4 for project questions, 2 for greetings)
    4. Re-append fallback suffix if original contained it
    """
    if not response or not response.strip():
        return _FALLBACK_DEFAULT

    # 1. Strip thinking markers
    response = _THINKING_RE.sub("", response).strip()

    if not response:
        return _FALLBACK_DEFAULT

    # 2. Strip leading affirmations
    response = _AFFIRMATION_RE.sub("", response).strip()

    if not response:
        return _FALLBACK_DEFAULT

    # 3. Check if response contains a fallback suffix (before truncation)
    has_fallback_1 = _FALLBACK_SUFFIX_1 in response
    has_fallback_2 = _FALLBACK_SUFFIX_2 in response

    # 4. Sentence truncation
    max_sentences = 2 if is_greeting else 4
    sentences = _split_sentences(response)

    if len(sentences) > max_sentences:
        # If the last sentence is a fallback, don't count it
        fallback_sentence = None
        if has_fallback_1 or has_fallback_2:
            # Find which sentence contains the fallback
            for i, s in enumerate(sentences):
                if _FALLBACK_SUFFIX_1 in s or _FALLBACK_SUFFIX_2 in s:
                    fallback_sentence = s
                    sentences = [x for j, x in enumerate(sentences) if j != i]
                    break

        # Truncate main content
        sentences = sentences[:max_sentences]
        response = " ".join(sentences)

        # Re-append fallback if it was in the original
        if fallback_sentence:
            response = response.rstrip(". ") + " " + fallback_sentence
    else:
        response = " ".join(sentences)

    return response


# ─────────────────────────────────────────────
# 7. LangGraph Agent Definition
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    """Typed state passed between LangGraph nodes."""

    question: str
    context: List[str]
    answer: str
    history: List[dict]  # Task 10: conversation history
    condensed_question: str  # Task 10: rewritten standalone question


def condense(state: AgentState) -> AgentState:
    """
    Node 0 (conditional) — Rewrite follow-up question into standalone question.
    Task 10: Only runs when history is non-empty. Uses one LLM call.
    Skipped entirely for single-shot questions (no history) to avoid
    unnecessary API calls.
    """
    history = state.get("history", [])
    if not history:
        # No history — use original question as-is
        return {**state, "condensed_question": state["question"]}

    logger.info("Node: condense — rewriting follow-up with history context")

    # Format history for the condense prompt
    history_text = "\n".join(
        f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        for turn in history
    )

    try:
        chain = get_condense_chain()
        condensed = chain.invoke(
            {"history": history_text, "question": state["question"]}
        )
        logger.info(f"Condensed query: {condensed:.100s}")
        return {**state, "condensed_question": condensed.strip()}
    except Exception as e:
        logger.warning(f"Condense failed, using original question: {e}")
        return {**state, "condensed_question": state["question"]}


def retrieve(state: AgentState) -> AgentState:
    """
    Node 1 — Embed the question and retrieve matching documents from SQLite.
    Uses NIM NV-Embed v1 embeddings (Task 1) with query-mode encoding (Task 1a).
    If retrieval returns no context, sets the fallback answer directly
    (Task 12: grade node logic collapsed into retrieve).
    """
    logger.info("Node: retrieve — embedding query via NIM and searching SQLite")

    # Use condensed question if available (Task 10)
    search_query = state.get("condensed_question") or state["question"]

    try:
        query_vector = nim_embed_single(search_query, input_type="query")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return {
            **state,
            "context": [],
            "answer": "Sorry, something went wrong. Please try again later.",
        }

    docs, embeddings_matrix = load_vectors()

    if len(docs) == 0:
        logger.warning("Vector store is empty — no documents to search")
        return {
            **state,
            "context": [],
            "answer": _FALLBACK_DEFAULT,
        }

    # Cosine similarity (both sides are L2-normalized — Task 1b)
    similarities = embeddings_matrix @ query_vector

    # Filter by threshold and get top-8 (Task 7: env-driven threshold)
    match_threshold = float(os.getenv("MATCH_THRESHOLD", "0.16"))
    match_count = int(os.getenv("MATCH_COUNT", "8"))
    scored = [
        (sim, doc) for sim, doc in zip(similarities, docs) if sim >= match_threshold
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = scored[:match_count]

    logger.info(
        f"Retrieved {len(top_docs)} documents (threshold={match_threshold})"
    )

    if not top_docs:
        # Task 12: grade node logic collapsed — no matching docs = fallback
        logger.warning("No documents above threshold — returning fallback")
        return {
            **state,
            "context": [],
            "answer": _FALLBACK_DEFAULT,
        }

    # Build context strings with source attribution
    context_chunks = []
    for sim, doc in top_docs:
        meta = doc.get("metadata") or {}
        source = meta.get("source", "Unknown")
        section = meta.get("section", "")
        prefix = f"[Source: {source}]"
        if section:
            prefix += f" [Section: {section}]"
        context_chunks.append(f"{prefix}\n{doc['content']}")

    return {**state, "context": context_chunks}


def rerank(state: AgentState) -> AgentState:
    """
    Node 2 — NIM reranker (Task 2).
    Scores each retrieved chunk against the question using nvidia/nv-rerankqa-mistral-4b-v3.
    Returns the top-3 highest-scoring chunks. No hard score threshold.
    Degrades gracefully on failure: passes top-3 by cosine similarity through. (Task 2, 6)
    """
    if not state["context"] or state.get("answer"):
        return state

    logger.info(f"Node: rerank — scoring {len(state['context'])} chunks via NIM")

    # Use condensed question for reranking too
    question = state.get("condensed_question") or state["question"]

    try:
        rankings = nim_rerank(question, state["context"])

        # Take top 3 by logit score
        top_indices = [r["index"] for r in rankings[:3]]
        filtered = [state["context"][i] for i in top_indices]

        # Log top-3 scores at INFO for debuggability (Task 2)
        top_scores = [(r["index"], r.get("logit", 0)) for r in rankings[:3]]
        logger.info(f"Rerank top-3 scores: {top_scores}")

        logger.info(
            f"Rerank: {len(state['context'])} → {len(filtered)} chunks"
        )
        return {**state, "context": filtered}

    except Exception as e:
        # Task 2/6: Graceful degradation — pass top-3 by cosine similarity through
        logger.warning(
            f"Reranker failed, degrading to cosine top-3: {e}"
        )
        return {**state, "context": state["context"][:3]}


def generate(state: AgentState) -> AgentState:
    """
    Node 3 — Build the prompt and call the LLM.
    Task 3: Uses Nemotron 3 via NIM.
    Task 10: Includes conversation history in prompt if available.
    Task 11: Post-processes response for brevity.
    """
    llm_model = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    logger.info(f"Node: generate — calling {llm_model} via NIM")

    # If retrieve already set an answer (e.g., fallback), skip generation
    if state.get("answer"):
        return state

    context_text = "\n\n".join(state["context"]) if state["context"] else ""
    question = state.get("condensed_question") or state["question"]
    history = state.get("history", [])

    try:
        if history:
            # Task 10: Include history in prompt
            history_text = "\n".join(
                f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                for turn in history
            )
            chain = get_rag_chain(with_history=True)
            answer = chain.invoke(
                {
                    "context": context_text,
                    "question": question,
                    "history": history_text,
                }
            )
        else:
            chain = get_rag_chain(with_history=False)
            answer = chain.invoke(
                {"context": context_text, "question": question}
            )

        # Task 3: Empty-content guard (reasoning models can consume all tokens on thinking)
        if not answer or not answer.strip():
            logger.error(
                "LLM returned empty content — reasoning may have consumed "
                "entire token budget. Returning fallback."
            )
            answer = _FALLBACK_DEFAULT

        # Task 11: Post-process for brevity
        answer = post_process_response(answer, is_greeting=False)

        logger.info("Generated response successfully")
        return {**state, "answer": answer}

    except Exception as e:
        logger.exception(f"LLM generation failed: {e}")
        return {
            **state,
            "answer": "Sorry, something went wrong. Please try again later.",
        }


# ─────────────────────────────────────────────
# 8. Compile the LangGraph StateGraph
# ─────────────────────────────────────────────
# Task 12: Graph simplified — grade node collapsed into retrieve.
# Flow: condense → retrieve → rerank → generate → END
# condense is wired unconditionally but skips internally when history is empty.

builder = StateGraph(AgentState)
builder.add_node("condense", condense)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)
builder.add_node("generate", generate)

builder.set_entry_point("condense")
builder.add_edge("condense", "retrieve")
builder.add_edge("retrieve", "rerank")


def route_after_rerank(state: AgentState) -> str:
    """Skip generate if retrieve already set a fallback answer."""
    if state.get("answer"):
        return "__end__"
    return "generate"


builder.add_conditional_edges(
    "rerank", route_after_rerank, {"generate": "generate", "__end__": END}
)
builder.add_edge("generate", END)

rag_graph = builder.compile()


# ─────────────────────────────────────────────
# 9. FastAPI Endpoints
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Task 10

    @field_validator("message")
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 1000:
            raise ValueError("Message too long (max 1000 characters)")
        return v.strip()


@app.get("/")
async def root():
    """Root route — basic service info."""
    return {"service": "RAG Agent", "docs": "/docs"}


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "RAG Agent"}


@app.post("/chat")
@limiter.limit("10/minute;100/day")  # Task 9
@traceable(name="chat_endpoint")  # LangSmith: traces this function as a top-level run
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Runs the LangGraph RAG pipeline for a user question.
    Intent classifier short-circuits greetings and out-of-scope queries.
    Task 10: Supports session_id for conversation history.
    Task 9: Rate limited to 10/min, 100/day per IP.
    """
    try:
        logger.info(f"Received query: {chat_request.message:.100s}...")

        # Generate or use provided session_id (Task 10)
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Classify intent before running the full pipeline (Task 12: confirmed zero network calls)
        intent = classify_intent(chat_request.message)

        if intent == "greeting":
            reply = "Hey! I'm Mithil's portfolio assistant — ask me about his projects, skills, or background!"
            # Still track in session history
            session_store.add_turn(session_id, chat_request.message, reply)
            return {"reply": reply, "session_id": session_id}

        if intent == "out_of_scope":
            reply = "That's a bit outside my expertise! I'm here to talk about Mithil's work — projects, skills, experience. What would you like to know?"
            session_store.add_turn(session_id, chat_request.message, reply)
            return {"reply": reply, "session_id": session_id}

        # portfolio_question — run full RAG pipeline
        # Task 10: Load session history
        history = session_store.get(session_id) or []

        result = rag_graph.invoke(
            {
                "question": chat_request.message,
                "context": [],
                "answer": "",
                "history": history,
                "condensed_question": "",
            }
        )

        reply = result["answer"]

        # Task 10: Store turn in session history
        session_store.add_turn(session_id, chat_request.message, reply)

        # Periodic cleanup of expired sessions
        session_store.cleanup_expired()

        return {"reply": reply, "session_id": session_id}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # NIM API errors (missing key, 403, etc.)
        logger.error(f"Runtime error: {e}")
        return {
            "reply": "Sorry, something went wrong. Please try again later.",
            "session_id": chat_request.session_id or str(uuid.uuid4()),
        }
    except Exception as e:
        logger.exception(f"Error in /chat endpoint: {e}")
        return {
            "reply": "Sorry, something went wrong. Please try again later.",
            "session_id": chat_request.session_id or str(uuid.uuid4()),
        }