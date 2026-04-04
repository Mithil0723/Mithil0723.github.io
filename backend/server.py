import os
import logging
from typing import TypedDict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# LangChain / LangGraph / LangSmith
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langsmith import traceable

# Supabase
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- CORS Middleware ---
# For development: allow all origins
# For production: replace with your actual domain via ALLOWED_ORIGINS env var
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 1. HuggingFace Embeddings (lazy-loaded)
# ─────────────────────────────────────────────
# Model is downloaded once (~90 MB) and cached in ~/.cache/huggingface/
# No API key needed — no rate limits, no quota.
# Lazy-loaded on first request so uvicorn binds the port immediately
# and Render doesn't time out waiting for a port.
_embeddings = None


def get_embeddings():
    """Lazy singleton — loads the HuggingFace model on first call."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading HuggingFace embeddings model (first request)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embeddings model loaded successfully")
    return _embeddings

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

# ─────────────────────────────────────────────
# 2. OpenRouter LLM (Gemma 4 31B)
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="google/gemma-4-31b-it",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_tokens=600,
)

# ─────────────────────────────────────────────
# 3. LangChain Prompt Template
# ─────────────────────────────────────────────
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

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "QUESTION: {question}\n\n"
     "Answer using only the facts in the context above. Follow all rules in the system prompt exactly.")
])

rag_chain = prompt_template | llm | StrOutputParser()

# ─────────────────────────────────────────────
# 4. Supabase client
# ─────────────────────────────────────────────
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY"),
)


# ─────────────────────────────────────────────
# 5. LangGraph Agent Definition
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    """Typed state passed between LangGraph nodes."""
    question: str
    context: List[str]
    answer: str


# NEW — Intent classifier to short-circuit trivial inputs
import re

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


def retrieve(state: AgentState) -> AgentState:
    """
    Node 1 — Embed the question and retrieve matching documents from Supabase.
    Uses HuggingFace embeddings (local, no API quota).
    Each retrieved chunk is prefixed with its source file and section
    so the LLM can reference specific projects by name.
    """
    logger.info("Node: retrieve — embedding query and searching Supabase")

    query_vector = get_embeddings().embed_query(state["question"])

    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.40,
        "match_count": 8,
    }).execute()

    docs = response.data or []
    logger.info(f"Retrieved {len(docs)} documents")

    # Build context strings with source attribution
    context_chunks = []
    for doc in docs:
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
    Node 2 — Cross-encoder reranking.
    Scores each retrieved chunk against the question using ms-marco-MiniLM-L-6-v2.
    Returns the top-3 highest-scoring chunks. No hard score threshold — ms-marco
    scores are not normalized around 0, so relevant portfolio content can score
    negative. The grade node handles truly empty retrieval.
    """
    if not state["context"]:
        return state

    logger.info(f"Node: rerank — scoring {len(state['context'])} chunks")
    question = state["question"]
    pairs = [[question, chunk] for chunk in state["context"]]
    scores = get_reranker().predict(pairs)

    scored = sorted(zip(scores, state["context"]), key=lambda x: x[0], reverse=True)
    filtered = [chunk for _, chunk in scored][:3]

    logger.info(f"Rerank: {len(state['context'])} → {len(filtered)} chunks after reranking")
    return {**state, "context": filtered}


def grade(state: AgentState) -> AgentState:
    """
    Node 3 — Check whether any documents were retrieved.
    Short-circuits to fallback when context is empty (hallucination prevention).
    """
    if not state["context"]:
        logger.warning("No matching documents found — short-circuiting to fallback")
        return {
            **state,
            "answer": "That one's outside my knowledge, but Mithil's email is always open!"
        }
    logger.info(f"Grade node: {len(state['context'])} chunks retrieved")
    return state


def generate(state: AgentState) -> AgentState:
    """
    Node 4 — Build the prompt and call the LLM via the LangChain chain.
    Now dynamically responds even if context is empty.
    """
    logger.info("Node: generate — calling Gemma 4 31B via OpenRouter")
    context_text = "\n\n".join(state["context"]) if state["context"] else ""

    answer = rag_chain.invoke({
        "context": context_text,
        "question": state["question"],
    })

    logger.info("Generated response successfully")
    return {**state, "answer": answer}


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


# ─────────────────────────────────────────────
# 6. FastAPI Endpoints
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
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
@traceable(name="chat_endpoint")  # LangSmith: traces this function as a top-level run
async def chat_endpoint(request: ChatRequest):
    """
    Runs the LangGraph RAG pipeline for a user question.
    Intent classifier short-circuits greetings and out-of-scope queries.
    The @traceable decorator sends the full execution trace to LangSmith.
    """
    try:
        logger.info(f"Received query: {request.message:.100s}...")

        # NEW — classify intent before running the full pipeline
        intent = classify_intent(request.message)

        if intent == "greeting":
            return {"reply": "Hey! I'm Mithil's portfolio assistant — ask me about his projects, skills, or background!"}

        if intent == "out_of_scope":
            return {"reply": "That's a bit outside my expertise! I'm here to talk about Mithil's work — projects, skills, experience. What would you like to know?"}

        # portfolio_question — run full RAG pipeline
        result = rag_graph.invoke({"question": request.message, "context": [], "answer": ""})
        return {"reply": result["answer"]}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in /chat endpoint: {e}")
        return {"reply": "Sorry, something went wrong. Please try again later."}