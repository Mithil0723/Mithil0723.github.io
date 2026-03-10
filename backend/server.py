import os
import logging
from typing import TypedDict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# LangChain / LangGraph / LangSmith
from langchain_huggingface import HuggingFaceEmbeddings
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
# 2. OpenRouter LLM (DeepSeek V3.2)
# ─────────────────────────────────────────────
llm = ChatOpenAI(
    model="deepseek/deepseek-v3.2",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=512,
)

# ─────────────────────────────────────────────
# 3. LangChain Prompt Template
# ─────────────────────────────────────────────
SYSTEM_INSTRUCTION = (
    "You are Mithil Ravulapalli's portfolio AI assistant — sharp, personable, "
    "and a little bit enthusiastic about what Mithil has built. "
    "Answer questions about Mithil's skills, projects, and Agentic AI work "
    "using ONLY the context provided in each message. "
    "IMPORTANT: The context below DOES contain real information about Mithil. "
    "Always mine the context thoroughly for relevant details before responding. "
    "Never say information is unavailable or missing if the context contains ANY relevant details. "
    "Vary how you open and structure each response — avoid starting with the same phrase twice. "
    "Use concrete details from the context rather than generic summaries. "
    "Show genuine interest: if something in the context is impressive, it's okay to say so. "
    "If the answer truly isn't in the context at all, say so naturally — something like "
    "'That one's outside my knowledge, but Mithil's email is always open!' "
    "Never use bullet points unless the question explicitly asks for a list. "
    "Prefer flowing, conversational prose."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_INSTRUCTION),
    ("human",
     "CONTEXT:\n{context}\n\n"
     "QUESTION: {question}\n\n"
     "Answer conversationally in your own words. "
     "Do not repeat phrases you might have used before. "
     "Draw on specific details from the context above.")
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


def retrieve(state: AgentState) -> AgentState:
    """
    Node 1 — Embed the question and retrieve matching documents from Supabase.
    Uses HuggingFace embeddings (local, no API quota).
    """
    logger.info("Node: retrieve — embedding query and searching Supabase")

    query_vector = get_embeddings().embed_query(state["question"])

    # Start with lower threshold (0.5) for better recall.
    # Increase to 0.7+ if getting irrelevant results.
    response = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_threshold": 0.35,
        "match_count": 5,
    }).execute()

    docs = response.data or []
    logger.info(f"Retrieved {len(docs)} documents")
    return {**state, "context": [doc["content"] for doc in docs]}


def grade(state: AgentState) -> AgentState:
    """
    Node 2 — Check whether any documents were retrieved.
    Sets a sentinel answer if the retrieval step returned nothing,
    which the generate node will pass through unchanged.
    """
    if not state["context"]:
        logger.warning("No matching documents found — returning fallback")
        return {**state, "answer": "__NO_CONTEXT__"}
    return state


def generate(state: AgentState) -> AgentState:
    """
    Node 3 — Build the prompt and call the LLM via the LangChain chain.
    Skipped if the grade node set the fallback sentinel.
    """
    if state.get("answer") == "__NO_CONTEXT__":
        return {
            **state,
            "answer": "I don't have enough info to answer that. Try emailing Mithil!"
        }

    logger.info("Node: generate — calling DeepSeek V3.2 via OpenRouter")
    context_text = "\n\n".join(state["context"])

    answer = rag_chain.invoke({
        "context": context_text,
        "question": state["question"],
    })

    logger.info("Generated response successfully")
    return {**state, "answer": answer}


# Compile the LangGraph StateGraph
# Nodes run in order: retrieve → grade → generate → END
builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "grade")
builder.add_edge("grade", "generate")
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
def chat_endpoint(request: ChatRequest):
    """
    Runs the LangGraph RAG pipeline for a user question.
    The @traceable decorator sends the full execution trace (inputs,
    outputs, latency) to LangSmith automatically.
    """
    try:
        logger.info(f"Received query: {request.message[:100]}...")

        # Invoke the compiled LangGraph — runs retrieve → grade → generate
        result = rag_graph.invoke({"question": request.message, "context": [], "answer": ""})
        return {"reply": result["answer"]}

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Use logger.exception to capture the full traceback — critical for debugging
        # transient issues (rate limits, timeouts) vs real bugs.
        # The failed run will also appear in LangSmith with its full trace.
        logger.exception(f"Error in /chat endpoint: {e}")
        return {"reply": "Sorry, something went wrong. Please try again later."}