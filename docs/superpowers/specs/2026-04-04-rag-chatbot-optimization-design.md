# RAG Chatbot Optimization Design

**Date:** 2026-04-04  
**Status:** Approved

---

## Problem Statement

The existing RAG chatbot exhibits three primary failure modes:

1. **Hallucination on retrieved context** — the LLM misrepresents or exaggerates facts present in chunks
2. **Fabrication on empty context** — the LLM sometimes invents information even when no relevant chunks are found
3. **Excessive verbosity** — even simple or conversational queries produce long, padded responses

Root causes identified:
- Similarity threshold `0.45` is too permissive; noisy, loosely-related chunks reach the LLM
- No reranking step — retrieved chunks are not ordered by actual relevance to the question
- Narrative "My Learnings" prose in `About_me.md` gives the LLM vague, enthusiastic text it amplifies
- DeepSeek V3.2 does not reliably enforce the brevity and grounding rules in the prompt
- Intent classifier has edge-case gaps (e.g. "who are you", "what can you do")

---

## Known System Constraints (Unchanged)

- **Stateless** — no conversation history; each message is processed independently
- **Keyword-based intent classifier** — hardcoded pattern matching; edge cases can slip through
- **Simulated streaming** — typing animation is client-side only; full response arrives as one block
- **Content is fixed** — `About_me.md` and project files will not be reformatted

---

## Architecture

### LangGraph Pipeline

```
retrieve → rerank → grade → generate → END
```

The new `rerank` node is inserted between `retrieve` and `grade`. All other nodes retain their current logic.

### Node Changes

**retrieve** (modified parameters only)
- `match_threshold`: `0.45` → `0.40` — cast a wider net intentionally; the reranker filters aggressively afterward
- `match_count`: `5` → `8` — more candidates for the reranker to work with

**rerank** (new node)
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally via `sentence-transformers`, already installed)
- Scores each retrieved chunk against the question using the CrossEncoder
- Filters out chunks with score ≤ `0.0` (eliminates truly irrelevant chunks)
- Returns at most top-3 chunks, sorted by descending score
- If all chunks score ≤ `0.0`, passes an empty context list (grade node handles fallback)

**grade** — unchanged logic; now operates on the smaller, reranked context

**generate** — same chain; prompt updated (see below); model swapped

---

## Model Change

| | Before | After |
|---|---|---|
| Model | `deepseek/deepseek-v3.2` | `google/gemma-4-31b-it` |
| Temperature | `0.3` | `0.3` |
| Max tokens | `512` | `600` |

**Rationale:** Gemma 4 31B is instruction-tuned and follows system prompt constraints more reliably. Max tokens raised from 512 to 600 to avoid truncating legitimate answers — prompt-enforced brevity handles length control instead of hard truncation.

---

## Prompt Hardening

Replace soft brevity guidance with explicit hard limits:

```
4. BE BRIEF. Hard limits on response length:
   - Conversational / greeting question → Maximum 2 sentences. Stop.
   - Factual question about a project or skill → Maximum 4 sentences. Stop.
   - Never use bullet points unless explicitly asked.
   - Never use filler phrases: "Great question!", "Certainly!", "Of course!", "Sure!", or similar.
   - Never repeat or summarize the question back to the user.
```

All other rules (grounding, fallback, source attribution) remain unchanged.

---

## Intent Classifier Improvements

Add the following pattern groups to `classify_intent`:

**New "greeting" patterns:**
- `"who are you"`, `"what are you"`, `"what can you do"`, `"help"` → return bot-description reply
- Single characters or standalone punctuation → treated as greeting

**New "greeting" reply for bot-identity queries:**
```
"I'm Mithil's portfolio assistant — ask me about his projects, skills, or background!"
```

---

## Files Changed

| File | Change |
|---|---|
| `backend/server.py` | Add `rerank` node; update retrieval params; swap model; update prompt; improve intent classifier |

No changes to `ingest.py`, frontend, or knowledge base files.

---

## What This Does Not Change

- Ingestion pipeline (`ingest.py`) — chunk size, overlap, and embedding model are unchanged
- Frontend — no changes; simulated streaming remains client-side
- Knowledge base content — `About_me.md` and project files are not modified
- Conversation statefulness — system remains fully stateless
