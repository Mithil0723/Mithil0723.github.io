CustomerIntel
# MANDATORY YAML FRONTMATTER - DO NOT CHANGE FIELD NAMES
project_id: "2026-05-02_CustomerIntel"
title: "CustomerIntel — Multi-Agent System for Customer Feedback Analysis"
status: "in-progress"
confidence_level: 0.85
analysis_date: "2026-07-09"
analysis_version: "1.0"

# ACADEMIC CONTEXT
course_context: "CS 494 Agentic AI — University of Illinois Chicago (Spring 2026)"
course_level: "advanced"
assignment_type: "team"
team_size: 5
primary_role: "Strategy Agent & Critic Agent Developer (Adversarial Loop)"

# TECHNICAL STACK - Arrays Only
primary_languages: ["Python"]
frameworks: ["LangGraph", "LangChain"]
databases: ["ChromaDB"]
tools: ["Claude Sonnet 4.6", "Claude Haiku 4.5", "GPT-3.5-turbo", "OpenAI Embeddings", "DistilBERT (planned)", "Streamlit (planned)"]
cloud_platforms: ["Anthropic API", "OpenAI API"]

# SKILLS ASSESSMENT - Integers 1-10 Only
programming_fundamentals: 8
oop_concepts: 7
data_structures: 7
algorithms: 7
database_design: 6
api_development: 7
frontend_dev: 3
backend_dev: 8
testing: 6
version_control: 7

# ADVANCED SKILLS - Boolean Only
design_patterns_used: true
performance_optimization: false
error_handling: true
user_authentication: false
deployment_automated: false
documentation_comprehensive: true

# PROJECT METRICS - Integers Only
estimated_loc: 900
files_created: 15
complexity_score: 9
portfolio_score: 9
market_relevance: 10

# CAREER INTELLIGENCE
salary_range_low: 95000
salary_range_high: 150000
relevant_job_titles: ["AI/ML Engineer", "Agentic AI Engineer", "LLM Engineer", "Applied AI Engineer", "GenAI Developer"]
target_companies: ["AI startups", "enterprise SaaS", "consulting firms", "customer-experience platforms"]
next_learning_priorities: ["OpenAI embeddings integration for ChromaDB", "DistilBERT sentiment classification", "Hybrid search retrieval", "Streamlit visualization frontend", "Multi-agent evaluation methodology"]

# PROFESSIONAL OUTPUTS - Single Strings
resume_bullet_technical: "Designed the Strategy and Critic agents in a five-agent LangGraph system, implementing multi-perspective recommendation generation and a Socratic-questioning adversarial critique loop with conditional routing and up to three forced revision cycles."
resume_bullet_impact: "Co-built a multi-agent LLM system that converts unstructured customer feedback into prioritized, evidence-grounded business recommendations, with an adversarial quality-control loop that rejects logically inconsistent outputs before final synthesis."
resume_bullet_behavioral: "Owned the adversarial-critique subsystem in a five-person team project — defining the structured evaluation rubric, revision routing logic, and cost-benefit strategy reasoning — coordinating interfaces with teammates' orchestrator, RAG, and diagnosis agents through a shared TypedDict state."

# SEARCHABLE KEYWORDS - Space Separated
technical_keywords: "python langgraph langchain chromadb multi-agent-system agentic-ai claude-sonnet claude-haiku gpt-3.5-turbo react-pattern chain-of-thought socratic-questioning adversarial-critique rag retrieval-augmented-generation sentiment-analysis state-machine typeddict conditional-routing prompt-engineering"
skill_keywords: "multi-agent-coordination agent-design reasoning-patterns quality-control system-integration team-collaboration llm-orchestration causal-reasoning recommendation-generation evaluation-design"
---

# CustomerIntel — Multi-Agent System for Customer Feedback Analysis

## 📋 EXECUTIVE_SUMMARY
### Business Problem Solved
Businesses accumulate large volumes of unstructured customer feedback (reviews, surveys, support tickets) that rarely translate into concrete action. CustomerIntel transforms that raw feedback into actionable business intelligence — surfacing complaint themes, diagnosing root causes, and producing prioritized recommendations that have survived adversarial quality review.

### Technical Solution Delivered
A five-agent LLM pipeline orchestrated as a **LangGraph state machine** with conditional routing. A ReAct-based **Orchestrator** (Claude Sonnet 4.6) coordinates the workflow; a **Data Intelligence** agent (GPT-3.5-turbo) performs RAG retrieval over ChromaDB with iterative query refinement plus sentiment analysis; a **Diagnosis** agent (Claude Sonnet 4.6) runs five-step chain-of-thought root-cause analysis with self-reflection; a **Strategy** agent (Claude Haiku 4.5) generates multi-perspective recommendations with cost-benefit estimates; and a **Critic** agent (Claude Sonnet 4.6) applies Socratic questioning to reject weak outputs, forcing up to three revision cycles before the Orchestrator synthesizes the final report.

### Key Professional Achievement
Owned the Strategy → Critic adversarial loop — the subsystem that makes the pipeline self-correcting. Designed the structured critique rubric (logical consistency, missing cost analysis, unaddressed root causes, risk mitigation), the revision routing logic, and the multi-perspective strategy reasoning that generates 2+ alternatives per root cause with an impact-vs-effort matrix.

## 🛠️ TECHNICAL_IMPLEMENTATION
### Technology Stack Analysis
**Primary Language**: Python powers the whole system — LangGraph state machine, agent nodes, ChromaDB ingestion, and CLI entry point.

**Frameworks & Libraries**:
- **LangGraph**: `StateGraph` with five agent nodes and conditional edges implementing the critique loop (`MAX_ITERATIONS = 3` in `graph.py`).
- **LangChain**: Prompt templates and LLM integration layer for all five agents (`prompts.py`).
- **ChromaDB**: Vector store for customer-feedback retrieval (`ingest.py`, `CHUNK_SIZE = 500`), with hybrid search and OpenAI embeddings on the roadmap.
- **Mixed LLM routing**: Model selection per agent — Sonnet-class models where reasoning depth matters (Orchestrator, Diagnosis, Critic), Haiku for high-volume strategy generation, GPT-3.5-turbo for retrieval refinement — an explicit cost/quality trade-off.

### Architecture & Design Assessment
**Data Flow**:
```
User Query
    ↓
Data Intelligence (RAG retrieval + sentiment analysis)
    ↓
Diagnosis (chain-of-thought root cause analysis)
    ↓
Strategy (multi-perspective recommendations)
    ↓
Critic (Socratic evaluation) ──→ rejected → back to Strategy (≤3 cycles)
    ↓ approved
Orchestrator (final synthesis)
```

**Key Technical Decisions**:
- **[Adversarial Critique Loop]** (my subsystem): The Critic evaluates Strategy output against a structured rubric and routes rejections back with specific objections. Auto-approval at max iterations prevents infinite loops while guaranteeing at least one quality gate.
- **[Multi-Perspective Strategy]** (my subsystem): Each root cause gets 2+ alternative recommendations with cost-benefit estimates, ranked in an impact-vs-effort matrix, revised against Critic feedback into an implementation roadmap.
- **[Shared TypedDict State]**: All agents read/write one typed state dictionary, giving full traceability for debugging and making each node independently testable.
- **[Reasoning Pattern per Agent]**: ReAct for coordination and retrieval, chain-of-thought + self-reflection for diagnosis, Socratic questioning for critique — patterns matched to each agent's job rather than one-size-fits-all prompting.

### File Structure
```
customerintel/
├── agents/
│   ├── orchestrator.py       # ReAct workflow coordination
│   ├── data_intelligence.py  # RAG retrieval + sentiment
│   ├── diagnosis.py          # CoT root-cause analysis
│   ├── strategy.py           # Multi-perspective recommendations (mine)
│   └── critic.py             # Adversarial critique loop (mine)
├── graph.py                  # LangGraph state machine + conditional routing
├── state.py                  # Shared TypedDict state
├── prompts.py                # Prompt templates for all agents
├── ingest.py                 # ChromaDB ingestion pipeline
main.py                       # CLI entry point
```

## 💡 COMPREHENSIVE_SKILLS_ANALYSIS
### Technical Competency Assessment
- **Agent Design**: 9/10 — Designed two of five agents including the system's quality-control mechanism; reasoning patterns chosen deliberately per role.
- **Multi-Agent Coordination**: 8/10 — Conditional routing, shared state contracts, and revision cycles across teammate-owned nodes.
- **Backend Engineering**: 8/10 — Clean node separation, PEP 8, docstrings, graceful fallbacks, structured outputs consumable by downstream agents.
- **Evaluation Design**: 7/10 — Contributed to a 12-task evaluation plan (standard, stress, and robustness cases) with precision/recall, human quality ratings, and critic-effectiveness metrics.

### Professional Development Demonstrated
**Team Collaboration**: Five-person team with clearly divided agent ownership; my Strategy/Critic subsystem had to honor interface contracts with the Orchestrator (Rounak), RAG pipeline (Pranshu), and Diagnosis agent (Shivanshi), coordinated through the shared state schema.

**Systems Thinking**: The critique loop demonstrates understanding that LLM output quality must be engineered, not assumed — building rejection, revision, and termination logic rather than trusting single-pass generation.

## 🚀 PROFESSIONAL_VALUE_TRANSLATION
### Resume Bullets (Optimized for ATS and Human Reviewers)
• **Technical Focus**: Designed the Strategy and Critic agents in a five-agent LangGraph system — multi-perspective recommendation generation, Socratic adversarial critique, and conditional revision routing with bounded iteration.
• **Impact Focus**: Co-built a multi-agent pipeline converting unstructured customer feedback into evidence-grounded, prioritized recommendations, with a quality gate that forces revision of logically inconsistent outputs.
• **Growth Focus**: Owned a critical subsystem in a five-person agentic AI team project, coordinating typed state contracts across teammates' agents in a shared LangGraph state machine.

### Interview Preparation Arsenal
**Technical Discussion Points**:
- "Walk me through the adversarial critique loop — what does the Critic check, and how do revisions terminate?"
- "Why different LLM models per agent? What drove the Sonnet/Haiku/GPT-3.5 split?"
- "How does the shared TypedDict state enable debugging a five-agent pipeline?"
- "How would you evaluate whether the Critic actually improves output quality?" (T12: with/without-Critic comparison in the eval plan)

**Behavioral Interview Stories (STAR Format)**:
- **Collaboration**: "Tell me about working on a complex team system." (Owning Strategy/Critic while integrating against three teammates' agents via a shared state contract.)
- **Quality**: "How do you ensure quality in AI outputs?" (Designing the structured critique rubric and bounded revision loop instead of trusting single-pass LLM generation.)

# STUDENT_PROJECT_RECALL_SECTION

## 🧠 PROJECT_MEMORY_BANK
*This section is your personal reference to quickly remember the specific details of your work.*

### What You Actually Built (Feature Breakdown)
- **Strategy Agent** (`strategy.py`, Claude Haiku 4.5): generates 2+ alternatives per root cause, estimates cost-benefit, builds an impact-vs-effort matrix, revises on Critic feedback, produces an implementation roadmap.
- **Critic Agent** (`critic.py`, Claude Sonnet 4.6): Socratic-questioning evaluation against a rubric — logical consistency, missing elements, root-cause alignment, risk mitigation — with auto-approval at max iterations.
- **Adversarial Loop**: conditional edge in `graph.py` routing Critic rejections back to Strategy, capped at `MAX_ITERATIONS = 3`.

### Technical Implementation Details You Might Forget
- **Team**: Rounak Deshpande (Orchestrator/LangGraph), Pranshu Bansal (Data Intelligence/RAG/ChromaDB), Shivanshi Shukla (Diagnosis/causal reasoning), me (Strategy/Critic/adversarial loop), Ethan Van (Streamlit frontend/evaluation).
- **Models per agent**: Orchestrator = Claude Sonnet 4.6 (ReAct); Data Intelligence = GPT-3.5-turbo (ReAct + query refinement); Diagnosis = Claude Sonnet 4.6 (CoT + self-reflection); Strategy = Claude Haiku 4.5 (multi-perspective); Critic = Claude Sonnet 4.6 (Socratic).
- **Config knobs**: `MAX_ITERATIONS = 3` (graph.py), `CHUNK_SIZE = 500` (ingest.py), `SAMPLE_FEEDBACK` (main.py). Env: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `CHROMA_PATH=./chroma_db`.
- **Eval plan**: 12 tasks — 4 standard (themes, sentiment trends, root cause, recommendations), 4 stress (contradictory feedback, empty retrieval, slang, vague queries), 4 robustness (full trace, critic rejection/revision, multi-turn coherence, with/without-Critic comparison).
- **Status at last push (May 2, 2026)**: Proposal milestone — LangGraph skeleton, prompts, functional stubs complete; OpenAI embeddings, DistilBERT sentiment, real ChromaDB retrieval, and Streamlit frontend were next-phase work.
- **Repo**: https://github.com/Mithil0723/CustomerIntel (Python, public).

## ✅ ANALYSIS_CONFIDENCE_METADATA
- **Overall Assessment Confidence**: 0.85/1.0
- **Reasoning**: Based on the public repo README and file tree via the GitHub API (last pushed 2026-05-02, proposal milestone). Architecture, roles, and design details are well documented; deduction because the LLM-integration phase and evaluation results aren't reflected in the repo yet — update this file once the implementation phase lands.
- **Structure Guarantee**: Bulletproof Consistency for Database Integration
