# Mithil Ravulapalli

(630) 632-7037 | vravu3@uic.edu | LinkedIn | GitHub | Website

## Professional summary

Data Science graduate (UIC, May 2026) who builds agentic AI systems. Runs a production RAG chatbot on a public site and recently migrated its entire embedding, reranking, and generation stack to a hosted inference API without downtime. Also brings applied machine learning and statistics from recommender, classification, and analytics work.

## Education

**University of Illinois Chicago** — Bachelor of Science in Data Science, May 2026

## Technical skills

**Languages:** Python, JavaScript, SQL, C++, Java, R

**Agentic AI and LLM systems:** LangGraph, LangChain, LangSmith, Retrieval-Augmented Generation, multi-agent orchestration, prompt engineering, embeddings and reranking, vector search, NVIDIA NIM, Anthropic Claude API, OpenAI API

**Machine learning and statistics:** Scikit-learn, Random Forests, Decision Trees, collaborative filtering, matrix factorization (SVD), hyperparameter tuning (GridSearchCV), hypothesis testing, exploratory data analysis, model evaluation (precision, recall, RMSE, ROC-AUC)

**Backend and data:** FastAPI, Pydantic, Docker, SQLite, ChromaDB, Pandas, NumPy, pytest, Git, GitHub Actions, Looker Studio, Tableau

## Projects

**AI Portfolio Chatbot with RAG** — GitHub | Live demo

- Built an agentic RAG backend on a LangGraph StateGraph: condense, retrieve, rerank, generate. A conditional edge skips generation when retrieval finds nothing, so out-of-scope questions never reach the LLM
- Served all three model calls through the NVIDIA NIM API using NV-Embed v1, nv-rerankqa-mistral-4b-v3, and Nemotron 3. Queries and passages are embedded asymmetrically, since NV-Embed is a bi-encoder
- Wrote the vector store directly against SQLite. Embeddings are stored as float32 BLOBs and searched in NumPy, which kept a managed vector database out of the deployment entirely
- Migrated the model stack off local HuggingFace models and Supabase pgvector. The embedding dimension went from 384 to 4096, so the retrieval threshold had to be recalibrated from 0.40 to 0.16 against real queries
- Hardened the service: retry with backoff on the inference API, reranker fallback to cosine ranking, per-IP rate limiting, CORS lockdown, and LangSmith tracing. Deployed on Render in Docker

**CustomerIntel: Multi-Agent Feedback Analysis** (CS 494 Agentic AI, 5-person team) — GitHub

- Designed the Strategy and Critic agents in a five-agent LangGraph system and owned the adversarial loop, which forces up to three revision cycles before synthesis
- Implemented Socratic-questioning critique against a structured rubric, with auto-approval once the iteration ceiling is hit so the graph cannot stall
- Routed each agent to a different model based on how much reasoning depth the role actually needed, and coordinated interfaces with teammates' orchestrator, retrieval, and diagnosis agents through a shared TypedDict state

**Amazon Recommendation System** — GitHub

- Implemented rank-based, user-user, item-item, and SVD matrix factorization recommenders on Amazon product ratings using Python and the Surprise library
- Tuned hyperparameters with GridSearchCV. RMSE dropped and F1 improved across the model suite
- Compared models on precision, recall, and RMSE, then analyzed where each one breaks down on cold-start and scale

**Lead Conversion Prediction** — GitHub

- Built Decision Tree and Random Forest classifiers that identify high-conversion leads for an EdTech client, reaching 84% accuracy on the held-out set
- Caught overfitting early: the first models hit 100% training accuracy. Pruned them through GridSearchCV to close the train-test gap
- Evaluated with confusion matrices and ROC-AUC, and ranked the features driving conversion so the client could focus outreach

**FoodHub Order Analysis** — GitHub

- Analyzed 1,898 food delivery orders against 17 business questions on restaurant demand, delivery time, and ratings
- Translated a tiered commission rule into a revenue function and applied it across the dataset to size total platform revenue
- Found that 10.5% of orders take over 60 minutes end to end, and flagged the volume of unrated orders as a gap worth incentivizing

**Tata Data Visualization Dashboard** — Looker Studio

- Built executive dashboards tracking operational metrics, delivered through the Tata Consultancy Services job simulation on Forage

## Certifications

Data Science and Machine Learning: Making Data-Driven Decisions — MIT IDSS, May–Sep 2024
