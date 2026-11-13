# Autonomous Multi-Agent Research & Evidence Engine

> **Autonomous Research & Intelligence Agent** — General-Purpose, Domain-Independent Research Engine Subsystem.

This repository implements a domain-independent multi-agent research engine for medicine,
energy, AI security, public policy, and other research topics. It discovers sources,
fetches and chunks documents, extracts verbatim evidence, verifies claims, performs
hybrid retrieval, builds an evidence graph, and produces citation-audited reports.

The backend is a FastAPI/LangGraph application and the frontend is a Vite/React
dashboard. Four research depths are supported by the existing UI: Quick, Balanced,
Thorough, and Deep.

## Running the application

Create a virtual environment, install `requirements.txt`, copy `.env.example` to
`.env`, and configure at least one real `GEMINI_API_KEY` or `GROQ_API_KEY`.
Research requests fail with HTTP 503 when no provider is configured or all
configured providers fail; deterministic extraction is not used as a silent
substitute for a missing LLM.

Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

In a second terminal:

```powershell
Set-Location frontend
npm install
npm run dev
```

The backend is available at `http://localhost:8000`, API docs at
`http://localhost:8000/docs`, and the frontend at `http://localhost:5173`.

The default Qdrant local store is
`./qdrant_data`, and LangGraph checkpoints are stored in
`./research_checkpoints.db`. Both are reopened automatically after a backend
restart using the same research session/thread ID.

`EMBEDDING_PROVIDER=auto` uses the configured FastEmbed ONNX model and verifies
the returned vector dimension. Hash embeddings are available only in development
or test mode (`EMBEDDING_PROVIDER=hash`) and are labeled as non-semantic
development embeddings.

## Validation

Run the backend suite with the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
Set-Location frontend
npm run build
```

The tests use local LLM stubs where network credentials are not appropriate.
Provider failover, bounded retries, Qdrant persistence, hybrid retrieval,
incremental indexing, citation grounding, prompt-injection protection, audit
logging, and orchestration recovery are covered. Live Gemini/Groq success
requires valid credentials and is reported separately from deterministic tests.

---

## Subsystem Architecture & Pipeline

```text
Research Task Request
        ↓
[ Source Discovery Agent ]  (Tavily Primary → DuckDuckGo Fallback)
        ↓
    Sources (Untrusted)
        ↓
[ Document Reader Agent ]   (SSRF Checks, HTML / PyMuPDF & PyPDF extraction)
        ↓
    Documents (Untrusted)
        ↓
[ Deterministic Chunker ]   (Boundary-aware, Page & Provenance Preserving)
        ↓
   Document Chunks
        ↓
[ Evidence Extractor ]      (Verbatim Substring Grounding, Zero Fake Quotes)
        ↓
Claims + Grounded Evidence
        ↓
[ Credibility Agent ]       (Transparent 7-Factor Scoring & Explanations)
        ↓
[ Evidence Store ]          (SQLAlchemy Relational Metadata + Qdrant Vectors)
        ↓
[ Hybrid RAG Retriever ]    (Dense Vectors + BM25 Okapi + Reciprocal Rank Fusion)
        ↓
[ Evidence Graph & Memory ] (Nodes & Typed Edges: contains, supports, contradicts, about)
```

---

## Core Invariants & Engineering Guarantees

1. **Domain-Independent**: Zero hard-coded assumptions of companies, market competitors, or financial ratios. Evaluated identically on microbiology, renewable energy, and cybersecurity.
2. **Search Snippets ≠ Evidence**: Discovery search snippets are strictly treated as preliminary triage metadata. Evidence must be extracted from the actual fetched document body.
3. **Zero Fabricated Citations**: The extractor lexically verifies that every extracted evidence quote is an exact substring of the source chunk. Hallucinated quotes are rejected.
4. **Untrusted Data Isolation**: All retrieved web and PDF text is treated as untrusted external data. Control tokens (`<|im_start|>`, `[INST]`) are neutralized, suspicious prompt injections are detected and isolated into passive XML data blocks.
5. **SSRF Protection**: Private IP ranges (RFC 1918), localhost/loopback (`127.0.0.1`, `::1`), cloud metadata endpoints (`169.254.169.254`), and dangerous URI schemes (`file://`, `gopher://`) are strictly blocked.
6. **Complete Traceability**: Every chunk, evidence excerpt, and claim maintains an unbroken provenance link:
   $$\text{Evidence} \rightarrow \text{Chunk} \rightarrow \text{Document} \rightarrow \text{Source} \rightarrow \text{URL}$$
7. **Explicit Security Outcomes**: User and retrieved text are classified as `SAFE`,
   `SUSPICIOUS`, `BLOCKED`, or `REVIEW`; blocked user requests do not reach tools or
   LLM providers.
8. **Structured Auditability**: Queries, sessions, sources, retrieval scores,
   retries, errors, security decisions, and execution outcomes are recorded and
   exportable as JSON or CSV.

---

## Integration Contract (For Teammate's LangGraph Layer)

The research service provides clean, decoupled async functions from
`backend.app.services.research_service`:

```python
from backend.app.services.research_service import (
    discover_sources,
    read_documents,
    extract_evidence,
    assess_source_credibility,
    retrieve_evidence,
    get_claim_evidence,
    get_research_context,
    get_evidence_graph,
)

# 1. Discover Sources (Tavily with DuckDuckGo fallback)
sources = await discover_sources({
    "task_id": "task-01",
    "query": "What are the major causes of antibiotic resistance?",
    "research_objective": "Identify biological and clinical drivers of AMR",
    "source_preferences": ["who.int", "cdc.gov"]
})

# 2. Assess Source Credibility (Transparent multi-factor score)
credibility_assessments = assess_source_credibility(sources)

# 3. Read & Chunk Documents (HTML or PDF)
documents = await read_documents(sources)

# 4. Extract Grounded Claims and Verbatim Evidence Quotes
extraction_result = await extract_evidence(
    documents=documents,
    objective="Identify biological and clinical drivers of AMR"
)

# 5. Hybrid Evidence Retrieval (BM25 + Dense Qdrant Vector Retrieval)
retrieval_hits = retrieve_evidence(
    query="agricultural subtherapeutic antibiotic use",
    top_k=5
)

# 6. Retrieve Backing Evidence for a Claim
claim_view = get_claim_evidence(claim_id=extraction_result.claims[0].id)

# 7. Query Evidence Graph (Nodes & Typed Edges)
graph = get_evidence_graph()
# graph.nodes -> list of Source, Document, Evidence, Claim, Entity nodes
# graph.edges -> contains, supports, contradicts, about, related_to

# 8. Research Session Memory Snapshot
context = get_research_context(session_id="session-01")
```

---

## Directory Structure

```text
.
├── backend/
│   └── app/
│       ├── config.py                 # Pydantic Settings & environment variables
│       ├── database.py               # SQLAlchemy engine, sessionmaker, and Base
│       ├── main.py                   # FastAPI REST API endpoints
│       ├── models/
│       │   └── entities.py           # 12 relational SQLAlchemy ORM models
│       ├── schemas/                  # Typed Pydantic schemas
│       │   ├── common.py
│       │   ├── source.py
│       │   ├── document.py
│       │   ├── claim.py
│       │   ├── evidence.py
│       │   ├── credibility.py
│       │   ├── session.py
│       │   └── graph.py
│       ├── agents/
│       │   ├── source_discovery.py   # Tavily + DuckDuckGo fallback agent
│       │   ├── document_reader.py    # Multi-format reader & chunking coordinator
│       │   ├── evidence_extractor.py # Verbatim grounded extractor with anti-hallucination
│       │   └── credibility.py        # 7-Factor transparent credibility scoring
│       ├── retrieval/
│       │   ├── chunking.py           # Deterministic boundary-aware chunker
│       │   ├── embeddings.py         # FastEmbed (ONNX) & Hash embeddings
│       │   ├── vector_store.py       # Qdrant client (:memory: or persistent)
│       │   └── hybrid.py             # BM25 + Vector Reciprocal Rank Fusion (RRF)
│       ├── tools/
│       │   ├── base.py               # BaseTool with timeouts, retries, validation
│       │   ├── registry.py           # Controlled tool registry & audit logger
│       │   ├── security.py           # SSRF defense & prompt injection guards
│       │   ├── web_search.py         # Tavily / DuckDuckGo search tool
│       │   ├── webpage_reader.py     # HTML reader & boilerplate cleaner
│       │   └── pdf_reader.py         # PyMuPDF / pypdf page-preserved reader
│       ├── evidence/
│       │   ├── store.py              # Relational persistence manager
│       │   ├── graph.py              # Relational graph generator (nodes & edges)
│       │   └── memory.py             # Cross-iteration session context & memory
│       ├── evaluation/
│       │   ├── metrics.py            # Recall@K, Precision@K, MRR, NDCG, Coverage
│       │   ├── benchmark_data.py     # Multi-domain benchmark test cases
│       │   └── evaluator.py          # Benchmark runner & security evaluator
│       └── services/
│           └── research_service.py   # Discovery, extraction, retrieval, and graph service
├── tests/                            # Backend regression and integration tests
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Running and evaluating the application

### 1. Run the Test Suite
```bash
.\.venv\Scripts\python.exe -m pytest -q
```
The suite runs without external network dependencies using in-memory stores,
deterministic embeddings, and mocked tool execution.

### 2. Run the FastAPI Service
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 3. Run the Frontend Dashboard
```bash
cd frontend
npm run dev
```
Access the interactive Autonomous Research Console at `http://localhost:5173`.

### 4. Run the Automated Evaluation Benchmark
```bash
.\.venv\Scripts\python.exe -c "
import asyncio
from backend.app.evaluation.evaluator import default_evaluator

async def main():
    report = await default_evaluator.run_benchmark()
    print('Benchmark Results:')
    print(f'Mean Recall@3: {report.mean_recall_at_3}')
    print(f'Mean Precision@3: {report.mean_precision_at_3}')
    print(f'Mean MRR: {report.mean_mrr}')
    print(f'Mean Evidence Coverage: {report.mean_evidence_coverage}')
    print(f'Mean Claim Support Rate: {report.mean_claim_support_rate}')
    print(f'Security Rejection Rate: {report.overall_security_rejection_rate}')
    print(f'Average Similarity: {report.average_similarity}')
    print(f'Successful Queries: {report.successful_queries}')
    print(f'Failed Queries: {report.failed_queries}')

asyncio.run(main())
"
```

### 5. Export audit events

After running the backend, export structured audit events with:

```text
GET /api/audit/export?format=json
GET /api/audit/export?format=csv&session_id=<session-id>
```

Audit data is kept in memory for the running process; export it before stopping
the backend if it is needed for an external report.

---

## Autonomous Orchestration & Decision-Making Layer (LangGraph)

The orchestration layer coordinates the multi-agent decision cycle over a stateful directed graph:

```text
[ User Query ]
       ↓
[ Query Understanding Agent ] (Scope, Concepts, Ambiguity Resolution)
       ↓
[ Research Planner Agent ]    (Multi-Task Decomposition, Parallel Batches)
       ↓
[ Research Execution Node ]   (Calls Research Intelligence Subsystem)
       ↓
[ Evidence Verifier Agent ]   (Provenance Check, Support Scoring, Contradiction Detection)
       ↓
[ Research Reviewer Agent ] ──(Gaps Found & Iterations < Max)──→ [ Replanning Node ]
       ↓ (Objective Satisfied OR Max Iterations Reached)                  ↓
[ Report Writer Agent ]                                       [ Research Execution Node ]
       ↓
[ Citation Validation Agent ] (100% Ground Truth Audit: Real Sources & Quotes)
       ↓
[ Final Validated Report ]
```

### Key Orchestration Components

- **Query Understanding Agent** (`backend.app.orchestration.agents.query_understanding`): Translates raw queries into validated specifications, extracts key concepts, formulates sub-questions, and handles under-specified or ambiguous queries.
- **Research Planner Agent** (`backend.app.orchestration.agents.planner`): Decomposes objectives into focused tasks with parallel/sequential dependency ordering and manages targeted replanning.
- **Evidence Verification Agent** (`backend.app.orchestration.agents.verifier`): Deterministically resolves evidence provenance to stored sources and chunks, evaluates lexical/semantic support strength, and flags directional contradictions.
- **Research Reviewer Agent** (`backend.app.orchestration.agents.reviewer`): Evaluates coverage against sub-questions, detects information gaps, enforces bounded loop stopping conditions, and triggers targeted replanning.
- **Report Writer Agent** (`backend.app.orchestration.agents.writer`): Synthesizes findings across research tasks into an evidence-grounded report with executive summary, methodology, key findings, contradictions, and disclosed limitations.
- **Citation Validation Agent** (`backend.app.orchestration.agents.validator`): Validates that every citation exists in persistent storage with exact verbatim passages, ensuring zero hallucinated citations.
- **LangGraph State Machine** (`backend.app.orchestration.graph`): Stateful LangGraph workflow with typed state, robust reducers, error resilience, and MemorySaver checkpointing.
- **FastAPI Endpoints**:
  - `POST /api/orchestration/research`: Submit research query and execute autonomous research workflow.
  - `GET /api/orchestration/sessions/{session_id}`: Inspect execution status, plan, verification report, and final research report.
