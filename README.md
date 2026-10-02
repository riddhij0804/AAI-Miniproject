# Research Intelligence & Evidence Subsystem

> **Autonomous Research & Intelligence Agent** — General-Purpose, Domain-Independent Research Engine Subsystem.

This repository implements the **Research Intelligence & Evidence** subsystem for the Autonomous Multi-Agent Research Engine. It is completely domain-independent (operating across medicine, energy, AI security, public policy, etc.) and provides clean, decoupled service interfaces for integration with the teammate's LangGraph orchestration layer.

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
3. **Zero Fabricated Citations**: The extractor cryptographically and lexically verifies that every extracted evidence quote is an exact or high-fidelity substring of the source chunk. Hallucinations are rejected.
4. **Untrusted Data Isolation**: All retrieved web and PDF text is treated as untrusted external data. Control tokens (`<|im_start|>`, `[INST]`) are neutralized, suspicious prompt injections are detected and isolated into passive XML data blocks.
5. **SSRF Protection**: Private IP ranges (RFC 1918), localhost/loopback (`127.0.0.1`, `::1`), cloud metadata endpoints (`169.254.169.254`), and dangerous URI schemes (`file://`, `gopher://`) are strictly blocked.
6. **Complete Traceability**: Every chunk, evidence excerpt, and claim maintains an unbroken provenance link:
   $$\text{Evidence} \rightarrow \text{Chunk} \rightarrow \text{Document} \rightarrow \text{Source} \rightarrow \text{URL}$$

---

## Integration Contract (For Teammate's LangGraph Layer)

The subsystem provides clean, decoupled async functions from `backend.app.services.research_service`:

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
│           └── research_service.py   # Integration interface for teammate's LangGraph
├── tests/                            # 28 Comprehensive pytest test suites
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Running the Subsystem

### 1. Run the Test Suite
```bash
pytest tests/ -v
```
All 28 tests run in ~3 seconds with zero external network dependencies (in-memory SQLite, in-memory Qdrant, deterministic feature embeddings).

### 2. Run the FastAPI Service
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at `http://localhost:8000/docs`.

### 3. Run the Automated Evaluation Benchmark
```bash
python -c "
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

asyncio.run(main())
"
```
