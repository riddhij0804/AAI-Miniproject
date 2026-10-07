import os
import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of execution CWD
_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from contextlib import asynccontextmanager
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.app.orchestration.state import ResearchWorkflowState
from backend.app.schemas.common import generate_uuid
import json
from fastapi import Request
from fastapi.responses import StreamingResponse

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.evaluation.evaluator import EvaluationReport, default_evaluator
from backend.app.schemas.claim import Claim
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.graph import GraphData
from backend.app.schemas.session import ResearchContextView
from backend.app.schemas.source import Source, SourceDiscoveryRequest
from backend.app.services.research_service import (
    ClaimEvidenceView,
    EvidenceExtractionResult,
    default_research_service,
)
from backend.app.orchestration.service import (
    ResearchExecutionRequest,
    ResearchExecutionResponse,
    default_orchestration_service,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database schemas...")
    init_db()
    logger.info("Subsystem initialized and ready.")
    yield
    logger.info("Shutting down Research Intelligence subsystem.")


app = FastAPI(
    title="Autonomous Research Intelligence & Evidence API",
    description="Domain-independent evidence discovery, acquisition, extraction, credibility, RAG, and graph subsystem.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response schemas for API
class EvidenceExtractRequest(BaseModel):
    documents: List[Document]
    objective: str
    session_id: Optional[str] = None
    task_id: Optional[str] = None


class EvidenceRetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    source_id_filter: Optional[str] = None


from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root access directly to interactive Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }


@app.post("/api/sources/discover", response_model=List[Source], tags=["Source Discovery"])
async def discover_sources_endpoint(task: SourceDiscoveryRequest):
    """Discover sources using Tavily with DuckDuckGo fallback."""
    try:
        return await default_research_service.discover_sources(task)
    except Exception as e:
        logger.error(f"Error discovering sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/read", response_model=List[Document], tags=["Document Reader"])
async def read_documents_endpoint(sources: List[Source]):
    """Acquire, parse, and chunk documents from sources."""
    try:
        return await default_research_service.read_documents(sources)
    except Exception as e:
        logger.error(f"Error reading documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evidence/extract", response_model=EvidenceExtractionResult, tags=["Evidence Extraction"])
async def extract_evidence_endpoint(payload: EvidenceExtractRequest):
    """Extract claims and grounded verbatim evidence quotes from document chunks."""
    try:
        return await default_research_service.extract_evidence(
            documents=payload.documents,
            objective=payload.objective,
            session_id=payload.session_id,
            task_id=payload.task_id,
        )
    except Exception as e:
        logger.error(f"Error extracting evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/credibility/assess", response_model=List[CredibilityAssessment], tags=["Credibility"])
def assess_credibility_endpoint(sources: List[Source]):
    """Evaluate multi-factor credibility scores and explanatory reasoning for sources."""
    try:
        return default_research_service.assess_source_credibility(sources)
    except Exception as e:
        logger.error(f"Error assessing credibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evidence/retrieve", tags=["RAG Retrieval"])
def retrieve_evidence_endpoint(payload: EvidenceRetrieveRequest):
    """Hybrid retrieval of relevant evidence chunks preserving provenance."""
    try:
        return default_research_service.retrieve_evidence(
            query=payload.query,
            top_k=payload.top_k,
            source_id_filter=payload.source_id_filter,
        )
    except Exception as e:
        logger.error(f"Error retrieving evidence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/claims/{claim_id}/evidence", response_model=ClaimEvidenceView, tags=["Evidence"])
def get_claim_evidence_endpoint(claim_id: str):
    """Retrieve all evidence snippets and sources backing a specific claim."""
    result = default_research_service.get_claim_evidence(claim_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")
    return result


@app.get("/api/sessions/{session_id}/context", response_model=ResearchContextView, tags=["Research Memory"])
def get_research_context_endpoint(session_id: str):
    """Fetch complete research memory and entity history for a session."""
    context = default_research_service.get_research_context(session_id)
    if not context:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return context


@app.get("/api/sessions/{session_id}/graph", response_model=GraphData, tags=["Evidence Graph"])
def get_evidence_graph_endpoint(session_id: str):
    """Retrieve evidence graph nodes and edges for visualization or relational reasoning."""
    return default_research_service.get_evidence_graph(session_id=session_id)


@app.post("/api/evaluate", response_model=EvaluationReport, tags=["Evaluation"])
async def run_evaluation_benchmark_endpoint():
    """Execute domain-independent benchmark suite evaluating retrieval, evidence, reliability, and security."""
    try:
        return await default_evaluator.run_benchmark()
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orchestration/research", response_model=ResearchExecutionResponse, tags=["Orchestration"])
async def run_autonomous_research_endpoint(payload: ResearchExecutionRequest):
    """Execute end-to-end autonomous research workflow via LangGraph orchestrator."""
    try:
        return await default_orchestration_service.execute_research(payload)
    except Exception as e:
        logger.error(f"Error in research orchestration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orchestration/sessions/{session_id}", response_model=ResearchExecutionResponse, tags=["Orchestration"])
def get_orchestration_session_endpoint(session_id: str):
    """Retrieve full execution status, plan, verification report, and final research report."""
    res = default_orchestration_service.get_session_state(session_id)
    if not res:
        raise HTTPException(status_code=404, detail=f"Orchestration session '{session_id}' not found.")
    return res
