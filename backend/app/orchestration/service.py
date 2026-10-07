"""High-level Orchestration Service coordinating execution sessions and API integration."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.orchestration.graph import (
    ResearchWorkflowOrchestrator,
    default_orchestrator,
)
from backend.app.orchestration.state import (
    CitationValidationResult,
    EvidenceVerificationReport,
    FinalResearchReport,
    ResearchPlan,
    ResearchReviewDecision,
    ResearchWorkflowState,
    StructuredQuery,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import generate_uuid
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Contradiction, Evidence
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class ResearchExecutionRequest(BaseModel):
    """API request model to trigger autonomous research workflow."""
    question: str = Field(..., min_length=3, description="The research question to investigate")
    session_id: Optional[str] = Field(None, description="Optional custom session identifier")
    max_iterations: int = Field(default=3, ge=1, le=5, description="Maximum research iterations")
    max_documents: int = Field(default=10, ge=1, le=100, description="Maximum documents to fetch and read")


class ResearchExecutionResponse(BaseModel):
    """Comprehensive API response containing the full research workflow results."""
    session_id: str
    original_question: str
    status: str
    iteration_count: int
    structured_query: Optional[StructuredQuery] = None
    plan: Optional[ResearchPlan] = None
    verification_report: Optional[EvidenceVerificationReport] = None
    contradictions: List[Contradiction] = Field(default_factory=list)
    review_decision: Optional[ResearchReviewDecision] = None
    final_report: Optional[FinalResearchReport] = None
    citation_validation: Optional[CitationValidationResult] = None
    discovered_sources_count: int = 0
    extracted_claims_count: int = 0
    verified_evidence_count: int = 0
    sources: List[Source] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    evidence_records: List[Evidence] = Field(default_factory=list)
    credibility_assessments: List[CredibilityAssessment] = Field(default_factory=list)
    audit_logs: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class OrchestrationService:
    """Manages autonomous research sessions and connects FastAPI endpoints to LangGraph."""

    def __init__(
        self,
        orchestrator: Optional[ResearchWorkflowOrchestrator] = None,
        store: Optional[EvidenceStore] = None,
    ):
        self.orchestrator = orchestrator or default_orchestrator
        self.store = store or default_evidence_store
        self._session_cache: Dict[str, ResearchWorkflowState] = {}

    async def execute_research(
        self,
        request: ResearchExecutionRequest,
    ) -> ResearchExecutionResponse:
        """Run autonomous research workflow end-to-end."""
        session_id = request.session_id or f"sess-{generate_uuid()[:8]}"
        logger.info(f"OrchestrationService: Starting research session '{session_id}' for query: '{request.question}'")

        state = await self.orchestrator.run(
            question=request.question,
            session_id=session_id,
            max_iterations=request.max_iterations,
            max_documents=request.max_documents,
        )

        self._session_cache[session_id] = state

        return self._format_response(state)

    def get_session_state(self, session_id: str) -> Optional[ResearchExecutionResponse]:
        """Retrieve stored execution state for a research session."""
        if session_id in self._session_cache:
            return self._format_response(self._session_cache[session_id])
        return None

    def _format_response(self, state: ResearchWorkflowState) -> ResearchExecutionResponse:
        return ResearchExecutionResponse(
            session_id=state.session_id,
            original_question=state.original_question,
            status="completed" if state.is_completed else state.current_stage,
            iteration_count=state.iteration,
            structured_query=state.structured_query,
            plan=state.plan,
            verification_report=state.verification_report,
            contradictions=state.contradictions,
            review_decision=state.review_decision,
            final_report=state.final_report,
            citation_validation=state.citation_validation,
            discovered_sources_count=len(state.discovered_sources),
            extracted_claims_count=len(state.claims),
            verified_evidence_count=len(state.evidence_records),
            sources=state.discovered_sources,
            documents=state.documents,
            claims=state.claims,
            evidence_records=state.evidence_records,
            credibility_assessments=state.credibility_assessments,
            audit_logs=state.audit_logs,
            errors=state.errors,
        )


default_orchestration_service = OrchestrationService()

