"""Clean integration service contract for teammate orchestration layer (e.g. LangGraph)."""

import logging
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from backend.app.agents.credibility import CredibilityAssessmentAgent
from backend.app.agents.document_reader import DocumentReaderAgent
from backend.app.agents.evidence_extractor import (
    EvidenceExtractionResult,
    EvidenceExtractorAgent,
)
from backend.app.agents.source_discovery import SourceDiscoveryAgent
from backend.app.evidence.graph import EvidenceGraphService, default_graph_service
from backend.app.evidence.memory import ResearchMemoryManager, default_memory_manager
from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.retrieval.hybrid import HybridRetriever, RetrievalResult, default_retriever
from backend.app.schemas.claim import Claim
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.graph import GraphData
from backend.app.schemas.session import ResearchContextView, ResearchSession, ResearchTask
from backend.app.schemas.source import Source, SourceDiscoveryItem, SourceDiscoveryRequest

logger = logging.getLogger(__name__)


class ClaimEvidenceView(BaseModel):
    """View model for a claim along with all its backing evidence and sources."""
    claim: Claim
    evidence: List[Evidence] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)


class ResearchIntelligenceService:
    """Core domain-independent service interface for the Research Intelligence & Evidence subsystem."""

    def __init__(
        self,
        discovery_agent: Optional[SourceDiscoveryAgent] = None,
        reader_agent: Optional[DocumentReaderAgent] = None,
        extractor_agent: Optional[EvidenceExtractorAgent] = None,
        credibility_agent: Optional[CredibilityAssessmentAgent] = None,
        retriever: Optional[HybridRetriever] = None,
        store: Optional[EvidenceStore] = None,
        graph_service: Optional[EvidenceGraphService] = None,
        memory_manager: Optional[ResearchMemoryManager] = None,
    ):
        self.discovery_agent = discovery_agent or SourceDiscoveryAgent()
        self.reader_agent = reader_agent or DocumentReaderAgent()
        self.extractor_agent = extractor_agent or EvidenceExtractorAgent()
        self.credibility_agent = credibility_agent or CredibilityAssessmentAgent()
        self.retriever = retriever or default_retriever
        self.store = store or default_evidence_store
        self.graph_service = graph_service or default_graph_service
        self.memory_manager = memory_manager or default_memory_manager

    # 1. Source Discovery
    async def discover_sources(
        self,
        task: Union[SourceDiscoveryRequest, Dict[str, Any]],
    ) -> List[Source]:
        """Discover external web sources relevant to a research task."""
        if isinstance(task, dict):
            task_req = SourceDiscoveryRequest(**task)
        else:
            task_req = task

        response = await self.discovery_agent.discover_sources(task_req)
        sources: List[Source] = []
        for item in response.sources:
            src = self.discovery_agent.convert_discovery_to_source(item)
            self.store.save_source(src)
            sources.append(src)

        return sources

    # 2. Document Reading
    async def read_documents(self, sources: List[Source]) -> List[Document]:
        """Acquire, parse, sanitize, and chunk documents from sources."""
        documents = await self.reader_agent.read_documents(sources)

        # Persist documents and index chunks into hybrid retriever
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            self.store.save_document(doc)
            all_chunks.extend(doc.chunks)

        if all_chunks:
            self.retriever.index_chunks(all_chunks)

        return documents

    # 3. Evidence Extraction
    async def extract_evidence(
        self,
        documents: List[Document],
        objective: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> EvidenceExtractionResult:
        """Extract grounded claims and verbatim evidence excerpts from documents."""
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            all_chunks.extend(doc.chunks)

        result = await self.extractor_agent.extract_evidence_from_chunks(
            chunks=all_chunks,
            research_objective=objective,
            session_id=session_id,
            task_id=task_id,
        )

        # Persist extracted claims and evidence
        for claim in result.claims:
            self.store.save_claim(claim)
        for ev in result.evidence_records:
            self.store.save_evidence(ev)

        return result

    # 4. Source Credibility Assessment
    def assess_source_credibility(self, sources: List[Source]) -> List[CredibilityAssessment]:
        """Assess multi-factor transparent credibility for a set of sources."""
        assessments: List[CredibilityAssessment] = []
        for src in sources:
            assessment = self.credibility_agent.assess_source(src, all_sources=sources)
            self.store.save_credibility(assessment)
            assessments.append(assessment)
        return assessments

    # 5. RAG / Evidence Retrieval
    def retrieve_evidence(
        self,
        query: str,
        top_k: int = 5,
        source_id_filter: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Retrieve top-k relevant evidence chunks preserving provenance."""
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            source_id_filter=source_id_filter,
            use_hybrid=True,
        )

    # 6. Claim Evidence View
    def get_claim_evidence(self, claim_id: str) -> Optional[ClaimEvidenceView]:
        """Get claim details with its backing evidence snippets and sources."""
        claim = self.store.get_claim(claim_id)
        if not claim:
            return None

        evidence_list = self.store.get_evidence_for_claim(claim_id)
        source_ids = {ev.source_id for ev in evidence_list if ev.source_id}
        sources: List[Source] = []
        for sid in source_ids:
            src = self.store.get_source(sid)
            if src:
                sources.append(src)

        return ClaimEvidenceView(
            claim=claim,
            evidence=evidence_list,
            sources=sources,
        )

    # 7. Research Memory / Context
    def get_research_context(self, session_id: str) -> Optional[ResearchContextView]:
        """Retrieve full research-session memory for follow-up query context."""
        return self.memory_manager.get_session_context(session_id)

    # 8. Evidence Graph
    def get_evidence_graph(self, session_id: Optional[str] = None) -> GraphData:
        """Return the evidence graph as nodes and edges for visualization and reasoning."""
        return self.graph_service.get_evidence_graph(session_id=session_id)


# Global default service instance
default_research_service = ResearchIntelligenceService()

# Module-level convenience functions matching target integration contract
async def discover_sources(task: Union[SourceDiscoveryRequest, Dict[str, Any]]) -> List[Source]:
    return await default_research_service.discover_sources(task)

async def read_documents(sources: List[Source]) -> List[Document]:
    return await default_research_service.read_documents(sources)

async def extract_evidence(
    documents: List[Document],
    objective: str,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> EvidenceExtractionResult:
    return await default_research_service.extract_evidence(documents, objective, session_id, task_id)

def assess_source_credibility(sources: List[Source]) -> List[CredibilityAssessment]:
    return default_research_service.assess_source_credibility(sources)

def retrieve_evidence(query: str, top_k: int = 5) -> List[RetrievalResult]:
    return default_research_service.retrieve_evidence(query, top_k)

def get_claim_evidence(claim_id: str) -> Optional[ClaimEvidenceView]:
    return default_research_service.get_claim_evidence(claim_id)

def get_research_context(session_id: str) -> Optional[ResearchContextView]:
    return default_research_service.get_research_context(session_id)

def get_evidence_graph(session_id: Optional[str] = None) -> GraphData:
    return default_research_service.get_evidence_graph(session_id)
