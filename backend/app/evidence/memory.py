"""Research Memory managing multi-turn session persistence and follow-up evidence retrieval."""

import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.models.entities import (
    ClaimModel,
    DocumentChunkModel,
    DocumentModel,
    EvidenceModel,
    ResearchIterationModel,
    ResearchSessionModel,
    ResearchTaskModel,
    SourceModel,
)
from backend.app.retrieval.hybrid import HybridRetriever, RetrievalResult, default_retriever
from backend.app.schemas.claim import Claim, ClaimStatus
from backend.app.schemas.common import DocumentType, SourceType
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.session import (
    ResearchContextView,
    ResearchIteration,
    ResearchSession,
    ResearchTask,
)
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class ResearchMemoryManager:
    """Manages cross-iteration state and contextual evidence recall for follow-up questions."""

    def __init__(
        self,
        store: EvidenceStore = default_evidence_store,
        retriever: HybridRetriever = default_retriever,
        db_session_factory=SessionLocal,
    ):
        self.store = store
        self.retriever = retriever
        self.session_factory = db_session_factory

    def get_session_context(self, session_id: str) -> Optional[ResearchContextView]:
        """Fetch the complete research memory snapshot for a session."""
        with self.session_factory() as db:
            sess_model = db.get(ResearchSessionModel, session_id)
            if not sess_model:
                return None

            session_obj = ResearchSession(
                id=sess_model.id,
                title=sess_model.title,
                original_question=sess_model.original_question,
                status=sess_model.status,
                created_at=sess_model.created_at,
                metadata=sess_model.metadata_json,
            )

            # Tasks
            task_stmt = select(ResearchTaskModel).where(ResearchTaskModel.session_id == session_id)
            task_models = db.execute(task_stmt).scalars().all()
            tasks = [
                ResearchTask(
                    id=t.id,
                    session_id=t.session_id,
                    query=t.query,
                    research_objective=t.research_objective,
                    status=t.status,
                    source_preferences=t.source_preferences,
                    iteration_index=t.iteration_index,
                    created_at=t.created_at,
                    metadata=t.metadata_json,
                )
                for t in task_models
            ]

            # Iterations
            iter_stmt = select(ResearchIterationModel).where(ResearchIterationModel.session_id == session_id)
            iter_models = db.execute(iter_stmt).scalars().all()
            iterations = [
                ResearchIteration(
                    id=i.id,
                    session_id=i.session_id,
                    iteration_number=i.iteration_number,
                    notes=i.notes,
                    created_at=i.created_at,
                    metadata=i.metadata_json,
                )
                for i in iter_models
            ]

            # Claims associated with session
            claim_stmt = select(ClaimModel).where(ClaimModel.session_id == session_id)
            claim_models = db.execute(claim_stmt).scalars().all()
            claims = [
                Claim(
                    id=c.id,
                    session_id=c.session_id,
                    research_task_id=c.research_task_id,
                    text=c.text,
                    confidence=c.confidence,
                    status=ClaimStatus(c.status) if c.status in [e.value for e in ClaimStatus] else ClaimStatus.EXTRACTED,
                    source_ids=c.source_ids,
                    evidence_ids=c.evidence_ids,
                    created_at=c.created_at,
                    metadata=c.metadata_json,
                )
                for c in claim_models
            ]

            claim_ids = [c.id for c in claims]

            # Evidence records
            evidences: List[Evidence] = []
            if claim_ids:
                ev_stmt = select(EvidenceModel).where(EvidenceModel.claim_id.in_(claim_ids))
                ev_models = db.execute(ev_stmt).scalars().all()
                evidences = [
                    Evidence(
                        id=e.id,
                        claim_id=e.claim_id,
                        source_id=e.source_id,
                        document_id=e.document_id,
                        chunk_id=e.chunk_id,
                        text=e.text,
                        location=e.location,
                        supports=e.supports,
                        extraction_confidence=e.extraction_confidence,
                        created_at=e.created_at,
                        metadata=e.metadata_json,
                    )
                    for e in ev_models
                ]

            # Documents and sources
            doc_ids = {e.document_id for e in evidences if e.document_id}
            documents: List[Document] = []
            source_ids = {e.source_id for e in evidences if e.source_id}

            if doc_ids:
                doc_stmt = select(DocumentModel).where(DocumentModel.id.in_(list(doc_ids)))
                doc_models = db.execute(doc_stmt).scalars().all()
                for dm in doc_models:
                    source_ids.add(dm.source_id)
                    documents.append(
                        Document(
                            id=dm.id,
                            source_id=dm.source_id,
                            title=dm.title,
                            content=dm.content,
                            document_type=DocumentType(dm.document_type) if dm.document_type in [e.value for e in DocumentType] else DocumentType.HTML,
                            publication_date=dm.publication_date,
                            created_at=dm.created_at,
                            metadata=dm.metadata_json,
                        )
                    )

            sources: List[Source] = []
            if source_ids:
                src_stmt = select(SourceModel).where(SourceModel.id.in_(list(source_ids)))
                src_models = db.execute(src_stmt).scalars().all()
                for sm in src_models:
                    sources.append(
                        Source(
                            id=sm.id,
                            url=sm.url,
                            title=sm.title,
                            publisher=sm.publisher,
                            source_type=SourceType(sm.source_type) if sm.source_type in [e.value for e in SourceType] else SourceType.WEBPAGE,
                            author=sm.author,
                            published_at=sm.published_at,
                            retrieved_at=sm.retrieved_at,
                            credibility_score=sm.credibility_score,
                            metadata=sm.metadata_json,
                        )
                    )

            return ResearchContextView(
                session=session_obj,
                tasks=tasks,
                iterations=iterations,
                sources=sources,
                documents=documents,
                claims=claims,
                evidence=evidences,
            )

    def retrieve_session_evidence(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieve relevant previously stored evidence within the scope of a research session."""
        # Use hybrid retrieval across indexed chunks
        return self.retriever.retrieve(query=query, top_k=top_k)


# Global default memory manager
default_memory_manager = ResearchMemoryManager()
