"""Persistent Evidence Store managing relational metadata and vector synchronization."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import delete, desc, select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal, get_db_context
from backend.app.models.entities import (
    CitationModel,
    ClaimModel,
    ContradictionModel,
    CredibilityAssessmentModel,
    DocumentChunkModel,
    DocumentModel,
    EntityModel,
    EvidenceModel,
    RelationshipModel,
    ResearchIterationModel,
    ResearchSessionModel,
    ResearchTaskModel,
    SourceModel,
)
from backend.app.schemas.claim import Claim, ClaimCreate
from backend.app.schemas.common import ClaimStatus, DocumentType, SourceType, utc_now
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Citation, Contradiction, Evidence
from backend.app.schemas.session import ResearchIteration, ResearchSession, ResearchTask
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class EvidenceStore:
    """Manages transactional persistence for all research entities in the relational database."""

    def __init__(self, db_session_factory=SessionLocal):
        self.session_factory = db_session_factory

    # --- Research Sessions & Tasks ---

    def create_session(self, title: str, original_question: str, metadata: Optional[Dict[str, Any]] = None) -> ResearchSession:
        with self.session_factory() as session:
            sess_model = ResearchSessionModel(
                id=ResearchSession(title=title, original_question=original_question).id,
                title=title,
                original_question=original_question,
                status="active",
                created_at=utc_now(),
                metadata_json=metadata or {},
            )
            session.add(sess_model)
            session.commit()
            session.refresh(sess_model)
            return ResearchSession(
                id=sess_model.id,
                title=sess_model.title,
                original_question=sess_model.original_question,
                status=sess_model.status,
                created_at=sess_model.created_at,
                metadata=sess_model.metadata_json,
            )

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        with self.session_factory() as session:
            stmt = select(ResearchSessionModel).where(ResearchSessionModel.id == session_id)
            model = session.execute(stmt).scalar_one_or_none()
            if not model:
                return None
            return ResearchSession(
                id=model.id,
                title=model.title,
                original_question=model.original_question,
                status=model.status,
                created_at=model.created_at,
                metadata=model.metadata_json,
            )

    def create_task(self, session_id: str, query: str, research_objective: str, preferences: Optional[List[str]] = None) -> ResearchTask:
        with self.session_factory() as session:
            task = ResearchTask(
                session_id=session_id,
                query=query,
                research_objective=research_objective,
                source_preferences=preferences or [],
            )
            model = ResearchTaskModel(
                id=task.id,
                session_id=task.session_id,
                query=task.query,
                research_objective=task.research_objective,
                status=task.status,
                source_preferences=task.source_preferences,
                iteration_index=task.iteration_index,
                created_at=task.created_at,
                metadata_json=task.metadata,
            )
            session.add(model)
            session.commit()
            return task

    def create_iteration(self, session_id: str, iteration_number: int, notes: Optional[str] = None) -> ResearchIteration:
        with self.session_factory() as session:
            iteration = ResearchIteration(
                session_id=session_id,
                iteration_number=iteration_number,
                notes=notes,
            )
            model = ResearchIterationModel(
                id=iteration.id,
                session_id=iteration.session_id,
                iteration_number=iteration.iteration_number,
                notes=iteration.notes,
                created_at=iteration.created_at,
                metadata_json=iteration.metadata,
            )
            session.add(model)
            session.commit()
            return iteration

    # --- Sources & Credibility ---

    def save_source(self, source: Source) -> Source:
        with self.session_factory() as session:
            existing = session.get(SourceModel, source.id)
            if existing:
                existing.url = source.url
                existing.title = source.title
                existing.publisher = source.publisher
                existing.source_type = source.source_type.value if hasattr(source.source_type, "value") else str(source.source_type)
                existing.author = source.author
                existing.published_at = source.published_at
                existing.credibility_score = source.credibility_score
                existing.metadata_json = source.metadata
            else:
                model = SourceModel(
                    id=source.id,
                    url=source.url,
                    title=source.title,
                    publisher=source.publisher,
                    source_type=source.source_type.value if hasattr(source.source_type, "value") else str(source.source_type),
                    author=source.author,
                    published_at=source.published_at,
                    retrieved_at=source.retrieved_at,
                    credibility_score=source.credibility_score,
                    metadata_json=source.metadata,
                )
                session.add(model)
            session.commit()
            return source

    def save_sources(self, sources: List[Source]) -> List[Source]:
        for s in sources:
            self.save_source(s)
        return sources

    def get_source(self, source_id: str) -> Optional[Source]:
        with self.session_factory() as session:
            model = session.get(SourceModel, source_id)
            if not model:
                return None
            return Source(
                id=model.id,
                url=model.url,
                title=model.title,
                publisher=model.publisher,
                source_type=SourceType(model.source_type) if model.source_type in [e.value for e in SourceType] else SourceType.WEBPAGE,
                author=model.author,
                published_at=model.published_at,
                retrieved_at=model.retrieved_at,
                credibility_score=model.credibility_score,
                metadata=model.metadata_json,
            )

    def save_credibility(self, assessment: CredibilityAssessment) -> CredibilityAssessment:
        with self.session_factory() as session:
            stmt = select(CredibilityAssessmentModel).where(CredibilityAssessmentModel.source_id == assessment.source_id)
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                existing.authority_score = assessment.authority_score
                existing.primary_source_score = assessment.primary_source_score
                existing.recency_score = assessment.recency_score
                existing.evidence_quality_score = assessment.evidence_quality_score
                existing.peer_review_score = assessment.peer_review_score
                existing.corroboration_score = assessment.corroboration_score
                existing.overall_score = assessment.overall_score
                existing.reasoning = assessment.reasoning
                existing.metadata_json = assessment.metadata
            else:
                model = CredibilityAssessmentModel(
                    id=assessment.id,
                    source_id=assessment.source_id,
                    authority_score=assessment.authority_score,
                    primary_source_score=assessment.primary_source_score,
                    recency_score=assessment.recency_score,
                    evidence_quality_score=assessment.evidence_quality_score,
                    peer_review_score=assessment.peer_review_score,
                    corroboration_score=assessment.corroboration_score,
                    overall_score=assessment.overall_score,
                    reasoning=assessment.reasoning,
                    assessed_at=assessment.assessed_at,
                    metadata_json=assessment.metadata,
                )
                session.add(model)

            # Update parent source credibility score
            src = session.get(SourceModel, assessment.source_id)
            if src:
                src.credibility_score = assessment.overall_score

            session.commit()
            return assessment

    # --- Documents & Chunks ---

    def save_document(self, document: Document) -> Document:
        with self.session_factory() as session:
            existing = session.get(DocumentModel, document.id)
            if not existing:
                doc_model = DocumentModel(
                    id=document.id,
                    source_id=document.source_id,
                    title=document.title,
                    content=document.content,
                    document_type=document.document_type.value if hasattr(document.document_type, "value") else str(document.document_type),
                    publication_date=document.publication_date,
                    created_at=document.created_at,
                    metadata_json=document.metadata,
                )
                session.add(doc_model)

            # Save chunks
            for chunk in document.chunks:
                chk_existing = session.get(DocumentChunkModel, chunk.id)
                if not chk_existing:
                    chk_model = DocumentChunkModel(
                        id=chunk.id,
                        document_id=document.id,
                        text=chunk.text,
                        chunk_index=chunk.chunk_index,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        embedding_id=chunk.embedding_id,
                        created_at=chunk.created_at,
                        metadata_json=chunk.metadata,
                    )
                    session.add(chk_model)

            session.commit()
            return document

    def get_document(self, document_id: str) -> Optional[Document]:
        with self.session_factory() as session:
            model = session.get(DocumentModel, document_id)
            if not model:
                return None
            chunks = [
                DocumentChunk(
                    id=c.id,
                    document_id=c.document_id,
                    text=c.text,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    section=c.section,
                    embedding_id=c.embedding_id,
                    created_at=c.created_at,
                    metadata=c.metadata_json,
                )
                for c in model.chunks
            ]
            return Document(
                id=model.id,
                source_id=model.source_id,
                title=model.title,
                content=model.content,
                document_type=DocumentType(model.document_type) if model.document_type in [e.value for e in DocumentType] else DocumentType.HTML,
                publication_date=model.publication_date,
                created_at=model.created_at,
                metadata=model.metadata_json,
                chunks=chunks,
            )

    # --- Claims & Evidence ---

    def save_claim(self, claim: Claim) -> Claim:
        with self.session_factory() as session:
            existing = session.get(ClaimModel, claim.id)
            if existing:
                existing.text = claim.text
                existing.confidence = claim.confidence
                existing.status = claim.status.value if hasattr(claim.status, "value") else str(claim.status)
                existing.source_ids = claim.source_ids
                existing.evidence_ids = claim.evidence_ids
                existing.metadata_json = claim.metadata
            else:
                model = ClaimModel(
                    id=claim.id,
                    session_id=claim.session_id,
                    research_task_id=claim.research_task_id,
                    text=claim.text,
                    confidence=claim.confidence,
                    status=claim.status.value if hasattr(claim.status, "value") else str(claim.status),
                    source_ids=claim.source_ids,
                    evidence_ids=claim.evidence_ids,
                    created_at=claim.created_at,
                    metadata_json=claim.metadata,
                )
                session.add(model)
            session.commit()
            return claim

    def save_evidence(self, evidence: Evidence) -> Evidence:
        with self.session_factory() as session:
            existing = session.get(EvidenceModel, evidence.id)
            if existing:
                existing.claim_id = evidence.claim_id
                existing.text = evidence.text
                existing.location = evidence.location
                existing.supports = evidence.supports
                existing.extraction_confidence = evidence.extraction_confidence
                existing.metadata_json = evidence.metadata
            else:
                model = EvidenceModel(
                    id=evidence.id,
                    claim_id=evidence.claim_id,
                    source_id=evidence.source_id,
                    document_id=evidence.document_id,
                    chunk_id=evidence.chunk_id,
                    text=evidence.text,
                    location=evidence.location,
                    supports=evidence.supports,
                    extraction_confidence=evidence.extraction_confidence,
                    created_at=evidence.created_at,
                    metadata_json=evidence.metadata,
                )
                session.add(model)
            session.commit()
            return evidence

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        with self.session_factory() as session:
            model = session.get(ClaimModel, claim_id)
            if not model:
                return None
            return Claim(
                id=model.id,
                session_id=model.session_id,
                research_task_id=model.research_task_id,
                text=model.text,
                confidence=model.confidence,
                status=ClaimStatus(model.status) if model.status in [e.value for e in ClaimStatus] else ClaimStatus.EXTRACTED,
                source_ids=model.source_ids,
                evidence_ids=model.evidence_ids,
                created_at=model.created_at,
                metadata=model.metadata_json,
            )

    def get_evidence_for_claim(self, claim_id: str) -> List[Evidence]:
        with self.session_factory() as session:
            stmt = select(EvidenceModel).where(EvidenceModel.claim_id == claim_id)
            models = session.execute(stmt).scalars().all()
            return [
                Evidence(
                    id=m.id,
                    claim_id=m.claim_id,
                    source_id=m.source_id,
                    document_id=m.document_id,
                    chunk_id=m.chunk_id,
                    text=m.text,
                    location=m.location,
                    supports=m.supports,
                    extraction_confidence=m.extraction_confidence,
                    created_at=m.created_at,
                    metadata=m.metadata_json,
                )
                for m in models
            ]

    # --- Contradictions & Citations ---

    def save_citation(self, citation: Citation) -> Citation:
        with self.session_factory() as session:
            model = CitationModel(
                id=citation.id,
                claim_id=citation.claim_id,
                evidence_id=citation.evidence_id,
                source_id=citation.source_id,
                formatted_citation=citation.formatted_citation,
                created_at=citation.created_at,
                metadata_json=citation.metadata,
            )
            session.add(model)
            session.commit()
            return citation

    def save_contradiction(self, contradiction: Contradiction) -> Contradiction:
        with self.session_factory() as session:
            model = ContradictionModel(
                id=contradiction.id,
                claim_a_id=contradiction.claim_a_id,
                claim_b_id=contradiction.claim_b_id,
                evidence_a_id=contradiction.evidence_a_id,
                evidence_b_id=contradiction.evidence_b_id,
                reasoning=contradiction.reasoning,
                severity=contradiction.severity,
                detected_at=contradiction.detected_at,
                metadata_json=contradiction.metadata,
            )
            session.add(model)
            session.commit()
            return contradiction


# Global default evidence store
default_evidence_store = EvidenceStore()
