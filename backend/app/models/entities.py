"""SQLAlchemy ORM models for the Research Intelligence & Evidence subsystem."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceModel(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    publisher: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    source_type: Mapped[str] = mapped_column(String(64), default="webpage")
    author: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    credibility_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    documents: Mapped[List["DocumentModel"]] = relationship(
        "DocumentModel", back_populates="source", cascade="all, delete-orphan"
    )
    credibility: Mapped[Optional["CredibilityAssessmentModel"]] = relationship(
        "CredibilityAssessmentModel", back_populates="source", uselist=False, cascade="all, delete-orphan"
    )


class DocumentModel(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sources.id", ondelete="CASCADE"), index=True, nullable=False
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(64), default="html")
    publication_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    source: Mapped["SourceModel"] = relationship("SourceModel", back_populates="documents")
    chunks: Mapped[List["DocumentChunkModel"]] = relationship(
        "DocumentChunkModel", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunkModel(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    document: Mapped["DocumentModel"] = relationship("DocumentModel", back_populates="chunks")


class ClaimModel(Base):
    __tablename__ = "claims"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("research_sessions.id", ondelete="SET NULL"), index=True, nullable=True
    )
    research_task_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("research_tasks.id", ondelete="SET NULL"), index=True, nullable=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(64), default="extracted")
    source_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    evidence_ids: Mapped[List[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    evidence_records: Mapped[List["EvidenceModel"]] = relationship(
        "EvidenceModel", back_populates="claim"
    )


class EvidenceModel(Base):
    __tablename__ = "evidence"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    claim_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("claims.id", ondelete="SET NULL"), index=True, nullable=True
    )
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sources.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("document_chunks.id", ondelete="CASCADE"), index=True, nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(String(256), nullable=False)
    supports: Mapped[bool] = mapped_column(Boolean, default=True)
    extraction_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    claim: Mapped[Optional["ClaimModel"]] = relationship("ClaimModel", back_populates="evidence_records")


class CitationModel(Base):
    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    claim_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("claims.id", ondelete="CASCADE"), index=True, nullable=False
    )
    evidence_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("evidence.id", ondelete="CASCADE"), index=True, nullable=False
    )
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sources.id", ondelete="CASCADE"), index=True, nullable=False
    )
    formatted_citation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class ContradictionModel(Base):
    __tablename__ = "contradictions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    claim_a_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("claims.id", ondelete="CASCADE"), index=True, nullable=False
    )
    claim_b_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("claims.id", ondelete="CASCADE"), index=True, nullable=False
    )
    evidence_a_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("evidence.id", ondelete="SET NULL"), nullable=True
    )
    evidence_b_id: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("evidence.id", ondelete="SET NULL"), nullable=True
    )
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[float] = mapped_column(Float, default=0.5)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class CredibilityAssessmentModel(Base):
    __tablename__ = "credibility_assessments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sources.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    authority_score: Mapped[float] = mapped_column(Float, nullable=False)
    primary_source_score: Mapped[float] = mapped_column(Float, nullable=False)
    recency_score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    peer_review_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corroboration_score: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    assessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    source: Mapped["SourceModel"] = relationship("SourceModel", back_populates="credibility")


class ResearchSessionModel(Base):
    __tablename__ = "research_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    original_question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    tasks: Mapped[List["ResearchTaskModel"]] = relationship(
        "ResearchTaskModel", back_populates="session", cascade="all, delete-orphan"
    )
    iterations: Mapped[List["ResearchIterationModel"]] = relationship(
        "ResearchIterationModel", back_populates="session", cascade="all, delete-orphan"
    )


class ResearchTaskModel(Base):
    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    query: Mapped[str] = mapped_column(Text, nullable=False)
    research_objective: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(64), default="pending")
    source_preferences: Mapped[List[str]] = mapped_column(JSON, default=list)
    iteration_index: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    session: Mapped["ResearchSessionModel"] = relationship("ResearchSessionModel", back_populates="tasks")


class ResearchIterationModel(Base):
    __tablename__ = "research_iterations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("research_sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    iteration_number: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    session: Mapped["ResearchSessionModel"] = relationship("ResearchSessionModel", back_populates="iterations")


class EntityModel(Base):
    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), index=True, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), default="concept")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class RelationshipModel(Base):
    __tablename__ = "relationships"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_node_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source_node_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    target_node_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    target_node_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    relation_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
