"""Pytest fixtures for the Research Intelligence & Evidence subsystem."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.evidence.store import EvidenceStore
from backend.app.evidence.graph import EvidenceGraphService
from backend.app.evidence.memory import ResearchMemoryManager
from backend.app.retrieval.chunking import DocumentChunker
from backend.app.retrieval.embeddings import HashEmbeddingService
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.vector_store import QdrantVectorStore
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.source import Source
from backend.app.services.research_service import ResearchIntelligenceService


@pytest.fixture
def in_memory_db_session():
    """Create an isolated in-memory SQLite database session for unit tests."""
    engine = create_engine("sqlite:///:memory:", echo=False, future=True)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def evidence_store(in_memory_db_session):
    """Evidence store using in-memory SQLite session factory."""
    session_factory = lambda: in_memory_db_session
    return EvidenceStore(db_session_factory=session_factory)


@pytest.fixture
def test_retriever():
    """Isolated in-memory hybrid retriever using fast Hash embeddings."""
    embedding_svc = HashEmbeddingService(dimension=384)
    vs = QdrantVectorStore(location=":memory:", collection_name=f"test_coll_{generate_uuid()[:8]}")
    return HybridRetriever(vector_store=vs, embedding_service=embedding_svc)


@pytest.fixture
def sample_source() -> Source:
    return Source(
        id=generate_uuid(),
        url="https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        title="WHO Antimicrobial Resistance Factsheet",
        publisher="World Health Organization",
        source_type=SourceType.GOVERNMENT,
        metadata={"category": "public_health"},
    )


@pytest.fixture
def sample_document(sample_source) -> Document:
    doc_id = generate_uuid()
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    content = (
        "Antimicrobial resistance occurs when bacteria change over time and no longer respond to medicines. "
        "Clinical overprescription of antibiotics accelerates this process dramatically. "
        "In livestock farming, subtherapeutic antibiotic use as growth promoters fuels resistant strains. "
        "Plasmids transfer resistance genes horizontally across diverse bacterial populations."
    )
    doc = Document(
        id=doc_id,
        source_id=sample_source.id,
        title=sample_source.title,
        content=content,
        document_type=DocumentType.HTML,
        metadata={"url": sample_source.url},
    )
    doc.chunks = chunker.chunk_document(doc, source_url=sample_source.url, source_id=sample_source.id)
    return doc
