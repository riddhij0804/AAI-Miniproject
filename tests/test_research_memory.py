"""Unit tests for Research Memory and follow-up evidence retrieval."""

from backend.app.evidence.memory import ResearchMemoryManager
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


def test_research_memory_session_snapshot(evidence_store, test_retriever, in_memory_db_session):
    memory_mgr = ResearchMemoryManager(
        store=evidence_store,
        retriever=test_retriever,
        db_session_factory=lambda: in_memory_db_session,
    )

    # 1. Create multi-turn session
    session = evidence_store.create_session("AI Security Study", "What are security risks in LLMs?")
    task1 = evidence_store.create_task(session.id, "Prompt Injection", "Analyze prompt injection risks")
    evidence_store.create_iteration(session.id, 1, notes="Initial literature discovery completed")

    src = Source(
        id=generate_uuid(),
        url="https://owasp.org/llm-top-10",
        title="OWASP Top 10 for LLMs",
        source_type=SourceType.ORGANIZATION,
    )
    evidence_store.save_source(src)

    doc = Document(
        id=generate_uuid(),
        source_id=src.id,
        title="OWASP LLM Security",
        content="Prompt injection allows untrusted user inputs to alter application logic.",
        document_type=DocumentType.HTML,
    )
    evidence_store.save_document(doc)

    claim = Claim(
        id=generate_uuid(),
        session_id=session.id,
        research_task_id=task1.id,
        text="Prompt injection subverts LLM application control flow.",
    )
    evidence_store.save_claim(claim)

    ev = Evidence(
        id=generate_uuid(),
        claim_id=claim.id,
        source_id=src.id,
        document_id=doc.id,
        chunk_id="chk-sec-1",
        text="Prompt injection allows untrusted user inputs to alter application logic.",
        location="Section 1",
    )
    evidence_store.save_evidence(ev)

    # 2. Retrieve session context view
    context = memory_mgr.get_session_context(session.id)
    assert context is not None
    assert context.session.id == session.id
    assert len(context.tasks) == 1
    assert len(context.iterations) == 1
    assert len(context.claims) == 1
    assert len(context.evidence) == 1
    assert context.claims[0].text == claim.text
