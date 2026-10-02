"""Unit tests for relational Evidence Store persistence."""

from backend.app.schemas.claim import Claim, ClaimStatus
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


def test_evidence_store_crud(evidence_store):
    # 1. Session and task creation
    sess = evidence_store.create_session("Test AMR Session", "What causes AMR?")
    assert sess.id is not None
    assert sess.title == "Test AMR Session"

    task = evidence_store.create_task(sess.id, "AMR causes", "Discover drivers of AMR")
    assert task.session_id == sess.id

    # 2. Source saving and retrieval
    src = Source(
        id=generate_uuid(),
        url="https://who.int/amr",
        title="WHO AMR",
        publisher="WHO",
        source_type=SourceType.GOVERNMENT,
    )
    evidence_store.save_source(src)
    fetched_src = evidence_store.get_source(src.id)
    assert fetched_src is not None
    assert fetched_src.url == src.url

    # 3. Document and chunk saving
    doc_id = generate_uuid()
    chk = DocumentChunk(
        id=generate_uuid(),
        document_id=doc_id,
        source_id=src.id,
        source_url=src.url,
        text="Overprescription is a primary driver.",
        chunk_index=0,
    )
    doc = Document(
        id=doc_id,
        source_id=src.id,
        title="WHO AMR Document",
        content="Overprescription is a primary driver.",
        document_type=DocumentType.HTML,
        chunks=[chk],
    )
    evidence_store.save_document(doc)
    fetched_doc = evidence_store.get_document(doc.id)
    assert fetched_doc is not None
    assert len(fetched_doc.chunks) == 1
    assert fetched_doc.chunks[0].id == chk.id

    # 4. Claim and Evidence saving
    claim = Claim(
        id=generate_uuid(),
        session_id=sess.id,
        research_task_id=task.id,
        text="Clinical overprescription causes antibiotic resistance.",
        confidence=0.92,
        status=ClaimStatus.EXTRACTED,
    )
    evidence_store.save_claim(claim)

    ev = Evidence(
        id=generate_uuid(),
        claim_id=claim.id,
        source_id=src.id,
        document_id=doc.id,
        chunk_id=chk.id,
        text="Overprescription is a primary driver.",
        location="Chunk 0",
        supports=True,
    )
    evidence_store.save_evidence(ev)

    ev_list = evidence_store.get_evidence_for_claim(claim.id)
    assert len(ev_list) == 1
    assert ev_list[0].id == ev.id
    assert ev_list[0].text == ev.text
