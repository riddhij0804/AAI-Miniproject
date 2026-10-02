"""Unit tests for deterministic document chunking and provenance tracking."""

from backend.app.retrieval.chunking import DocumentChunker
from backend.app.schemas.common import DocumentType, generate_uuid
from backend.app.schemas.document import Document


def test_chunking_determinism():
    chunker = DocumentChunker(chunk_size=150, chunk_overlap=30)
    text = (
        "Renewable energy capacity in India has expanded rapidly. "
        "Solar power contributes the largest proportion of additions. "
        "The tariffs have plummeted due to competitive reverse bidding auctions."
    )
    doc_id = "doc-fixed-123"
    doc = Document(
        id=doc_id,
        source_id="src-fixed-456",
        title="Solar Growth",
        content=text,
        document_type=DocumentType.HTML,
    )

    chunks_first_run = chunker.chunk_document(doc, source_url="https://mnre.gov.in", source_id="src-fixed-456")
    chunks_second_run = chunker.chunk_document(doc, source_url="https://mnre.gov.in", source_id="src-fixed-456")

    assert len(chunks_first_run) == len(chunks_second_run)
    for c1, c2 in zip(chunks_first_run, chunks_second_run):
        assert c1.id == c2.id
        assert c1.text == c2.text
        assert c1.chunk_index == c2.chunk_index
        assert c1.source_id == "src-fixed-456"
        assert c1.source_url == "https://mnre.gov.in"
        assert c1.document_id == doc_id


def test_chunking_paged_pdf_metadata():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    pages = [
        {"page_number": 1, "text": "Page one: Antibiotic resistance genetic mechanisms."},
        {"page_number": 2, "text": "Page two: Hospital acquired infections and mitigation strategies."},
    ]
    doc = Document(
        id="doc-paged-1",
        source_id="src-who-1",
        title="AMR Guidelines",
        content="Full text...",
        document_type=DocumentType.PDF,
        metadata={"pages": pages, "url": "https://who.int/amr.pdf"},
    )

    chunks = chunker.chunk_document(doc, source_url="https://who.int/amr.pdf", source_id="src-who-1")
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Page one" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "Page two" in chunks[1].text
