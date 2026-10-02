"""Unit tests for Evidence Extraction Agent and grounding verification."""

import pytest

from backend.app.agents.evidence_extractor import EvidenceExtractorAgent
from backend.app.schemas.document import DocumentChunk


def test_quote_grounding_verification():
    extractor = EvidenceExtractorAgent()
    chunk_text = (
        "Bacterial pathogens frequently exchange resistance plasmids via horizontal gene transfer. "
        "Overuse of broad-spectrum antimicrobials in agriculture creates massive evolutionary selection pressure."
    )

    # Valid exact quote
    valid_quote = "Overuse of broad-spectrum antimicrobials in agriculture creates massive evolutionary selection pressure."
    is_valid, matched = extractor.verify_quote_grounding(valid_quote, chunk_text)
    assert is_valid is True
    assert "evolutionary selection pressure" in matched

    # Hallucinated quote not present in chunk
    hallucinated_quote = "Penicillin was discovered in 1928 by Alexander Fleming in Scotland."
    is_valid, _ = extractor.verify_quote_grounding(hallucinated_quote, chunk_text)
    assert is_valid is False


@pytest.mark.asyncio
async def test_evidence_extraction_from_chunk():
    extractor = EvidenceExtractorAgent()
    chunk = DocumentChunk(
        id="chk-test-01",
        document_id="doc-test-01",
        source_id="src-test-01",
        source_url="https://who.int/amr",
        text=(
            "Clinical overprescription of antibiotics for viral infections is a primary driver of resistance. "
            "In 2019, approximately 1.27 million deaths were directly attributed to antimicrobial resistance."
        ),
        chunk_index=0,
        section="Page 2",
    )

    result = await extractor.extract_evidence_from_chunks(
        chunks=[chunk],
        research_objective="Determine causes and mortality burden of antibiotic resistance",
    )

    assert len(result.claims) > 0
    assert len(result.evidence_records) > 0
    assert result.verified_quote_count > 0
    assert result.rejected_hallucinated_quotes == 0

    ev = result.evidence_records[0]
    assert ev.chunk_id == "chk-test-01"
    assert ev.document_id == "doc-test-01"
    assert ev.source_id == "src-test-01"
    assert ev.text in chunk.text
    assert ev.metadata["grounding_verified"] is True
