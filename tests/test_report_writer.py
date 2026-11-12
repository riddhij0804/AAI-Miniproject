import pytest

from backend.app.config import settings
from backend.app.orchestration.agents.writer import ReportWriterAgent
from backend.app.orchestration.state import (
    EvidenceVerificationReport,
    StructuredQuery,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


@pytest.mark.asyncio
async def test_report_writer_bounds_citations(sample_source):
    claims = []
    evidence = []
    for index in range(settings.MAX_REPORT_CITATIONS + 10):
        claim_id = f"claim-{index}"
        evidence_id = f"evidence-{index}"
        claims.append(Claim(id=claim_id, text=f"Grounded finding {index}", evidence_ids=[evidence_id]))
        evidence.append(
            Evidence(
                id=evidence_id,
                claim_id=claim_id,
                source_id=sample_source.id,
                document_id="document-1",
                chunk_id="chunk-1",
                text=f"Grounded source text {index}",
                location="body",
            )
        )
    report = await ReportWriterAgent().generate_report(
        structured_query=StructuredQuery(
            original_query="Test research question",
            research_objective="Test objective",
            scope="Test scope",
        ),
        claims=claims,
        evidence_records=evidence,
        sources=[sample_source],
        verification_report=EvidenceVerificationReport(
            total_claims_checked=len(claims),
            supported_claims_count=len(claims),
            claim_findings=[],
        ),
        contradictions=[],
    )
    assert len(report.references) <= settings.MAX_REPORT_CITATIONS
