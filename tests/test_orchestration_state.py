"""Unit tests for shared orchestration state models and reducers."""

import pytest
from backend.app.orchestration.state import (
    PlanTask,
    ResearchPlan,
    ResearchWorkflowState,
    StructuredQuery,
    merge_claims,
    merge_contradictions,
    merge_documents,
    merge_errors,
    merge_evidence,
    merge_sources,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Contradiction, Evidence
from backend.app.schemas.source import Source


def test_merge_sources_deduplication():
    s1 = Source(id="src-1", url="https://who.int/amr", title="WHO", source_type=SourceType.GOVERNMENT)
    s2 = Source(id="src-2", url="https://cdc.gov/amr", title="CDC", source_type=SourceType.GOVERNMENT)
    s1_dup = Source(id="src-3", url="https://who.int/amr", title="WHO Updated", source_type=SourceType.GOVERNMENT)

    merged = merge_sources([s1, s2], [s1_dup])
    assert len(merged) == 2
    assert any(s.url == "https://who.int/amr" for s in merged)
    assert any(s.url == "https://cdc.gov/amr" for s in merged)


def test_merge_claims_deduplication():
    c1 = Claim(id="claim-1", text="Antibiotic overuse accelerates resistance.", evidence_ids=["ev-1"])
    c2 = Claim(id="claim-2", text="Horizontal gene transfer spreads beta-lactamase.", evidence_ids=["ev-2"])
    c1_dup = Claim(id="claim-3", text="Antibiotic overuse accelerates resistance.", evidence_ids=["ev-3"])

    merged = merge_claims([c1, c2], [c1_dup])
    assert len(merged) == 2
    # Check evidence_ids merged
    c1_found = next(c for c in merged if "overuse accelerates" in c.text)
    assert "ev-1" in c1_found.evidence_ids
    assert "ev-3" in c1_found.evidence_ids


def test_merge_evidence_deduplication():
    e1 = Evidence(
        id="ev-1",
        source_id="src-1",
        document_id="doc-1",
        chunk_id="chk-1",
        text="Overuse of antibiotics in livestock fuels resistance.",
        location="p. 1",
    )
    e2 = Evidence(
        id="ev-2",
        source_id="src-1",
        document_id="doc-1",
        chunk_id="chk-1",
        text="Overuse of antibiotics in livestock fuels resistance.",
        location="p. 1",
    )
    merged = merge_evidence([e1], [e2])
    assert len(merged) == 1


def test_merge_contradictions():
    c1 = Contradiction(
        id="con-1",
        claim_a_id="c-1",
        claim_b_id="c-2",
        reasoning="Directional conflict",
        severity=0.8,
    )
    c1_dup = Contradiction(
        id="con-2",
        claim_a_id="c-2",
        claim_b_id="c-1",
        reasoning="Duplicate reversed order",
        severity=0.8,
    )
    merged = merge_contradictions([c1], [c1_dup])
    assert len(merged) == 1


def test_merge_errors():
    merged = merge_errors(["Timeout on src-1", "Parser failed"], ["Timeout on src-1", "Disk full"])
    assert len(merged) == 3
    assert merged == ["Timeout on src-1", "Parser failed", "Disk full"]


def test_workflow_state_initialization():
    state = ResearchWorkflowState(
        session_id="sess-test-01",
        original_question="What causes antimicrobial resistance?",
        max_iterations=3,
    )
    assert state.session_id == "sess-test-01"
    assert state.iteration == 1
    assert state.is_completed is False
    assert state.current_stage == "query_understanding"
    assert len(state.discovered_sources) == 0
    assert len(state.claims) == 0

