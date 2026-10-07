"""Unit tests for individual orchestration agents."""

import pytest
from backend.app.orchestration.agents.planner import ResearchPlannerAgent
from backend.app.orchestration.agents.query_understanding import QueryUnderstandingAgent
from backend.app.orchestration.agents.reviewer import ResearchReviewerAgent
from backend.app.orchestration.agents.validator import CitationValidationAgent
from backend.app.orchestration.agents.verifier import EvidenceVerificationAgent
from backend.app.orchestration.agents.writer import ReportWriterAgent
from backend.app.orchestration.state import (
    ClaimVerificationFinding,
    EvidenceVerificationReport,
    FinalResearchReport,
    KeyFindingTopic,
    PlanTask,
    ReportCitation,
    ResearchPlan,
    StructuredQuery,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Contradiction, Evidence
from backend.app.schemas.source import Source


@pytest.mark.asyncio
async def test_query_understanding_clear_query():
    agent = QueryUnderstandingAgent()
    query = "What are the major causes of antibiotic resistance in clinical settings?"
    structured = await agent.analyze_query(query)

    assert structured.original_query == query
    assert len(structured.key_concepts) > 0
    assert "antibiotic" in [c.lower() for c in structured.key_concepts]
    assert structured.is_ambiguous is False
    assert len(structured.sub_questions) >= 2


@pytest.mark.asyncio
async def test_query_understanding_ambiguous_query():
    agent = QueryUnderstandingAgent()
    query = "AI"  # Highly ambiguous short query
    structured = await agent.analyze_query(query)

    assert structured.is_ambiguous is True
    assert len(structured.ambiguity_reasons) > 0
    assert structured.clarification_strategy is not None


@pytest.mark.asyncio
async def test_research_planner_decomposition():
    planner = ResearchPlannerAgent()
    structured = StructuredQuery(
        original_query="What causes antibiotic resistance?",
        research_objective="Identify biological and agricultural drivers of AMR",
        scope="Global",
        key_concepts=["antibiotic", "resistance", "causes"],
        sub_questions=["What are primary mechanisms?", "What empirical data exists?"],
    )
    plan = await planner.create_plan(structured, max_iterations=3)

    assert len(plan.tasks) >= 2
    assert plan.max_iterations == 3
    assert plan.current_iteration == 1
    assert any(t.can_run_in_parallel for t in plan.tasks)
    assert len(plan.parallel_batches) > 0


def test_research_planner_replanning():
    planner = ResearchPlannerAgent()
    initial_plan = ResearchPlan(
        objective="Analyze AMR",
        tasks=[
            PlanTask(title="Initial Task", query="AMR causes", objective="Identify causes")
        ],
        max_iterations=3,
        current_iteration=1,
    )
    replanned = planner.replan_with_gaps(
        current_plan=initial_plan,
        unanswered_questions=["What is the role of agricultural runoff?"],
        missing_areas=["Agricultural runoff"],
        new_iteration=2,
    )

    assert replanned.current_iteration == 2
    assert replanned.replanned_count == 1
    assert len(replanned.tasks) == 2
    assert any("runoff" in t.query.lower() for t in replanned.tasks)


@pytest.mark.asyncio
async def test_evidence_verification_agent(evidence_store, sample_source, sample_document):
    # Save source and doc in store
    evidence_store.save_source(sample_source)
    evidence_store.save_document(sample_document)

    chunk = sample_document.chunks[0]
    claim_id = generate_uuid()
    ev_id = generate_uuid()

    claim = Claim(
        id=claim_id,
        text="Clinical overprescription of antibiotics accelerates resistance.",
        evidence_ids=[ev_id],
    )
    evidence = Evidence(
        id=ev_id,
        claim_id=claim_id,
        source_id=sample_source.id,
        document_id=sample_document.id,
        chunk_id=chunk.id,
        text="Clinical overprescription of antibiotics accelerates this process dramatically.",
        location="p. 1",
        supports=True,
    )

    verifier = EvidenceVerificationAgent(store=evidence_store)
    report = await verifier.verify_evidence(
        claims=[claim],
        evidence_records=[evidence],
        sources=[sample_source],
        documents=[sample_document],
    )

    assert report.total_claims_checked == 1
    assert report.supported_claims_count == 1
    assert report.provenance_integrity_passed is True
    assert len(report.claim_findings) == 1
    assert report.claim_findings[0].is_supported is True
    assert report.claim_findings[0].has_valid_provenance is True


@pytest.mark.asyncio
async def test_evidence_verification_detects_contradiction(evidence_store, sample_source):
    c1 = Claim(
        id="c-pos",
        text="Agricultural antibiotic usage increased pathogen transmission risks.",
        evidence_ids=[],
    )
    c2 = Claim(
        id="c-neg",
        text="Agricultural antibiotic usage decreased pathogen transmission risks.",
        evidence_ids=[],
    )

    verifier = EvidenceVerificationAgent(store=evidence_store)
    report = await verifier.verify_evidence(
        claims=[c1, c2],
        evidence_records=[],
        sources=[sample_source],
        documents=[],
    )

    assert len(report.detected_contradictions) == 1
    assert report.detected_contradictions[0].severity >= 0.5


@pytest.mark.asyncio
async def test_research_reviewer_stopping_at_max_iterations():
    reviewer = ResearchReviewerAgent()
    structured = StructuredQuery(
        original_query="Analyze AMR",
        research_objective="Investigate AMR",
        scope="Global",
        key_concepts=["AMR"],
        sub_questions=["Question 1", "Question 2"],
    )
    plan = ResearchPlan(objective="Investigate AMR", tasks=[], max_iterations=2)
    ver_report = EvidenceVerificationReport(total_claims_checked=1, supported_claims_count=1)

    decision = await reviewer.review_research_progress(
        structured_query=structured,
        current_plan=plan,
        verification_report=ver_report,
        claims=[Claim(text="Claim 1")],
        evidence_records=[],
        sources=[],
        current_iteration=2,
        max_iterations=2,
    )

    assert decision.is_sufficient is False
    assert decision.should_replan is False
    assert decision.stopping_reason == "max_iterations_reached"


@pytest.mark.asyncio
async def test_report_writer_synthesis(sample_source, sample_document):
    writer = ReportWriterAgent()
    structured = StructuredQuery(
        original_query="What causes antibiotic resistance?",
        research_objective="Identify causes of AMR",
        scope="Global literature",
        key_concepts=["antibiotic", "resistance"],
        sub_questions=[],
    )
    claim = Claim(
        id="c-1",
        text="Subtherapeutic livestock antibiotics fuel resistance.",
        evidence_ids=["ev-1"],
    )
    ev = Evidence(
        id="ev-1",
        claim_id="c-1",
        source_id=sample_source.id,
        document_id=sample_document.id,
        chunk_id=sample_document.chunks[0].id,
        text="In livestock farming, subtherapeutic antibiotic use fuels resistant strains.",
        location="section 2",
    )
    ver_report = EvidenceVerificationReport(
        total_claims_checked=1,
        supported_claims_count=1,
        claim_findings=[
            ClaimVerificationFinding(
                claim_id="c-1",
                claim_text=claim.text,
                is_supported=True,
                support_strength=0.9,
                has_valid_provenance=True,
                reasoning="Strong grounding",
            )
        ],
    )

    report = await writer.generate_report(
        structured_query=structured,
        claims=[claim],
        evidence_records=[ev],
        sources=[sample_source],
        verification_report=ver_report,
        contradictions=[],
    )

    assert "antibiotic resistance" in report.title.lower()
    assert len(report.executive_summary) > 30
    assert len(report.research_scope_and_methodology) > 20
    assert len(report.key_findings) > 0
    assert len(report.references) == 1
    assert report.references[0].source_url == sample_source.url
    assert len(report.conclusion) > 20


def test_citation_validator_detects_invalid_source():
    validator = CitationValidationAgent()
    report = FinalResearchReport(
        title="Test Report",
        executive_summary="Summary",
        research_scope_and_methodology="Methodology",
        supporting_evidence_summary="Evidence",
        conclusion="Conclusion",
        references=[
            ReportCitation(
                citation_id="cite-fake",
                claim_text="Unfounded assertion",
                source_title="Nonexistent Source",
                source_url="https://fabricated-hallucination-domain.org/fake",
                exact_quote="Fake quote never written anywhere.",
                location="p. 99",
            )
        ],
    )
    val_result = validator.validate_report(
        report=report,
        sources=[],
        documents=[],
        evidence_records=[],
        verification_report=EvidenceVerificationReport(),
    )

    assert val_result.is_valid is False
    assert len(val_result.invalid_citations) > 0
    assert "https://fabricated-hallucination-domain.org/fake" in val_result.missing_sources
