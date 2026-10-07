"""Typed workflow state and state reducers for LangGraph orchestration."""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.claim import Claim
from backend.app.schemas.common import BaseSchema, generate_uuid, utc_now
from backend.app.schemas.credibility import CredibilityAssessment
from backend.app.schemas.document import Document
from backend.app.schemas.evidence import Citation, Contradiction, Evidence
from backend.app.schemas.source import Source


# --- Structured Query Understanding Models ---

class StructuredQuery(BaseSchema):
    """Structured, validated representation of the user's research request."""
    original_query: str
    research_objective: str
    scope: str
    key_concepts: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    expected_output_format: str = "comprehensive_report"
    source_preferences: List[str] = Field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reasons: List[str] = Field(default_factory=list)
    clarification_strategy: Optional[str] = None
    sub_questions: List[str] = Field(default_factory=list)


# --- Research Planner Models ---

class PlanTask(BaseSchema):
    """Individual research task generated during planning."""
    task_id: str = Field(default_factory=generate_uuid)
    title: str
    query: str
    objective: str
    dependencies: List[str] = Field(default_factory=list)
    can_run_in_parallel: bool = True
    evidence_needed: List[str] = Field(default_factory=list)
    source_preferences: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result_summary: Optional[str] = None
    iteration: int = 1


class ResearchPlan(BaseSchema):
    """Coordinated research plan with dependencies and stopping limits."""
    plan_id: str = Field(default_factory=generate_uuid)
    objective: str
    tasks: List[PlanTask] = Field(default_factory=list)
    parallel_batches: List[List[str]] = Field(default_factory=list)
    max_iterations: int = 3
    current_iteration: int = 1
    replanned_count: int = 0
    plan_rationale: Optional[str] = None


# --- Evidence Verification Models ---

class ClaimVerificationFinding(BaseSchema):
    """Detailed verification assessment for an individual claim."""
    claim_id: str
    claim_text: str
    is_supported: bool
    support_strength: float = Field(ge=0.0, le=1.0)  # 0.0 unsupported, 1.0 strong
    has_valid_provenance: bool
    evidence_ids: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)
    flags: List[str] = Field(default_factory=list)  # "weakly_supported", "unsupported", "outdated", "no_provenance"
    reasoning: str


class EvidenceVerificationReport(BaseSchema):
    """Aggregated verification results across all research findings."""
    total_claims_checked: int = 0
    supported_claims_count: int = 0
    unsupported_claims_count: int = 0
    weakly_supported_claims_count: int = 0
    claim_findings: List[ClaimVerificationFinding] = Field(default_factory=list)
    detected_contradictions: List[Contradiction] = Field(default_factory=list)
    verified_evidence_count: int = 0
    provenance_integrity_passed: bool = True
    verification_notes: Optional[str] = None


# --- Research Reviewer Models ---

class ResearchReviewDecision(BaseSchema):
    """Assessment of whether research objective is met or replanning is required."""
    is_sufficient: bool = False
    evidence_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unanswered_questions: List[str] = Field(default_factory=list)
    missing_evidence_areas: List[str] = Field(default_factory=list)
    followup_tasks: List[PlanTask] = Field(default_factory=list)
    should_replan: bool = False
    stopping_reason: Optional[str] = None  # "objective_achieved", "max_iterations_reached", "no_new_info", "unresolvable_gap"
    reviewer_rationale: str = ""


# --- Final Report Models ---

class KeyFindingTopic(BaseSchema):
    """Thematic section of findings in final report."""
    topic_title: str
    summary: str
    claims: List[str] = Field(default_factory=list)
    supporting_citations: List[str] = Field(default_factory=list)


class ReportCitation(BaseSchema):
    """Grounding reference cited in final report."""
    citation_id: str
    claim_text: str
    source_title: str
    source_url: str
    publisher: Optional[str] = None
    exact_quote: str
    location: str
    credibility_score: Optional[float] = None


class FinalResearchReport(BaseSchema):
    """Structured, evidence-grounded research report."""
    report_id: str = Field(default_factory=generate_uuid)
    title: str
    executive_summary: str
    research_scope_and_methodology: str
    key_findings: List[KeyFindingTopic] = Field(default_factory=list)
    supporting_evidence_summary: str
    contradictory_findings: List[str] = Field(default_factory=list)
    limitations_and_uncertainties: List[str] = Field(default_factory=list)
    conclusion: str
    references: List[ReportCitation] = Field(default_factory=list)
    total_sources_cited: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class CitationValidationResult(BaseSchema):
    """Validation report checking every citation against stored ground truth."""
    is_valid: bool = True
    total_citations_checked: int = 0
    valid_citations_count: int = 0
    invalid_citations: List[str] = Field(default_factory=list)
    unsupported_claims_flagged: List[str] = Field(default_factory=list)
    missing_sources: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


# --- Reducers for State Merging ---

def merge_sources(existing: List[Source], new_items: List[Source]) -> List[Source]:
    """Reducer: Merge sources deduplicated by ID and URL."""
    merged: Dict[str, Source] = {s.id: s for s in existing or []}
    url_to_id = {s.url: s.id for s in merged.values() if s.url}
    for item in new_items or []:
        if item.id in merged:
            merged[item.id] = item
        elif item.url and item.url in url_to_id:
            existing_id = url_to_id[item.url]
            merged[existing_id] = item
        else:
            merged[item.id] = item
            if item.url:
                url_to_id[item.url] = item.id
    return list(merged.values())


def merge_documents(existing: List[Document], new_items: List[Document]) -> List[Document]:
    """Reducer: Merge documents deduplicated by ID."""
    merged: Dict[str, Document] = {d.id: d for d in existing or []}
    for item in new_items or []:
        merged[item.id] = item
    return list(merged.values())


def merge_claims(existing: List[Claim], new_items: List[Claim]) -> List[Claim]:
    """Reducer: Merge claims deduplicated by ID and normalized text."""
    merged: Dict[str, Claim] = {c.id: c for c in existing or []}
    text_to_id = {c.text.strip().lower(): c.id for c in merged.values()}
    for item in new_items or []:
        norm_text = item.text.strip().lower()
        if item.id in merged:
            merged[item.id] = item
        elif norm_text in text_to_id:
            existing_id = text_to_id[norm_text]
            # Merge evidence_ids
            existing_claim = merged[existing_id]
            combined_ev = list(set(existing_claim.evidence_ids + item.evidence_ids))
            existing_claim.evidence_ids = combined_ev
        else:
            merged[item.id] = item
            text_to_id[norm_text] = item.id
    return list(merged.values())


def merge_evidence(existing: List[Evidence], new_items: List[Evidence]) -> List[Evidence]:
    """Reducer: Merge evidence records deduplicated by ID and quote text."""
    merged: Dict[str, Evidence] = {e.id: e for e in existing or []}
    quote_to_id = {e.text.strip().lower(): e.id for e in merged.values()}
    for item in new_items or []:
        norm_quote = item.text.strip().lower()
        if item.id in merged:
            merged[item.id] = item
        elif norm_quote in quote_to_id:
            continue  # avoid duplicate exact quotes
        else:
            merged[item.id] = item
            quote_to_id[norm_quote] = item.id
    return list(merged.values())


def merge_credibility(existing: List[CredibilityAssessment], new_items: List[CredibilityAssessment]) -> List[CredibilityAssessment]:
    """Reducer: Merge credibility assessments by source ID."""
    merged: Dict[str, CredibilityAssessment] = {c.source_id: c for c in existing or []}
    for item in new_items or []:
        merged[item.source_id] = item
    return list(merged.values())


def merge_contradictions(existing: List[Contradiction], new_items: List[Contradiction]) -> List[Contradiction]:
    """Reducer: Merge contradictions deduplicated by claim pairs."""
    seen_pairs = set()
    result: List[Contradiction] = []
    for c in (existing or []) + (new_items or []):
        pair = tuple(sorted([c.claim_a_id, c.claim_b_id]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            result.append(c)
    return result


def merge_errors(existing: List[str], new_items: List[str]) -> List[str]:
    """Reducer: Merge unique error messages."""
    return list(dict.fromkeys((existing or []) + (new_items or [])))


def merge_audit_logs(existing: List[Dict[str, Any]], new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reducer: Append audit logs preserving temporal sequence."""
    return (existing or []) + (new_items or [])


# --- Main LangGraph Workflow State Schema ---

class ResearchWorkflowState(BaseModel):
    """Unified shared state for the multi-agent research workflow."""
    # Identification
    session_id: str = Field(default_factory=generate_uuid)
    original_question: str = ""

    # Agent outputs & workflow stages
    structured_query: Optional[StructuredQuery] = None
    plan: Optional[ResearchPlan] = None
    active_task_id: Optional[str] = None

    # Research Intelligence Evidence Entities (with merge reducers)
    discovered_sources: Annotated[List[Source], merge_sources] = Field(default_factory=list)
    documents: Annotated[List[Document], merge_documents] = Field(default_factory=list)
    claims: Annotated[List[Claim], merge_claims] = Field(default_factory=list)
    evidence_records: Annotated[List[Evidence], merge_evidence] = Field(default_factory=list)
    credibility_assessments: Annotated[List[CredibilityAssessment], merge_credibility] = Field(default_factory=list)

    # Verification & Review
    verification_report: Optional[EvidenceVerificationReport] = None
    contradictions: Annotated[List[Contradiction], merge_contradictions] = Field(default_factory=list)
    review_decision: Optional[ResearchReviewDecision] = None

    # Final Output & Validation
    final_report: Optional[FinalResearchReport] = None
    citation_validation: Optional[CitationValidationResult] = None

    # Flow Control & Metadata
    iteration: int = 1
    max_iterations: int = 3
    max_documents: int = 10
    is_completed: bool = False
    current_stage: str = "query_understanding"
    errors: Annotated[List[str], merge_errors] = Field(default_factory=list)
    audit_logs: Annotated[List[Dict[str, Any]], merge_audit_logs] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

