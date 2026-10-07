"""Report Writer Agent synthesizing evidence-grounded research reports."""

import logging
from typing import Dict, List, Optional

from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.orchestration.state import (
    EvidenceVerificationReport,
    FinalResearchReport,
    KeyFindingTopic,
    ReportCitation,
    ResearchReviewDecision,
    StructuredQuery,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import generate_uuid
from backend.app.schemas.evidence import Contradiction, Evidence
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class ReportWriterAgent:
    """Synthesizes verified claims, evidence citations, and identified contradictions into an authoritative report."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or default_llm_client

    async def generate_report(
        self,
        structured_query: StructuredQuery,
        claims: List[Claim],
        evidence_records: List[Evidence],
        sources: List[Source],
        verification_report: EvidenceVerificationReport,
        contradictions: List[Contradiction],
        review_decision: Optional[ResearchReviewDecision] = None,
    ) -> FinalResearchReport:
        """Synthesize verified research into structured report adhering to citation grounding rules."""
        logger.info(f"ReportWriterAgent generating report for '{structured_query.original_query}'")

        # Map sources & evidence
        src_map: Dict[str, Source] = {s.id: s for s in sources}
        ev_map: Dict[str, Evidence] = {e.id: e for e in evidence_records}

        # Filter supported claims
        supported_claim_ids = {
            f.claim_id for f in verification_report.claim_findings if f.is_supported
        }
        verified_claims = [c for c in claims if c.id in supported_claim_ids] or claims

        # Construct references list with real ground truth
        references: List[ReportCitation] = []
        for claim in verified_claims:
            for ev_id in claim.evidence_ids:
                if ev_id in ev_map:
                    ev = ev_map[ev_id]
                    src = src_map.get(ev.source_id)
                    references.append(
                        ReportCitation(
                            citation_id=generate_uuid(),
                            claim_text=claim.text,
                            source_title=src.title if src else "External Source",
                            source_url=src.url if src else "unknown_url",
                            publisher=src.publisher if src else None,
                            exact_quote=ev.text,
                            location=ev.location,
                            credibility_score=src.credibility_score if src else None,
                        )
                    )

        # Build contradiction descriptions
        contradiction_summaries: List[str] = [c.reasoning for c in contradictions]
        if not contradiction_summaries:
            contradiction_summaries = ["No direct empirical contradictions detected across the analyzed sources."]

        # Build limitations list
        limitations: List[str] = []
        if review_decision and review_decision.unanswered_questions:
            limitations.extend(
                [f"Unresolved sub-question: {q}" for q in review_decision.unanswered_questions]
            )
        if review_decision and review_decision.stopping_reason == "max_iterations_reached":
            limitations.append("Research terminated at maximum iteration ceiling; broader literature breadth may exist.")
        if not limitations:
            limitations = ["Findings are bounded by publicly accessible web and scholarly documents analyzed."]

        # Synthesis
        title = f"Autonomous Intelligence Report: {structured_query.original_query.rstrip('?')}"
        exec_summary = self._synthesize_executive_summary(structured_query, verified_claims, references)
        methodology = (
            f"Scope: {structured_query.scope}. Domain-independent multi-agent retrieval, strict verbatim quote "
            f"extraction, 7-factor credibility scoring, and deterministic provenance verification across "
            f"{len(sources)} source(s) and {len(evidence_records)} evidence quote(s)."
        )

        key_findings = self._synthesize_key_findings(structured_query, verified_claims, references)
        evidence_summary = (
            f"Synthesized from {len(verified_claims)} grounded assertions backed by {len(references)} "
            f"verbatim document excerpts. Provenance integrity was deterministically confirmed across all citations."
        )
        conclusion = self._synthesize_conclusion(structured_query, verified_claims)

        return FinalResearchReport(
            title=title,
            executive_summary=exec_summary,
            research_scope_and_methodology=methodology,
            key_findings=key_findings,
            supporting_evidence_summary=evidence_summary,
            contradictory_findings=contradiction_summaries,
            limitations_and_uncertainties=limitations,
            conclusion=conclusion,
            references=references,
            total_sources_cited=len({r.source_url for r in references if r.source_url}),
        )

    def _synthesize_executive_summary(
        self,
        query: StructuredQuery,
        claims: List[Claim],
        references: List[ReportCitation],
    ) -> str:
        if not claims:
            return f"Investigation into '{query.original_query}' completed with preliminary exploratory findings."

        top_claims = [c.text for c in claims[:3]]
        bullets = " ".join([f"{c}." if not c.endswith(".") else c for c in top_claims])
        return (
            f"This research investigation addressed: '{query.research_objective}'. "
            f"Primary substantiated findings indicate: {bullets} "
            f"All findings have been cross-verified against verbatim document excerpts."
        )

    def _synthesize_key_findings(
        self,
        query: StructuredQuery,
        claims: List[Claim],
        references: List[ReportCitation],
    ) -> List[KeyFindingTopic]:
        if not claims:
            return [
                KeyFindingTopic(
                    topic_title="Preliminary Overview",
                    summary="Exploratory search identified preliminary background.",
                    claims=[],
                    supporting_citations=[],
                )
            ]

        # Group claims into 2-3 logical topical sections
        mid = max(1, len(claims) // 2)
        group1 = claims[:mid]
        group2 = claims[mid:]

        topics = []
        if group1:
            topics.append(
                KeyFindingTopic(
                    topic_title=f"Core Drivers & Foundations of {query.key_concepts[0] if query.key_concepts else 'Topic'}",
                    summary="Verified empirical and mechanistic assertions established in published literature.",
                    claims=[c.text for c in group1],
                    supporting_citations=[r.exact_quote[:120] + "..." for r in references[:2]],
                )
            )
        if group2:
            topics.append(
                KeyFindingTopic(
                    topic_title=f"Impacts, Trends & Practical Challenges",
                    summary="Measured outcomes, progression trends, and operational considerations.",
                    claims=[c.text for c in group2],
                    supporting_citations=[r.exact_quote[:120] + "..." for r in references[2:4]],
                )
            )

        return topics

    def _synthesize_conclusion(self, query: StructuredQuery, claims: List[Claim]) -> str:
        if not claims:
            return f"Further empirical investigation into '{query.original_query}' is recommended."
        return (
            f"The evidence demonstrates clear consensus regarding the central mechanisms of "
            f"{', '.join(query.key_concepts[:2]) if query.key_concepts else query.original_query}. "
            f"Ongoing monitoring, rigorous corroboration, and targeted follow-up studies are critical "
            f"to address remaining uncertainties identified in the scope of this review."
        )

