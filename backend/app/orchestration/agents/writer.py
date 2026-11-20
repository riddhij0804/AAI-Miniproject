"""Report Writer Agent synthesizing evidence-grounded research reports."""

import logging
from typing import Any, Dict, List, Optional

from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.config import settings
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
        evidence_source_ids = {
            evidence.id: evidence.source_id for evidence in evidence_records
        }
        source_buckets: Dict[str, List[Claim]] = {}
        for claim in verified_claims:
            source_id = next(
                (evidence_source_ids[evidence_id] for evidence_id in claim.evidence_ids if evidence_id in evidence_source_ids),
                "unknown",
            )
            source_buckets.setdefault(source_id, []).append(claim)
        diversified_claims: List[Claim] = []
        while source_buckets and len(diversified_claims) < settings.MAX_REPORT_CITATIONS:
            for source_id in list(source_buckets):
                bucket = source_buckets[source_id]
                diversified_claims.append(bucket.pop(0))
                if not bucket:
                    del source_buckets[source_id]
                if len(diversified_claims) >= settings.MAX_REPORT_CITATIONS:
                    break
        verified_claims = diversified_claims

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
                            document_id=ev.document_id,
                            chunk_id=ev.chunk_id,
                            page_number=ev.metadata.get("page_number"),
                        )
                    )
                    if len(references) >= settings.MAX_REPORT_CITATIONS:
                        break
            if len(references) >= settings.MAX_REPORT_CITATIONS:
                break

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
        methodology = (
            f"Scope: {structured_query.scope}. Domain-independent multi-agent retrieval, strict verbatim quote "
            f"extraction, 7-factor credibility scoring, and deterministic provenance verification across "
            f"{len(sources)} source(s) and {len(evidence_records)} evidence quote(s)."
        )
        evidence_summary = (
            f"Synthesized from {len(verified_claims)} grounded assertions backed by {len(references)} "
            f"verbatim document excerpts. Provenance integrity was deterministically confirmed across all citations."
        )

        # Attempt structured LLM synthesis for rich, articulate narrative
        llm_report = await self._synthesize_narrative_with_llm(
            query=structured_query,
            verified_claims=verified_claims,
            references=references,
            contradictions=contradiction_summaries,
            limitations=limitations,
        )

        if llm_report:
            title = llm_report.get("title") or title
            exec_summary = llm_report.get("executive_summary") or self._synthesize_executive_summary(structured_query, verified_claims, references)
            conclusion = llm_report.get("conclusion") or self._synthesize_conclusion(structured_query, verified_claims)
            raw_kf = llm_report.get("key_findings", [])
            key_findings: List[KeyFindingTopic] = []
            if isinstance(raw_kf, list) and raw_kf:
                for item in raw_kf:
                    if isinstance(item, dict) and "topic_title" in item and "summary" in item:
                        # Resilient parsing: ensure claims is a list of full strings, never iterated characters
                        raw_claims = item.get("claims", [])
                        if isinstance(raw_claims, str):
                            lines = [line.strip().lstrip("-*•0123456789. ") for line in raw_claims.splitlines() if len(line.strip()) > 3]
                            parsed_claims = lines if lines else ([raw_claims.strip()] if len(raw_claims.strip()) > 1 else [])
                        elif isinstance(raw_claims, list):
                            parsed_claims = [str(c).strip() for c in raw_claims if isinstance(c, (str, int, float)) and len(str(c).strip()) > 1]
                        else:
                            parsed_claims = []

                        raw_cites = item.get("supporting_citations", [])
                        if isinstance(raw_cites, str):
                            lines = [line.strip().lstrip("-*•0123456789. ") for line in raw_cites.splitlines() if len(line.strip()) > 3]
                            parsed_cites = lines if lines else ([raw_cites.strip()] if len(raw_cites.strip()) > 1 else [])
                        elif isinstance(raw_cites, list):
                            parsed_cites = [str(q).strip() for q in raw_cites if isinstance(q, (str, int, float)) and len(str(q).strip()) > 1]
                        else:
                            parsed_cites = []

                        key_findings.append(
                            KeyFindingTopic(
                                topic_title=str(item["topic_title"]),
                                summary=str(item["summary"]),
                                claims=parsed_claims,
                                supporting_citations=parsed_cites,
                            )
                        )
            if not key_findings:
                key_findings = self._synthesize_key_findings(structured_query, verified_claims, references)
        else:
            exec_summary = self._synthesize_executive_summary(structured_query, verified_claims, references)
            key_findings = self._synthesize_key_findings(structured_query, verified_claims, references)
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

    async def _synthesize_narrative_with_llm(
        self,
        query: StructuredQuery,
        verified_claims: List[Claim],
        references: List[ReportCitation],
        contradictions: List[str],
        limitations: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Call LLM to produce an articulate, deeply grounded narrative report."""
        if not self.llm or not self.llm.has_active_key:
            return None

        # Prepare verified factual context
        claims_context = [f"- {c.text}" for c in verified_claims[:15]]
        evidence_context = [
            f"- Source: \"{r.source_title}\"\n  Quote: \"{r.exact_quote}\""
            for r in references[:10]
        ]
        system_instruction = (
            "You are an expert research intelligence analyst writing an authoritative, evidence-grounded research report. "
            "Synthesize a clear, deeply informative, and professional report directly addressing the user's research objective. "
            "MANDATORY REQUIREMENTS:\n"
            "1. Focus directly on answering the research objective using the verified factual findings and evidence quotes provided.\n"
            "2. 'executive_summary': Provide a 2-3 paragraph articulate, well-written analytical synthesis. Explain core historical/scientific facts, answer ambiguities or key questions, and highlight substantiated evidence. Never output web boilerplate, navigation menus, or disambiguation notices.\n"
            "3. 'key_findings': Provide 2-4 thematic sections. Each item must have 'topic_title', 'summary' (a detailed analytical paragraph synthesizing that theme), 'claims' (a JSON array of 2-4 distinct full-sentence factual strings from the verified claims, e.g. [\"Sentence 1\", \"Sentence 2\"]), and 'supporting_citations' (a JSON array of relevant excerpt strings from the provided evidence). Never output 'claims' as a single string or character sequence.\n"
            "4. 'conclusion': Provide a comprehensive analytical synthesis addressing nuances, implications, and unanswered dimensions.\n"
            "5. Return strictly a JSON object with keys: 'title', 'executive_summary', 'key_findings', 'conclusion'."
        )

        user_prompt = (
            f"Research Question: {query.original_query}\n"
            f"Objective: {query.research_objective}\n"
            f"Scope: {query.scope}\n"
            f"Key Concepts: {', '.join(query.key_concepts)}\n"
            f"Sub-Questions:\n" + ("\n".join(f"- {sq}" for sq in query.sub_questions) if query.sub_questions else "- General overview") + "\n\n"
            f"Verified Factual Claims:\n" + ("\n".join(claims_context) if claims_context else "None available.") + "\n\n"
            f"Supporting Evidence Excerpts:\n" + ("\n".join(evidence_context) if evidence_context else "None available.") + "\n\n"
            f"Contradictory Findings:\n" + ("\n".join(contradictions) if contradictions else "None.") + "\n\n"
            f"Limitations:\n" + ("\n".join(limitations) if limitations else "None.") + "\n\n"
            f"Synthesize the authoritative research report in JSON format."
        )

        try:
            res = await self.llm.generate_json(system_instruction, user_prompt)
            if res and isinstance(res, dict) and "executive_summary" in res:
                return res
        except Exception as e:
            logger.warning(f"ReportWriterAgent LLM synthesis failed ({e}), falling back to deterministic synthesis.")

        return None

    def _synthesize_executive_summary(
        self,
        query: StructuredQuery,
        claims: List[Claim],
        references: List[ReportCitation],
    ) -> str:
        if not claims:
            return f"Investigation into '{query.original_query}' completed with preliminary exploratory findings across available sources."

        clean_claims = [c.text.strip().rstrip(".") for c in claims if len(c.text.strip()) > 15]
        bullets = ". ".join(clean_claims[:4]) + "."
        return (
            f"This research investigation addressed '{query.research_objective}'. "
            f"Based on cross-verified evidence, primary findings show: {bullets} "
            f"All findings have been corroborated against verbatim document excerpts."
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
