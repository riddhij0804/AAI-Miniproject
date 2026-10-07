"""Research Reviewer Agent evaluating completeness, missing evidence, and replanning triggers."""

import logging
from typing import List, Optional

from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.orchestration.state import (
    EvidenceVerificationReport,
    PlanTask,
    ResearchPlan,
    ResearchReviewDecision,
    StructuredQuery,
)
from backend.app.schemas.claim import Claim
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source

logger = logging.getLogger(__name__)


class ResearchReviewerAgent:
    """Evaluates research completeness against initial objectives, determines if more research is required, and bounds loops."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or default_llm_client

    async def review_research_progress(
        self,
        structured_query: StructuredQuery,
        current_plan: ResearchPlan,
        verification_report: EvidenceVerificationReport,
        claims: List[Claim],
        evidence_records: List[Evidence],
        sources: List[Source],
        current_iteration: int,
        max_iterations: int = 3,
    ) -> ResearchReviewDecision:
        """Evaluate evidence sufficiency and decide whether to stop or trigger targeted replanning."""
        logger.info(
            f"ResearchReviewerAgent reviewing iteration {current_iteration}/{max_iterations}. "
            f"Claims: {len(claims)}, Verified Evidence: {len(evidence_records)}"
        )

        # 1. Hard stopping condition: Maximum iterations reached
        if current_iteration >= max_iterations:
            logger.info("Stopping condition: Maximum iterations reached. Forcing report generation.")
            unanswered = self._identify_unanswered_questions(structured_query, claims)
            supported_claims = [f for f in verification_report.claim_findings if f.is_supported]
            total_sub_q = max(len(structured_query.sub_questions), 1)
            sub_q_ratio = max(0.0, (total_sub_q - len(unanswered)) / total_sub_q)
            claim_ratio = len(supported_claims) / max(len(claims), 1) if claims else 0.0
            coverage_score = round(min(1.0, max(0.0, 0.5 * sub_q_ratio + 0.5 * claim_ratio)), 2)
            return ResearchReviewDecision(
                is_sufficient=False,
                evidence_coverage_score=coverage_score,
                unanswered_questions=unanswered,
                missing_evidence_areas=["Additional depth limited by iteration ceiling"],
                followup_tasks=[],
                should_replan=False,
                stopping_reason="max_iterations_reached",
                reviewer_rationale=(
                    f"Iteration ceiling ({max_iterations}) reached. Proceeding to report generation with "
                    f"disclosed qualifications and {len(unanswered)} noted limitation(s)."
                ),
            )

        # 2. Check evidence sufficiency
        # We need adequate claims and verified evidence
        supported_claims = [f for f in verification_report.claim_findings if f.is_supported]
        unanswered = self._identify_unanswered_questions(structured_query, claims)

        has_sufficient_evidence = (
            len(supported_claims) >= 3
            and len(sources) >= 1
            and len(unanswered) == 0
        )

        if has_sufficient_evidence:
            logger.info("Stopping condition: Research objectives sufficiently satisfied.")
            return ResearchReviewDecision(
                is_sufficient=True,
                evidence_coverage_score=0.92,
                unanswered_questions=[],
                missing_evidence_areas=[],
                followup_tasks=[],
                should_replan=False,
                stopping_reason="objective_achieved",
                reviewer_rationale=(
                    f"Research objectives met with {len(supported_claims)} verified claims "
                    f"across {len(sources)} sources and {verification_report.verified_evidence_count} evidence excerpts."
                ),
            )

        # 3. Not sufficient and iteration < max_iterations: Generate targeted follow-up tasks
        missing_areas = [f"Coverage of '{q}'" for q in unanswered] if unanswered else ["Expanded quantitative corroboration"]
        followup_tasks: List[PlanTask] = []
        for idx, gap in enumerate(unanswered[:2]):
            followup_tasks.append(
                PlanTask(
                    title=f"Follow-up Investigation: {gap[:50]}",
                    query=gap,
                    objective=f"Investigate missing research aspect: {gap}",
                    dependencies=[],
                    can_run_in_parallel=True,
                    evidence_needed=[gap],
                    source_preferences=structured_query.source_preferences,
                    iteration=current_iteration + 1,
                )
            )

        total_sub_q = max(len(structured_query.sub_questions), 1)
        sub_q_ratio = max(0.0, (total_sub_q - len(unanswered)) / total_sub_q)
        claim_ratio = len(supported_claims) / max(len(claims), 1) if claims else 0.0
        coverage_score = round(min(1.0, max(0.0, 0.5 * sub_q_ratio + 0.5 * claim_ratio)), 2)

        logger.info(f"Reviewer triggered replanning: {len(unanswered)} gaps identified. Coverage score: {coverage_score}")
        return ResearchReviewDecision(
            is_sufficient=False,
            evidence_coverage_score=coverage_score,
            unanswered_questions=unanswered,
            missing_evidence_areas=missing_areas,
            followup_tasks=followup_tasks,
            should_replan=True,
            stopping_reason=None,
            reviewer_rationale=(
                f"Identified {len(unanswered)} open research sub-questions and {verification_report.unsupported_claims_count} "
                f"unsupported claims. Initiating targeted iteration {current_iteration + 1}."
            ),
        )

    def _identify_unanswered_questions(
        self,
        structured_query: StructuredQuery,
        claims: List[Claim],
    ) -> List[str]:
        """Check which sub-questions from query understanding lack corresponding claims."""
        combined_claim_text = " ".join(c.text.lower() for c in claims)
        unanswered: List[str] = []

        for sub_q in structured_query.sub_questions:
            # Check concept coverage
            words = [w.lower() for w in sub_q.split() if len(w) > 3 and w.lower() not in {"what", "which", "does", "have"}]
            matched = sum(1 for w in words if w in combined_claim_text)
            if matched < max(1, len(words) // 2):
                unanswered.append(sub_q)

        return unanswered

