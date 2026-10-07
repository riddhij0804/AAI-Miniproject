"""LangGraph stateful workflow coordinator for the Autonomous Research Agent."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from backend.app.evidence.store import EvidenceStore, default_evidence_store
from backend.app.orchestration.agents.planner import ResearchPlannerAgent
from backend.app.orchestration.agents.query_understanding import QueryUnderstandingAgent
from backend.app.orchestration.agents.reviewer import ResearchReviewerAgent
from backend.app.orchestration.agents.validator import CitationValidationAgent
from backend.app.orchestration.agents.verifier import EvidenceVerificationAgent
from backend.app.orchestration.agents.writer import ReportWriterAgent
from backend.app.orchestration.state import (
    PlanTask,
    ResearchPlan,
    ResearchWorkflowState,
)
from backend.app.schemas.source import SourceDiscoveryRequest
from backend.app.services.research_service import (
    ResearchIntelligenceService,
    default_research_service,
)

logger = logging.getLogger(__name__)


class ResearchWorkflowOrchestrator:
    """Coordinates the full multi-agent research lifecycle via a stateful LangGraph workflow."""

    def __init__(
        self,
        research_service: Optional[ResearchIntelligenceService] = None,
        query_agent: Optional[QueryUnderstandingAgent] = None,
        planner_agent: Optional[ResearchPlannerAgent] = None,
        verifier_agent: Optional[EvidenceVerificationAgent] = None,
        reviewer_agent: Optional[ResearchReviewerAgent] = None,
        writer_agent: Optional[ReportWriterAgent] = None,
        validator_agent: Optional[CitationValidationAgent] = None,
        store: Optional[EvidenceStore] = None,
        checkpointer: Optional[Any] = None,
    ):
        self.research_service = research_service or default_research_service
        self.query_agent = query_agent or QueryUnderstandingAgent()
        self.planner_agent = planner_agent or ResearchPlannerAgent()
        self.verifier_agent = verifier_agent or EvidenceVerificationAgent(store=store)
        self.reviewer_agent = reviewer_agent or ResearchReviewerAgent()
        self.writer_agent = writer_agent or ReportWriterAgent()
        self.validator_agent = validator_agent or CitationValidationAgent(store=store)
        self.store = store or default_evidence_store
        self.checkpointer = checkpointer or MemorySaver()

        # Build and compile graph
        self.graph = self._build_graph()

    def _build_graph(self):
        """Construct the stateful LangGraph workflow with nodes and conditional branching."""
        builder = StateGraph(ResearchWorkflowState)

        # 1. Add Nodes
        builder.add_node("query_understanding", self._query_understanding_node)
        builder.add_node("planning", self._planning_node)
        builder.add_node("research_execution", self._research_execution_node)
        builder.add_node("evidence_verification", self._evidence_verification_node)
        builder.add_node("research_review", self._research_review_node)
        builder.add_node("replanning", self._replanning_node)
        builder.add_node("report_writing", self._report_writing_node)
        builder.add_node("citation_validation", self._citation_validation_node)

        # 2. Add Fixed Edges
        builder.add_edge(START, "query_understanding")
        builder.add_edge("query_understanding", "planning")
        builder.add_edge("planning", "research_execution")
        builder.add_edge("research_execution", "evidence_verification")
        builder.add_edge("evidence_verification", "research_review")

        # 3. Add Conditional Edge for Reviewer Decision (Replan vs Generate Report)
        builder.add_conditional_edges(
            "research_review",
            self._reviewer_decision_edge,
            {
                "replanning": "replanning",
                "report_writing": "report_writing",
            },
        )

        # Replanning loops back to research execution
        builder.add_edge("replanning", "research_execution")

        # Report writing proceeds to citation validation, then completes
        builder.add_edge("report_writing", "citation_validation")
        builder.add_edge("citation_validation", END)

        return builder.compile(checkpointer=self.checkpointer)

    # --- Node Implementations ---

    async def _query_understanding_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 1: Analyze user question, extract scope, constraints, and handle ambiguity."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Query Understanding")

        structured_q = await self.query_agent.analyze_query(state.original_question)
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "query_understanding",
            "duration_ms": round(duration_ms, 2),
            "concepts_extracted": structured_q.key_concepts,
            "is_ambiguous": structured_q.is_ambiguous,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "structured_query": structured_q,
            "current_stage": "planning",
            "audit_logs": [audit_entry],
        }

    async def _planning_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 2: Decompose objective into actionable parallel/dependent tasks."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Research Planning")

        # Create session in persistent store if not exists
        try:
            self.store.create_session(
                title=state.structured_query.original_query[:60],
                original_question=state.original_question,
                metadata={"session_id": state.session_id},
            )
        except Exception as e:
            logger.debug(f"Session already initialized or DB store note: {e}")

        plan = await self.planner_agent.create_plan(
            state.structured_query,
            max_iterations=state.max_iterations,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "planning",
            "duration_ms": round(duration_ms, 2),
            "task_count": len(plan.tasks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "plan": plan,
            "current_stage": "research_execution",
            "audit_logs": [audit_entry],
        }

    async def _research_execution_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 3: Execute tasks using existing research service (discovery, reading, extraction)."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Research Execution (Iteration {state.iteration})")

        # Identify pending tasks
        pending_tasks = [t for t in state.plan.tasks if t.status == "pending"]
        logger.info(f"Executing {len(pending_tasks)} pending research task(s).")

        new_sources = []
        new_docs = []
        new_claims = []
        new_evidences = []
        new_cred = []
        errors = []

        # Execute tasks concurrently or sequentially depending on task parallel flag
        for task in pending_tasks:
            task.status = "in_progress"
            try:
                # 1. Discover Sources
                disc_req = SourceDiscoveryRequest(
                    task_id=task.task_id,
                    query=task.query,
                    research_objective=task.objective,
                    source_preferences=task.source_preferences,
                    max_results=4,
                )
                sources = await self.research_service.discover_sources(disc_req)
                new_sources.extend(sources)

                if sources:
                    # 2. Assess Credibility
                    cred_assessments = self.research_service.assess_source_credibility(sources)
                    new_cred.extend(cred_assessments)

                    # 3. Read Documents
                    docs = await self.research_service.read_documents(sources)
                    new_docs.extend(docs)

                    # 4. Extract Grounded Evidence
                    ext_res = await self.research_service.extract_evidence(
                        documents=docs,
                        objective=task.objective,
                        session_id=state.session_id,
                        task_id=task.task_id,
                    )
                    new_claims.extend(ext_res.claims)
                    new_evidences.extend(ext_res.evidence_records)

                task.status = "completed"
                task.result_summary = f"Gathered {len(sources)} source(s), {len(new_claims)} claim(s)."

            except Exception as e:
                logger.error(f"Error executing task '{task.task_id}': {e}")
                task.status = "failed"
                errors.append(f"Task '{task.title}' failed: {str(e)}")

        duration_ms = (time.perf_counter() - start_t) * 1000
        audit_entry = {
            "stage": "research_execution",
            "iteration": state.iteration,
            "duration_ms": round(duration_ms, 2),
            "sources_discovered": len(new_sources),
            "claims_extracted": len(new_claims),
            "evidence_extracted": len(new_evidences),
            "errors": errors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "plan": state.plan,
            "discovered_sources": new_sources,
            "documents": new_docs,
            "claims": new_claims,
            "evidence_records": new_evidences,
            "credibility_assessments": new_cred,
            "errors": errors,
            "current_stage": "evidence_verification",
            "audit_logs": [audit_entry],
        }

    async def _evidence_verification_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 4: Verify provenance, claim support, and detect contradictions."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Evidence Verification")

        report = await self.verifier_agent.verify_evidence(
            claims=state.claims,
            evidence_records=state.evidence_records,
            sources=state.discovered_sources,
            documents=state.documents,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "evidence_verification",
            "duration_ms": round(duration_ms, 2),
            "supported_claims": report.supported_claims_count,
            "unsupported_claims": report.unsupported_claims_count,
            "contradictions_found": len(report.detected_contradictions),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "verification_report": report,
            "contradictions": report.detected_contradictions,
            "current_stage": "research_review",
            "audit_logs": [audit_entry],
        }

    async def _research_review_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 5: Review research completeness against initial query and decide next route."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Research Review")

        decision = await self.reviewer_agent.review_research_progress(
            structured_query=state.structured_query,
            current_plan=state.plan,
            verification_report=state.verification_report,
            claims=state.claims,
            evidence_records=state.evidence_records,
            sources=state.discovered_sources,
            current_iteration=state.iteration,
            max_iterations=state.max_iterations,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "research_review",
            "iteration": state.iteration,
            "duration_ms": round(duration_ms, 2),
            "is_sufficient": decision.is_sufficient,
            "should_replan": decision.should_replan,
            "stopping_reason": decision.stopping_reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "review_decision": decision,
            "audit_logs": [audit_entry],
        }

    def _reviewer_decision_edge(self, state: ResearchWorkflowState) -> str:
        """Conditional routing: replan or proceed to report generation."""
        if state.review_decision and state.review_decision.should_replan and state.iteration < state.max_iterations:
            logger.info(f"Reviewer decision: Replan for iteration {state.iteration + 1}")
            return "replanning"
        else:
            logger.info("Reviewer decision: Proceed to report generation")
            return "report_writing"

    async def _replanning_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 6: Targeted replanning to address information gaps."""
        start_t = time.perf_counter()
        new_iter = state.iteration + 1
        logger.info(f"Workflow [{state.session_id}]: Replanning for iteration {new_iter}")

        replanned = self.planner_agent.replan_with_gaps(
            current_plan=state.plan,
            unanswered_questions=state.review_decision.unanswered_questions,
            missing_areas=state.review_decision.missing_evidence_areas,
            new_iteration=new_iter,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "replanning",
            "iteration": new_iter,
            "duration_ms": round(duration_ms, 2),
            "new_tasks_added": len(replanned.tasks) - len(state.plan.tasks),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "plan": replanned,
            "iteration": new_iter,
            "current_stage": "research_execution",
            "audit_logs": [audit_entry],
        }

    async def _report_writing_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 7: Synthesize grounded research report."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Report Writing")

        report = await self.writer_agent.generate_report(
            structured_query=state.structured_query,
            claims=state.claims,
            evidence_records=state.evidence_records,
            sources=state.discovered_sources,
            verification_report=state.verification_report,
            contradictions=state.contradictions,
            review_decision=state.review_decision,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "report_writing",
            "duration_ms": round(duration_ms, 2),
            "sections_count": len(report.key_findings),
            "references_count": len(report.references),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "final_report": report,
            "current_stage": "citation_validation",
            "audit_logs": [audit_entry],
        }

    async def _citation_validation_node(self, state: ResearchWorkflowState) -> Dict[str, Any]:
        """Node 8: Final validation auditing every citation against stored ground truth."""
        start_t = time.perf_counter()
        logger.info(f"Workflow [{state.session_id}]: Entering Citation Validation")

        val_result = self.validator_agent.validate_report(
            report=state.final_report,
            sources=state.discovered_sources,
            documents=state.documents,
            evidence_records=state.evidence_records,
            verification_report=state.verification_report,
        )
        duration_ms = (time.perf_counter() - start_t) * 1000

        audit_entry = {
            "stage": "citation_validation",
            "duration_ms": round(duration_ms, 2),
            "is_valid": val_result.is_valid,
            "citations_checked": val_result.total_citations_checked,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "citation_validation": val_result,
            "is_completed": True,
            "current_stage": "completed",
            "audit_logs": [audit_entry],
        }

    # --- Execution Helpers ---

    async def run(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_iterations: int = 3,
    ) -> ResearchWorkflowState:
        """Execute end-to-end research workflow from user question to final validated report."""
        import uuid
        sess_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        initial_state = ResearchWorkflowState(
            session_id=sess_id,
            original_question=question,
            max_iterations=max_iterations,
        )

        config = {"configurable": {"thread_id": sess_id}}
        final_output_dict = await self.graph.ainvoke(initial_state, config=config)

        # Reconstruct typed state model from LangGraph output dictionary
        return ResearchWorkflowState.model_validate(final_output_dict)


default_orchestrator = ResearchWorkflowOrchestrator()

