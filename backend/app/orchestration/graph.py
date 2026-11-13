"""LangGraph stateful workflow coordinator for the Autonomous Research Agent."""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from backend.app.config import settings

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
from backend.app.orchestration.llm import LLMUnavailableError, default_llm_client
from backend.app.audit import record_audit_event
from backend.app.tools.security import PromptInjectionGuard

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
        self._checkpointer_context = None
        self.checkpointer = checkpointer

        # Build and compile graph
        self.graph = self._build_graph() if self.checkpointer is not None else None

    def _llm_metrics(self) -> Dict[str, int]:
        """Aggregate counters from the workflow agents without exposing provider secrets."""
        clients = {
            getattr(agent, "llm", None)
            for agent in (
                self.query_agent,
                self.planner_agent,
                self.reviewer_agent,
                self.writer_agent,
                self.verifier_agent,
                self.validator_agent,
            )
        }
        metrics = {"calls": 0, "retries": 0, "provider_failures": 0}
        for client in clients:
            for key in metrics:
                metrics[key] += int(getattr(client, "metrics", {}).get(key, 0))
        return metrics

    def close(self) -> None:
        """Close the durable checkpoint connection owned by this orchestrator."""
        self._checkpointer_context = None

    def _build_graph(self, checkpointer: Optional[Any] = None):
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

        return builder.compile(checkpointer=checkpointer or self.checkpointer)

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
            "llm_metrics": self._llm_metrics(),
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
            "llm_metrics": self._llm_metrics(),
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
        remaining_documents = max(0, state.max_documents - len(state.documents))
        seen_source_urls = {source.url for source in state.discovered_sources if source.url}

        # Execute dependency-ready tasks in genuine concurrent batches. A task is
        # eligible only when every declared dependency has completed.
        completed_ids = {t.task_id for t in state.plan.tasks if t.status == "completed"}
        while remaining_documents > 0:
            ready = [
                task for task in pending_tasks
                if task.status == "pending"
                and all(dep in completed_ids for dep in task.dependencies)
            ]
            if not ready:
                unresolved = [task.task_id for task in pending_tasks if task.status == "pending"]
                if unresolved:
                    errors.append(f"Unresolvable task dependencies: {', '.join(unresolved)}")
                break
            batch = ready if all(task.can_run_in_parallel for task in ready) else ready[:1]
            budget_each = max(1, remaining_documents // len(batch))
            results = await asyncio.gather(
                *(self._execute_research_task(task, state.session_id, budget_each) for task in batch),
                return_exceptions=True,
            )
            for task, result in zip(batch, results):
                if isinstance(result, Exception):
                    task.status = "failed"
                    errors.append(f"Task '{task.title}' failed: {result}")
                    continue
                task_result = result
                task.status = "completed"
                completed_ids.add(task.task_id)
                task.result_summary = task_result["summary"]
                new_sources.extend(task_result["sources"])
                new_docs.extend(task_result["documents"])
                new_claims.extend(task_result["claims"])
                new_evidences.extend(task_result["evidence"])
                new_cred.extend(task_result["credibility"])
                remaining_documents -= task_result["selected_count"]
            if not any(task.status == "pending" for task in pending_tasks):
                break

        duration_ms = (time.perf_counter() - start_t) * 1000
        audit_entry = {
            "stage": "research_execution",
            "iteration": state.iteration,
            "duration_ms": round(duration_ms, 2),
            "sources_discovered": len(new_sources),
            "claims_extracted": len(new_claims),
            "evidence_extracted": len(new_evidences),
            "documents_processed": len(new_docs),
            "chunks_processed": sum(len(doc.chunks) for doc in new_docs),
            "timings": [result.get("timings", {}) for result in results if isinstance(result, dict)],
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

    async def _execute_research_task(
        self, task: PlanTask, session_id: str, document_budget: int
    ) -> Dict[str, Any]:
        """Run discovery, reading, and extraction for one independent task."""
        task.status = "in_progress"
        started = time.perf_counter()
        request = SourceDiscoveryRequest(
            task_id=task.task_id,
            query=task.query,
            research_objective=task.objective,
            source_preferences=task.source_preferences,
            max_results=min(4, document_budget),
        )
        discovery_started = time.perf_counter()
        sources = await self.research_service.discover_sources(request)
        discovery_ms = (time.perf_counter() - discovery_started) * 1000
        selected = [source for source in sources if source.url][:document_budget]
        credibility = self.research_service.assess_source_credibility(selected)
        read_started = time.perf_counter()
        documents = await self.research_service.read_documents(selected)
        read_ms = (time.perf_counter() - read_started) * 1000
        extraction_started = time.perf_counter()
        extraction = await self.research_service.extract_evidence(
            documents=documents,
            objective=task.objective,
            session_id=session_id,
            task_id=task.task_id,
        )
        extraction_ms = (time.perf_counter() - extraction_started) * 1000
        logger.info(
            "research_task_timing task_id=%s discovery_ms=%.2f read_ms=%.2f "
            "extraction_ms=%.2f total_ms=%.2f sources=%d documents=%d chunks=%d",
            task.task_id,
            discovery_ms,
            read_ms,
            extraction_ms,
            (time.perf_counter() - started) * 1000,
            len(sources),
            len(documents),
            sum(len(doc.chunks) for doc in documents),
        )
        return {
            "sources": sources,
            "documents": documents,
            "claims": extraction.claims,
            "evidence": extraction.evidence_records,
            "credibility": credibility,
            "selected_count": len(selected),
            "timings": {
                "discovery_ms": round(discovery_ms, 2),
                "read_ms": round(read_ms, 2),
                "extraction_ms": round(extraction_ms, 2),
            },
            "summary": (
                f"Gathered {len(sources)} source(s), read {len(selected)} document(s), "
                f"{len(extraction.claims)} claim(s)."
            ),
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
            "llm_metrics": self._llm_metrics(),
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
            "llm_metrics": self._llm_metrics(),
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
        has_grounded_evidence = bool(state.claims and state.evidence_records)
        if not has_grounded_evidence:
            val_result.is_valid = False
            val_result.notes = (
                "Insufficient grounded evidence: report completion was blocked because "
                "no verified claims and evidence records were produced."
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
            "is_completed": has_grounded_evidence,
            "current_stage": "completed" if has_grounded_evidence else "insufficient_evidence",
            "errors": [] if has_grounded_evidence else [
                "Research completed without grounded evidence; report is not considered successful."
            ],
            "audit_logs": [audit_entry],
        }

    # --- Execution Helpers ---

    async def run(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_iterations: int = 3,
        max_documents: int = 10,
    ) -> ResearchWorkflowState:
        """Execute end-to-end research workflow from user question to final validated report."""
        import uuid
        sess_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        security_decision = PromptInjectionGuard.assess_text(question, user_input=True)
        record_audit_event("security_decision", session_id=sess_id, decision=security_decision, query=question)
        if security_decision["outcome"] == "BLOCKED":
            return ResearchWorkflowState(
                session_id=sess_id,
                original_question=question,
                max_iterations=max_iterations,
                max_documents=max_documents,
                current_stage="security_blocked",
                security_decision=security_decision,
                errors=["Research blocked by input security policy."],
                audit_logs=[{
                    "stage": "security",
                    "duration_ms": 0.0,
                    "decision": security_decision,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
            )
        if not default_llm_client.has_active_key:
            raise LLMUnavailableError(
                "Research execution is blocked: configure GEMINI_API_KEY or GROQ_API_KEY."
            )
        record_audit_event("research_started", session_id=sess_id, query=question)
        initial_state = ResearchWorkflowState(
            session_id=sess_id,
            original_question=question,
            max_iterations=max_iterations,
            max_documents=max_documents,
        )

        config = {"configurable": {"thread_id": sess_id}}
        existing_state = await self.graph.aget_state(config) if self.graph is not None else None
        if existing_state and existing_state.values and existing_state.values.get("is_completed"):
            return ResearchWorkflowState.model_validate(existing_state.values)
        graph_input: Optional[ResearchWorkflowState] = initial_state
        if existing_state and existing_state.next:
            graph_input = None
        if self.graph is not None:
            final_output_dict = await self.graph.ainvoke(graph_input, config=config)
        else:
            async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DATABASE_PATH) as checkpointer:
                self.graph = self._build_graph(checkpointer)
                try:
                    existing_state = await self.graph.aget_state(config)
                    if existing_state and existing_state.values and existing_state.values.get("is_completed"):
                        return ResearchWorkflowState.model_validate(existing_state.values)
                    graph_input = None if existing_state and existing_state.next else initial_state
                    final_output_dict = await self.graph.ainvoke(graph_input, config=config)
                finally:
                    self.graph = None

        # Reconstruct typed state model from LangGraph output dictionary
        return ResearchWorkflowState.model_validate(final_output_dict)

    async def load_checkpoint(self, session_id: str) -> Optional[ResearchWorkflowState]:
        """Load a persisted workflow snapshot without executing any nodes."""
        config = {"configurable": {"thread_id": session_id}}
        async with AsyncSqliteSaver.from_conn_string(settings.CHECKPOINT_DATABASE_PATH) as checkpointer:
            graph = self._build_graph(checkpointer)
            snapshot = await graph.aget_state(config)
            if not snapshot or not snapshot.values:
                return None
            return ResearchWorkflowState.model_validate(snapshot.values)


default_orchestrator = ResearchWorkflowOrchestrator()
