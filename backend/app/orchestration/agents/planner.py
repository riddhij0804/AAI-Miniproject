"""Research Planner Agent decomposing objectives into actionable parallel/dependent tasks."""

import logging
import re
from typing import List, Optional

from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.orchestration.state import PlanTask, ResearchPlan, StructuredQuery
from backend.app.schemas.common import generate_uuid

logger = logging.getLogger(__name__)


class ResearchPlannerAgent:
    """Decomposes structured research objectives into directed task graphs with parallel stages."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or default_llm_client

    async def create_plan(
        self,
        structured_query: StructuredQuery,
        max_iterations: int = 3,
    ) -> ResearchPlan:
        """Create an initial multi-task research plan."""
        logger.info(f"ResearchPlannerAgent planning for: '{structured_query.research_objective}'")

        system_prompt = (
            "You are an expert Research Planning Agent. "
            "Given a research objective, scope, key concepts, and sub-questions, decompose the research "
            "into 2 to 4 actionable, focused research tasks. "
            "CRITICAL SEARCH QUERY RULE: The 'query' field in each task MUST be a concise 3 to 6 keyword search query "
            "anchored directly to the primary subject matter (e.g. 'renewable energy adoption trends', 'quantum computing error mitigation'). "
            "Never generate isolated dictionary terms (like 'prevalence' or 'distinguishes') or conversational questions as the search query. "
            "For each task, specify dependencies, if it can run in parallel, and evidence needed. "
            "Output JSON with format:\n"
            "{\n"
            '  "tasks": [\n'
            '    {\n'
            '      "task_id": string,\n'
            '      "title": string,\n'
            '      "query": string,\n'
            '      "objective": string,\n'
            '      "dependencies": [string],\n'
            '      "can_run_in_parallel": boolean,\n'
            '      "evidence_needed": [string],\n'
            '      "source_preferences": [string]\n'
            '    }\n'
            '  ],\n'
            '  "plan_rationale": string\n'
            "}"
        )

        user_prompt = (
            f"Original Query: {structured_query.original_query}\n"
            f"Objective: {structured_query.research_objective}\n"
            f"Concepts: {', '.join(structured_query.key_concepts)}\n"
            f"Sub-questions: {', '.join(structured_query.sub_questions)}\n"
            f"Preferences: {', '.join(structured_query.source_preferences)}"
        )

        def fallback():
            return self._heuristic_plan(structured_query)

        data = await self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_handler=fallback,
        )

        raw_tasks = data.get("tasks", [])
        tasks: List[PlanTask] = []
        primary_topic = structured_query.original_query.strip().rstrip("?").strip()
        topic_words = set(re.findall(r"\w+", primary_topic.lower())) - {"what", "how", "the", "are", "and", "for", "with", "from"}

        for t in raw_tasks:
            raw_q = t.get("query", structured_query.original_query).strip()
            q_words = set(re.findall(r"\w+", raw_q.lower()))
            # Anchor query if it doesn't already contain the primary topic keywords
            if not (topic_words & q_words):
                anchored_query = f"{primary_topic} {raw_q}".strip()
            else:
                anchored_query = raw_q

            tasks.append(
                PlanTask(
                    task_id=t.get("task_id") or generate_uuid(),
                    title=t.get("title", "Research Task"),
                    query=anchored_query,
                    objective=t.get("objective", structured_query.research_objective),
                    dependencies=t.get("dependencies", []),
                    can_run_in_parallel=t.get("can_run_in_parallel", True),
                    evidence_needed=t.get("evidence_needed", []),
                    source_preferences=t.get("source_preferences", structured_query.source_preferences),
                    status="pending",
                    iteration=1,
                )
            )

        if not tasks:
            fallback_data = self._heuristic_plan(structured_query)
            tasks = [PlanTask(**t) for t in fallback_data["tasks"]]

        batches = self._compute_parallel_batches(tasks)

        return ResearchPlan(
            objective=structured_query.research_objective,
            tasks=tasks,
            parallel_batches=batches,
            max_iterations=max_iterations,
            current_iteration=1,
            replanned_count=0,
            plan_rationale=data.get("plan_rationale", "Systematic decomposition into background, empirical data, and analysis."),
        )

    def replan_with_gaps(
        self,
        current_plan: ResearchPlan,
        unanswered_questions: List[str],
        missing_areas: List[str],
        new_iteration: int,
    ) -> ResearchPlan:
        """Targeted replanning to address missing evidence and unresolved gaps."""
        logger.info(f"ResearchPlannerAgent replanning for iteration {new_iteration} with {len(unanswered_questions)} gaps.")
        new_tasks: List[PlanTask] = list(current_plan.tasks)

        for idx, gap in enumerate(unanswered_questions[:3]):
            area = missing_areas[idx] if idx < len(missing_areas) else "Targeted gap"
            clean_gap = re.sub(r"^(what|how|which|why|when|where|is|are|does|do)\s+(is|are|the|a|an)?\s*", "", gap, flags=re.IGNORECASE).strip("? ")
            anchored_gap_q = f"{current_plan.objective[:40]} {clean_gap}".strip()
            task = PlanTask(
                task_id=generate_uuid(),
                title=f"Targeted Follow-up: {area}",
                query=anchored_gap_q,
                objective=f"Resolve empirical gap: {gap}",
                dependencies=[],
                can_run_in_parallel=True,
                evidence_needed=[gap],
                source_preferences=[],
                status="pending",
                iteration=new_iteration,
            )
            new_tasks.append(task)

        batches = self._compute_parallel_batches(new_tasks)

        return ResearchPlan(
            plan_id=current_plan.plan_id,
            objective=current_plan.objective,
            tasks=new_tasks,
            parallel_batches=batches,
            max_iterations=current_plan.max_iterations,
            current_iteration=new_iteration,
            replanned_count=current_plan.replanned_count + 1,
            plan_rationale=f"Replanned iteration {new_iteration} to address gaps: {', '.join(unanswered_questions[:2])}",
        )

    def _compute_parallel_batches(self, tasks: List[PlanTask]) -> List[List[str]]:
        """Compute execution order batches based on task dependencies."""
        independent = [t.task_id for t in tasks if not t.dependencies]
        dependent = [t.task_id for t in tasks if t.dependencies]
        batches = []
        if independent:
            batches.append(independent)
        if dependent:
            batches.append(dependent)
        if not batches:
            batches = [[t.task_id for t in tasks]]
        return batches

    def _heuristic_plan(self, structured_query: StructuredQuery) -> dict:
        """Generate domain-independent structured 3-stage plan."""
        q = structured_query.original_query
        concepts = structured_query.key_concepts
        c_str = " ".join(concepts[:3])

        task1_id = generate_uuid()
        task2_id = generate_uuid()
        task3_id = generate_uuid()

        return {
            "tasks": [
                {
                    "task_id": task1_id,
                    "title": f"Core Foundations & Drivers of {concepts[0] if concepts else 'Topic'}",
                    "query": f"{q} causes drivers mechanisms",
                    "objective": f"Identify fundamental mechanisms and established background for {c_str}",
                    "dependencies": [],
                    "can_run_in_parallel": True,
                    "evidence_needed": ["Primary definitions", "Causal drivers", "Key mechanisms"],
                    "source_preferences": structured_query.source_preferences,
                },
                {
                    "task_id": task2_id,
                    "title": f"Empirical Data & Quantitative Findings",
                    "query": f"{c_str} statistics data studies trends",
                    "objective": f"Gather verifiable quantitative data, empirical studies, and real-world statistics for {c_str}",
                    "dependencies": [],
                    "can_run_in_parallel": True,
                    "evidence_needed": ["Empirical statistics", "Measured outcomes", "Published studies"],
                    "source_preferences": structured_query.source_preferences,
                },
                {
                    "task_id": task3_id,
                    "title": f"Impacts, Contradictions & Future Challenges",
                    "query": f"{c_str} challenges risks debates future impact",
                    "objective": f"Analyze complex impacts, opposing viewpoints, and unanswered challenges for {c_str}",
                    "dependencies": [task1_id],
                    "can_run_in_parallel": False,
                    "evidence_needed": ["Disputed findings", "Risk assessments", "Policy/practical challenges"],
                    "source_preferences": structured_query.source_preferences,
                },
            ],
            "plan_rationale": "Decomposed into foundational mechanisms, empirical metrics, and critical debate/challenges.",
        }

