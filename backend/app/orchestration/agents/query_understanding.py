"""Query Understanding Agent interpreting and structuring research questions."""

import logging
import re
from typing import List, Optional

from backend.app.orchestration.llm import LLMClient, default_llm_client
from backend.app.orchestration.state import StructuredQuery

logger = logging.getLogger(__name__)


class QueryUnderstandingAgent:
    """Interprets user queries, extracts research scope, concepts, and handles ambiguity."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or default_llm_client

    async def analyze_query(self, raw_query: str) -> StructuredQuery:
        """Parse raw user query into structured, validated research request."""
        clean_query = raw_query.strip()
        logger.info(f"QueryUnderstandingAgent analyzing: '{clean_query[:80]}...'")

        # System prompt for LLM analysis
        system_prompt = (
            "You are an expert Research Query Analysis agent. "
            "Your job is to analyze a user's research question and produce a structured specification. "
            "Evaluate whether the query is clear or ambiguous/under-specified. "
            "Output a JSON object with: "
            "{\n"
            '  "original_query": string,\n'
            '  "research_objective": string,\n'
            '  "scope": string,\n'
            '  "key_concepts": [string],\n'
            '  "constraints": [string],\n'
            '  "expected_output_format": "comprehensive_report",\n'
            '  "source_preferences": [string],\n'
            '  "is_ambiguous": boolean,\n'
            '  "ambiguity_reasons": [string],\n'
            '  "clarification_strategy": string or null,\n'
            '  "sub_questions": [string]\n'
            "}"
        )

        user_prompt = f"User Research Question:\n{clean_query}"

        # Deterministic fallback handler
        def fallback():
            return self._heuristic_analyze(clean_query)

        data = await self.llm.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_handler=fallback,
        )

        if isinstance(data, StructuredQuery):
            return data
        if isinstance(data, dict) and data:
            try:
                return StructuredQuery(**data)
            except Exception as e:
                logger.warning(f"Failed to parse LLM structured query ({e}); falling back to heuristic.")
        return self._heuristic_analyze(clean_query)

    def _heuristic_analyze(self, query: str) -> StructuredQuery:
        """Domain-independent rule-based query parser and ambiguity detector."""
        words = [w for w in re.findall(r"\w+", query) if len(w) > 2]
        is_ambiguous = len(words) < 4

        ambiguity_reasons = []
        clarification_strategy = None
        if is_ambiguous:
            ambiguity_reasons.append("Query is under-specified; broad overview inferred.")
            clarification_strategy = "Expanded scope across fundamental concepts, current mechanisms, and impacts."

        # Extract stopword-filtered key concepts
        stopwords = {
            "what", "how", "why", "when", "where", "which", "who", "whom",
            "the", "and", "for", "with", "from", "about", "into", "over",
            "that", "this", "these", "those", "are", "is", "was", "were",
            "can", "could", "should", "would", "does", "did", "tell", "give",
            "explain", "overview", "detail", "details"
        }
        key_concepts = [w for w in words if w.lower() not in stopwords]
        if not key_concepts:
            key_concepts = words[:3] if words else ["general_overview"]

        # Synthesize research objective
        if query.lower().startswith("what are the causes of"):
            objective = f"Identify and analyze primary drivers, mechanisms, and contributing factors of {query[23:].strip(' ?')}."
        elif query.lower().startswith("how"):
            objective = f"Investigate mechanisms, progression, and real-world implications of {query[4:].strip(' ?')}."
        else:
            objective = f"Conduct comprehensive evidence-based investigation into {query.strip(' ?')}."

        # Formulate sub-questions for thorough decomposition
        sub_questions = [
            f"What are the foundational definitions and mechanisms of {', '.join(key_concepts[:2])}?",
            f"What empirical evidence and quantitative data describe {', '.join(key_concepts[:2])}?",
            f"What are the key challenges, disagreements, or future outlook regarding {', '.join(key_concepts[:2])}?",
        ]

        # Domain/source preferences detection heuristics
        source_preferences = []
        if any(w in query.lower() for w in ["antibiotic", "medical", "disease", "health", "clinical", "virus"]):
            source_preferences = ["who.int", "cdc.gov", "nih.gov"]
        elif any(w in query.lower() for w in ["energy", "solar", "wind", "grid", "electricity", "india"]):
            source_preferences = ["iea.org", "gov.in", "irena.org"]
        elif any(w in query.lower() for w in ["llm", "security", "ai", "cyber", "injection", "vulnerability"]):
            source_preferences = ["arxiv.org", "owasp.org", "nist.gov"]

        return StructuredQuery(
            original_query=query,
            research_objective=objective,
            scope="Global multi-perspective literature review",
            key_concepts=key_concepts[:6],
            constraints=["Rely on verified, primary, or high-credibility evidence"],
            expected_output_format="comprehensive_report",
            source_preferences=source_preferences,
            is_ambiguous=is_ambiguous,
            ambiguity_reasons=ambiguity_reasons,
            clarification_strategy=clarification_strategy,
            sub_questions=sub_questions,
        )
