"""Source Discovery Agent responsible for discovering, filtering, and structuring web sources."""

import logging
from typing import Any, Dict, List, Optional

from backend.app.schemas.common import SourceType, generate_uuid
from backend.app.schemas.source import (
    Source,
    SourceDiscoveryItem,
    SourceDiscoveryRequest,
    SourceDiscoveryResponse,
)
from backend.app.tools.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)


class SourceDiscoveryAgent:
    """Agent that identifies and triages external web sources given a research task and objective."""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or default_registry

    async def discover_sources(self, task: SourceDiscoveryRequest) -> SourceDiscoveryResponse:
        """Execute source discovery using approved search tools with fallback."""
        logger.info(f"SourceDiscoveryAgent starting search for task '{task.task_id}': '{task.query}'")

        # Construct optimized query string taking preferences into account
        query = task.query
        if task.source_preferences:
            prefs = " OR ".join(f"site:{pref}" if "." in pref else pref for pref in task.source_preferences)
            query = f"{query} ({prefs})"

        # Execute web search through the controlled tool registry
        exec_result = await self.registry.execute_tool(
            "search_web",
            query=query,
            max_results=task.max_results,
        )

        if not exec_result.success:
            logger.error(f"Source discovery failed: {exec_result.error}")
            return SourceDiscoveryResponse(
                task_id=task.task_id,
                query=task.query,
                provider_used="none",
                sources=[],
                error_message=exec_result.error,
            )

        data = exec_result.data or {}
        raw_sources = data.get("sources", [])
        provider = data.get("provider_used", "unknown")

        sources: List[SourceDiscoveryItem] = []
        for s in raw_sources:
            item = SourceDiscoveryItem(
                source_id=s.get("source_id", generate_uuid()),
                title=s.get("title", "Untitled Source"),
                url=s.get("url", ""),
                publisher=s.get("publisher"),
                source_type=s.get("source_type", SourceType.WEBPAGE),
                published_at=s.get("published_at"),
                snippet=s.get("snippet", ""),
                score=s.get("score"),
                metadata={
                    **s.get("metadata", {}),
                    "task_id": task.task_id,
                    "research_objective": task.research_objective,
                    "search_snippet_only": True,
                    "is_verified_evidence": False,  # Explicit invariant
                },
            )
            sources.append(item)

        logger.info(f"SourceDiscoveryAgent found {len(sources)} sources via {provider}")
        return SourceDiscoveryResponse(
            task_id=task.task_id,
            query=task.query,
            provider_used=provider,
            sources=sources,
        )

    def convert_discovery_to_source(self, item: SourceDiscoveryItem) -> Source:
        """Convert a preliminary discovery item into a persistent Source model."""
        return Source(
            id=item.source_id,
            url=item.url,
            title=item.title,
            publisher=item.publisher,
            source_type=item.source_type,
            published_at=None,
            credibility_score=None,
            metadata={
                **item.metadata,
                "discovery_snippet": item.snippet,
            },
        )
