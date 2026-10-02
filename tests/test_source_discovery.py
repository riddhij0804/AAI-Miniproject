"""Unit tests for Source Discovery Agent and web search tools."""

import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agents.source_discovery import SourceDiscoveryAgent
from backend.app.schemas.common import SourceType
from backend.app.schemas.source import SourceDiscoveryItem, SourceDiscoveryRequest
from backend.app.tools.base import ToolExecutionResult
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.web_search import infer_publisher_and_type


def test_infer_publisher_and_type():
    domain, stype = infer_publisher_and_type("https://www.nature.com/articles/s41586-021-03819-2")
    assert "nature.com" in domain
    assert stype == SourceType.ACADEMIC_JOURNAL

    domain, stype = infer_publisher_and_type("https://www.cdc.gov/drugresistance/about.html")
    assert stype == SourceType.GOVERNMENT

    domain, stype = infer_publisher_and_type("https://cs.stanford.edu/research")
    assert stype == SourceType.EDUCATIONAL

    domain, stype = infer_publisher_and_type("https://www.reuters.com/business/energy")
    assert stype == SourceType.NEWS


@pytest.mark.asyncio
async def test_source_discovery_agent_success():
    registry = ToolRegistry()
    mock_tool = AsyncMock()
    mock_tool.name = "search_web"
    mock_tool.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="search_web",
            success=True,
            data={
                "provider_used": "duckduckgo",
                "sources": [
                    {
                        "source_id": "src-1",
                        "title": "Causes of Antibiotic Resistance",
                        "url": "https://www.who.int/antimicrobial-resistance",
                        "publisher": "who.int",
                        "source_type": "government",
                        "snippet": "Overuse of antibiotics in humans and animals leads to resistance.",
                    }
                ],
            },
        )
    )
    registry.register_tool(mock_tool)

    agent = SourceDiscoveryAgent(registry=registry)
    task = SourceDiscoveryRequest(
        task_id="task-101",
        query="What are the major causes of antibiotic resistance?",
        research_objective="Determine drivers of AMR",
        source_preferences=["who.int", "cdc.gov"],
        max_results=3,
    )

    response = await agent.discover_sources(task)
    assert response.provider_used == "duckduckgo"
    assert len(response.sources) == 1

    item = response.sources[0]
    assert item.title == "Causes of Antibiotic Resistance"
    assert item.url == "https://www.who.int/antimicrobial-resistance"
    # CRITICAL INVARIANT: Snippet is not verified evidence
    assert item.metadata["is_verified_evidence"] is False
    assert item.metadata["search_snippet_only"] is True

    # Convert to persistent source
    source = agent.convert_discovery_to_source(item)
    assert source.id == "src-1"
    assert source.url == item.url
