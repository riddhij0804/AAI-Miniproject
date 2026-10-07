"""End-to-end integration tests for LangGraph workflow orchestrator."""

import pytest
from unittest.mock import AsyncMock

from backend.app.agents.document_reader import DocumentReaderAgent
from backend.app.agents.source_discovery import SourceDiscoveryAgent
from backend.app.orchestration.graph import ResearchWorkflowOrchestrator
from backend.app.schemas.source import SourceDiscoveryRequest
from backend.app.services.research_service import ResearchIntelligenceService
from backend.app.tools.base import ToolExecutionResult
from backend.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_langgraph_orchestrator_end_to_end(evidence_store, test_retriever, in_memory_db_session):
    # Setup mock tools in test registry
    registry = ToolRegistry()
    mock_search = AsyncMock()
    mock_search.name = "search_web"
    mock_search.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="search_web",
            success=True,
            data={
                "provider_used": "duckduckgo",
                "sources": [
                    {
                        "source_id": "src-orch-1",
                        "title": "Causes of Antibiotic Resistance",
                        "url": "https://who.int/amr-overview",
                        "publisher": "who.int",
                        "source_type": "government",
                        "snippet": "Antimicrobial resistance is accelerated by overprescription in humans.",
                    }
                ],
            },
        )
    )
    registry.register_tool(mock_search)

    mock_reader = AsyncMock()
    mock_reader.name = "read_webpage"
    mock_reader.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read_webpage",
            success=True,
            data={
                "title": "Causes of Antibiotic Resistance",
                "content": (
                    "Overuse of antibiotics in clinical medicine accelerates the emergence of resistant bacteria. "
                    "Agricultural usage of antibiotics as animal growth promoters generates selective pressure. "
                    "Horizontal plasmid transfer spreads resistant beta-lactamase genes across diverse strains."
                ),
                "document_type": "html",
                "metadata": {"url": "https://who.int/amr-overview"},
            },
        )
    )
    registry.register_tool(mock_reader)

    # Initialize Service
    discovery_agent = SourceDiscoveryAgent(registry=registry)
    reader_agent = DocumentReaderAgent(registry=registry)
    res_service = ResearchIntelligenceService(
        discovery_agent=discovery_agent,
        reader_agent=reader_agent,
        store=evidence_store,
        retriever=test_retriever,
    )

    orchestrator = ResearchWorkflowOrchestrator(
        research_service=res_service,
        store=evidence_store,
    )

    # Run workflow
    final_state = await orchestrator.run(
        question="What are the major causes of antibiotic resistance?",
        session_id="session-test-e2e",
        max_iterations=2,
    )

    # Assertions on final state
    assert final_state.is_completed is True
    assert final_state.structured_query is not None
    assert final_state.plan is not None
    assert len(final_state.discovered_sources) >= 1
    assert len(final_state.claims) >= 1
    assert len(final_state.evidence_records) >= 1
    assert final_state.verification_report is not None
    assert final_state.final_report is not None
    assert final_state.citation_validation is not None
    assert len(final_state.audit_logs) >= 5

    # Check report fields
    report = final_state.final_report
    assert "antibiotic resistance" in report.title.lower()
    assert len(report.executive_summary) > 20
    assert len(report.key_findings) > 0
    assert len(report.references) >= 1
    assert report.references[0].source_url == "https://who.int/amr-overview"


@pytest.mark.asyncio
async def test_langgraph_orchestrator_tool_failure_handling(evidence_store, test_retriever):
    """Workflow gracefully handles tool failure without crashing or unhandled exceptions."""
    registry = ToolRegistry()
    mock_search = AsyncMock()
    mock_search.name = "search_web"
    mock_search.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="search_web",
            success=False,
            error="Rate limit exceeded: 429 Too Many Requests",
        )
    )
    registry.register_tool(mock_search)

    discovery_agent = SourceDiscoveryAgent(registry=registry)
    res_service = ResearchIntelligenceService(
        discovery_agent=discovery_agent,
        store=evidence_store,
        retriever=test_retriever,
    )

    orchestrator = ResearchWorkflowOrchestrator(
        research_service=res_service,
        store=evidence_store,
    )

    final_state = await orchestrator.run(
        question="What are the causes of AMR?",
        session_id="session-fail-test",
        max_iterations=1,
    )

    assert final_state.is_completed is True
    assert final_state.final_report is not None
    assert len(final_state.audit_logs) > 0
