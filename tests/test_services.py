"""Unit tests for Research Intelligence Service integration contract."""

import pytest
from unittest.mock import AsyncMock

from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.source import Source, SourceDiscoveryItem, SourceDiscoveryRequest
from backend.app.services.research_service import ResearchIntelligenceService
from backend.app.tools.base import ToolExecutionResult
from backend.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_research_service_integration_contract(evidence_store, test_retriever, in_memory_db_session):
    # Setup mock tools in a test registry
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
                        "source_id": "src-contract-1",
                        "title": "Causes of Antibiotic Resistance",
                        "url": "https://who.int/amr-facts",
                        "publisher": "who.int",
                        "source_type": "government",
                        "snippet": "Overuse of antimicrobials drives resistance genes.",
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
                    "Agricultural overuse of antibiotics as growth promoters causes evolutionary pressure. "
                    "Horizontal plasmid transfer spreads resistant beta-lactamase genes across bacterial species."
                ),
                "document_type": "html",
                "metadata": {"url": "https://who.int/amr-facts"},
            },
        )
    )
    registry.register_tool(mock_reader)

    # Initialize Service with test fixtures
    from backend.app.agents.document_reader import DocumentReaderAgent
    from backend.app.agents.source_discovery import SourceDiscoveryAgent
    from backend.app.evidence.graph import EvidenceGraphService
    from backend.app.evidence.memory import ResearchMemoryManager

    discovery_agent = SourceDiscoveryAgent(registry=registry)
    reader_agent = DocumentReaderAgent(registry=registry)
    graph_svc = EvidenceGraphService(db_session_factory=lambda: in_memory_db_session)
    mem_mgr = ResearchMemoryManager(
        store=evidence_store,
        retriever=test_retriever,
        db_session_factory=lambda: in_memory_db_session,
    )

    service = ResearchIntelligenceService(
        discovery_agent=discovery_agent,
        reader_agent=reader_agent,
        store=evidence_store,
        retriever=test_retriever,
        graph_service=graph_svc,
        memory_manager=mem_mgr,
    )

    # 1. Step 1: discover_sources(task)
    task_req = SourceDiscoveryRequest(
        task_id="task-amr-01",
        query="What are the major causes of antibiotic resistance?",
        research_objective="Determine AMR root causes",
    )
    sources = await service.discover_sources(task_req)
    assert len(sources) == 1
    assert sources[0].url == "https://who.int/amr-facts"

    # 2. Step 2: assess_source_credibility(sources)
    assessments = service.assess_source_credibility(sources)
    assert len(assessments) == 1
    assert assessments[0].overall_score > 0.70

    # 3. Step 3: read_documents(sources)
    documents = await service.read_documents(sources)
    assert len(documents) == 1
    assert len(documents[0].chunks) > 0

    # 4. Step 4: extract_evidence(documents, objective)
    extract_result = await service.extract_evidence(
        documents=documents,
        objective="Determine agricultural and genetic causes of AMR",
    )
    assert len(extract_result.claims) > 0
    assert len(extract_result.evidence_records) > 0
    assert extract_result.verified_quote_count > 0

    # 5. Step 5: retrieve_evidence(query, top_k)
    retrieved_hits = service.retrieve_evidence("agricultural overuse antibiotics", top_k=3)
    assert len(retrieved_hits) > 0
    assert retrieved_hits[0].source_id == sources[0].id
    assert "agricultural" in retrieved_hits[0].text.lower()

    # 6. Step 6: get_claim_evidence(claim_id)
    claim_id = extract_result.claims[0].id
    claim_view = service.get_claim_evidence(claim_id)
    assert claim_view is not None
    assert claim_view.claim.id == claim_id
    assert len(claim_view.evidence) > 0

    # 7. Step 7: get_evidence_graph()
    graph = service.get_evidence_graph()
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
