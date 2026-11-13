import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from backend.app.agents.document_reader import DocumentReaderAgent
from backend.app.agents.evidence_extractor import EvidenceExtractorAgent
from backend.app.orchestration.llm import LLMClient, LLMUnavailableError
from backend.app.retrieval.embeddings import HashEmbeddingService
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.vector_store import QdrantVectorStore
from backend.app.services.research_service import ResearchIntelligenceService
from backend.app.schemas.common import SourceType
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.source import Source
from backend.app.tools.base import ToolExecutionResult
from backend.app.tools.base import BaseTool
from backend.app.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_extraction_runs_chunks_concurrently(monkeypatch):
    extractor = EvidenceExtractorAgent()
    chunks = [
        DocumentChunk(
            id=f"chunk-{index}",
            document_id="doc",
            source_id="source",
            text=f"Climate change evidence sentence number {index} is documented.",
            chunk_index=index,
        )
        for index in range(4)
    ]

    async def delayed_extract(chunk, _objective):
        await asyncio.sleep(0.03)
        return []

    monkeypatch.setattr(extractor, "_extract_from_chunk", delayed_extract)
    started = time.perf_counter()
    result = await extractor.extract_evidence_from_chunks(chunks, "climate change")

    assert result.execution_notes.startswith("Processed 4 chunks")
    assert time.perf_counter() - started < 0.10


@pytest.mark.asyncio
async def test_document_reader_deduplicates_concurrent_urls():
    registry = ToolRegistry()
    tool = AsyncMock()
    tool.name = "read_webpage"
    tool.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read_webpage",
            success=True,
            data={"title": "Climate", "content": "Climate evidence is documented."},
        )
    )
    registry.register_tool(tool)
    agent = DocumentReaderAgent(registry=registry)
    sources = [
        Source(id=f"source-{i}", url=url, title="Climate", source_type=SourceType.WEBPAGE)
        for i, url in enumerate(
            ["https://example.com/article", "https://EXAMPLE.com/article/"]
        )
    ]

    documents = await agent.read_documents(sources)

    assert len(documents) == 1
    assert tool.run.await_count == 1


def test_incremental_indexing_reuses_embeddings():
    class CountingEmbeddingService(HashEmbeddingService):
        def __init__(self):
            super().__init__(dimension=384)
            self.batch_calls = 0

        def embed_batch(self, texts):
            self.batch_calls += 1
            return super().embed_batch(texts)

    embedding_service = CountingEmbeddingService()
    retriever = HybridRetriever(
        vector_store=QdrantVectorStore(location=":memory:", collection_name="perf-test"),
        embedding_service=embedding_service,
    )
    chunk = DocumentChunk(
        id="chunk-1",
        document_id="doc-1",
        source_id="source-1",
        source_url="https://example.com",
        text="Global warming is driven by greenhouse gas emissions.",
        chunk_index=0,
    )

    retriever.index_chunks([chunk])
    retriever.index_chunks([chunk])

    assert embedding_service.batch_calls == 1
    assert len(retriever._bm25_corpus) == 1


@pytest.mark.asyncio
async def test_llm_does_not_retry_permanent_http_error(monkeypatch):
    client = LLMClient(provider="groq", groq_key="test-key", gemini_key="", openai_key="")
    calls = 0

    async def bad_request(_system, _user):
        nonlocal calls
        calls += 1
        raise ValueError("invalid request")

    monkeypatch.setattr(client, "_call_groq_json", bad_request)
    with pytest.raises(LLMUnavailableError):
        await client.generate_json("system", "user")

    assert calls == 1


@pytest.mark.asyncio
async def test_tool_does_not_retry_permanent_http_error():
    class ForbiddenTool(BaseTool):
        name = "forbidden"
        class Input(BaseModel):
            pass

        input_schema = Input

        async def _execute(self, _params):
            raise RuntimeError("HTTP error 403 fetching document")

    result = await ForbiddenTool(max_retries=3).run()

    assert result.success is False
    assert "403" in result.error


@pytest.mark.asyncio
async def test_index_failure_does_not_drop_documents_before_extraction():
    class Reader:
        async def read_documents(self, _sources):
            return [
                Document(
                    id="doc-index-failure",
                    source_id="source-index-failure",
                    title="Evidence",
                    content="Climate change causes warming through greenhouse gases.",
                    chunks=[
                        DocumentChunk(
                            id="chunk-index-failure",
                            document_id="doc-index-failure",
                            source_id="source-index-failure",
                            source_url="https://example.com/evidence",
                            text="Climate change causes warming through greenhouse gases.",
                            chunk_index=0,
                        )
                    ],
                )
            ]

    class BrokenRetriever:
        def index_chunks(self, _chunks):
            raise RuntimeError("persistent Qdrant directory is locked")

    service = ResearchIntelligenceService(
        reader_agent=Reader(),
        retriever=BrokenRetriever(),
        store=MagicMock(),
    )
    source = Source(
        id="source-index-failure",
        url="https://example.com/evidence",
        title="Evidence",
        source_type=SourceType.WEBPAGE,
    )

    documents = await service.read_documents([source])

    assert len(documents) == 1
    assert documents[0].chunks
