"""Unit tests for Document Reader Agent and webpage/PDF tools."""

import io
import pytest
from unittest.mock import AsyncMock, patch

from backend.app.agents.document_reader import DocumentReaderAgent
from backend.app.schemas.common import DocumentType, SourceType, generate_uuid
from backend.app.schemas.source import Source
from backend.app.tools.base import ToolExecutionResult
from backend.app.tools.pdf_reader import PDFReaderTool
from backend.app.tools.registry import ToolRegistry
from backend.app.tools.webpage_reader import WebpageReaderTool


def test_html_parsing_removes_boilerplate():
    reader = WebpageReaderTool()
    sample_html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Renewable Energy in India</title>
        <meta name="author" content="Dr. A. Sharma">
      </head>
      <body>
        <nav><a href="/home">Home</a><a href="/menu">Menu</a></nav>
        <script>console.log("analytics tracking");</script>
        <style>body { background: white; }</style>
        <main>
          <h1>Solar Surge in India</h1>
          <p>India added 15 GW of solar capacity during the last fiscal year.</p>
        </main>
        <footer>Copyright 2024</footer>
      </body>
    </html>
    """
    parsed = reader._parse_html(sample_html, "https://example.org/energy")
    assert parsed["title"] == "Renewable Energy in India"
    assert parsed["author"] == "Dr. A. Sharma"
    assert "Solar Surge in India" in parsed["content"]
    assert "India added 15 GW" in parsed["content"]
    # Check that boilerplate was stripped
    assert "analytics tracking" not in parsed["content"]
    assert "Copyright 2024" not in parsed["content"]


def test_pdf_page_preservation():
    # Generate a lightweight in-memory PDF using reportlab or PyMuPDF
    import fitz  # PyMuPDF
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 72), "Page 1 Content: Research into antibiotic resistance.")
    page2 = doc.new_page()
    page2.insert_text((50, 72), "Page 2 Content: Overprescription statistics and data.")
    pdf_bytes = doc.tobytes()
    doc.close()

    reader = PDFReaderTool()
    extracted = reader._extract_pdf_pages(pdf_bytes, "test_doc.pdf")
    assert extracted["document_type"] == DocumentType.PDF.value
    assert len(extracted["pages"]) == 2
    assert extracted["pages"][0]["page_number"] == 1
    assert "Page 1 Content" in extracted["pages"][0]["text"]
    assert extracted["pages"][1]["page_number"] == 2
    assert "Page 2 Content" in extracted["pages"][1]["text"]


@pytest.mark.asyncio
async def test_document_reader_agent_provenance_and_chunking():
    registry = ToolRegistry()
    mock_tool = AsyncMock()
    mock_tool.name = "read_webpage"
    mock_tool.run = AsyncMock(
        return_value=ToolExecutionResult(
            tool_name="read_webpage",
            success=True,
            data={
                "title": "Antibiotic Resistance Overview",
                "content": "Paragraph 1: Antibiotic resistance is a global health threat. Paragraph 2: Hospital transmission is rapid.",
                "document_type": "html",
                "metadata": {"url": "https://example.com/amr"},
            },
        )
    )
    registry.register_tool(mock_tool)

    agent = DocumentReaderAgent(registry=registry)
    source = Source(
        id="src-test-01",
        url="https://example.com/amr",
        title="Antibiotic Resistance Overview",
        source_type=SourceType.WEBPAGE,
    )

    docs = await agent.read_documents([source])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_id == "src-test-01"
    assert len(doc.chunks) > 0

    # Test complete chunk provenance
    for chunk in doc.chunks:
        assert chunk.document_id == doc.id
        assert chunk.source_id == "src-test-01"
        assert chunk.source_url == "https://example.com/amr"
