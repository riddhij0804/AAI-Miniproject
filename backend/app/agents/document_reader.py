"""Document Reader Agent for reading, parsing, and chunking documents from sources."""

import asyncio
from datetime import datetime
import logging
from typing import List, Optional
from urllib.parse import urlparse
from backend.app.agents.source_discovery import canonicalize_url

from backend.app.retrieval.chunking import DocumentChunker
from backend.app.schemas.common import DocumentType, generate_uuid, utc_now
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.source import Source
from backend.app.tools.registry import ToolRegistry, default_registry

logger = logging.getLogger(__name__)


class DocumentReaderAgent:
    """Agent responsible for acquiring, reading, parsing, and chunking external documents."""

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.registry = registry or default_registry
        self.chunker = chunker or DocumentChunker()
        self._document_cache = {}
        self._cache_lock = asyncio.Lock()
        self._url_locks = {}

    async def read_documents(self, sources: List[Source]) -> List[Document]:
        """Read and chunk documents from multiple sources concurrently with graceful error handling."""
        unique_sources = []
        seen_urls = set()
        for source in sources:
            key = canonicalize_url(source.url)
            if key in seen_urls:
                continue
            seen_urls.add(key)
            unique_sources.append(source)
        tasks = [self.read_single_document(source) for source in unique_sources]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return [doc for doc in results if doc is not None]

    async def read_single_document(self, source: Source) -> Optional[Document]:
        """Acquire a document from a single source, parse content, and chunk it."""
        url = source.url
        cache_key = canonicalize_url(url)
        async with self._cache_lock:
            url_lock = self._url_locks.setdefault(cache_key, asyncio.Lock())
        async with url_lock:
            async with self._cache_lock:
                cached = self._document_cache.get(cache_key)
            if cached is not None:
                logger.info("document_cache_hit url=%s", url)
                return cached
            logger.info(f"DocumentReaderAgent reading source '{source.id}' from URL: '{url}'")

            is_pdf = self._is_pdf_url(url)
            doc_id = generate_uuid()
            try:
                if is_pdf:
                    exec_result = await self.registry.execute_tool("read_pdf", url_or_path=url, source_id=source.id)
                else:
                    exec_result = await self.registry.execute_tool("read_webpage", url=url, source_id=source.id)

                if not exec_result.success:
                    logger.warning(f"Failed to read document from '{url}': {exec_result.error}")
                    return Document(
                        id=doc_id,
                        source_id=source.id,
                        title=f"Failed to load: {source.title}",
                        content="",
                        document_type=DocumentType.PDF if is_pdf else DocumentType.HTML,
                        metadata={
                            "url": url,
                            "read_status": "failed",
                            "error": exec_result.error,
                            "source_title": source.title,
                        },
                        chunks=[],
                    )

                data = exec_result.data or {}
                content = data.get("content", "")
                title = data.get("title") or source.title or "Untitled Document"
                doc_type = DocumentType.PDF if is_pdf else DocumentType.HTML
                pub_date: Optional[datetime] = None
                raw_pub_date = data.get("publication_date")
                if raw_pub_date and isinstance(raw_pub_date, str):
                    try:
                        pub_date = datetime.fromisoformat(raw_pub_date.replace("Z", "+00:00"))
                    except ValueError:
                        pub_date = None

                document = Document(
                    id=doc_id,
                    source_id=source.id,
                    title=title,
                    content=content,
                    document_type=doc_type,
                    publication_date=pub_date,
                    metadata={
                        **data.get("metadata", {}),
                        "url": url,
                        "pages": data.get("pages"),
                        "author": data.get("author") or source.author,
                        "read_status": "success",
                    },
                )
                chunks = self.chunker.chunk_document(
                    document=document,
                    source_url=url,
                    source_id=source.id,
                )
                document.chunks = chunks
                async with self._cache_lock:
                    self._document_cache[cache_key] = document
                logger.info(f"DocumentReaderAgent successfully processed '{url}' into {len(chunks)} chunks")
                return document

            except Exception as e:
                logger.error(f"Unexpected error while reading document from '{url}': {e}")
                return Document(
                    id=doc_id,
                    source_id=source.id,
                    title=f"Error reading: {source.title}",
                    content="",
                    document_type=DocumentType.PDF if is_pdf else DocumentType.HTML,
                    metadata={
                        "url": url,
                        "read_status": "error",
                        "error": str(e),
                    },
                    chunks=[],
                )

    @staticmethod
    def _is_pdf_url(url: str) -> bool:
        """Check if URL points to a PDF file."""
        try:
            path = urlparse(url).path.lower()
            return path.endswith(".pdf") or "pdf" in path.split("/")[-1]
        except Exception:
            return False
