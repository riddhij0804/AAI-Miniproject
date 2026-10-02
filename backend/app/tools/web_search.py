"""Web search tool supporting Tavily primary with DuckDuckGo fallback."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.schemas.common import SourceType, generate_uuid
from backend.app.schemas.source import SourceDiscoveryItem
from backend.app.tools.base import BaseTool

logger = logging.getLogger(__name__)


def infer_publisher_and_type(url: str, raw_title: str = "") -> tuple[str, SourceType]:
    """Domain-independent heuristic to infer publisher name and source type from URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # Categorize by top-level or domain patterns
        if domain.endswith(".edu") or ".edu." in domain or "ac.uk" in domain:
            return domain, SourceType.EDUCATIONAL
        if domain.endswith(".gov") or ".gov." in domain:
            return domain, SourceType.GOVERNMENT
        if any(academic in domain for academic in [
            "arxiv.org", "biorxiv.org", "medrxiv.org", "ncbi.nlm.nih.gov",
            "nature.com", "sciencedirect.com", "springer.com", "ieee.org",
            "acm.org", "thelancet.com", "cell.com", "pnas.org", "wiley.com"
        ]):
            return domain, SourceType.ACADEMIC_JOURNAL
        if domain.endswith(".org"):
            return domain, SourceType.ORGANIZATION
        if any(news in domain for news in [
            "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
            "thehindu.com", "nytimes.com", "wsj.com", "theguardian.com", "nature.com/news"
        ]):
            return domain, SourceType.NEWS

        return domain, SourceType.WEBPAGE
    except Exception:
        return "Unknown Publisher", SourceType.UNKNOWN


class WebSearchInput(BaseModel):
    query: str = Field(..., min_length=2, description="Search query string")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum search results to return")


class WebSearchTool(BaseTool):
    """Web search tool with Tavily primary and DuckDuckGo fallback."""

    name: str = "search_web"
    description: str = "Search the web for research sources given a natural-language query."
    input_schema = WebSearchInput
    timeout: float = settings.SEARCH_TIMEOUT_SECONDS

    async def _execute(self, params: WebSearchInput) -> Dict[str, Any]:
        results: List[SourceDiscoveryItem] = []
        provider_used = "none"
        errors: List[str] = []

        # 1. Attempt Tavily if API key is present
        if settings.TAVILY_API_KEY:
            try:
                results = await self._search_tavily(params.query, params.max_results)
                provider_used = "tavily"
                logger.info(f"Tavily search successful for query '{params.query}': {len(results)} results")
            except Exception as e:
                err = f"Tavily search failed: {e}"
                logger.warning(err)
                errors.append(err)

        # 2. Fallback to DuckDuckGo / Open Search if Tavily wasn't used or failed
        if not results:
            try:
                results = await self._search_duckduckgo(params.query, params.max_results)
                if results:
                    provider_used = "duckduckgo"
                    logger.info(f"DuckDuckGo search successful for query '{params.query}': {len(results)} results")
            except Exception as e:
                err = f"DuckDuckGo search failed: {e}"
                logger.warning(err)
                errors.append(err)

        # 3. Tertiary fallback to Wikipedia OpenSearch if DDG returned 0 results
        if not results:
            try:
                results = await self._search_wikipedia(params.query, params.max_results)
                if results:
                    provider_used = "wikipedia_opensearch"
                    logger.info(f"Wikipedia fallback search successful: {len(results)} results")
            except Exception as e:
                err = f"Wikipedia fallback search failed: {e}"
                logger.warning(err)
                errors.append(err)

        # Convert to dictionary output format
        return {
            "query": params.query,
            "provider_used": provider_used,
            "results_count": len(results),
            "sources": [r.model_dump() for r in results],
            "fallback_errors": errors if errors else None,
        }

    async def _search_tavily(self, query: str, max_results: int) -> List[SourceDiscoveryItem]:
        """Search via Tavily Client asynchronously in a worker thread."""
        from tavily import TavilyClient

        def _sync_tavily():
            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            return client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=False,
            )

        response = await asyncio.to_thread(_sync_tavily)
        raw_results = response.get("results", [])

        items: List[SourceDiscoveryItem] = []
        for r in raw_results:
            url = r.get("url", "")
            title = r.get("title", "Untitled Source")
            snippet = r.get("content", "")
            publisher, source_type = infer_publisher_and_type(url, title)

            items.append(
                SourceDiscoveryItem(
                    source_id=generate_uuid(),
                    title=title,
                    url=url,
                    publisher=publisher,
                    source_type=source_type,
                    published_at=r.get("published_date"),
                    snippet=snippet,
                    score=r.get("score"),
                    metadata={"provider": "tavily", "raw": r},
                )
            )
        return items

    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SourceDiscoveryItem]:
        """Search via DuckDuckGo asynchronously using ddgs with automatic query simplification."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def _sync_ddg(q: str):
            with DDGS() as ddgs:
                return list(ddgs.text(q, max_results=max_results))

        raw_results = []
        try:
            raw_results = await asyncio.to_thread(_sync_ddg, query)
        except Exception as e:
            logger.warning(f"DDGS attempt failed on '{query}': {e}")

        # If zero results and query contained complex site/boolean filters, retry with clean core query
        if not raw_results and ("(" in query or "site:" in query.lower()):
            clean_q = re.sub(r"\(.*?\)", "", query).strip()
            clean_q = re.sub(r"site:\S+", "", clean_q).strip()
            if clean_q:
                logger.info(f"Retrying DDGS with simplified query: '{clean_q}'")
                try:
                    raw_results = await asyncio.to_thread(_sync_ddg, clean_q)
                except Exception as e:
                    logger.warning(f"Simplified DDGS search failed: {e}")

        items: List[SourceDiscoveryItem] = []
        for r in raw_results:
            url = r.get("href", "")
            title = r.get("title", "Untitled Source")
            snippet = r.get("body", "")
            publisher, source_type = infer_publisher_and_type(url, title)

            items.append(
                SourceDiscoveryItem(
                    source_id=generate_uuid(),
                    title=title,
                    url=url,
                    publisher=publisher,
                    source_type=source_type,
                    snippet=snippet,
                    metadata={"provider": "duckduckgo"},
                )
            )
        return items

    async def _search_wikipedia(self, query: str, max_results: int) -> List[SourceDiscoveryItem]:
        """Free, robust tertiary search via Wikipedia OpenSearch API."""
        import httpx
        clean_q = re.sub(r"\(.*?\)", "", query).strip()
        clean_q = re.sub(r"site:\S+", "", clean_q).strip()
        headers = {"User-Agent": settings.DOC_READER_USER_AGENT}

        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "opensearch",
                    "search": clean_q,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            titles = data[1] if len(data) > 1 else []
            descriptions = data[2] if len(data) > 2 else []
            urls = data[3] if len(data) > 3 else []

            items: List[SourceDiscoveryItem] = []
            for t, desc, u in zip(titles, descriptions, urls):
                items.append(
                    SourceDiscoveryItem(
                        source_id=generate_uuid(),
                        title=t,
                        url=u,
                        publisher="Wikipedia",
                        source_type=SourceType.EDUCATIONAL,
                        snippet=desc or f"Wikipedia overview article for {t}",
                        metadata={"provider": "wikipedia_opensearch"},
                    )
                )
            return items
