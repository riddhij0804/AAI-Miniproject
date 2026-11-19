"""Web search tool supporting Tavily primary with DuckDuckGo fallback."""

import asyncio
from datetime import datetime, timezone
import logging
import re
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
                if results:
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

        # 3. For biomedical/clinical queries or when DDG returned 0, query NCBI / PubMed Central (PMC)
        is_biomedical = any(k in params.query.lower() for k in [
            "cause", "treatment", "symptom", "disease", "syndrome", "therapy",
            "clinical", "etiology", "cancer", "health", "medical", "patient",
            "drug", "diagnosis", "disorder", "biological", "pharmaceutical", "pathology"
        ])
        if (is_biomedical or not results) and len(results) < params.max_results:
            try:
                pmc_results = await self._search_pubmed(params.query, params.max_results - len(results))
                if pmc_results:
                    results.extend(pmc_results)
                    if provider_used == "none":
                        provider_used = "ncbi_pmc"
                    logger.info(f"PubMed PMC search added {len(pmc_results)} scientific sources for query '{params.query}'")
            except Exception as e:
                err = f"PubMed search failed: {e}"
                logger.warning(err)
                errors.append(err)

        # 4. Tertiary fallback to Wikipedia Full-Text Search if more sources needed
        if len(results) < params.max_results:
            try:
                wiki_results = await self._search_wikipedia(params.query, params.max_results - len(results))
                if wiki_results:
                    results.extend(wiki_results)
                    if provider_used == "none":
                        provider_used = "wikipedia_search"
                    logger.info(f"Wikipedia full-text search added {len(wiki_results)} results")
            except Exception as e:
                err = f"Wikipedia full-text search failed: {e}"
                logger.warning(err)
                errors.append(err)

        # 4. Filter out dictionary spam, definition sites, and translation scrapers
        # unless user query specifically asks for a definition
        is_def_query = any(w in params.query.lower() for w in ["define", "definition", "meaning of", "what does"])
        if not is_def_query and results:
            noise_domains = (
                "merriam-webster.com", "dictionary.cambridge.org",
                "dictionary.com", "wiktionary.org", "thefreedictionary.com",
                "collinsdictionary.com", "vocabulary.com", "dictall.com",
                "iask.sina.com.cn", "iciba.com", "youdao.com", "dict.cn",
                "thesaurus.com", "synonyms.com", "wordhippo.com",
            )
            clean_results = [
                r for r in results
                if not any(noise in r.url.lower() for noise in noise_domains)
            ]
            if clean_results:
                results = clean_results

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
        """Search via DuckDuckGo asynchronously with multi-backend resilience for datacenter IPs."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def _sync_ddg(q: str):
            # Try multiple backends — default HTML scraping gets Cloudflare-blocked from datacenter IPs
            for backend in ("lite", "api", "html"):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(q, max_results=max_results, backend=backend))
                    if results:
                        logger.info(f"DDG backend '{backend}' returned {len(results)} results")
                        return results
                except TypeError:
                    # Older library version doesn't support backend param — use default once
                    try:
                        with DDGS() as ddgs:
                            return list(ddgs.text(q, max_results=max_results))
                    except Exception:
                        return []
                except Exception as e:
                    logger.debug(f"DDG backend '{backend}' failed: {e}")
                    continue
            return []

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
        """Authoritative search via Wikipedia Full-Text Search API with relevance filtering."""
        import httpx
        clean_q = re.sub(r"\(.*?\)", "", query).strip()
        clean_q = re.sub(r"site:\S+", "", clean_q).strip()
        headers = {"User-Agent": settings.DOC_READER_USER_AGENT}

        # Stop words for relevance scoring
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "between", "out", "over", "under",
            "again", "then", "here", "there", "when", "where", "why", "how", "all", "each",
            "both", "more", "most", "other", "some", "no", "not", "only", "so", "than",
            "very", "just", "and", "or", "if", "but", "what", "which", "who", "it", "its",
            "this", "that", "about", "up", "down", "many", "much",
        }

        def _stem(w: str) -> str:
            """Basic suffix stripping for plural/verb matching."""
            for sfx in ("tion", "sion", "ment", "ness", "ity", "ing", "ies", "ous", "ive", "ed", "er", "ly", "es", "s"):
                if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                    return w[:-len(sfx)]
            return w

        query_words = set(re.findall(r"\w+", clean_q.lower())) - _STOP
        query_stems = {_stem(w) for w in query_words}
        min_overlap = 2 if len(query_words) >= 3 else 1

        # Over-fetch for filtering headroom
        fetch_limit = max(max_results * 3, 15)

        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": clean_q,
                    "srlimit": fetch_limit,
                    "utf8": "1",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            search_items = data.get("query", {}).get("search", [])

            # Hard filter: exclude disambiguation, media, list pages
            hard_ok = []
            for item in search_items:
                t = item.get("title", "")
                t_lower = t.lower()
                if "(disambiguation)" in t_lower or t_lower.startswith("list of "):
                    continue
                if any(k in t_lower for k in ["(film)", "(album)", "(song)", "(tv series)", "(character)", "(band)", "(video game)"]):
                    continue
                hard_ok.append(item)

            # Relevance scoring by stem overlap with title
            scored = []
            for item in hard_ok:
                t = item.get("title", "")
                title_words = set(re.findall(r"\w+", t.lower())) - _STOP
                title_stems = {_stem(w) for w in title_words}
                overlap = len(query_stems & title_stems)
                scored.append((overlap, item))

            scored.sort(key=lambda x: x[0], reverse=True)

            # Apply soft relevance threshold
            passing = [(sc, it) for sc, it in scored if sc >= min_overlap]

            # Fallback: if nothing passes, keep top 2 from CirrusSearch ranking
            if not passing and scored:
                passing = scored[:min(2, max_results)]

            items: List[SourceDiscoveryItem] = []
            for _, item in passing[:max_results]:
                t = item.get("title", "")
                raw_snippet = item.get("snippet", "")
                clean_snippet = re.sub(r"<.*?>", "", raw_snippet).strip()
                article_url = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"

                items.append(
                    SourceDiscoveryItem(
                        source_id=generate_uuid(),
                        title=t,
                        url=article_url,
                        publisher="Wikipedia",
                        source_type=SourceType.EDUCATIONAL,
                        snippet=clean_snippet or f"Wikipedia overview article for {t}",
                        metadata={"provider": "wikipedia_search", "wordcount": item.get("wordcount")},
                    )
                )
            return items

    async def _search_pubmed(self, query: str, max_results: int) -> List[SourceDiscoveryItem]:
        """Free, authoritative biomedical search via NCBI / PubMed Central (PMC) Open Access API."""
        import httpx
        clean_q = re.sub(r"\(.*?\)", "", query).strip()
        clean_q = re.sub(r"site:\S+", "", clean_q).strip()
        headers = {"User-Agent": settings.DOC_READER_USER_AGENT}

        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            # 1. Query PMC for relevant IDs
            search_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pmc",
                    "term": f"{clean_q} AND open access[filter]",
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "pub_date",
                },
            )
            if search_resp.status_code != 200:
                return []
            id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # 2. Query PMC for article summary data
            summary_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pmc",
                    "id": ",".join(id_list),
                    "retmode": "json",
                },
            )
            if summary_resp.status_code != 200:
                return []

            result_dict = summary_resp.json().get("result", {})
            items: List[SourceDiscoveryItem] = []
            for pmc_id in id_list:
                doc = result_dict.get(pmc_id)
                if not doc or not isinstance(doc, dict):
                    continue
                title = doc.get("title", f"PMC Article {pmc_id}")
                source_journal = doc.get("source", "PubMed Central")
                pub_date = doc.get("pubdate")
                article_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"

                items.append(
                    SourceDiscoveryItem(
                        source_id=generate_uuid(),
                        title=title,
                        url=article_url,
                        publisher=f"NCBI / {source_journal}",
                        source_type=SourceType.PEER_REVIEWED,
                        published_at=pub_date,
                        snippet=f"Peer-reviewed study from {source_journal} ({pub_date}): {title}",
                        metadata={"provider": "ncbi_pmc", "pmc_id": pmc_id},
                    )
                )
            return items
