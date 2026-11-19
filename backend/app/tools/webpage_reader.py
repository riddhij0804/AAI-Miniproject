"""Webpage reader tool for fetching, sanitizing, and extracting readable HTML content."""

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.schemas.common import DocumentType, generate_uuid
from backend.app.schemas.document import Document
from backend.app.tools.base import BaseTool
from backend.app.tools.security import PromptInjectionGuard, SecurityError, URLValidator

logger = logging.getLogger(__name__)


class WebpageReaderInput(BaseModel):
    url: str = Field(..., description="Target webpage URL to read")
    source_id: Optional[str] = Field(None, description="Associated Source UUID if known")
    timeout_seconds: Optional[float] = Field(None, description="Custom timeout override")


class WebpageReaderTool(BaseTool):
    """Secure webpage reading tool with HTML cleaning and metadata extraction."""

    name: str = "read_webpage"
    description: str = "Fetch a webpage, remove noise/boilerplate, and extract clean readable text."
    input_schema = WebpageReaderInput
    timeout: float = settings.DOC_READER_TIMEOUT_SECONDS

    async def _execute(self, params: WebpageReaderInput) -> Dict[str, Any]:
        # 1. SSRF & URL safety check
        is_safe, error_reason = URLValidator.is_safe_url(params.url)
        if not is_safe:
            raise SecurityError(f"Rejected unsafe URL '{params.url}': {error_reason}")

        # 2. Check if Wikipedia URL - use official extracts API for 100% reliability and bot immunity
        timeout = params.timeout_seconds or self.timeout
        if "wikipedia.org/wiki/" in params.url.lower():
            wiki_data = await self._fetch_wikipedia_extract(params.url, timeout)
            if wiki_data:
                return wiki_data

        # 3. Fetch page using httpx with size limits & streaming
        headers = {
            "User-Agent": settings.DOC_READER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }

        fetch_url = params.url
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=headers) as client:
            for _ in range(3):
                try:
                    response = await client.get(fetch_url)
                except httpx.HTTPError as e:
                    raise RuntimeError(f"HTTP request failed for '{fetch_url}': {e}") from e
                if not response.is_redirect:
                    break
                location = response.headers.get("location")
                redirect_url = str(httpx.URL(fetch_url).join(location)) if location else ""
                is_safe, error_reason = URLValidator.is_safe_url(redirect_url)
                if not is_safe:
                    raise SecurityError(f"Rejected unsafe redirect: {error_reason}")
                fetch_url = redirect_url
            else:
                raise RuntimeError(f"Too many redirects fetching '{params.url}'")

            if response.status_code >= 400:
                raise RuntimeError(f"HTTP error {response.status_code} fetching '{fetch_url}'")

            content_bytes = response.content
            if len(content_bytes) > settings.DOC_READER_MAX_BYTES:
                raise ValueError(
                    f"Webpage content exceeds maximum allowed size "
                    f"({len(content_bytes)} > {settings.DOC_READER_MAX_BYTES} bytes)"
                )

            html_text = response.text

        # 3. Parse HTML and extract clean readable text
        extracted_data = self._parse_html(html_text, fetch_url)

        # 4. Check for prompt injection in extracted text (warn and sanitize)
        security_decision = PromptInjectionGuard.assess_text(extracted_data["content"])
        if security_decision["outcome"] != "SAFE":
            logger.warning(
                f"Suspicious prompt injection pattern detected in {params.url}: "
                f"'{security_decision['matched_pattern']}'. "
                f"Isolating as untrusted data."
            )
            extracted_data["metadata"]["prompt_injection_flag"] = True
            extracted_data["metadata"]["injection_pattern"] = security_decision["matched_pattern"]
        extracted_data["metadata"]["security_decision"] = security_decision

        sanitized_content = PromptInjectionGuard.sanitize_untrusted_content(extracted_data["content"])
        extracted_data["content"] = sanitized_content

        return extracted_data
    async def _fetch_wikipedia_extract(self, url: str, timeout: float) -> Optional[Dict[str, Any]]:
        """Fetch article extract directly from Wikipedia API to avoid bot blocking and HTML scraping issues."""
        from urllib.parse import unquote, urlparse
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split("/wiki/")
            if len(path_parts) < 2:
                return None
            raw_title = unquote(path_parts[-1].split("#")[0]).strip()
            if not raw_title:
                return None

            api_url = f"https://{parsed.netloc}/w/api.php"
            headers = {"User-Agent": settings.WIKIPEDIA_USER_AGENT, "Accept": "application/json"}
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                resp = await client.get(
                    api_url,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": "1",
                        "titles": raw_title,
                        "format": "json",
                        "redirects": "1",
                    },
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page_id, page_info in pages.items():
                    if str(page_id) == "-1":
                        continue
                    extract = page_info.get("extract", "").strip()
                    article_title = page_info.get("title", raw_title.replace("_", " "))
                    if len(extract) > 100:
                        logger.info(f"Successfully loaded Wikipedia article '{article_title}' via API ({len(extract)} chars)")
                        return {
                            "title": article_title,
                            "url": url,
                            "content": extract[:settings.DOC_READER_MAX_BYTES],
                            "document_type": DocumentType.HTML.value,
                            "author": "Wikipedia Contributors",
                            "publication_date": None,
                            "metadata": {
                                "url": url,
                                "provider": "wikipedia_api",
                                "character_count": len(extract),
                                "extracted_at": datetime.now(timezone.utc).isoformat(),
                            },
                        }
        except Exception as e:
            logger.warning(f"Failed to fetch Wikipedia extract for '{url}': {e}")
        return None

    def _parse_html(self, html: str, url: str) -> Dict[str, Any]:
        """Extract title, metadata, and clean body text from HTML."""
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = "Untitled Webpage"
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            h1 = soup.find("h1")
            if h1 and h1.text:
                title = h1.text.strip()

        # Extract metadata (author, published time)
        author: Optional[str] = None
        author_meta = soup.find("meta", attrs={"name": lambda x: x and x.lower() in ("author", "byl", "article:author")})
        if author_meta and author_meta.get("content"):
            author = author_meta["content"].strip()

        published_date: Optional[str] = None
        date_meta = soup.find(
            "meta",
            attrs={"property": lambda x: x and x.lower() in ("article:published_time", "og:pubdate", "datePublished")},
        )
        if not date_meta:
            date_meta = soup.find("meta", attrs={"name": lambda x: x and x.lower() in ("date", "pubdate")})
        if date_meta and date_meta.get("content"):
            published_date = date_meta["content"].strip()

        # Strip non-content and noisy elements
        noisy_selectors = [
            "script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg",
            ".hatnote", ".mw-disambig", ".mw-editsection", ".toc", "#toc", ".navbox",
            ".vertical-navbox", ".sidebar", ".catlinks", ".printfooter", ".mw-indicators",
            "#siteNotice", ".mw-jump-link", ".vector-header", ".vector-column-start",
            ".vector-dropdown", "#mw-navigation", "#p-personal", "#p-navigation", "#p-tb",
            "#article-nav", "#article-banner", "#article-left", ".menu",
            "[role='navigation']", "[aria-label*='navigation' i]",
            ".social-share", ".cookie-banner", ".breadcrumbs", ".breadcrumb"
        ]
        for sel in noisy_selectors:
            for element in soup.select(sel):
                element.decompose()

        # Target main content containers if available
        content_container = (
            soup.find("div", {"id": "mw-content-text"})
            or soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=lambda c: c and any(k in str(c).lower() for k in ["content", "post", "entry"]))
            or soup.body
            or soup
        )

        # Extract clean text, filtering noisy lines
        noise_prefixes = (
            "from wikipedia, the free encyclopedia",
            "for other uses, see",
            "for other people named",
            "jump to navigation",
            "jump to search",
            "table of contents",
            "stanford encyclopedia of philosophy menu",
        )
        lines = []
        for raw_line in content_container.get_text().splitlines():
            line_str = raw_line.strip()
            if not line_str:
                continue
            lower_line = line_str.lower()
            if any(lower_line.startswith(prefix) for prefix in noise_prefixes):
                continue
            lines.append(line_str)

        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = "\n".join(chunk for chunk in chunks if chunk)

        return {
            "title": title,
            "url": url,
            "content": clean_text,
            "document_type": DocumentType.HTML.value,
            "author": author,
            "publication_date": published_date,
            "metadata": {
                "url": url,
                "character_count": len(clean_text),
                "extracted_at": datetime.now(timezone.utc).isoformat(),
            },
        }
