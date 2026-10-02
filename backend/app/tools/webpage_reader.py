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

        # 2. Fetch page using httpx with size limits & streaming
        timeout = params.timeout_seconds or self.timeout
        headers = {"User-Agent": settings.DOC_READER_USER_AGENT}

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            try:
                response = await client.get(params.url)
            except httpx.HTTPError as e:
                raise RuntimeError(f"HTTP request failed for '{params.url}': {e}") from e

            if response.status_code >= 400:
                raise RuntimeError(f"HTTP error {response.status_code} fetching '{params.url}'")

            content_bytes = response.content
            if len(content_bytes) > settings.DOC_READER_MAX_BYTES:
                raise ValueError(
                    f"Webpage content exceeds maximum allowed size "
                    f"({len(content_bytes)} > {settings.DOC_READER_MAX_BYTES} bytes)"
                )

            html_text = response.text

        # 3. Parse HTML and extract clean readable text
        extracted_data = self._parse_html(html_text, params.url)

        # 4. Check for prompt injection in extracted text (warn and sanitize)
        has_injection, pattern = PromptInjectionGuard.contains_injection_pattern(extracted_data["content"])
        if has_injection:
            logger.warning(
                f"Suspicious prompt injection pattern detected in {params.url}: '{pattern}'. "
                f"Isolating as untrusted data."
            )
            extracted_data["metadata"]["prompt_injection_flag"] = True
            extracted_data["metadata"]["injection_pattern"] = pattern

        sanitized_content = PromptInjectionGuard.sanitize_untrusted_content(extracted_data["content"])
        extracted_data["content"] = sanitized_content

        return extracted_data
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
        for element in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form", "svg"]):
            element.decompose()

        # Target main content containers if available
        content_container = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=lambda c: c and any(k in str(c).lower() for k in ["content", "post", "entry"]))
            or soup.body
            or soup
        )

        # Extract clean text
        lines = (line.strip() for line in content_container.get_text().splitlines())
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
