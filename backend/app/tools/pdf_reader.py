"""PDF reader tool for downloading, parsing, and extracting page-preserved PDF content."""

import asyncio
from datetime import datetime, timezone
import io
import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.schemas.common import DocumentType, generate_uuid
from backend.app.tools.base import BaseTool
from backend.app.tools.security import PromptInjectionGuard, SecurityError, URLValidator

logger = logging.getLogger(__name__)


class PDFReaderInput(BaseModel):
    url_or_path: str = Field(..., description="Target PDF URL or local filesystem path")
    source_id: Optional[str] = Field(None, description="Associated Source UUID if known")
    timeout_seconds: Optional[float] = Field(None, description="Custom timeout override")


class PDFReaderTool(BaseTool):
    """Secure PDF reader tool preserving page numbers and document metadata."""

    name: str = "read_pdf"
    description: str = "Download/read a PDF file and extract clean text with page number preservation."
    input_schema = PDFReaderInput
    timeout: float = settings.DOC_READER_TIMEOUT_SECONDS

    async def _execute(self, params: PDFReaderInput) -> Dict[str, Any]:
        target = params.url_or_path

        # 1. Fetch or load PDF bytes
        if target.startswith("http://") or target.startswith("https://"):
            is_safe, error_reason = URLValidator.is_safe_url(target)
            if not is_safe:
                raise SecurityError(f"Rejected unsafe URL '{target}': {error_reason}")

            timeout = params.timeout_seconds or self.timeout
            headers = {"User-Agent": settings.DOC_READER_USER_AGENT}
            fetch_url = target
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=headers) as client:
                for _ in range(3):
                    try:
                        response = await client.get(fetch_url)
                    except httpx.HTTPError as e:
                        raise RuntimeError(f"Failed to fetch PDF from '{fetch_url}': {e}") from e
                    if not response.is_redirect:
                        break
                    location = response.headers.get("location")
                    redirect_url = str(httpx.URL(fetch_url).join(location)) if location else ""
                    is_safe, error_reason = URLValidator.is_safe_url(redirect_url)
                    if not is_safe:
                        raise SecurityError(f"Rejected unsafe PDF redirect: {error_reason}")
                    fetch_url = redirect_url
                else:
                    raise RuntimeError(f"Too many redirects fetching '{target}'")

                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP error {response.status_code} fetching PDF '{fetch_url}'")

                pdf_bytes = response.content
        else:
            raise SecurityError("Only public HTTP(S) PDF URLs are accepted by the research API.")

        if len(pdf_bytes) > settings.DOC_READER_MAX_BYTES:
            raise ValueError(
                f"PDF file size exceeds limit ({len(pdf_bytes)} > {settings.DOC_READER_MAX_BYTES} bytes)"
            )

        if not pdf_bytes.startswith(b"%PDF"):
            raise ValueError(f"Target '{target}' does not appear to be a valid PDF (invalid magic header)")

        # 2. Extract page-by-page text
        return await asyncio.to_thread(self._extract_pdf_pages, pdf_bytes, fetch_url if target.startswith(("http://", "https://")) else target)

    def _extract_pdf_pages(self, pdf_bytes: bytes, target_ref: str) -> Dict[str, Any]:
        """Extract text page by page using PyMuPDF (fitz) with pypdf fallback."""
        pages_data: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []
        title: str = os.path.basename(target_ref)
        author: Optional[str] = None
        creation_date: Optional[str] = None
        extractor_used = "pymupdf"

        # Try PyMuPDF first
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            meta = doc.metadata or {}
            if meta.get("title") and meta["title"].strip():
                title = meta["title"].strip()
            if meta.get("author") and meta["author"].strip():
                author = meta["author"].strip()
            if meta.get("creationDate"):
                creation_date = meta["creationDate"]

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                page_number = page_idx + 1
                pages_data.append({
                    "page_number": page_number,
                    "text": page_text,
                    "character_count": len(page_text)
                })
                if page_text:
                    full_text_parts.append(f"--- [Page {page_number}] ---\n{page_text}")
            doc.close()
        except Exception as fitz_err:
            logger.warning(f"PyMuPDF failed on {target_ref} ({fitz_err}), attempting pypdf fallback...")
            extractor_used = "pypdf"
            pages_data = []
            full_text_parts = []
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                doc_info = reader.metadata
                if doc_info and doc_info.title:
                    title = doc_info.title
                if doc_info and doc_info.author:
                    author = doc_info.author

                for page_idx, page in enumerate(reader.pages):
                    page_number = page_idx + 1
                    page_text = (page.extract_text() or "").strip()
                    pages_data.append({
                        "page_number": page_number,
                        "text": page_text,
                        "character_count": len(page_text)
                    })
                    if page_text:
                        full_text_parts.append(f"--- [Page {page_number}] ---\n{page_text}")
            except Exception as pypdf_err:
                raise RuntimeError(
                    f"Both PyMuPDF and pypdf failed to extract PDF content: {fitz_err} | {pypdf_err}"
                ) from pypdf_err

        combined_text = "\n\n".join(full_text_parts)

        # Check for prompt injection
        has_injection, pattern = PromptInjectionGuard.contains_injection_pattern(combined_text)
        metadata = {
            "source_ref": target_ref,
            "total_pages": len(pages_data),
            "extractor": extractor_used,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        if has_injection:
            logger.warning(f"Prompt injection detected in PDF '{target_ref}': '{pattern}'")
            metadata["prompt_injection_flag"] = True
            metadata["injection_pattern"] = pattern

        sanitized_content = PromptInjectionGuard.sanitize_untrusted_content(combined_text)

        return {
            "title": title,
            "url": target_ref,
            "content": sanitized_content,
            "document_type": DocumentType.PDF.value,
            "author": author,
            "publication_date": creation_date,
            "pages": pages_data,
            "metadata": metadata,
        }
