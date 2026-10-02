"""Deterministic document chunking preserving provenance and page/section boundaries."""

import hashlib
import re
from typing import Any, Dict, List, Optional

from backend.app.config import settings
from backend.app.schemas.document import Document, DocumentChunk


class DocumentChunker:
    """Chunks documents deterministically with boundary awareness and full provenance tracing."""

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def generate_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
        """Create a deterministic chunk ID based on document ID, index, and text content hash."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"chk_{document_id}_{chunk_index}_{content_hash}"

    def chunk_document(
        self,
        document: Document,
        source_url: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """Split a Document into deterministic chunks preserving provenance."""
        sid = source_id or document.source_id
        surl = source_url or document.metadata.get("url")

        content = document.content
        if not content or not content.strip():
            return []

        # If document has pages data stored in metadata, chunk by page
        pages_data = document.metadata.get("pages")
        if pages_data and isinstance(pages_data, list):
            return self._chunk_paged_document(
                document_id=document.id,
                pages_data=pages_data,
                source_id=sid,
                source_url=surl,
                doc_metadata=document.metadata,
            )

        # Standard linear text chunking with sentence/paragraph boundaries
        return self._chunk_linear_text(
            document_id=document.id,
            text=content,
            source_id=sid,
            source_url=surl,
            doc_metadata=document.metadata,
        )

    def _chunk_paged_document(
        self,
        document_id: str,
        pages_data: List[Dict[str, Any]],
        source_id: str,
        source_url: Optional[str],
        doc_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Chunk a document that has page-level segmentation (e.g. PDF)."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages_data:
            page_num = page.get("page_number", 1)
            page_text = page.get("text", "").strip()
            if not page_text:
                continue

            page_chunks = self._split_text(page_text)
            for chunk_str in page_chunks:
                chunk_id = self.generate_chunk_id(document_id, global_chunk_idx, chunk_str)
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=document_id,
                        text=chunk_str,
                        chunk_index=global_chunk_idx,
                        page_number=page_num,
                        section=f"Page {page_num}",
                        source_id=source_id,
                        source_url=source_url,
                        metadata={
                            "page": page_num,
                            "char_count": len(chunk_str),
                            "doc_title": doc_metadata.get("title", ""),
                            "source_id": source_id,
                            "source_url": source_url,
                        },
                    )
                )
                global_chunk_idx += 1

        return chunks

    def _chunk_linear_text(
        self,
        document_id: str,
        text: str,
        source_id: str,
        source_url: Optional[str],
        doc_metadata: Dict[str, Any],
    ) -> List[DocumentChunk]:
        """Chunk standard un-paged text (e.g. HTML/plain text) with boundary preservation."""
        chunks: List[DocumentChunk] = []
        raw_chunks = self._split_text(text)

        for idx, chunk_str in enumerate(raw_chunks):
            chunk_id = self.generate_chunk_id(document_id, idx, chunk_str)
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    document_id=document_id,
                    text=chunk_str,
                    chunk_index=idx,
                    page_number=None,
                    section=f"Section {idx + 1}",
                    source_id=source_id,
                    source_url=source_url,
                    metadata={
                        "char_count": len(chunk_str),
                        "doc_title": doc_metadata.get("title", ""),
                        "source_id": source_id,
                        "source_url": source_url,
                    },
                )
            )

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text into segments of roughly chunk_size with chunk_overlap, breaking on paragraphs/sentences."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_len = 0

        for p in paragraphs:
            # If paragraph itself is longer than chunk_size, split by sentences
            if len(p) > self.chunk_size:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                for s in sentences:
                    if current_len + len(s) + 1 > self.chunk_size and current_chunk_parts:
                        chunk_text = " ".join(current_chunk_parts)
                        chunks.append(chunk_text)
                        # Build overlap from the tail of current parts
                        overlap_parts = []
                        overlap_len = 0
                        for part in reversed(current_chunk_parts):
                            if overlap_len + len(part) <= self.chunk_overlap:
                                overlap_parts.insert(0, part)
                                overlap_len += len(part)
                            else:
                                break
                        current_chunk_parts = overlap_parts
                        current_len = sum(len(x) + 1 for x in current_chunk_parts)

                    current_chunk_parts.append(s)
                    current_len += len(s) + 1
            else:
                if current_len + len(p) + 1 > self.chunk_size and current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts)
                    chunks.append(chunk_text)
                    current_chunk_parts = []
                    current_len = 0

                current_chunk_parts.append(p)
                current_len += len(p) + 1

        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        return [c.strip() for c in chunks if c.strip()]
