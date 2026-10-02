"""Document and DocumentChunk schemas with provenance tracking."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field
from backend.app.schemas.common import BaseSchema, DocumentType, generate_uuid, utc_now


class DocumentChunkBase(BaseSchema):
    document_id: str
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    section: Optional[str] = None
    embedding_id: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunkCreate(DocumentChunkBase):
    pass


class DocumentChunk(DocumentChunkBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class DocumentBase(BaseSchema):
    source_id: str
    title: str
    content: str
    document_type: DocumentType = DocumentType.HTML
    publication_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentCreate(DocumentBase):
    pass


class Document(DocumentBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)
    chunks: List[DocumentChunk] = Field(default_factory=list)
