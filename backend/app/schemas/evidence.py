"""Evidence, Citation, and Contradiction schemas."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import Field
from backend.app.schemas.common import BaseSchema, generate_uuid, utc_now


class EvidenceBase(BaseSchema):
    claim_id: Optional[str] = None
    source_id: str
    document_id: str
    chunk_id: str
    text: str = Field(..., description="Exact textual excerpt from the document chunk")
    location: str = Field(..., description="Location in source, e.g. 'page 12', 'section 2'")
    supports: bool = Field(default=True, description="True if evidence supports claim, False if refutes")
    extraction_confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceCreate(EvidenceBase):
    pass


class Evidence(EvidenceBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class CitationBase(BaseSchema):
    claim_id: str
    evidence_id: str
    source_id: str
    formatted_citation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationCreate(CitationBase):
    pass


class Citation(CitationBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class ContradictionBase(BaseSchema):
    claim_a_id: str
    claim_b_id: str
    evidence_a_id: Optional[str] = None
    evidence_b_id: Optional[str] = None
    reasoning: str
    severity: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContradictionCreate(ContradictionBase):
    pass


class Contradiction(ContradictionBase):
    id: str = Field(default_factory=generate_uuid)
    detected_at: datetime = Field(default_factory=utc_now)
