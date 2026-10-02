"""Claim schemas for research assertions and conclusions."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field
from backend.app.schemas.common import BaseSchema, ClaimStatus, generate_uuid, utc_now


class ClaimBase(BaseSchema):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    status: ClaimStatus = ClaimStatus.EXTRACTED
    source_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    research_task_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimCreate(ClaimBase):
    pass


class Claim(ClaimBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)
