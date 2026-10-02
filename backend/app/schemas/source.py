"""Source schemas for discovery and source management."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field, HttpUrl
from backend.app.schemas.common import BaseSchema, SourceType, generate_uuid, utc_now


class SourceBase(BaseSchema):
    url: str
    title: str
    publisher: Optional[str] = None
    source_type: SourceType = SourceType.WEBPAGE
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    credibility_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceCreate(SourceBase):
    pass


class Source(SourceBase):
    id: str = Field(default_factory=generate_uuid)
    retrieved_at: datetime = Field(default_factory=utc_now)


class SourceDiscoveryItem(BaseSchema):
    """Preliminary source item returned directly from web discovery (snippet != final evidence)."""
    source_id: str = Field(default_factory=generate_uuid)
    title: str
    url: str
    publisher: Optional[str] = None
    source_type: SourceType = SourceType.WEBPAGE
    published_at: Optional[str] = None
    snippet: str = Field(
        ...,
        description="Search snippet for relevance triage only; NOT considered verified evidence."
    )
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourceDiscoveryRequest(BaseSchema):
    task_id: str
    query: str
    research_objective: str
    source_preferences: List[str] = Field(default_factory=list)
    max_results: int = 5


class SourceDiscoveryResponse(BaseSchema):
    task_id: str
    query: str
    provider_used: str
    sources: List[SourceDiscoveryItem] = Field(default_factory=list)
    error_message: Optional[str] = None
