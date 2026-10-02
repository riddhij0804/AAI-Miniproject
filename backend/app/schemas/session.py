"""Research session, task, and iteration schemas for memory persistence."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field
from backend.app.schemas.common import BaseSchema, generate_uuid, utc_now
from backend.app.schemas.source import Source
from backend.app.schemas.document import Document
from backend.app.schemas.claim import Claim
from backend.app.schemas.evidence import Evidence


class ResearchIterationBase(BaseSchema):
    session_id: str
    iteration_number: int = 1
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchIterationCreate(ResearchIterationBase):
    pass


class ResearchIteration(ResearchIterationBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class ResearchTaskBase(BaseSchema):
    session_id: str
    query: str
    research_objective: str
    status: str = "pending"
    source_preferences: List[str] = Field(default_factory=list)
    iteration_index: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchTaskCreate(ResearchTaskBase):
    pass


class ResearchTask(ResearchTaskBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class ResearchSessionBase(BaseSchema):
    title: str
    original_question: str
    status: str = "active"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResearchSessionCreate(ResearchSessionBase):
    pass


class ResearchSession(ResearchSessionBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)
    tasks: List[ResearchTask] = Field(default_factory=list)
    iterations: List[ResearchIteration] = Field(default_factory=list)


class ResearchContextView(BaseSchema):
    """Full snapshot of research memory for follow-up query context."""
    session: ResearchSession
    tasks: List[ResearchTask] = Field(default_factory=list)
    iterations: List[ResearchIteration] = Field(default_factory=list)
    sources: List[Source] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
