"""Credibility assessment schemas with transparent factor evaluation."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import Field
from backend.app.schemas.common import BaseSchema, generate_uuid, utc_now


class CredibilityFactorScores(BaseSchema):
    authority: float = Field(ge=0.0, le=1.0, description="Domain authority, institutional reputation")
    primary_source: float = Field(ge=0.0, le=1.0, description="Original data/finding vs secondary reporting")
    recency: float = Field(ge=0.0, le=1.0, description="Publication freshness relative to field velocity")
    evidence_quality: float = Field(ge=0.0, le=1.0, description="Methodology, data citations, quantitative rigor")
    peer_review: Optional[float] = Field(None, ge=0.0, le=1.0, description="Formal peer-review status if applicable")
    corroboration: float = Field(ge=0.0, le=1.0, description="Consensus with other independent sources")


class CredibilityAssessmentBase(BaseSchema):
    source_id: str
    authority_score: float = Field(ge=0.0, le=1.0)
    primary_source_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    evidence_quality_score: float = Field(ge=0.0, le=1.0)
    peer_review_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    corroboration_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CredibilityAssessmentCreate(CredibilityAssessmentBase):
    pass


class CredibilityAssessment(CredibilityAssessmentBase):
    id: str = Field(default_factory=generate_uuid)
    assessed_at: datetime = Field(default_factory=utc_now)
