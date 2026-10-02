"""Common schemas and enumerations for the Research Intelligence subsystem."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    """Return current UTC datetime with timezone awareness."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate standard UUID4 string."""
    return str(uuid.uuid4())


class SourceType(str, Enum):
    ACADEMIC_JOURNAL = "academic_journal"
    PREPRINT = "preprint"
    GOVERNMENT = "government"
    EDUCATIONAL = "educational"
    NEWS = "news"
    ORGANIZATION = "organization"
    TECHNICAL_REPORT = "technical_report"
    WEBPAGE = "webpage"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    UNKNOWN = "unknown"


class ClaimStatus(str, Enum):
    EXTRACTED = "extracted"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    UNVERIFIED = "unverified"


class RelationshipType(str, Enum):
    CONTAINS = "contains"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ABOUT = "about"
    RELATED_TO = "related_to"


class EntityType(str, Enum):
    CONCEPT = "concept"
    PATHOGEN = "pathogen"
    TECHNOLOGY = "technology"
    ORGANIZATION = "organization"
    LOCATION = "location"
    METRIC = "metric"
    POLICY = "policy"
    EVENT = "event"
    PERSON = "person"
    OTHER = "other"


class BaseSchema(BaseModel):
    """Base Pydantic model with modern V2 configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )
