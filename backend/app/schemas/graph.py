"""Graph and entity schemas for representing research evidence relationships."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import Field
from backend.app.schemas.common import (
    BaseSchema,
    EntityType,
    RelationshipType,
    generate_uuid,
    utc_now,
)


class EntityBase(BaseSchema):
    name: str
    entity_type: EntityType = EntityType.CONCEPT
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EntityCreate(EntityBase):
    pass


class Entity(EntityBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class RelationshipBase(BaseSchema):
    source_node_type: str
    source_node_id: str
    target_node_type: str
    target_node_id: str
    relation_type: RelationshipType
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RelationshipCreate(RelationshipBase):
    pass


class Relationship(RelationshipBase):
    id: str = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=utc_now)


class GraphNode(BaseSchema):
    id: str
    label: str
    type: str  # source, document, evidence, claim, entity
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseSchema):
    id: str = Field(default_factory=generate_uuid)
    source: str
    target: str
    relation: str  # contains, supports, contradicts, about, related_to
    weight: float = 1.0
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseSchema):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
