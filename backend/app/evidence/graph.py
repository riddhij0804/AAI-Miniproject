"""Evidence Graph service constructing node-and-edge graphs from research entities and relationships."""

import logging
from typing import Dict, List, Optional, Set
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models.entities import (
    ClaimModel,
    DocumentChunkModel,
    DocumentModel,
    EntityModel,
    EvidenceModel,
    RelationshipModel,
    SourceModel,
)
from backend.app.schemas.common import EntityType, RelationshipType, generate_uuid, utc_now
from backend.app.schemas.graph import Entity, GraphData, GraphEdge, GraphNode, Relationship

logger = logging.getLogger(__name__)


class EvidenceGraphService:
    """Constructs and queries the research evidence graph across sources, documents, evidence, claims, and entities."""

    def __init__(self, db_session_factory=SessionLocal):
        self.session_factory = db_session_factory

    def record_relationship(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        relation_type: RelationshipType,
        weight: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> Relationship:
        """Persist an explicit graph edge in the relationships table."""
        with self.session_factory() as session:
            rel = Relationship(
                source_node_type=source_type,
                source_node_id=source_id,
                target_node_type=target_type,
                target_node_id=target_id,
                relation_type=relation_type,
                weight=weight,
                metadata=metadata or {},
            )
            model = RelationshipModel(
                id=rel.id,
                source_node_type=rel.source_node_type,
                source_node_id=rel.source_node_id,
                target_node_type=rel.target_node_type,
                target_node_id=rel.target_node_id,
                relation_type=rel.relation_type.value,
                weight=rel.weight,
                created_at=rel.created_at,
                metadata_json=rel.metadata,
            )
            session.add(model)
            session.commit()
            return rel

    def create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.CONCEPT,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Entity:
        """Create a domain entity node (concept, pathogen, technology, etc.)."""
        with self.session_factory() as session:
            entity = Entity(
                name=name,
                entity_type=entity_type,
                description=description,
                metadata=metadata or {},
            )
            model = EntityModel(
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type.value,
                description=entity.description,
                created_at=entity.created_at,
                metadata_json=entity.metadata,
            )
            session.add(model)
            session.commit()
            return entity

    def link_claim_to_entity(self, claim_id: str, entity_id: str) -> Relationship:
        """Record a Claim --about--> Entity edge."""
        return self.record_relationship(
            source_type="claim",
            source_id=claim_id,
            target_type="entity",
            target_id=entity_id,
            relation_type=RelationshipType.ABOUT,
        )

    def link_claims(self, claim_a_id: str, claim_b_id: str, relation: RelationshipType = RelationshipType.RELATED_TO) -> Relationship:
        """Record a Claim --related_to--> Claim edge."""
        return self.record_relationship(
            source_type="claim",
            source_id=claim_a_id,
            target_type="claim",
            target_id=claim_b_id,
            relation_type=relation,
        )

    def get_evidence_graph(
        self,
        session_id: Optional[str] = None,
        claim_id: Optional[str] = None,
    ) -> GraphData:
        """Assemble the complete graph of nodes and edges for visualization or downstream reasoning."""
        nodes: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        with self.session_factory() as session:
            # 1. Fetch claims matching filter
            claim_query = select(ClaimModel)
            if session_id:
                claim_query = claim_query.where(ClaimModel.session_id == session_id)
            if claim_id:
                claim_query = claim_query.where(ClaimModel.id == claim_id)

            claims = session.execute(claim_query).scalars().all()
            for c in claims:
                nodes[c.id] = GraphNode(
                    id=c.id,
                    label=c.text[:60] + "..." if len(c.text) > 60 else c.text,
                    type="claim",
                    properties={"confidence": c.confidence, "status": c.status, "full_text": c.text},
                )

            # 2. Fetch evidence for claims
            claim_ids = list(nodes.keys())
            if claim_ids:
                ev_query = select(EvidenceModel).where(EvidenceModel.claim_id.in_(claim_ids))
            else:
                ev_query = select(EvidenceModel).limit(100)

            evidences = session.execute(ev_query).scalars().all()
            doc_ids_needed: Set[str] = set()

            for ev in evidences:
                nodes[ev.id] = GraphNode(
                    id=ev.id,
                    label=f"Evidence ({ev.location})",
                    type="evidence",
                    properties={"text": ev.text, "supports": ev.supports, "confidence": ev.extraction_confidence},
                )
                if ev.claim_id and ev.claim_id in nodes:
                    relation = "supports" if ev.supports else "contradicts"
                    edges.append(
                        GraphEdge(
                            source=ev.id,
                            target=ev.claim_id,
                            relation=relation,
                            weight=ev.extraction_confidence,
                            properties={"location": ev.location},
                        )
                    )
                if ev.document_id:
                    doc_ids_needed.add(ev.document_id)
                    # Document --contains--> Evidence
                    edges.append(
                        GraphEdge(
                            source=ev.document_id,
                            target=ev.id,
                            relation="contains",
                            weight=1.0,
                        )
                    )

            # 3. Fetch documents
            source_ids_needed: Set[str] = set()
            if doc_ids_needed:
                doc_query = select(DocumentModel).where(DocumentModel.id.in_(list(doc_ids_needed)))
                documents = session.execute(doc_query).scalars().all()
                for doc in documents:
                    nodes[doc.id] = GraphNode(
                        id=doc.id,
                        label=doc.title[:50],
                        type="document",
                        properties={"type": doc.document_type, "created_at": doc.created_at.isoformat()},
                    )
                    source_ids_needed.add(doc.source_id)
                    # Source --contains--> Document
                    edges.append(
                        GraphEdge(
                            source=doc.source_id,
                            target=doc.id,
                            relation="contains",
                            weight=1.0,
                        )
                    )

            # 4. Fetch sources
            if source_ids_needed:
                src_query = select(SourceModel).where(SourceModel.id.in_(list(source_ids_needed)))
                sources = session.execute(src_query).scalars().all()
                for src in sources:
                    nodes[src.id] = GraphNode(
                        id=src.id,
                        label=src.title[:50] or src.url[:50],
                        type="source",
                        properties={
                            "url": src.url,
                            "publisher": src.publisher,
                            "credibility_score": src.credibility_score,
                            "source_type": src.source_type,
                        },
                    )

            # 5. Fetch explicit relationships (Entity links, Claim-to-Claim links)
            rel_query = select(RelationshipModel)
            if claim_ids:
                rel_query = rel_query.where(
                    (RelationshipModel.source_node_id.in_(claim_ids))
                    | (RelationshipModel.target_node_id.in_(claim_ids))
                )
            relationships = session.execute(rel_query).scalars().all()
            for r in relationships:
                edges.append(
                    GraphEdge(
                        id=r.id,
                        source=r.source_node_id,
                        target=r.target_node_id,
                        relation=r.relation_type,
                        weight=r.weight,
                        properties=r.metadata_json,
                    )
                )

        return GraphData(
            nodes=list(nodes.values()),
            edges=edges,
        )


# Global default graph service
default_graph_service = EvidenceGraphService()
