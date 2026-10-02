"""Unit tests for Evidence Graph representation and relational edges."""

from backend.app.evidence.graph import EvidenceGraphService
from backend.app.schemas.claim import Claim
from backend.app.schemas.common import DocumentType, EntityType, RelationshipType, SourceType, generate_uuid
from backend.app.schemas.document import Document, DocumentChunk
from backend.app.schemas.evidence import Evidence
from backend.app.schemas.source import Source


def test_evidence_graph_generation(evidence_store, in_memory_db_session):
    graph_svc = EvidenceGraphService(db_session_factory=lambda: in_memory_db_session)

    # Setup graph topology
    sess = evidence_store.create_session("Solar Energy India", "How has solar grown?")
    src = Source(
        id=generate_uuid(),
        url="https://mnre.gov.in/solar",
        title="MNRE Solar Report",
        source_type=SourceType.GOVERNMENT,
    )
    evidence_store.save_source(src)

    doc_id = generate_uuid()
    doc = Document(
        id=doc_id,
        source_id=src.id,
        title="Solar Report 2024",
        content="Tariffs decreased below 2.50 INR per unit.",
        document_type=DocumentType.HTML,
    )
    evidence_store.save_document(doc)

    claim = Claim(
        id=generate_uuid(),
        session_id=sess.id,
        text="Solar tariffs in India dropped below 2.50 INR/kWh.",
    )
    evidence_store.save_claim(claim)

    ev = Evidence(
        id=generate_uuid(),
        claim_id=claim.id,
        source_id=src.id,
        document_id=doc.id,
        chunk_id="chk-1",
        text="Tariffs decreased below 2.50 INR per unit.",
        location="Page 1",
        supports=True,
    )
    evidence_store.save_evidence(ev)

    # Link an Entity to the Claim
    entity = graph_svc.create_entity(
        name="Solar Photovoltaic Tariff",
        entity_type=EntityType.METRIC,
        description="Electricity generation cost metric",
    )
    graph_svc.link_claim_to_entity(claim.id, entity.id)

    # Query graph
    graph_data = graph_svc.get_evidence_graph(session_id=sess.id)
    assert len(graph_data.nodes) >= 3  # Source, Document, Evidence, Claim
    assert len(graph_data.edges) >= 3

    edge_relations = {e.relation for e in graph_data.edges}
    assert "supports" in edge_relations
    assert "contains" in edge_relations
    assert "about" in edge_relations
