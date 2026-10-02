"""Unit tests for embeddings, vector store, and hybrid retrieval."""

import pytest
from backend.app.retrieval.embeddings import FastEmbedService, HashEmbeddingService
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.vector_store import QdrantVectorStore
from backend.app.schemas.document import DocumentChunk


def test_hash_embedding_properties():
    svc = HashEmbeddingService(dimension=384)
    v1 = svc.embed_text("Antimicrobial resistance in clinical hospitals")
    v2 = svc.embed_text("Clinical hospital antimicrobial resistance")
    v3 = svc.embed_text("Solar PV tariff trends in Gujarat India")

    assert len(v1) == 384
    assert len(v2) == 384

    # v1 and v2 should have high cosine similarity (close to 1.0)
    import numpy as np
    sim_similar = np.dot(v1, v2)
    sim_unrelated = np.dot(v1, v3)

    assert sim_similar > 0.85
    assert sim_similar > sim_unrelated


def test_qdrant_vector_store_in_memory():
    vs = QdrantVectorStore(location=":memory:", collection_name="test_qdrant")
    svc = HashEmbeddingService(dimension=384)

    chunk = DocumentChunk(
        id="chk-v-1",
        document_id="doc-v-1",
        source_id="src-v-1",
        source_url="https://mnre.gov.in/solar",
        text="India's solar capacity exceeded 70 GW in recent years.",
        chunk_index=0,
        page_number=3,
    )

    emb = svc.embed_text(chunk.text)
    vs.upsert_chunks([chunk], [emb])
    assert vs.count() == 1

    query_emb = svc.embed_text("solar capacity additions")
    hits = vs.search(query_emb, top_k=3)
    assert len(hits) == 1
    assert hits[0].chunk_id == "chk-v-1"
    assert hits[0].source_url == "https://mnre.gov.in/solar"
    assert hits[0].page_number == 3


def test_hybrid_retriever_provenance(test_retriever):
    c1 = DocumentChunk(
        id="chk-10",
        document_id="doc-10",
        source_id="src-10",
        source_url="https://who.int/amr",
        text="Bacterial mutation and plasmid transfer drive drug resistance.",
        chunk_index=0,
    )
    c2 = DocumentChunk(
        id="chk-20",
        document_id="doc-20",
        source_id="src-20",
        source_url="https://energy.gov/solar",
        text="Grid parity reached for utility-scale photovoltaic solar parks.",
        chunk_index=0,
    )

    test_retriever.index_chunks([c1, c2])

    results = test_retriever.retrieve(query="bacterial resistance plasmids", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "chk-10"
    assert top.source_id == "src-10"
    assert top.source_url == "https://who.int/amr"
    assert "bacterial" in top.text.lower()
    assert top.relevance_score > 0.0
