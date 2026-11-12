from pathlib import Path

from backend.app.retrieval.embeddings import HashEmbeddingService
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.vector_store import QdrantVectorStore
from backend.app.schemas.document import DocumentChunk


def test_qdrant_local_vectors_survive_reopen(tmp_path: Path):
    location = str(tmp_path / "qdrant")
    chunk = DocumentChunk(
        id="chunk-persist-1",
        document_id="doc-persist-1",
        source_id="source-persist-1",
        source_url="https://example.com/persist",
        text="Persistent semantic evidence about vector databases.",
        chunk_index=0,
    )
    embedding = HashEmbeddingService(384)
    first = QdrantVectorStore(location=location, collection_name="persist_test")
    first.upsert_chunks([chunk], [embedding.embed_text(chunk.text)])
    assert first.count() == 1
    first._client.close()

    second = QdrantVectorStore(location=location, collection_name="persist_test")
    results = second.search(embedding.embed_text("vector database evidence"), top_k=1)
    assert results and results[0].chunk_id == chunk.id
    assert results[0].document_id == chunk.document_id
    second._client.close()


def test_hybrid_marks_single_backend_results():
    retriever = HybridRetriever(
        vector_store=QdrantVectorStore(location=":memory:", collection_name="single_backend_test"),
        embedding_service=HashEmbeddingService(384),
    )
    chunk = DocumentChunk(
        id="chunk-hybrid-1",
        document_id="doc-hybrid-1",
        source_id="source-hybrid-1",
        source_url="https://example.com/hybrid",
        text="Unique keyword evidence for BM25 retrieval.",
        chunk_index=0,
    )
    retriever.index_chunks([chunk])
    results = retriever.retrieve("Unique keyword evidence", top_k=1)
    # A single-document BM25 corpus has zero discriminative IDF, so dense
    # retrieval is the only contributing backend for this repeatable case.
    assert results and results[0].retrieval_method == "vector"
