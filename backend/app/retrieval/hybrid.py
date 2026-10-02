"""Hybrid retrieval combining dense vector similarity with sparse BM25 keyword matching."""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from backend.app.retrieval.embeddings import BaseEmbeddingService, default_embedding_service
from backend.app.retrieval.vector_store import QdrantVectorStore, VectorSearchResult, default_vector_store
from backend.app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Normalized evidence retrieval result preserving complete provenance."""
    chunk_id: str
    document_id: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    text: str
    page_number: Optional[int] = None
    relevance_score: float = Field(..., description="Normalized fused relevance score [0.0 - 1.0]")
    retrieval_method: str = "hybrid"  # 'vector', 'bm25', 'hybrid'
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HybridRetriever:
    """Hybrid retrieval engine combining BM25 keyword matching and dense vector search."""

    def __init__(
        self,
        vector_store: QdrantVectorStore = default_vector_store,
        embedding_service: BaseEmbeddingService = default_embedding_service,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        rrf_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        # In-memory BM25 index over currently registered chunks
        self._bm25_corpus: List[DocumentChunk] = []
        self._bm25_index: Optional[BM25Okapi] = None
        self._tokenized_corpus: List[List[str]] = []

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into both vector store and local BM25 index."""
        if not chunks:
            return

        # 1. Vector Store Indexing
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        self.vector_store.upsert_chunks(chunks, embeddings)

        # 2. BM25 Indexing
        for c in chunks:
            self._bm25_corpus.append(c)
            tokens = self._tokenize(c.text)
            self._tokenized_corpus.append(tokens)

        if self._tokenized_corpus:
            self._bm25_index = BM25Okapi(self._tokenized_corpus)
            logger.info(f"HybridRetriever indexed {len(chunks)} chunks (Total in BM25: {len(self._bm25_corpus)})")

    def _tokenize(self, text: str) -> List[str]:
        """Simple clean whitespace and punctuation tokenizer for BM25."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 1]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_id_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
        use_hybrid: bool = True,
    ) -> List[RetrievalResult]:
        """Retrieve relevant evidence chunks for a research query with provenance."""
        if not query or not query.strip():
            return []

        # 1. Vector Search
        query_vector = self.embedding_service.embed_text(query)
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=max(top_k * 2, 10),
            source_id_filter=source_id_filter,
            document_id_filter=document_id_filter,
        )

        if not use_hybrid or not self._bm25_index or len(self._bm25_corpus) == 0:
            # Fall back to pure vector retrieval
            out: List[RetrievalResult] = []
            for item in vector_results[:top_k]:
                out.append(
                    RetrievalResult(
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        source_id=item.source_id,
                        source_url=item.source_url,
                        text=item.text,
                        page_number=item.page_number,
                        relevance_score=max(0.0, min(1.0, (item.score + 1.0) / 2.0)),  # Map cosine to [0,1]
                        retrieval_method="vector",
                        metadata=item.metadata,
                    )
                )
            return out

        # 2. BM25 Search
        query_tokens = self._tokenize(query)
        bm25_scores = self._bm25_index.get_scores(query_tokens)

        # Rank BM25 hits
        scored_bm25 = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                chunk = self._bm25_corpus[idx]
                if source_id_filter and chunk.source_id != source_id_filter:
                    continue
                if document_id_filter and chunk.document_id != document_id_filter:
                    continue
                scored_bm25.append((chunk, float(score)))

        scored_bm25.sort(key=lambda x: x[1], reverse=True)
        top_bm25 = scored_bm25[: max(top_k * 2, 10)]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Any] = {}

        # Process vector ranks
        for rank, item in enumerate(vector_results):
            cid = item.chunk_id
            chunk_map[cid] = {
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "source_id": item.source_id,
                "source_url": item.source_url,
                "text": item.text,
                "page_number": item.page_number,
                "metadata": item.metadata,
            }
            score_contrib = self.vector_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

        # Process BM25 ranks
        for rank, (chunk, _) in enumerate(top_bm25):
            cid = chunk.id
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "source_id": chunk.source_id,
                    "source_url": chunk.source_url,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "metadata": chunk.metadata,
                }
            score_contrib = self.bm25_weight * (1.0 / (self.rrf_k + rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score_contrib

        # Sort fused items by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
        final_results: List[RetrievalResult] = []

        for cid in sorted_cids[:top_k]:
            data = chunk_map[cid]
            norm_score = round(rrf_scores[cid] / max_rrf, 4) if max_rrf > 0 else 0.0
            final_results.append(
                RetrievalResult(
                    chunk_id=data["chunk_id"],
                    document_id=data["document_id"],
                    source_id=data["source_id"],
                    source_url=data["source_url"],
                    text=data["text"],
                    page_number=data["page_number"],
                    relevance_score=norm_score,
                    retrieval_method="hybrid",
                    metadata=data["metadata"],
                )
            )

        return final_results


# Global default hybrid retriever
default_retriever = HybridRetriever()
