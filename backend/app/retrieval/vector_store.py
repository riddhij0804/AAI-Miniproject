"""Vector store implementation using Qdrant with full provenance preservation."""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend.app.config import settings
from backend.app.schemas.document import DocumentChunk

logger = logging.getLogger(__name__)


class VectorSearchResult(BaseModel):
    """Result item returned from vector retrieval preserving full provenance."""
    chunk_id: str
    document_id: str
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    text: str
    page_number: Optional[int] = None
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QdrantVectorStore:
    """Manages chunk vector embeddings and similarity queries in Qdrant."""

    def __init__(
        self,
        location: str = settings.QDRANT_LOCATION,
        collection_name: str = settings.QDRANT_COLLECTION,
        vector_size: int = settings.VECTOR_DIMENSION,
        api_key: Optional[str] = settings.QDRANT_API_KEY,
    ):
        self.location = location
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.api_key = api_key
        self._client: Optional[QdrantClient] = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Qdrant client and ensure collection exists."""
        try:
            if self.location == ":memory:":
                self._client = QdrantClient(location=":memory:")
            elif self.location.startswith("http://") or self.location.startswith("https://"):
                self._client = QdrantClient(url=self.location, api_key=self.api_key)
            else:
                self._client = QdrantClient(path=self.location)

            self._ensure_collection()
            logger.info(f"QdrantVectorStore initialized at '{self.location}', collection: '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantVectorStore: {e}")
            self._client = None

    def _ensure_collection(self) -> None:
        """Create collection if it does not already exist."""
        if not self._client:
            return
        collections = self._client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}'")

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> bool:
        """Upsert document chunks and their dense embeddings into Qdrant."""
        if not self._client:
            raise RuntimeError("Qdrant client is not available")
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})")
        if not chunks:
            return True

        points: List[qmodels.PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            # Ensure valid 128-bit UUID or integer for Qdrant point ID
            import uuid
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))

            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "source_id": chunk.source_id,
                "source_url": chunk.source_url,
                "text": chunk.text,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "metadata": chunk.metadata,
            }

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            logger.info(f"Upserted {len(points)} chunks into Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error upserting chunks to Qdrant: {e}")
            raise RuntimeError(f"Vector store upsert failed: {e}") from e

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        source_id_filter: Optional[str] = None,
        document_id_filter: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search nearest chunk embeddings by cosine similarity."""
        if not self._client:
            logger.warning("Qdrant client unavailable during search. Returning empty results.")
            return []

        # Build filter if requested
        query_filter = None
        conditions = []
        if source_id_filter:
            conditions.append(
                qmodels.FieldCondition(
                    key="source_id",
                    match=qmodels.MatchValue(value=source_id_filter),
                )
            )
        if document_id_filter:
            conditions.append(
                qmodels.FieldCondition(
                    key="document_id",
                    match=qmodels.MatchValue(value=document_id_filter),
                )
            )
        if conditions:
            query_filter = qmodels.Filter(must=conditions)

        try:
            # In Qdrant client 1.10+, query_points or search can be used
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                )
                hits = response.points
            else:
                hits = self._client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                )

            results: List[VectorSearchResult] = []
            for hit in hits:
                payload = hit.payload or {}
                results.append(
                    VectorSearchResult(
                        chunk_id=payload.get("chunk_id", str(hit.id)),
                        document_id=payload.get("document_id", ""),
                        source_id=payload.get("source_id"),
                        source_url=payload.get("source_url"),
                        text=payload.get("text", ""),
                        page_number=payload.get("page_number"),
                        score=float(hit.score),
                        metadata=payload.get("metadata", {}),
                    )
                )
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def count(self) -> int:
        """Count total vectors in collection."""
        if not self._client:
            return 0
        try:
            return self._client.count(collection_name=self.collection_name).count
        except Exception:
            return 0


# Global default vector store instance
default_vector_store = QdrantVectorStore()
