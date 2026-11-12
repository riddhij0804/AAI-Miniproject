"""Embedding generation services supporting fast local ONNX models and deterministic fallback."""

from abc import ABC, abstractmethod
import hashlib
import logging
import math
import re
from typing import List, Optional
import numpy as np

from backend.app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Abstract base class for vector embedding generators."""

    def __init__(self, dimension: int = settings.VECTOR_DIMENSION):
        self.dimension = dimension

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate a dense vector embedding for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of strings."""
        pass

    @staticmethod
    def normalize_vector(vec: np.ndarray) -> np.ndarray:
        """L2-normalize a vector so dot product equals cosine similarity."""
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return np.zeros_like(vec)
        return vec / norm


class FastEmbedService(BaseEmbeddingService):
    """High-performance local embeddings using FastEmbed and ONNX Runtime."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME, dimension: int = 384):
        super().__init__(dimension=dimension)
        self.model_name = model_name
        self._model = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info(f"Initialized FastEmbed model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"FastEmbed model '{self.model_name}' could not be initialized: {e}") from e

    def embed_text(self, text: str) -> List[float]:
        if self._model is None:
            raise RuntimeError("FastEmbed model is not initialized")
        embeddings = list(self._model.embed([text]))
        vec = np.array(embeddings[0], dtype=np.float32)
        if vec.size != self.dimension:
            raise ValueError(f"Embedding dimension {vec.size} does not match configured {self.dimension}")
        return self.normalize_vector(vec).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("FastEmbed model is not initialized")
        embeddings = list(self._model.embed(texts))
        if len(embeddings) != len(texts):
            raise ValueError("Embedding provider returned an incorrect batch size")
        result = []
        for embedding in embeddings:
            vec = np.array(embedding, dtype=np.float32)
            if vec.size != self.dimension:
                raise ValueError(f"Embedding dimension {vec.size} does not match configured {self.dimension}")
            result.append(self.normalize_vector(vec).tolist())
        return result


class HashEmbeddingService(BaseEmbeddingService):
    """Deterministic, infallible n-gram feature hashing embedding.

    Produces normalized 384-dimensional dense vectors with semantic preservation for
    token overlap. Zero external model downloads, sub-millisecond execution, ideal for
    fast tests and offline environments.
    """

    def embed_text(self, text: str) -> List[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        tokens = cleaned.split()
        if not tokens:
            return vec.tolist()

        # Token unigrams and bigrams
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i+1]}")

        # Subword character n-grams (3-to-5 grams) for morphological invariance (e.g. singular/plural)
        for t in tokens:
            if len(t) >= 4:
                for n in (3, 4, 5):
                    for start in range(len(t) - n + 1):
                        features.append(f"c_{t[start:start+n]}")

        for feat in features:
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sign

        return self.normalize_vector(vec).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_service() -> BaseEmbeddingService:
    """Factory creating the configured embedding service."""
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider in {"hash", "mock"}:
        if settings.ENVIRONMENT.lower() not in {"development", "test"}:
            raise RuntimeError("Hash embeddings are restricted to development and test environments.")
        return HashEmbeddingService(settings.VECTOR_DIMENSION)
    try:
        return FastEmbedService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            dimension=settings.VECTOR_DIMENSION,
        )
    except Exception:
        if settings.ENVIRONMENT.lower() in {"development", "test"} and provider == "auto":
            logger.warning("FastEmbed unavailable; using explicitly labeled development hash embeddings.")
            return HashEmbeddingService(settings.VECTOR_DIMENSION)
        raise


# Global default embedding service
default_embedding_service = get_embedding_service()
