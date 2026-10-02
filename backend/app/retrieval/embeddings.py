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

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384):
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
            logger.warning(f"Could not initialize FastEmbed ({e}). Will fall back to HashEmbeddingService.")
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        if not self._model:
            return HashEmbeddingService(self.dimension).embed_text(text)
        embeddings = list(self._model.embed([text]))
        vec = np.array(embeddings[0], dtype=np.float32)
        return self.normalize_vector(vec).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._model:
            return HashEmbeddingService(self.dimension).embed_batch(texts)
        embeddings = list(self._model.embed(texts))
        return [self.normalize_vector(np.array(e, dtype=np.float32)).tolist() for e in embeddings]


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
    if provider == "hash" or provider == "mock":
        return HashEmbeddingService(settings.VECTOR_DIMENSION)
    try:
        service = FastEmbedService(dimension=settings.VECTOR_DIMENSION)
        if service._model is not None:
            return service
    except Exception as e:
        logger.warning(f"Failed to create FastEmbedService ({e}), falling back to HashEmbeddingService")
    return HashEmbeddingService(settings.VECTOR_DIMENSION)


# Global default embedding service
default_embedding_service = get_embedding_service()
