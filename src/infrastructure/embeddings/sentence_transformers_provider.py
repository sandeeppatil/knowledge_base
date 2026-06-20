"""SentenceTransformers embedding provider.

Uses the sentence-transformers library to generate dense embeddings locally.
Supports batching, device selection, and optional vector normalization.
"""

from __future__ import annotations

import asyncio
from functools import cached_property
from typing import Any

import numpy as np

from src.domain.interfaces import EmbeddingProvider
from src.monitoring.logging import get_logger
from src.monitoring.metrics import EMBEDDING_LATENCY_SECONDS, EMBEDDING_REQUESTS_TOTAL
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class SentenceTransformersProvider(EmbeddingProvider):
    """EmbeddingProvider backed by sentence-transformers.

    Args:
        model_name: HuggingFace model name (e.g. "BAAI/bge-m3").
        device: Torch device ("cpu", "cuda", "mps").
        batch_size: Number of texts per encode call.
        normalize: Whether to L2-normalise output vectors.
        cache_dir: Optional local directory for model weights.

    Example:
        >>> provider = SentenceTransformersProvider("BAAI/bge-m3")
        >>> vector = await provider.embed_text("What is ISO 9001?")
        >>> print(len(vector))  # 1024
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 64,
        normalize: bool = True,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize = normalize
        self._cache_dir = cache_dir
        self._model: Any = None  # lazy-loaded

    def _load_model(self) -> Any:
        """Load the SentenceTransformer model (lazy, thread-safe)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading embedding model",
                model=self._model_name,
                device=self._device,
            )
            self._model = SentenceTransformer(
                self._model_name,
                device=self._device,
                cache_folder=self._cache_dir,
            )
            logger.info("Embedding model loaded", dimension=self.dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding vector dimension."""
        model = self._load_model()
        return int(model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input string.

        Returns:
            Dense embedding vector.
        """
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts using batch processing.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        with tracer.start_as_current_span("embed_batch") as span:
            span.set_attribute("batch_size", len(texts))
            span.set_attribute("model", self._model_name)

            with EMBEDDING_LATENCY_SECONDS.labels(provider="sentence_transformers").time():
                EMBEDDING_REQUESTS_TOTAL.labels(
                    provider="sentence_transformers", model=self._model_name
                ).inc()

                loop = asyncio.get_event_loop()
                vectors = await loop.run_in_executor(
                    None, self._encode_sync, texts
                )

        return vectors

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encode call — runs in a thread executor."""
        model = self._load_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
