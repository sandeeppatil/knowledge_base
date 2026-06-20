"""Ollama embedding provider — calls local Ollama REST API."""

from __future__ import annotations

import httpx

from src.domain.interfaces import EmbeddingProvider
from src.monitoring.logging import get_logger
from src.monitoring.metrics import EMBEDDING_LATENCY_SECONDS, EMBEDDING_REQUESTS_TOTAL

logger = get_logger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider that calls the Ollama /api/embed endpoint.

    Args:
        model_name: Ollama model identifier (e.g. "nomic-embed-text").
        base_url: Ollama server URL.
        timeout: Request timeout in seconds.

    Example:
        >>> provider = OllamaEmbeddingProvider("nomic-embed-text")
        >>> vector = await provider.embed_text("Hello world")
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimension unknown until first embed_text call")
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        with EMBEDDING_LATENCY_SECONDS.labels(provider="ollama").time():
            EMBEDDING_REQUESTS_TOTAL.labels(
                provider="ollama", model=self._model_name
            ).inc()

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for text in texts:
                    response = await client.post(
                        f"{self._base_url}/api/embed",
                        json={"model": self._model_name, "input": text},
                    )
                    response.raise_for_status()
                    data = response.json()
                    vector: list[float] = data["embeddings"][0]
                    results.append(vector)
                    if self._dim is None:
                        self._dim = len(vector)

        return results
