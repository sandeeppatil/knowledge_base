"""OpenAI-compatible embedding provider.

Works with any API that implements the OpenAI /v1/embeddings endpoint,
including OpenAI, Azure OpenAI, and local servers (e.g. LiteLLM, LocalAI).
"""

from __future__ import annotations

import httpx

from src.domain.interfaces import EmbeddingProvider
from src.monitoring.logging import get_logger
from src.monitoring.metrics import EMBEDDING_LATENCY_SECONDS, EMBEDDING_REQUESTS_TOTAL

logger = get_logger(__name__)


class OpenAICompatibleProvider(EmbeddingProvider):
    """EmbeddingProvider for OpenAI-compatible /v1/embeddings endpoints.

    Args:
        model_name: Embedding model name (e.g. "text-embedding-3-small").
        base_url: API base URL.
        api_key: Bearer token for Authorization header.
        batch_size: Max texts per API request.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com",
        api_key: str = "",
        batch_size: int = 100,
        timeout: float = 60.0,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._batch_size = batch_size
        self._timeout = timeout
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimension unknown until first embed call")
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

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        results: list[list[float]] = []

        with EMBEDDING_LATENCY_SECONDS.labels(provider="openai_compatible").time():
            EMBEDDING_REQUESTS_TOTAL.labels(
                provider="openai_compatible", model=self._model_name
            ).inc()

            async with httpx.AsyncClient(timeout=self._timeout) as client:
                for i in range(0, len(texts), self._batch_size):
                    batch = texts[i : i + self._batch_size]
                    response = await client.post(
                        f"{self._base_url}/v1/embeddings",
                        headers=headers,
                        json={"input": batch, "model": self._model_name},
                    )
                    response.raise_for_status()
                    data = response.json()
                    batch_vectors = [item["embedding"] for item in data["data"]]
                    results.extend(batch_vectors)
                    if self._dim is None and batch_vectors:
                        self._dim = len(batch_vectors[0])

        return results
