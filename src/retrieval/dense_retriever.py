"""Dense retriever — wraps a VectorStore for semantic similarity search."""

from __future__ import annotations

from src.domain.interfaces import EmbeddingProvider, VectorStore
from src.domain.models.chunk import ChunkWithScore
from src.monitoring.logging import get_logger
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class DenseRetriever:
    """Perform ANN similarity search using an EmbeddingProvider + VectorStore.

    Args:
        vector_store: VectorStore to query.
        embedding_provider: Provider to embed the query.

    Example:
        >>> retriever = DenseRetriever(store, embedder)
        >>> results = await retriever.search("kb_iso_123", "quality management", 20)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._store = vector_store
        self._embedder = embedding_provider

    async def search(
        self,
        collection: str,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[ChunkWithScore]:
        """Embed the query and perform ANN search.

        Args:
            collection: Vector store collection name.
            query: Query string.
            top_k: Number of candidates to retrieve.
            filters: Optional payload filters.

        Returns:
            Ranked list of ChunkWithScore.
        """
        with tracer.start_as_current_span("dense_retriever.search") as span:
            span.set_attribute("collection", collection)
            span.set_attribute("top_k", top_k)

            query_vector = await self._embedder.embed_text(query)
            results = await self._store.search(
                collection=collection,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )

        logger.debug(
            "Dense retrieval",
            collection=collection,
            results=len(results),
        )
        return results
