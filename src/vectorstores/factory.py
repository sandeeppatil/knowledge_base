"""Vector store factory."""
from __future__ import annotations

from src.config.settings import VectorStoreConfig
from src.domain.interfaces import VectorStore


def create_vector_store(config: VectorStoreConfig) -> VectorStore:
    """Instantiate the correct VectorStore from configuration.

    Args:
        config: Vector store configuration block.

    Returns:
        Configured VectorStore.

    Raises:
        ValueError: If provider is not recognised.
    """
    if config.provider == "qdrant":
        from src.infrastructure.vector_stores.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(
            url=config.qdrant.url,
            api_key=config.qdrant.api_key,
            prefer_grpc=config.qdrant.prefer_grpc,
            timeout=config.qdrant.timeout,
        )

    if config.provider == "chroma":
        from src.infrastructure.vector_stores.chroma_store import ChromaVectorStore

        return ChromaVectorStore(host=config.chroma.host, port=config.chroma.port)

    if config.provider == "faiss":
        from src.infrastructure.vector_stores.faiss_store import FaissVectorStore

        return FaissVectorStore(index_path=config.faiss.index_path)

    raise ValueError(f"Unknown vector store provider: {config.provider!r}")
