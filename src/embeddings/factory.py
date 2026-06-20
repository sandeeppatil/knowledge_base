"""Embedding provider factory."""
from __future__ import annotations

from src.config.settings import EmbeddingConfig
from src.domain.interfaces import EmbeddingProvider


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    """Instantiate the correct EmbeddingProvider from configuration.

    Args:
        config: Embedding configuration block.

    Returns:
        Configured EmbeddingProvider.

    Raises:
        ValueError: If the provider name is not recognised.
    """
    if config.provider == "sentence_transformers":
        from src.infrastructure.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(
            model_name=config.model,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize,
            cache_dir=config.cache_dir,
        )

    if config.provider == "ollama":
        from src.infrastructure.embeddings.ollama_provider import OllamaEmbeddingProvider

        return OllamaEmbeddingProvider(model_name=config.model)

    if config.provider == "openai_compatible":
        from src.infrastructure.embeddings.openai_compatible_provider import (
            OpenAICompatibleProvider,
        )

        return OpenAICompatibleProvider(model_name=config.model)

    raise ValueError(f"Unknown embedding provider: {config.provider!r}")
