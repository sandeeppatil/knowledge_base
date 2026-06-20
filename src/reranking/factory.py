"""Reranker factory."""
from __future__ import annotations

from src.config.settings import RerankerConfig
from src.domain.interfaces import Reranker


def create_reranker(config: RerankerConfig) -> Reranker | None:
    """Instantiate a reranker from config, or return None if disabled.

    Args:
        config: Reranker configuration block.

    Returns:
        Configured Reranker, or None if reranker.enabled is False.
    """
    if not config.enabled:
        return None

    model = config.model.lower()

    if "bge-reranker" in model or "bge" in model:
        from src.reranking.rerankers import BGEReranker

        return BGEReranker(
            model_name=config.model,
            device=config.device,
            batch_size=config.batch_size,
        )

    # Default: sentence-transformers CrossEncoder
    from src.reranking.rerankers import CrossEncoderReranker

    return CrossEncoderReranker(
        model_name=config.model,
        device=config.device,
        batch_size=config.batch_size,
    )
