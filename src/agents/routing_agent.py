"""KB Routing Agent — selects the most relevant knowledge base for a query.

The router uses semantic similarity between the query and KB descriptions
to shortlist candidate KBs, avoiding exhaustive search of every KB.

Algorithm:
    1. Embed the query.
    2. Embed all KB descriptions (cached).
    3. Compute cosine similarity.
    4. Return the top-k most relevant KBs.
"""

from __future__ import annotations

import math
from typing import Any

from src.domain.interfaces import EmbeddingProvider, KBRepository
from src.domain.models.knowledge_base import KnowledgeBase
from src.monitoring.logging import get_logger
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KBRouter:
    """Routes a query to the most relevant knowledge base(s).

    Args:
        kb_repository: Source of KB records.
        embedding_provider: For embedding queries and descriptions.
        min_score: Minimum cosine similarity to include a KB.

    Example:
        >>> router = KBRouter(repo, embedder)
        >>> candidates = await router.route("ISO 9001 requirements", top_k=2)
        >>> for kb in candidates:
        ...     print(kb.name, kb.description)
    """

    def __init__(
        self,
        kb_repository: KBRepository,
        embedding_provider: EmbeddingProvider,
        min_score: float = 0.3,
    ) -> None:
        self._repo = kb_repository
        self._embedder = embedding_provider
        self._min_score = min_score
        # Description embedding cache: kb_id → vector
        self._desc_cache: dict[str, list[float]] = {}

    async def route(
        self,
        query: str,
        top_k: int = 3,
        candidate_kb_ids: list[str] | None = None,
    ) -> list[tuple[KnowledgeBase, float]]:
        """Select the most relevant KBs for a query.

        Args:
            query: User query string.
            top_k: Maximum number of KBs to return.
            candidate_kb_ids: Pre-filter to these KB IDs (None = all active).

        Returns:
            List of (KnowledgeBase, similarity_score) tuples, sorted by score.
        """
        with tracer.start_as_current_span("kb_router.route") as span:
            span.set_attribute("top_k", top_k)

            kbs = await self._repo.list_kbs()
            if not kbs:
                return []

            if candidate_kb_ids:
                kbs = [kb for kb in kbs if kb.id in candidate_kb_ids]

            # Embed query
            query_vector = await self._embedder.embed_text(query)

            # Embed descriptions (using cache)
            scored: list[tuple[KnowledgeBase, float]] = []
            for kb in kbs:
                desc_vector = await self._get_description_vector(kb)
                score = _cosine_similarity(query_vector, desc_vector)
                if score >= self._min_score:
                    scored.append((kb, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            result = scored[:top_k]

        logger.debug(
            "KB routing complete",
            query=query[:60],
            candidates=len(result),
            top_kb=result[0][0].name if result else "none",
        )
        return result

    async def _get_description_vector(self, kb: KnowledgeBase) -> list[float]:
        """Return cached (or freshly computed) description embedding."""
        cache_key = f"{kb.id}:{kb.version}"
        if cache_key not in self._desc_cache:
            self._desc_cache[cache_key] = await self._embedder.embed_text(
                f"Knowledge base: {kb.name}. {kb.description}"
            )
        return self._desc_cache[cache_key]

    def invalidate_cache(self, kb_id: str | None = None) -> None:
        """Clear the description embedding cache."""
        if kb_id:
            to_remove = [k for k in self._desc_cache if k.startswith(kb_id)]
            for k in to_remove:
                del self._desc_cache[k]
        else:
            self._desc_cache.clear()
