"""Reciprocal Rank Fusion (RRF) for merging dense and BM25 result lists.

RRF is a simple, parameter-free fusion algorithm that is robust to differences
in score distributions between retrievers.

Reference:
    Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods"

Formula:
    RRF(d) = Σ_r  1 / (k + rank_r(d))

where k = 60 (empirically recommended default).
"""

from __future__ import annotations

from collections import defaultdict

from src.domain.models.chunk import Chunk, ChunkWithScore
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


def reciprocal_rank_fusion(
    result_lists: list[list[ChunkWithScore]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[ChunkWithScore]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_lists: Multiple ranked lists of ChunkWithScore.
        k: RRF constant (default 60, recommended in literature).
        weights: Optional per-list weights (must match len of result_lists).
            Defaults to equal weights.

    Returns:
        Fused and re-ranked list of ChunkWithScore with RRF scores.

    Example:
        >>> dense_results = await dense_retriever.search(...)
        >>> bm25_results  = await bm25_retriever.search(...)
        >>> fused = reciprocal_rank_fusion([dense_results, bm25_results])
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError("weights length must match result_lists length")

    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = defaultdict(float)
    # chunk_id → Chunk object (keep first seen)
    chunks: dict[str, Chunk] = {}

    for result_list, weight in zip(result_lists, weights):
        for result in result_list:
            cid = result.chunk.id
            rank = result.rank  # 1-based

            rrf_scores[cid] += weight * (1.0 / (k + rank))

            if cid not in chunks:
                chunks[cid] = result.chunk

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Normalise scores to [0, 1]
    max_score = sorted_items[0][1] if sorted_items else 1.0

    return [
        ChunkWithScore(
            chunk=chunks[cid],
            score=min(1.0, score / max_score),
            rank=rank + 1,
            retrieval_method="rrf",
        )
        for rank, (cid, score) in enumerate(sorted_items)
    ]
