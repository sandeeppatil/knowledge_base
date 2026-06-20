"""BM25 retriever for sparse keyword-based retrieval.

BM25 complements dense retrieval by capturing exact keyword matches that
semantic search may miss (e.g. product codes, IDs, technical terms).
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.domain.models.chunk import Chunk, ChunkWithScore
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """In-memory BM25 retriever built from a corpus of chunks.

    The index is rebuilt each time a knowledge base is loaded or updated.
    For very large KBs (>100k chunks), consider persisting the index.

    Args:
        chunks: List of Chunk objects forming the search corpus.

    Example:
        >>> retriever = BM25Retriever(chunks)
        >>> results = await retriever.search("ISO 9001 quality management", top_k=10)
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.content) for c in chunks]
        self._index = BM25Okapi(tokenized) if tokenized else None
        logger.debug("BM25 index built", corpus_size=len(chunks))

    async def search(self, query: str, top_k: int) -> list[ChunkWithScore]:
        """BM25 search over the in-memory corpus.

        Args:
            query: User query string.
            top_k: Number of top results to return.

        Returns:
            List of ChunkWithScore, ordered by descending BM25 score.
        """
        if self._index is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        raw_scores: list[float] = self._index.get_scores(tokens).tolist()

        # Normalise to [0, 1]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            return []

        scored = [
            (self._chunks[i], raw_scores[i] / max_score)
            for i in range(len(self._chunks))
            if raw_scores[i] > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return [
            ChunkWithScore(
                chunk=chunk,
                score=score,
                rank=rank + 1,
                retrieval_method="bm25",
            )
            for rank, (chunk, score) in enumerate(top)
        ]
