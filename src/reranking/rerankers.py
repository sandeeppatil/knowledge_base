"""BGE Reranker implementation using FlagEmbedding.

Uses BAAI/bge-reranker-v2-m3 cross-encoder for high-quality reranking.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.domain.interfaces import Reranker
from src.domain.models.chunk import ChunkWithScore
from src.monitoring.logging import get_logger
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class BGEReranker(Reranker):
    """Cross-encoder reranker using FlagEmbedding BGE models.

    Args:
        model_name: HuggingFace model name.
        device: Torch device ("cpu", "cuda", "mps").
        batch_size: Batch size for inference.

    Example:
        >>> reranker = BGEReranker("BAAI/bge-reranker-v2-m3")
        >>> reranked = await reranker.rerank(query, candidates, top_k=10)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info("Loading reranker model", model=self._model_name)
            self._model = FlagReranker(
                self._model_name,
                use_fp16=self._device != "cpu",
            )
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[ChunkWithScore],
        top_k: int,
    ) -> list[ChunkWithScore]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: User query.
            candidates: Pre-retrieved candidate chunks.
            top_k: Maximum results to return.

        Returns:
            Reranked list, highest cross-encoder score first.
        """
        if not candidates:
            return []

        with tracer.start_as_current_span("bge_reranker.rerank") as span:
            span.set_attribute("candidates", len(candidates))
            span.set_attribute("top_k", top_k)

            pairs = [[query, c.chunk.content] for c in candidates]

            scores: list[float] = await asyncio.get_event_loop().run_in_executor(
                None, self._score_sync, pairs
            )

        # Pair original candidates with new scores
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        max_score = top[0][1] if top else 1.0
        if max_score <= 0:
            max_score = 1.0

        return [
            ChunkWithScore(
                chunk=cws.chunk,
                score=min(1.0, max(0.0, score / max_score)),
                rank=rank + 1,
                retrieval_method="reranked",
            )
            for rank, (cws, score) in enumerate(top)
        ]

    def _score_sync(self, pairs: list[list[str]]) -> list[float]:
        model = self._load_model()
        scores = model.compute_score(pairs, batch_size=self._batch_size)
        if isinstance(scores, float):
            return [scores]
        return [float(s) for s in scores]


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers CrossEncoder.

    Suitable for models like "cross-encoder/ms-marco-MiniLM-L-6-v2".

    Args:
        model_name: HuggingFace cross-encoder model.
        device: Torch device.
        batch_size: Inference batch size.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading CrossEncoder", model=self._model_name)
            self._model = CrossEncoder(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[ChunkWithScore],
        top_k: int,
    ) -> list[ChunkWithScore]:
        if not candidates:
            return []

        pairs = [[query, c.chunk.content] for c in candidates]

        scores: list[float] = await asyncio.get_event_loop().run_in_executor(
            None, self._score_sync, pairs
        )

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        max_score = max((s for _, s in top), default=1.0)
        if max_score <= 0:
            max_score = 1.0

        return [
            ChunkWithScore(
                chunk=cws.chunk,
                score=min(1.0, max(0.0, score / max_score)),
                rank=rank + 1,
                retrieval_method="reranked",
            )
            for rank, (cws, score) in enumerate(top)
        ]

    def _score_sync(self, pairs: list[list[str]]) -> list[float]:
        model = self._load_model()
        scores = model.predict(pairs, batch_size=self._batch_size)
        return [float(s) for s in scores]
