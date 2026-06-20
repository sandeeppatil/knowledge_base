"""Retrieval pipeline — orchestrates dense retrieval, BM25, RRF, and reranking.

Pipeline stages:
    Query
      ↓  Dense Retrieval (ANN)
      ↓  BM25 Retrieval
      ↓  RRF Fusion
      ↓  Reranker (optional)
      ↓  Final top-k
"""

from __future__ import annotations

import time
from typing import Any

from src.domain.interfaces import EmbeddingProvider, KBRepository, Reranker, VectorStore
from src.domain.models.chunk import ChunkWithScore
from src.domain.models.retrieval import RetrievalResult
from src.monitoring.logging import get_logger
from src.monitoring.metrics import (
    RETRIEVAL_CHUNKS_RETURNED,
    RETRIEVAL_LATENCY_SECONDS,
    RETRIEVAL_REQUESTS_TOTAL,
)
from src.monitoring.tracing import get_tracer
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf_fusion import reciprocal_rank_fusion

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class RetrievalPipeline:
    """End-to-end retrieval pipeline for a single knowledge base.

    Args:
        vector_store: Configured VectorStore.
        embedding_provider: Configured EmbeddingProvider.
        kb_repository: KBRepository for chunk/document lookup.
        reranker: Optional Reranker (skipped when None).
        top_k_dense: Number of dense retrieval candidates.
        top_k_bm25: Number of BM25 candidates.
        dense_weight: RRF weight for dense results.
        bm25_weight: RRF weight for BM25 results.
        final_top_k: Number of results after reranking.
        rrf_k: RRF constant.

    Example:
        >>> pipeline = RetrievalPipeline(store, embedder, repo, reranker)
        >>> result = await pipeline.retrieve(kb, "What is ISO 9001?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3,
        final_top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._dense_retriever = DenseRetriever(vector_store, embedding_provider)
        self._reranker = reranker
        self._top_k_dense = top_k_dense
        self._top_k_bm25 = top_k_bm25
        self._dense_weight = dense_weight
        self._bm25_weight = bm25_weight
        self._final_top_k = final_top_k
        self._rrf_k = rrf_k
        # BM25 index cache: collection_name → BM25Retriever
        self._bm25_cache: dict[str, BM25Retriever] = {}

    async def retrieve(
        self,
        collection: str,
        kb_name: str,
        kb_id: str,
        query: str,
        top_k: int | None = None,
        bm25_corpus: list | None = None,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Args:
            collection: Vector store collection name.
            kb_name: Human-readable KB name (for citations).
            kb_id: Knowledge base ID.
            query: User query string.
            top_k: Override final_top_k for this call.
            bm25_corpus: Pre-loaded chunk list for BM25 index.

        Returns:
            RetrievalResult — either with chunks/citations or answer_found=False.
        """
        if not query.strip():
            return RetrievalResult.not_found(query)

        final_k = top_k or self._final_top_k

        with tracer.start_as_current_span("retrieval_pipeline") as span:
            span.set_attribute("kb_id", kb_id)
            span.set_attribute("query_len", len(query))

            t0 = time.monotonic()

            # ── Stage 1: Dense retrieval ─────────────────────────────────
            dense_results: list[ChunkWithScore] = []
            try:
                dense_results = await self._dense_retriever.search(
                    collection=collection,
                    query=query,
                    top_k=self._top_k_dense,
                )
            except Exception as exc:
                logger.warning("Dense retrieval failed", error=str(exc))

            # ── Stage 2: BM25 retrieval ──────────────────────────────────
            bm25_results: list[ChunkWithScore] = []
            if bm25_corpus:
                try:
                    bm25_retriever = self._get_bm25_retriever(collection, bm25_corpus)
                    bm25_results = await bm25_retriever.search(
                        query=query, top_k=self._top_k_bm25
                    )
                except Exception as exc:
                    logger.warning("BM25 retrieval failed", error=str(exc))

            # ── Stage 3: RRF fusion ──────────────────────────────────────
            if dense_results and bm25_results:
                fused = reciprocal_rank_fusion(
                    [dense_results, bm25_results],
                    k=self._rrf_k,
                    weights=[self._dense_weight, self._bm25_weight],
                )
            else:
                fused = dense_results or bm25_results

            if not fused:
                latency = (time.monotonic() - t0) * 1000
                RETRIEVAL_REQUESTS_TOTAL.labels(
                    knowledge_base=kb_name, answer_found="false"
                ).inc()
                return RetrievalResult.not_found(query)

            # ── Stage 4: Reranking ───────────────────────────────────────
            if self._reranker:
                try:
                    fused = await self._reranker.rerank(
                        query=query, candidates=fused, top_k=final_k
                    )
                except Exception as exc:
                    logger.warning("Reranking failed, using RRF order", error=str(exc))
                    fused = fused[:final_k]
            else:
                fused = fused[:final_k]

            # ── Populate KB name on all chunks ────────────────────────────
            for result in fused:
                result.chunk.metadata.knowledge_base_name = kb_name

            latency_ms = (time.monotonic() - t0) * 1000

            # ── Metrics ───────────────────────────────────────────────────
            RETRIEVAL_REQUESTS_TOTAL.labels(
                knowledge_base=kb_name, answer_found="true"
            ).inc()
            RETRIEVAL_LATENCY_SECONDS.labels(knowledge_base=kb_name).observe(
                latency_ms / 1000
            )
            RETRIEVAL_CHUNKS_RETURNED.labels(knowledge_base=kb_name).observe(len(fused))

            logger.info(
                "Retrieval complete",
                kb=kb_name,
                query=query[:80],
                results=len(fused),
                latency_ms=round(latency_ms, 1),
            )

            result = RetrievalResult.from_chunks(
                query=query,
                kb_id=kb_id,
                kb_name=kb_name,
                chunks=fused,
            )
            result.retrieval_metadata = {
                "dense_count": len(dense_results),
                "bm25_count": len(bm25_results),
                "fused_count": len(fused),
                "latency_ms": round(latency_ms, 1),
                "reranker_used": self._reranker is not None,
            }
            return result

    def _get_bm25_retriever(
        self, collection: str, corpus: list
    ) -> BM25Retriever:
        """Return or build a BM25 index for the collection."""
        if collection not in self._bm25_cache:
            self._bm25_cache[collection] = BM25Retriever(corpus)
        return self._bm25_cache[collection]

    def invalidate_bm25_cache(self, collection: str) -> None:
        """Invalidate the BM25 index for a collection (call after ingestion)."""
        self._bm25_cache.pop(collection, None)
