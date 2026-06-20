"""Dependency injection container and FastAPI dependency functions.

All shared application components are created once here and injected into
routes via FastAPI's dependency system.  No business logic lives here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Request

from src.application.services.services import (
    IngestionService,
    KnowledgeBaseService,
    RetrievalService,
)
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.config.settings import Settings
from src.domain.interfaces import EmbeddingProvider, KBRepository, VectorStore
from src.embeddings.factory import create_embedding_provider
from src.infrastructure.persistence.sqlite_kb_repository import SQLiteKBRepository
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring.logging import get_logger
from src.parsers.registry import build_pdf_parser_registry
from src.reranking.factory import create_reranker
from src.retrieval.pipeline import RetrievalPipeline
from src.vectorstores.factory import create_vector_store

logger = get_logger(__name__)


class Container:
    """IoC container — owns the lifecycle of all shared singletons.

    Args:
        settings: Application settings (injected at startup).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Infrastructure singletons
        self.embedding_provider: EmbeddingProvider = create_embedding_provider(
            settings.embedding
        )
        self.vector_store: VectorStore = create_vector_store(settings.vector_store)
        self.kb_repository: KBRepository = SQLiteKBRepository(settings.database.url)

        # Parsers and chunker
        self.parser_registry = build_pdf_parser_registry(
            primary=settings.parsers.pdf.primary,
            ocr_enabled=settings.parsers.ocr.enabled,
            ocr_engine=settings.parsers.ocr.engine,
        )
        self.chunker = HierarchicalChunker(
            chunk_size=settings.chunking.chunk_size,
            chunk_overlap=settings.chunking.chunk_overlap,
            min_chunk_size=settings.chunking.min_chunk_size,
        )

        # Reranker (optional)
        self.reranker = create_reranker(settings.reranker)

        # Pipelines
        self.ingestion_pipeline = IngestionPipeline(
            parser_registry=self.parser_registry,
            chunker=self.chunker,
            embedding_provider=self.embedding_provider,
            vector_store=self.vector_store,
            kb_repository=self.kb_repository,
        )
        self.retrieval_pipeline = RetrievalPipeline(
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            reranker=self.reranker,
            top_k_dense=settings.retrieval.top_k,
            top_k_bm25=settings.retrieval.top_k,
            dense_weight=settings.retrieval.dense_weight,
            bm25_weight=settings.retrieval.bm25_weight,
            final_top_k=settings.retrieval.final_top_k,
            rrf_k=settings.retrieval.rrf_k,
        )

        # Application services
        self.kb_service = KnowledgeBaseService(
            kb_repository=self.kb_repository,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
        )
        self.ingestion_service = IngestionService(
            ingestion_pipeline=self.ingestion_pipeline,
            kb_repository=self.kb_repository,
        )
        self.retrieval_service = RetrievalService(
            retrieval_pipeline=self.retrieval_pipeline,
            kb_repository=self.kb_repository,
            vector_store=self.vector_store,
        )

    async def initialise(self) -> None:
        """Initialise database and ensure directories exist."""
        await self.kb_repository.initialise()  # type: ignore[attr-defined]
        logger.info("Container initialised")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Container shutdown complete")


# ─── FastAPI dependency functions ────────────────────────────────────────────


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


def get_kb_service(
    container: Annotated[Container, Depends(get_container)],
) -> KnowledgeBaseService:
    return container.kb_service


def get_ingestion_service(
    container: Annotated[Container, Depends(get_container)],
) -> IngestionService:
    return container.ingestion_service


def get_retrieval_service(
    container: Annotated[Container, Depends(get_container)],
) -> RetrievalService:
    return container.retrieval_service
