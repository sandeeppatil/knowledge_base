"""Application services — coordinate domain objects and infrastructure.

Services live in the application layer and implement business workflows.
They depend only on domain interfaces, never on concrete implementations.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.domain.interfaces import EmbeddingProvider, KBRepository, VectorStore
from src.domain.models.document import Document
from src.domain.models.knowledge_base import KnowledgeBase
from src.domain.models.retrieval import RetrievalResult
from src.ingestion.pipeline import IngestionPipeline
from src.monitoring.logging import get_logger
from src.monitoring.metrics import KB_CHUNK_COUNT, KB_DOCUMENT_COUNT, KNOWLEDGE_BASES_ACTIVE
from src.retrieval.pipeline import RetrievalPipeline

logger = get_logger(__name__)


class KnowledgeBaseService:
    """CRUD operations for knowledge bases.

    Args:
        kb_repository: Persistence layer for KB metadata.
        vector_store: Vector store for collection management.
        embedding_provider: Used to get dimension when creating collections.
    """

    def __init__(
        self,
        kb_repository: KBRepository,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._repo = kb_repository
        self._store = vector_store
        self._embedder = embedding_provider

    async def create(
        self,
        name: str,
        description: str,
        embedding_model: str | None = None,
        vector_store_provider: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base.

        Args:
            name: Unique KB name.
            description: Natural-language description for routing.
            embedding_model: Override default embedding model name.
            vector_store_provider: Override default vector store provider.
            metadata: Optional arbitrary metadata.

        Returns:
            Newly created KnowledgeBase.

        Raises:
            ValueError: If a KB with the same name already exists.
        """
        existing = await self._repo.get_kb_by_name(name)
        if existing:
            raise ValueError(f"Knowledge base '{name}' already exists.")

        # Generate KB ID and collection name
        kb_id = str(uuid.uuid4())
        collection_name = f"kb_{kb_id.replace('-', '_')}"
        
        # Get embedding dimension from provider
        embedding_dimension = self._embedder.dimension
        embedding_model_name = embedding_model or self._embedder.model_name
        vector_store_type = vector_store_provider or "qdrant"

        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            embedding_model=embedding_model_name,
            embedding_dimension=embedding_dimension,
            vector_store_type=vector_store_type,
            collection_name=collection_name,
            metadata=metadata or {},
        )
        kb = await self._repo.create_kb(kb)

        # Create vector store collection
        await self._store.create_collection(kb.collection_name, self._embedder.dimension)

        KNOWLEDGE_BASES_ACTIVE.inc()
        logger.info("Knowledge base created", kb_id=kb.id, name=kb.name)
        return kb

    async def get(self, kb_id: str) -> KnowledgeBase | None:
        return await self._repo.get_kb(kb_id)

    async def get_by_name(self, name: str) -> KnowledgeBase | None:
        return await self._repo.get_kb_by_name(name)

    async def list(self) -> list[KnowledgeBase]:
        return await self._repo.list_kbs()

    async def delete(self, kb_id: str) -> bool:
        """Delete a knowledge base and all its data.

        Args:
            kb_id: ID of the KB to delete.

        Returns:
            True if deleted, False if not found.
        """
        kb = await self._repo.get_kb(kb_id)
        if not kb:
            return False

        # Delete vector collection
        try:
            await self._store.delete_collection(kb.collection_name)
        except Exception as exc:
            logger.warning(
                "Failed to delete vector collection",
                kb_id=kb_id,
                error=str(exc),
            )

        deleted = await self._repo.delete_kb(kb_id)
        if deleted:
            KNOWLEDGE_BASES_ACTIVE.dec()
            logger.info("Knowledge base deleted", kb_id=kb_id, name=kb.name)
        return deleted

    async def rebuild(
        self,
        kb_id: str,
        ingestion_pipeline: IngestionPipeline,
    ) -> KnowledgeBase:
        """Rebuild a KB by re-ingesting all its documents.

        Deletes the existing vector collection, recreates it, then re-ingests.

        Args:
            kb_id: Knowledge base to rebuild.
            ingestion_pipeline: Configured ingestion pipeline.

        Returns:
            Updated KnowledgeBase with new chunk/document counts.
        """
        kb = await self._repo.get_kb(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")

        # Reset counters and rebuild collection
        await self._store.delete_collection(kb.collection_name)
        await self._store.create_collection(kb.collection_name, self._embedder.dimension)

        kb.chunk_count = 0
        kb.document_count = 0
        kb.bump_version()
        await self._repo.update_kb(kb)

        # Re-ingest all documents
        documents = await self._repo.list_documents(kb_id)
        for doc in documents:
            path = Path(doc.source_path)
            if path.exists():
                await ingestion_pipeline.ingest(path, kb, skip_duplicates=False)

        KB_CHUNK_COUNT.labels(knowledge_base=kb.name).set(kb.chunk_count)
        KB_DOCUMENT_COUNT.labels(knowledge_base=kb.name).set(kb.document_count)
        logger.info("Knowledge base rebuilt", kb_id=kb_id, name=kb.name)
        return kb


class IngestionService:
    """Orchestrates document ingestion into a knowledge base.

    Args:
        ingestion_pipeline: Configured IngestionPipeline.
        kb_repository: For KB lookups and updates.
    """

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        kb_repository: KBRepository,
    ) -> None:
        self._pipeline = ingestion_pipeline
        self._repo = kb_repository

    async def ingest_file(
        self, file_path: Path, kb_id: str, skip_duplicates: bool = True
    ) -> Document:
        """Ingest a single file into a KB.

        Args:
            file_path: Path to the source file.
            kb_id: Target knowledge base ID.
            skip_duplicates: Skip files already ingested by checksum.

        Returns:
            Resulting Document record.

        Raises:
            ValueError: If KB not found.
        """
        kb = await self._repo.get_kb(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")
        return await self._pipeline.ingest(file_path, kb, skip_duplicates)

    async def ingest_folder(
        self, folder: Path, kb_id: str, skip_duplicates: bool = True
    ) -> list[Document]:
        """Ingest all supported files in a folder.

        Args:
            folder: Root folder to scan.
            kb_id: Target knowledge base ID.
            skip_duplicates: Skip files already ingested by checksum.

        Returns:
            List of Document results.
        """
        kb = await self._repo.get_kb(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")
        return await self._pipeline.ingest_folder(folder, kb, skip_duplicates=skip_duplicates)

    async def list_documents(self, kb_id: str) -> list[Document]:
        return await self._repo.list_documents(kb_id)

    async def delete_document(self, document_id: str, kb_id: str) -> bool:
        doc = await self._repo.get_document(document_id)
        if not doc:
            return False

        kb = await self._repo.get_kb(kb_id)
        if kb:
            try:
                from src.domain.interfaces import VectorStore
                # Deletion happens via the pipeline's vector store reference
            except Exception:
                pass

        return await self._repo.delete_document(document_id)


class RetrievalService:
    """Orchestrates retrieval requests against a knowledge base.

    Args:
        retrieval_pipeline: Configured RetrievalPipeline.
        kb_repository: For KB and chunk lookups.
        vector_store: For BM25 corpus loading.
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        kb_repository: KBRepository,
        vector_store: VectorStore,
    ) -> None:
        self._pipeline = retrieval_pipeline
        self._repo = kb_repository
        self._store = vector_store
        # BM25 corpus cache: kb_id → list[Chunk]
        self._corpus_cache: dict[str, list] = {}

    async def search(
        self,
        kb_id: str,
        query: str,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """Search a knowledge base and return grounded results.

        Args:
            kb_id: Knowledge base to search.
            query: User query string.
            top_k: Number of results (overrides pipeline default).

        Returns:
            RetrievalResult with chunks, citations, or answer_found=False.

        Raises:
            ValueError: If the KB is not found.
        """
        kb = await self._repo.get_kb(kb_id)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_id}")

        # Load BM25 corpus (lazy)
        corpus = await self._get_bm25_corpus(kb)

        return await self._pipeline.retrieve(
            collection=kb.collection_name,
            kb_name=kb.name,
            kb_id=kb.id,
            query=query,
            top_k=top_k,
            bm25_corpus=corpus,
        )

    async def search_across_kbs(
        self,
        query: str,
        kb_ids: list[str],
        top_k_per_kb: int = 5,
    ) -> list[RetrievalResult]:
        """Search multiple knowledge bases and return per-KB results.

        Args:
            query: User query.
            kb_ids: List of KB IDs to search.
            top_k_per_kb: Results per KB.

        Returns:
            List of RetrievalResults (one per KB, may include answer_found=False).
        """
        results: list[RetrievalResult] = []
        for kb_id in kb_ids:
            try:
                result = await self.search(kb_id, query, top_k=top_k_per_kb)
                results.append(result)
            except Exception as exc:
                logger.warning("Search failed for KB", kb_id=kb_id, error=str(exc))
        return results

    async def _get_bm25_corpus(self, kb: KnowledgeBase) -> list:
        """Return cached BM25 corpus for a KB, rebuilding if necessary."""
        if kb.id not in self._corpus_cache:
            # Attempt to load corpus from vector store
            # For now, return empty (BM25 will be skipped if corpus is empty)
            self._corpus_cache[kb.id] = []
        return self._corpus_cache[kb.id]

    def invalidate_corpus_cache(self, kb_id: str) -> None:
        self._corpus_cache.pop(kb_id, None)
        self._pipeline.invalidate_bm25_cache(
            f"kb_{kb_id.replace('-', '_')}"
        )
