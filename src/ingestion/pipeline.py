"""Ingestion pipeline — orchestrates parsing, chunking, embedding, and indexing.

Pipeline stages:
    File
      ↓  Validate & checksum
      ↓  Parse (Docling → PyMuPDF → OCR)
      ↓  Chunk (hierarchical)
      ↓  Embed (batch)
      ↓  Index (vector store upsert)
      ↓  Persist metadata (SQLite)
"""

from __future__ import annotations

import hashlib
import time
import uuid
import mimetypes
from pathlib import Path

from src.domain.interfaces import EmbeddingProvider, KBRepository, VectorStore
from src.domain.models.document import Document, DocumentStatus, DocumentType
from src.domain.models.knowledge_base import KnowledgeBase
from src.domain.events.domain_events import DocumentIngested, DocumentIngestionFailed
from src.monitoring.logging import get_logger
from src.monitoring.metrics import (
    CHUNKS_CREATED_TOTAL,
    DOCUMENTS_FAILED_TOTAL,
    DOCUMENTS_INGESTED_TOTAL,
    INGESTION_DURATION_SECONDS,
)
from src.monitoring.tracing import get_tracer
from src.parsers.registry import ParserRegistry
from src.chunking.hierarchical_chunker import HierarchicalChunker

logger = get_logger(__name__)
tracer = get_tracer(__name__)

# Maximum file size guard (configurable)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


class IngestionPipeline:
    """End-to-end document ingestion pipeline.

    Args:
        parser_registry: Registry of document parsers.
        chunker: Chunking strategy implementation.
        embedding_provider: Provider for generating embeddings.
        vector_store: Vector store for indexing chunks.
        kb_repository: Repository for persisting metadata.
        batch_embed_size: Number of chunks to embed per batch.

    Example:
        >>> pipeline = IngestionPipeline(registry, chunker, embedder, store, repo)
        >>> doc = await pipeline.ingest(path, kb)
    """

    def __init__(
        self,
        parser_registry: ParserRegistry,
        chunker: HierarchicalChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        kb_repository: KBRepository,
        batch_embed_size: int = 64,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedder = embedding_provider
        self._vector_store = vector_store
        self._repo = kb_repository
        self._batch_embed_size = batch_embed_size

    async def ingest(
        self,
        file_path: Path,
        kb: KnowledgeBase,
        skip_duplicates: bool = True,
    ) -> Document:
        """Ingest a single document into the knowledge base.

        Args:
            file_path: Absolute path to the source file.
            kb: Target KnowledgeBase.
            skip_duplicates: Skip documents with matching SHA-256 checksum.

        Returns:
            Document with updated status (COMPLETED or FAILED).

        Raises:
            ValueError: If file is invalid or exceeds size limits.
        """
        t0 = time.monotonic()

        # ── Validate file ────────────────────────────────────────────────
        self._validate_file(file_path)

        checksum = self._compute_checksum(file_path)

        # ── Create document record ───────────────────────────────────────
        doc_id = str(uuid.uuid4())
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        
        # Determine document type from file extension
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            doc_type = DocumentType.PDF
        elif suffix in {".md", ".markdown"}:
            doc_type = DocumentType.MARKDOWN
        elif suffix in {".txt"}:
            doc_type = DocumentType.TEXT
        elif suffix in {".html", ".htm"}:
            doc_type = DocumentType.HTML
        elif suffix in {".docx"}:
            doc_type = DocumentType.DOCX
        else:
            doc_type = DocumentType.UNKNOWN
        
        document = Document(
            id=doc_id,
            kb_id=kb.id,
            filename=file_path.name,
            file_path=str(file_path),
            checksum=checksum,
            size_bytes=file_path.stat().st_size,
            status=DocumentStatus.PENDING,
            document_type=doc_type,
            mime_type=mime_type,
        )
        document = await self._repo.create_document(document)

        # ── Duplicate check ──────────────────────────────────────────────
        if skip_duplicates:
            existing = await self._repo.get_document_by_checksum(kb.id, checksum)
            if existing and existing.id != document.id and existing.status == DocumentStatus.INDEXED:
                logger.info(
                    "Skipping duplicate document",
                    name=file_path.name,
                    existing_id=existing.id,
                )
                document.status = DocumentStatus.FAILED
                document.error_message = "Duplicate document — already ingested."
                await self._repo.update_document(document)
                return document

        with tracer.start_as_current_span("ingestion_pipeline") as span:
            span.set_attribute("document", file_path.name)
            span.set_attribute("kb_id", kb.id)

            try:
                # ── Select parser ─────────────────────────────────────────
                parser = self._parser_registry.select(file_path)
                if parser is None:
                    raise ValueError(f"No parser found for file type: {file_path.suffix}")

                logger.info(
                    "Ingesting document",
                    name=file_path.name,
                    parser=parser.name,
                    kb=kb.name,
                )

                # ── Parse ─────────────────────────────────────────────────
                parsed_doc = await parser.parse(file_path, document)

                # ── Chunk ─────────────────────────────────────────────────
                chunks = await self._chunker.chunk(parsed_doc, document)

                if not chunks:
                    raise ValueError("Parser produced no chunks — document may be empty.")

                # Populate KB name on all chunks
                for chunk in chunks:
                    chunk.metadata.knowledge_base_name = kb.name

                # ── Embed in batches ──────────────────────────────────────
                await self._embed_chunks(chunks)

                # ── Index into vector store ───────────────────────────────
                collection = kb.collection_name
                if not await self._vector_store.collection_exists(collection):
                    await self._vector_store.create_collection(
                        collection, self._embedder.dimension
                    )

                for i in range(0, len(chunks), self._batch_embed_size):
                    batch = chunks[i : i + self._batch_embed_size]
                    await self._vector_store.upsert(collection, batch)

                # ── Update document status ────────────────────────────────
                document.mark_completed(
                    chunk_count=len(chunks),
                    page_count=getattr(parsed_doc, "page_count", 0),
                    parser_used=parser.name,
                )
                await self._repo.update_document(document)

                # ── Update KB counters ────────────────────────────────────
                kb.document_count += 1
                kb.chunk_count += len(chunks)
                kb.touch()
                await self._repo.update_kb(kb)

                duration = time.monotonic() - t0

                # ── Metrics ───────────────────────────────────────────────
                DOCUMENTS_INGESTED_TOTAL.labels(
                    knowledge_base=kb.name,
                    document_type=document.document_type.value,
                    parser=parser.name,
                ).inc()
                for chunk in chunks:
                    CHUNKS_CREATED_TOTAL.labels(
                        knowledge_base=kb.name,
                        content_type=chunk.metadata.content_type.value,
                    ).inc()
                INGESTION_DURATION_SECONDS.labels(
                    knowledge_base=kb.name, parser=parser.name
                ).observe(duration)

                logger.info(
                    "Document ingested successfully",
                    name=file_path.name,
                    chunks=len(chunks),
                    pages=document.page_count,
                    duration_s=round(duration, 2),
                )

            except Exception as exc:
                logger.error(
                    "Document ingestion failed",
                    name=file_path.name,
                    error=str(exc),
                )
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)
                await self._repo.update_document(document)

                DOCUMENTS_FAILED_TOTAL.labels(
                    knowledge_base=kb.name,
                    error_type=type(exc).__name__,
                ).inc()

        return document

    async def ingest_folder(
        self,
        folder: Path,
        kb: KnowledgeBase,
        glob_pattern: str = "**/*.pdf",
        skip_duplicates: bool = True,
    ) -> list[Document]:
        """Ingest all matching files from a folder.

        Args:
            folder: Root folder to scan.
            kb: Target knowledge base.
            glob_pattern: Glob pattern for file matching.
            skip_duplicates: Skip duplicate files by checksum.

        Returns:
            List of Document objects (one per file found).
        """
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        files = list(folder.glob(glob_pattern))
        logger.info(
            "Starting folder ingestion",
            folder=str(folder),
            file_count=len(files),
            kb=kb.name,
        )

        results: list[Document] = []
        for file_path in files:
            if file_path.is_file():
                doc = await self.ingest(file_path, kb, skip_duplicates)
                results.append(doc)

        completed = sum(1 for d in results if d.status == DocumentStatus.COMPLETED)
        failed = sum(1 for d in results if d.status == DocumentStatus.FAILED)
        logger.info(
            "Folder ingestion complete",
            total=len(results),
            completed=completed,
            failed=failed,
        )
        return results

    async def _embed_chunks(self, chunks: list) -> None:
        """Embed all chunks in-place using batched encoding."""
        texts = [c.content for c in chunks]
        all_vectors = await self._embedder.embed_batch(texts)
        for chunk, vector in zip(chunks, all_vectors):
            chunk.embedding = vector

    @staticmethod
    def _validate_file(path: Path) -> None:
        """Validate a file before ingestion.

        Args:
            path: Path to validate.

        Raises:
            ValueError: If file is missing, too large, or is a path traversal.
        """
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a regular file: {path}")
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File exceeds maximum size ({MAX_FILE_SIZE_BYTES // 1024 // 1024} MB): {path}"
            )
        # Path traversal protection
        try:
            path.resolve().relative_to(path.resolve().anchor)
        except ValueError:
            raise ValueError(f"Invalid path: {path}")

    @staticmethod
    def _compute_checksum(path: Path) -> str:
        """Compute SHA-256 checksum of a file.

        Args:
            path: Path to the file.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
