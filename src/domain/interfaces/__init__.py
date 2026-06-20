"""Domain interfaces: abstract contracts for all pluggable components.

Every concrete implementation (Qdrant, ChromaDB, FAISS, SentenceTransformers,
Docling, etc.) must implement the appropriate interface defined here.  Business
logic and API routes depend ONLY on these interfaces, never on concrete classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models.chunk import Chunk, ChunkWithScore
from ..models.document import Document
from ..models.knowledge_base import KnowledgeBase


# ─── Vector Store ─────────────────────────────────────────────────────────────


class VectorStore(ABC):
    """Abstract vector store — manage collections and perform similarity search.

    One collection per knowledge base.  Collection names are derived from the
    KnowledgeBase.collection_name property.
    """

    @abstractmethod
    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection for a knowledge base.

        Args:
            name: Collection name (KB.collection_name).
            dimension: Embedding dimensionality.

        Raises:
            VectorStoreError: If creation fails.
        """
        ...

    @abstractmethod
    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its vectors.

        Args:
            name: Collection name to delete.
        """
        ...

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Return True if a collection with the given name exists."""
        ...

    @abstractmethod
    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Insert or update chunks in the collection.

        Args:
            collection: Target collection name.
            chunks: Chunks with populated embeddings.

        Raises:
            ValueError: If any chunk is missing an embedding.
        """
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkWithScore]:
        """Perform approximate-nearest-neighbour search.

        Args:
            collection: Collection to search.
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filters: Optional payload filters (provider-specific).

        Returns:
            List of ChunkWithScore ordered by descending relevance.
        """
        ...

    @abstractmethod
    async def delete_by_document(self, collection: str, document_id: str) -> int:
        """Delete all chunks belonging to a document.

        Args:
            collection: Collection name.
            document_id: Document whose chunks should be removed.

        Returns:
            Number of chunks deleted.
        """
        ...

    @abstractmethod
    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Return metadata about a collection (point count, dimension, etc.)."""
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Return all collection names."""
        ...


# ─── Embedding Provider ───────────────────────────────────────────────────────


class EmbeddingProvider(ABC):
    """Abstract embedding provider — convert text to dense vectors."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input string to embed.

        Returns:
            Dense embedding vector as a Python list of floats.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, same order as input.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying embedding model."""
        ...


# ─── Document Parser ──────────────────────────────────────────────────────────


class ParsedPage(ABC):
    """Represents a single parsed page from a document."""


class ParsedDocument(ABC):
    """Represents the full parsed output of a document."""

    @property
    @abstractmethod
    def pages(self) -> list[Any]:
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        ...


class DocumentParser(ABC):
    """Abstract document parser — extract structured content from a file."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return True if this parser can handle the given file.

        Args:
            path: Path to the source file.
        """
        ...

    @abstractmethod
    async def parse(self, path: Path, document: Document) -> ParsedDocument:
        """Parse a file and return structured content.

        Args:
            path: Absolute path to the source file.
            document: Domain Document being processed (metadata, IDs).

        Returns:
            ParsedDocument containing pages, tables, figures, metadata.

        Raises:
            ParserError: If parsing fails.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique parser name (e.g. 'docling', 'pymupdf', 'ocr')."""
        ...


# ─── Chunker ─────────────────────────────────────────────────────────────────


class Chunker(ABC):
    """Abstract chunker — split parsed content into vector-store-ready chunks."""

    @abstractmethod
    async def chunk(self, parsed_doc: ParsedDocument, document: Document) -> list[Chunk]:
        """Produce chunks from a parsed document.

        Args:
            parsed_doc: Output from a DocumentParser.
            document: Source Document for metadata population.

        Returns:
            List of Chunk objects (without embeddings — those are added next).
        """
        ...


# ─── Reranker ────────────────────────────────────────────────────────────────


class Reranker(ABC):
    """Abstract reranker — reorder a candidate list by cross-encoder relevance."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[ChunkWithScore],
        top_k: int,
    ) -> list[ChunkWithScore]:
        """Rerank candidates and return the top-k results.

        Args:
            query: Original user query.
            candidates: Pre-retrieved candidates to rerank.
            top_k: Maximum number of results to return.

        Returns:
            Reranked list (highest cross-encoder score first), len <= top_k.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the underlying reranker model."""
        ...


# ─── KB Repository ────────────────────────────────────────────────────────────


class KBRepository(ABC):
    """Abstract repository for KnowledgeBase and Document persistence."""

    # ── Knowledge Base operations ──────────────────────────────────────────
    @abstractmethod
    async def create_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Persist a new KnowledgeBase record."""
        ...

    @abstractmethod
    async def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        """Return the KB with the given id, or None if not found."""
        ...

    @abstractmethod
    async def get_kb_by_name(self, name: str) -> KnowledgeBase | None:
        """Return the KB with the given name, or None."""
        ...

    @abstractmethod
    async def list_kbs(self) -> list[KnowledgeBase]:
        """Return all active knowledge bases."""
        ...

    @abstractmethod
    async def update_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Persist changes to an existing KnowledgeBase."""
        ...

    @abstractmethod
    async def delete_kb(self, kb_id: str) -> bool:
        """Delete a KnowledgeBase and all associated documents.

        Returns:
            True if deleted, False if not found.
        """
        ...

    # ── Document operations ────────────────────────────────────────────────
    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Persist a new Document record."""
        ...

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        """Return the Document with the given id, or None."""
        ...

    @abstractmethod
    async def list_documents(self, kb_id: str) -> list[Document]:
        """Return all documents belonging to the given KB."""
        ...

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Persist changes to an existing Document."""
        ...

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a Document record.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def get_document_by_checksum(self, kb_id: str, checksum: str) -> Document | None:
        """Return a document with matching checksum in the given KB (dedup)."""
        ...
