"""Qdrant vector store implementation.

Qdrant is the recommended default vector store.  This implementation uses
the official qdrant-client library with async support.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.domain.interfaces import VectorStore
from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.monitoring.logging import get_logger
from src.monitoring.tracing import get_tracer

logger = get_logger(__name__)
tracer = get_tracer(__name__)


class QdrantVectorStore(VectorStore):
    """VectorStore implementation backed by Qdrant.

    Args:
        url: Qdrant server URL.
        api_key: Optional API key for authentication.
        prefer_grpc: Use gRPC transport when True.
        timeout: Request timeout in seconds.

    Example:
        >>> store = QdrantVectorStore(url="http://localhost:6333")
        >>> await store.create_collection("kb_iso_standards", dimension=1024)
        >>> await store.upsert("kb_iso_standards", chunks)
        >>> results = await store.search("kb_iso_standards", query_vector, top_k=10)
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
    ) -> None:
        self._client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
        )

    async def create_collection(self, name: str, dimension: int) -> None:
        """Create a Qdrant collection with HNSW index.

        Args:
            name: Collection name.
            dimension: Vector dimension.
        """
        with tracer.start_as_current_span("qdrant.create_collection"):
            try:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=dimension,
                        distance=qmodels.Distance.COSINE,
                        on_disk=False,
                    ),
                    hnsw_config=qmodels.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10000,
                    ),
                    optimizers_config=qmodels.OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                )
                logger.info("Qdrant collection created", name=name, dimension=dimension)
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    logger.debug("Collection already exists", name=name)
                else:
                    raise

    async def delete_collection(self, name: str) -> None:
        with tracer.start_as_current_span("qdrant.delete_collection"):
            await self._client.delete_collection(collection_name=name)
            logger.info("Qdrant collection deleted", name=name)

    async def collection_exists(self, name: str) -> bool:
        try:
            await self._client.get_collection(collection_name=name)
            return True
        except Exception:
            return False

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        """Upsert chunks into a Qdrant collection.

        Args:
            collection: Collection name.
            chunks: Chunks that must have non-None embeddings.

        Raises:
            ValueError: If any chunk is missing an embedding.
        """
        if not chunks:
            return

        missing = [c.id for c in chunks if not c.has_embedding()]
        if missing:
            raise ValueError(f"Chunks missing embeddings: {missing}")

        points = []
        for chunk in chunks:
            payload = self._chunk_to_payload(chunk)
            points.append(
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id)),
                    vector=chunk.embedding,  # type: ignore[arg-type]
                    payload=payload,
                )
            )

        with tracer.start_as_current_span("qdrant.upsert") as span:
            span.set_attribute("collection", collection)
            span.set_attribute("count", len(points))
            await self._client.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )
        logger.debug("Upserted chunks", collection=collection, count=len(chunks))

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkWithScore]:
        """Cosine similarity search in Qdrant.

        Args:
            collection: Collection name.
            query_vector: Query embedding.
            top_k: Number of results.
            filters: Optional Qdrant payload filter dict.

        Returns:
            List of ChunkWithScore ordered by descending score.
        """
        qdrant_filter = None
        if filters:
            qdrant_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=k,
                        match=qmodels.MatchValue(value=v),
                    )
                    for k, v in filters.items()
                ]
            )

        with tracer.start_as_current_span("qdrant.search") as span:
            span.set_attribute("collection", collection)
            span.set_attribute("top_k", top_k)
            results = await self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        return [
            ChunkWithScore(
                chunk=self._payload_to_chunk(r.payload or {}),
                score=max(0.0, min(1.0, float(r.score))),
                rank=idx + 1,
                retrieval_method="dense",
            )
            for idx, r in enumerate(results)
        ]

    async def delete_by_document(self, collection: str, document_id: str) -> int:
        result = await self._client.delete(
            collection_name=collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        deleted = result.result if result.result else 0
        logger.info(
            "Deleted chunks by document",
            collection=collection,
            document_id=document_id,
            deleted=deleted,
        )
        return int(deleted)

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        info = await self._client.get_collection(collection_name=collection)
        return {
            "name": collection,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }

    async def list_collections(self) -> list[str]:
        result = await self._client.get_collections()
        return [c.name for c in result.collections]

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            # Top-level for fast payload filtering
            "chunk_id": chunk.id,
            "document_id": meta.document_id,
            "document_name": meta.document_name,
            "knowledge_base_id": meta.knowledge_base_id,
            "knowledge_base_name": meta.knowledge_base_name,
            "content_type": meta.content_type.value,
            # Full content
            "content": chunk.content,
            "content_markdown": chunk.content_markdown,
            "content_json": json.dumps(chunk.content_json) if chunk.content_json else None,
            "token_count": chunk.token_count,
            # Provenance
            "page_numbers": meta.page_numbers,
            "section_title": meta.section_title,
            "subsection_title": meta.subsection_title,
            "source_path": meta.source_path,
            "heading_path": meta.heading_path,
            "table_id": meta.table_id,
            "table_title": meta.table_title,
            "figure_id": meta.figure_id,
            "figure_caption": meta.figure_caption,
            "created_at": meta.created_at.isoformat(),
        }

    @staticmethod
    def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
        content_json_raw = payload.get("content_json")
        content_json = json.loads(content_json_raw) if content_json_raw else None

        meta = ChunkMetadata(
            chunk_id=payload.get("chunk_id", ""),
            document_id=payload.get("document_id", ""),
            document_name=payload.get("document_name", ""),
            knowledge_base_id=payload.get("knowledge_base_id", ""),
            knowledge_base_name=payload.get("knowledge_base_name", ""),
            page_numbers=payload.get("page_numbers", []),
            section_title=payload.get("section_title"),
            subsection_title=payload.get("subsection_title"),
            content_type=ContentType(payload.get("content_type", "text")),
            source_path=payload.get("source_path", ""),
            heading_path=payload.get("heading_path", []),
            table_id=payload.get("table_id"),
            table_title=payload.get("table_title"),
            figure_id=payload.get("figure_id"),
            figure_caption=payload.get("figure_caption"),
        )

        return Chunk(
            id=payload.get("chunk_id", ""),
            content=payload.get("content", ""),
            content_markdown=payload.get("content_markdown"),
            content_json=content_json,
            token_count=payload.get("token_count", 0),
            metadata=meta,
        )
