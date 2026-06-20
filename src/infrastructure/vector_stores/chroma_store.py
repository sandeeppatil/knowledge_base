"""ChromaDB vector store implementation."""

from __future__ import annotations

import json
from typing import Any

from src.domain.interfaces import VectorStore
from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(VectorStore):
    """VectorStore implementation backed by ChromaDB.

    Args:
        host: ChromaDB server host.
        port: ChromaDB server port.

    Example:
        >>> store = ChromaVectorStore(host="localhost", port=8001)
        >>> await store.create_collection("kb_research", dimension=1024)
    """

    def __init__(self, host: str = "localhost", port: int = 8001) -> None:
        self._host = host
        self._port = port
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import chromadb

            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        return self._client

    async def create_collection(self, name: str, dimension: int) -> None:
        client = self._get_client()
        client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine", "dimension": dimension},
        )
        logger.info("Chroma collection created/ensured", name=name)

    async def delete_collection(self, name: str) -> None:
        client = self._get_client()
        client.delete_collection(name=name)
        logger.info("Chroma collection deleted", name=name)

    async def collection_exists(self, name: str) -> bool:
        try:
            client = self._get_client()
            client.get_collection(name=name)
            return True
        except Exception:
            return False

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        client = self._get_client()
        col = client.get_collection(name=collection)

        ids = [c.id for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding]
        documents = [c.content for c in chunks]
        metadatas = [self._chunk_to_meta(c) for c in chunks]

        col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.debug("Chroma upsert", collection=collection, count=len(chunks))

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkWithScore]:
        client = self._get_client()
        col = client.get_collection(name=collection)

        where = {f"${k}": v for k, v in filters.items()} if filters else None
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        output: list[ChunkWithScore] = []
        for idx, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            chunk = self._meta_to_chunk(doc_id, doc, meta)
            output.append(
                ChunkWithScore(chunk=chunk, score=score, rank=idx + 1, retrieval_method="dense")
            )
        return output

    async def delete_by_document(self, collection: str, document_id: str) -> int:
        client = self._get_client()
        col = client.get_collection(name=collection)
        existing = col.get(where={"document_id": document_id})
        ids = existing["ids"]
        if ids:
            col.delete(ids=ids)
        return len(ids)

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        client = self._get_client()
        col = client.get_collection(name=collection)
        return {"name": collection, "count": col.count()}

    async def list_collections(self) -> list[str]:
        client = self._get_client()
        return [c.name for c in client.list_collections()]

    @staticmethod
    def _chunk_to_meta(chunk: Chunk) -> dict[str, Any]:
        meta = chunk.metadata
        return {
            "chunk_id": chunk.id,
            "document_id": meta.document_id,
            "document_name": meta.document_name,
            "knowledge_base_id": meta.knowledge_base_id,
            "knowledge_base_name": meta.knowledge_base_name,
            "content_type": meta.content_type.value,
            "page_numbers": json.dumps(meta.page_numbers),
            "section_title": meta.section_title or "",
            "subsection_title": meta.subsection_title or "",
            "source_path": meta.source_path,
        }

    @staticmethod
    def _meta_to_chunk(doc_id: str, content: str, meta: dict[str, Any]) -> Chunk:
        page_numbers = json.loads(meta.get("page_numbers", "[]"))
        chroma_meta = ChunkMetadata(
            chunk_id=doc_id,
            document_id=meta.get("document_id", ""),
            document_name=meta.get("document_name", ""),
            knowledge_base_id=meta.get("knowledge_base_id", ""),
            knowledge_base_name=meta.get("knowledge_base_name", ""),
            page_numbers=page_numbers,
            section_title=meta.get("section_title") or None,
            subsection_title=meta.get("subsection_title") or None,
            content_type=ContentType(meta.get("content_type", "text")),
            source_path=meta.get("source_path", ""),
        )
        return Chunk(id=doc_id, content=content, metadata=chroma_meta)
