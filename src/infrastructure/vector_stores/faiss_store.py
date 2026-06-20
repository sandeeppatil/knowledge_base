"""FAISS vector store implementation — fully local, in-process."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.interfaces import VectorStore
from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


class FaissVectorStore(VectorStore):
    """VectorStore backed by FAISS (fully local, no external service required).

    Each collection is stored as two files:
    - ``<index_path>/<collection>.index``   — FAISS index
    - ``<index_path>/<collection>.meta.pkl`` — chunk payloads and IDs

    Args:
        index_path: Directory where index files are persisted.

    Example:
        >>> store = FaissVectorStore(index_path="/data/faiss")
        >>> await store.create_collection("kb_company", dimension=768)
    """

    def __init__(self, index_path: str = "./data/faiss_indices") -> None:
        self._base_path = Path(index_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
        # In-memory cache: collection_name -> (faiss.Index, {id: chunk_payload})
        self._indices: dict[str, Any] = {}
        self._payloads: dict[str, dict[str, dict[str, Any]]] = {}

    def _index_file(self, name: str) -> Path:
        return self._base_path / f"{name}.index"

    def _meta_file(self, name: str) -> Path:
        return self._base_path / f"{name}.meta.pkl"

    def _load_collection(self, name: str) -> tuple[Any, dict[str, dict[str, Any]]]:
        import faiss

        if name in self._indices:
            return self._indices[name], self._payloads[name]

        idx_path = self._index_file(name)
        meta_path = self._meta_file(name)

        if idx_path.exists() and meta_path.exists():
            index = faiss.read_index(str(idx_path))
            with meta_path.open("rb") as f:
                payloads: dict[str, dict[str, Any]] = pickle.load(f)
            self._indices[name] = index
            self._payloads[name] = payloads
            return index, payloads

        raise KeyError(f"Collection '{name}' not found")

    def _save_collection(self, name: str) -> None:
        import faiss

        index = self._indices[name]
        payloads = self._payloads[name]
        faiss.write_index(index, str(self._index_file(name)))
        with self._meta_file(name).open("wb") as f:
            pickle.dump(payloads, f)

    async def create_collection(self, name: str, dimension: int) -> None:
        import faiss

        if name not in self._indices:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
            self._indices[name] = index
            self._payloads[name] = {}
            self._save_collection(name)
        logger.info("FAISS collection created", name=name, dimension=dimension)

    async def delete_collection(self, name: str) -> None:
        self._indices.pop(name, None)
        self._payloads.pop(name, None)
        for path in [self._index_file(name), self._meta_file(name)]:
            path.unlink(missing_ok=True)
        logger.info("FAISS collection deleted", name=name)

    async def collection_exists(self, name: str) -> bool:
        return self._index_file(name).exists() or name in self._indices

    async def upsert(self, collection: str, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        index, payloads = self._load_collection(collection)
        import faiss

        vectors = np.array([c.embedding for c in chunks if c.embedding], dtype="float32")
        # Use a stable integer ID derived from the chunk UUID
        ids = np.array(
            [abs(hash(c.id)) % (2**63) for c in chunks], dtype="int64"
        )

        faiss.normalize_L2(vectors)
        index.add_with_ids(vectors, ids)

        for chunk, cid in zip(chunks, ids):
            payloads[str(cid)] = {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "content_markdown": chunk.content_markdown,
                "metadata": chunk.metadata.model_dump(mode="json"),
            }

        self._save_collection(collection)
        logger.debug("FAISS upsert", collection=collection, count=len(chunks))

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[ChunkWithScore]:
        import faiss

        index, payloads = self._load_collection(collection)
        q = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, ids = index.search(q, top_k)

        results: list[ChunkWithScore] = []
        for rank, (score, fid) in enumerate(zip(scores[0], ids[0])):
            if fid < 0:
                continue
            payload = payloads.get(str(fid))
            if not payload:
                continue
            chunk = self._payload_to_chunk(payload)
            results.append(
                ChunkWithScore(
                    chunk=chunk,
                    score=max(0.0, min(1.0, float(score))),
                    rank=rank + 1,
                    retrieval_method="dense",
                )
            )
        return results

    async def delete_by_document(self, collection: str, document_id: str) -> int:
        # FAISS IndexIDMap supports selective removal via IDSelector
        index, payloads = self._load_collection(collection)
        to_delete = [
            int(fid)
            for fid, p in payloads.items()
            if p.get("metadata", {}).get("document_id") == document_id
        ]
        if to_delete:
            import faiss

            selector = faiss.IDSelectorBatch(len(to_delete), to_delete)
            index.remove_ids(selector)
            for fid in to_delete:
                payloads.pop(str(fid), None)
            self._save_collection(collection)
        return len(to_delete)

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        index, payloads = self._load_collection(collection)
        return {"name": collection, "count": index.ntotal}

    async def list_collections(self) -> list[str]:
        on_disk = [p.stem for p in self._base_path.glob("*.index")]
        in_memory = list(self._indices.keys())
        return list(set(on_disk + in_memory))

    @staticmethod
    def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
        raw_meta = payload.get("metadata", {})
        meta = ChunkMetadata(
            chunk_id=raw_meta.get("chunk_id", payload.get("chunk_id", "")),
            document_id=raw_meta.get("document_id", ""),
            document_name=raw_meta.get("document_name", ""),
            knowledge_base_id=raw_meta.get("knowledge_base_id", ""),
            knowledge_base_name=raw_meta.get("knowledge_base_name", ""),
            page_numbers=raw_meta.get("page_numbers", []),
            section_title=raw_meta.get("section_title"),
            subsection_title=raw_meta.get("subsection_title"),
            content_type=ContentType(raw_meta.get("content_type", "text")),
            source_path=raw_meta.get("source_path", ""),
        )
        return Chunk(
            id=payload.get("chunk_id", ""),
            content=payload.get("content", ""),
            content_markdown=payload.get("content_markdown"),
            metadata=meta,
        )
