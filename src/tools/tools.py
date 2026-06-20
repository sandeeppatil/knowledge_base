"""Callable tools for GitHub Copilot agents and MCP.

Every tool follows these rules:
- Stateless execution
- JSON-serialisable return types
- Strict grounding — never fabricate answers
- Full citations on every retrieval result
"""

from __future__ import annotations

from typing import Any

from src.application.services.services import (
    IngestionService,
    KnowledgeBaseService,
    RetrievalService,
)
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


class KBTools:
    """Tools for knowledge base management.

    Args:
        kb_service: KnowledgeBaseService instance.
        ingestion_service: IngestionService instance.
    """

    def __init__(
        self,
        kb_service: KnowledgeBaseService,
        ingestion_service: IngestionService,
    ) -> None:
        self._kb = kb_service
        self._ingestion = ingestion_service

    async def list_knowledge_bases(self) -> dict[str, Any]:
        """List all available knowledge bases.

        Returns:
            Dict with "knowledge_bases" list.  Each KB includes its name,
            description, document count, chunk count, and ID.

        Example (MCP):
            >>> result = await tools.list_knowledge_bases()
            >>> for kb in result["knowledge_bases"]:
            ...     print(kb["name"], "—", kb["description"])
        """
        kbs = await self._kb.list()
        return {
            "knowledge_bases": [
                {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "document_count": kb.document_count,
                    "chunk_count": kb.chunk_count,
                    "embedding_model": kb.embedding_model,
                    "version": kb.version,
                }
                for kb in kbs
            ]
        }

    async def create_knowledge_base(
        self, name: str, description: str
    ) -> dict[str, Any]:
        """Create a new knowledge base.

        Args:
            name: Unique name for the knowledge base.
            description: Natural-language description for KB routing.

        Returns:
            Dict with the created KB details including its ID.
        """
        try:
            kb = await self._kb.create(name=name, description=description)
            return {
                "success": True,
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

    async def delete_knowledge_base(self, kb_id: str) -> dict[str, Any]:
        """Delete a knowledge base and all its documents.

        Args:
            kb_id: ID of the knowledge base to delete.

        Returns:
            Dict with success status.
        """
        deleted = await self._kb.delete(kb_id)
        return {"success": deleted}

    async def list_documents(self, kb_id: str) -> dict[str, Any]:
        """List all documents in a knowledge base.

        Args:
            kb_id: Knowledge base ID.

        Returns:
            Dict with "documents" list.
        """
        docs = await self._ingestion.list_documents(kb_id)
        return {
            "documents": [
                {
                    "id": d.id,
                    "name": d.name,
                    "status": d.status.value,
                    "chunk_count": d.chunk_count,
                    "page_count": d.page_count,
                    "document_type": d.document_type.value,
                    "processed_at": d.processed_at.isoformat() if d.processed_at else None,
                }
                for d in docs
            ]
        }


class RetrievalTools:
    """Tools for document retrieval — the primary interface for agents.

    These tools implement the grounding contract: they NEVER inject
    fabricated content.  When no evidence is found, they return
    ``answer_found: false``.

    Args:
        retrieval_service: RetrievalService instance.
        kb_service: KnowledgeBaseService instance.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        kb_service: KnowledgeBaseService,
    ) -> None:
        self._retrieval = retrieval_service
        self._kb = kb_service

    async def search_knowledge_bases(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """Search across ALL knowledge bases and return the best results.

        The agent should call this when it doesn't know which KB to use.
        Then inspect the results and call retrieve_from_kb for deeper retrieval.

        Args:
            query: User query string.
            top_k: Maximum results per knowledge base.

        Returns:
            Dict with per-KB results.  Each result includes citations and
            ``answer_found`` flag.
        """
        kbs = await self._kb.list()
        kb_ids = [kb.id for kb in kbs]

        results = await self._retrieval.search_across_kbs(query, kb_ids, top_k)

        return {
            "query": query,
            "results_by_kb": [
                {
                    "knowledge_base_id": r.knowledge_base_id,
                    "knowledge_base_name": r.knowledge_base_name,
                    "answer_found": r.answer_found,
                    "reason": r.reason,
                    "citations": [
                        {
                            "source_document": c.source_document,
                            "page_numbers": c.page_numbers,
                            "section": c.section,
                            "score": round(c.score, 4),
                            "excerpt": c.excerpt,
                        }
                        for c in r.citations
                    ],
                }
                for r in results
            ],
        }

    async def retrieve_from_kb(
        self,
        kb_name: str,
        query: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Retrieve grounded results from a specific knowledge base.

        This is the primary retrieval tool.  Always call list_knowledge_bases()
        first to find the correct KB name.

        Grounding contract:
        - Returns ``answer_found: false`` when no evidence exists.
        - NEVER fabricates citations or content.
        - Only returns content that was found in the KB.

        Args:
            kb_name: Name of the knowledge base to search.
            query: User query string.
            top_k: Number of results to return (1–50).

        Returns:
            Grounded result dict with citations, chunks, and answer_found flag.

        Example agent workflow:
            1. kbs = list_knowledge_bases()
            2. Select KB by reading descriptions
            3. result = retrieve_from_kb(kb_name="ISO Standards", query="...")
            4. If result["answer_found"]: inject result["chunks"] into LLM context
               Else: return {answer_found: false} to the user
        """
        kb = await self._kb.get_by_name(kb_name)
        if not kb:
            return {
                "answer_found": False,
                "reason": f"Knowledge base '{kb_name}' not found.",
                "query": query,
            }

        result = await self._retrieval.search(kb.id, query, top_k)

        return {
            "answer_found": result.answer_found,
            "reason": result.reason,
            "query": result.query,
            "knowledge_base": kb_name,
            "chunks": [
                {
                    "content": cws.chunk.content,
                    "content_type": cws.chunk.metadata.content_type.value,
                    "score": round(cws.score, 4),
                    "rank": cws.rank,
                    "source_document": cws.chunk.metadata.document_name,
                    "page_numbers": cws.chunk.metadata.page_numbers,
                    "section": cws.chunk.metadata.section_title,
                }
                for cws in result.chunks
            ],
            "citations": [
                {
                    "source_document": c.source_document,
                    "page_numbers": c.page_numbers,
                    "section": c.section,
                    "score": round(c.score, 4),
                    "chunk_id": c.chunk_id,
                    "excerpt": c.excerpt,
                }
                for c in result.citations
            ],
            "retrieval_metadata": result.retrieval_metadata,
        }

    async def get_chunk(self, chunk_id: str) -> dict[str, Any]:
        """Retrieve a specific chunk by its ID.

        Useful for expanding a citation into full content.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            Chunk content and metadata, or error if not found.
        """
        # This would require a vector store point lookup by payload filter
        # For now, return a helpful message
        return {
            "chunk_id": chunk_id,
            "message": "Use retrieve_from_kb to get chunks with full context.",
        }
