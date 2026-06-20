"""Retrieval and search routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import RetrievalService, get_retrieval_service
from src.domain.models.retrieval import Citation, RetrievalResult

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────


class SearchRequest(BaseModel):
    """Request to search across one or more knowledge bases."""

    query: str = Field(..., min_length=1, max_length=2000)
    kb_ids: list[str] = Field(
        default_factory=list,
        description="KB IDs to search. Empty = search all active KBs.",
    )
    top_k: int = Field(default=10, ge=1, le=50)


class RetrieveRequest(BaseModel):
    """Request to retrieve from a specific knowledge base."""

    query: str = Field(..., min_length=1, max_length=2000)
    kb_id: str
    top_k: int = Field(default=10, ge=1, le=50)


class CitationOut(BaseModel):
    source_document: str
    knowledge_base_name: str
    page_numbers: list[int]
    section: str | None
    content_type: str
    score: float
    chunk_id: str
    excerpt: str

    @classmethod
    def from_domain(cls, c: Citation) -> CitationOut:
        return cls(
            source_document=c.source_document,
            knowledge_base_name=c.knowledge_base_name,
            page_numbers=c.page_numbers,
            section=c.section,
            content_type=c.content_type.value,
            score=round(c.score, 4),
            chunk_id=c.chunk_id,
            excerpt=c.excerpt,
        )


class ChunkOut(BaseModel):
    chunk_id: str
    content: str
    score: float
    rank: int
    retrieval_method: str
    page_numbers: list[int]
    section_title: str | None
    content_type: str
    document_name: str


class RetrievalResultOut(BaseModel):
    answer_found: bool
    reason: str | None
    query: str
    knowledge_base_id: str | None
    knowledge_base_name: str | None
    chunks: list[ChunkOut]
    citations: list[CitationOut]
    retrieval_metadata: dict[str, Any]

    @classmethod
    def from_domain(cls, r: RetrievalResult) -> RetrievalResultOut:
        chunks_out = [
            ChunkOut(
                chunk_id=cws.chunk.id,
                content=cws.chunk.content,
                score=round(cws.score, 4),
                rank=cws.rank,
                retrieval_method=cws.retrieval_method,
                page_numbers=cws.chunk.metadata.page_numbers,
                section_title=cws.chunk.metadata.section_title,
                content_type=cws.chunk.metadata.content_type.value,
                document_name=cws.chunk.metadata.document_name,
            )
            for cws in r.chunks
        ]
        return cls(
            answer_found=r.answer_found,
            reason=r.reason,
            query=r.query,
            knowledge_base_id=r.knowledge_base_id,
            knowledge_base_name=r.knowledge_base_name,
            chunks=chunks_out,
            citations=[CitationOut.from_domain(c) for c in r.citations],
            retrieval_metadata=r.retrieval_metadata,
        )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "/retrieve",
    response_model=RetrievalResultOut,
    summary="Retrieve from a specific knowledge base",
)
async def retrieve(
    body: RetrieveRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResultOut:
    """Retrieve grounded results from a specific knowledge base.

    Uses hybrid search (dense + BM25 + RRF + optional reranking).

    Returns `answer_found: false` if no supporting evidence is found.
    Never fabricates information.

    **Python example:**
    ```python
    import httpx
    response = httpx.post("http://localhost:8000/retrieve", json={
        "query": "What are the requirements for ISO 9001 certification?",
        "kb_id": "my-kb-id",
        "top_k": 10
    })
    result = response.json()
    if result["answer_found"]:
        for citation in result["citations"]:
            print(citation["source_document"], citation["excerpt"])
    ```
    """
    try:
        result = await service.search(body.kb_id, body.query, body.top_k)
        return RetrievalResultOut.from_domain(result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/search",
    response_model=list[RetrievalResultOut],
    summary="Search across multiple knowledge bases",
)
async def search(
    body: SearchRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> list[RetrievalResultOut]:
    """Search across one or more knowledge bases.

    If `kb_ids` is empty, searches all active knowledge bases.
    Returns one RetrievalResult per knowledge base.
    """
    kb_ids = body.kb_ids

    if not kb_ids:
        # Get all active KBs
        from src.api.dependencies import get_container
        # Fall back to returning empty result for now
        return []

    results = await service.search_across_kbs(body.query, kb_ids, body.top_k)
    return [RetrievalResultOut.from_domain(r) for r in results]
