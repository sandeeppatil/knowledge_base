"""Knowledge base CRUD routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import KnowledgeBaseService, get_kb_service
from src.domain.models.knowledge_base import KnowledgeBase

router = APIRouter()


# ─── Request / Response schemas ───────────────────────────────────────────────


class CreateKBRequest(BaseModel):
    """Request body for creating a knowledge base."""

    name: str = Field(..., min_length=1, max_length=200, examples=["ISO Standards"])
    description: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        examples=["ISO quality management and safety standards documents."],
    )
    embedding_model: str | None = Field(
        default=None,
        examples=["BAAI/bge-m3"],
        description="Override the default embedding model for this KB.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class KBResponse(BaseModel):
    """Knowledge base response schema."""

    id: str
    name: str
    description: str
    document_count: int
    chunk_count: int
    created_at: str
    updated_at: str
    embedding_model: str
    vector_store: str
    version: str
    is_active: bool
    metadata: dict[str, Any]

    @classmethod
    def from_domain(cls, kb: KnowledgeBase) -> KBResponse:
        return cls(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            document_count=kb.document_count,
            chunk_count=kb.chunk_count,
            created_at=kb.created_at.isoformat(),
            updated_at=kb.updated_at.isoformat(),
            embedding_model=kb.embedding_model,
            vector_store=kb.vector_store,
            version=kb.version,
            is_active=kb.is_active,
            metadata=kb.metadata,
        )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=KBResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a knowledge base",
)
async def create_knowledge_base(
    body: CreateKBRequest,
    service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
) -> KBResponse:
    """Create a new, empty knowledge base.

    The KB description is used by the routing agent to select the most
    relevant KB for a given query.  Write a clear, informative description.

    **Example request:**
    ```json
    {
      "name": "ISO Standards",
      "description": "ISO quality, safety, and environmental management standards."
    }
    ```
    """
    try:
        kb = await service.create(
            name=body.name,
            description=body.description,
            embedding_model=body.embedding_model,
            metadata=body.metadata,
        )
        return KBResponse.from_domain(kb)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.get("", response_model=list[KBResponse], summary="List all knowledge bases")
async def list_knowledge_bases(
    service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
) -> list[KBResponse]:
    """Return all active knowledge bases."""
    kbs = await service.list()
    return [KBResponse.from_domain(kb) for kb in kbs]


@router.get("/{kb_id}", response_model=KBResponse, summary="Get a knowledge base")
async def get_knowledge_base(
    kb_id: str,
    service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
) -> KBResponse:
    """Return a single knowledge base by ID."""
    kb = await service.get(kb_id)
    if not kb:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found.")
    return KBResponse.from_domain(kb)


@router.delete(
    "/{kb_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a knowledge base",
)
async def delete_knowledge_base(
    kb_id: str,
    service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
) -> None:
    """Permanently delete a knowledge base and all its documents and vectors."""
    deleted = await service.delete(kb_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge base not found.")
