"""Retrieval result model for the knowledge base platform."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional

from .chunk import ChunkWithScore


class Citation(BaseModel):
    """Citation information for a chunk."""

    source_document: str = Field(..., description="Source document filename or ID")
    page_numbers: list[int] = Field(default_factory=list, description="Page numbers")
    section_title: Optional[str] = Field(None, description="Section title if available")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk in document")


class RetrievalResult(BaseModel):
    """Result from retrieval pipeline."""

    query: str = Field(..., description="Original query")
    kb_id: str = Field(..., description="Knowledge base searched")
    answer_found: bool = Field(..., description="Whether supporting evidence was found")
    chunks: list[ChunkWithScore] = Field(default_factory=list, description="Retrieved chunks")
    reason: Optional[str] = Field(None, description="Explanation if answer_found is False")
    total_candidates_searched: int = Field(default=0, description="Total chunks considered")
    retrieval_time_ms: float = Field(default=0.0, description="Retrieval latency in ms")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "query": "What is the policy on remote work?",
                "kb_id": "kb_123",
                "answer_found": True,
                "chunks": [],
                "total_candidates_searched": 100,
                "retrieval_time_ms": 245.5,
            }
        }

    @classmethod
    def not_found(cls, query: str, kb_id: str = "", reason: str = "") -> RetrievalResult:
        """Create a result indicating no supporting evidence was found."""
        return cls(
            query=query,
            kb_id=kb_id,
            answer_found=False,
            reason=reason or "No supporting evidence found in selected knowledge base.",
            chunks=[],
        )
