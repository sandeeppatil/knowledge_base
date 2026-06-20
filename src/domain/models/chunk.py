"""Chunk model for the knowledge base platform."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ContentType(str, Enum):
    """Type of content in a chunk."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"


class ChunkMetadata(BaseModel):
    """Metadata about a chunk."""

    content_type: ContentType = Field(default=ContentType.TEXT)
    is_synthetic: bool = Field(default=False, description="Generated or original")
    confidence: Optional[float] = Field(None, description="Extraction confidence score")
    custom_fields: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """Represents a single chunk of content from a document."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    kb_id: str = Field(..., description="Knowledge base ID")
    content: str = Field(..., description="Text content of the chunk")
    embedding: Optional[list[float]] = Field(None, description="Dense vector embedding")
    page_numbers: Optional[list[int]] = Field(None, description="Source page numbers")
    section_title: Optional[str] = Field(None, description="Section heading")
    heading_path: Optional[list[str]] = Field(None, description="Hierarchy of headings")
    source_document: str = Field(..., description="Document filename or reference")
    chunk_index: int = Field(0, description="Index within document")
    metadata: dict = Field(default_factory=dict, description="Custom metadata")
    chunk_metadata: Optional[ChunkMetadata] = Field(None, description="Structured chunk metadata")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "chunk_123",
                "document_id": "doc_456",
                "kb_id": "kb_789",
                "content": "This is a chunk of text...",
                "page_numbers": [1],
                "section_title": "Introduction",
                "source_document": "document.pdf",
                "chunk_index": 0,
            }
        }


class ChunkWithScore(BaseModel):
    """Chunk with relevance score from retrieval."""

    chunk: Chunk
    score: float = Field(..., description="Relevance score (higher is better)")
    rank: int = Field(default=0, description="Result rank position")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "chunk": {},
                "score": 0.95,
                "rank": 1,
            }
        }
