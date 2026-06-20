"""Document model for the knowledge base platform."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document type."""

    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    DOCX = "docx"
    UNKNOWN = "unknown"


class Document(BaseModel):
    """Represents a document in a knowledge base."""

    id: str = Field(..., description="Unique document ID")
    kb_id: str = Field(..., description="Parent knowledge base ID")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path or URI to the file")
    checksum: str = Field(..., description="SHA-256 checksum for deduplication")
    size_bytes: int = Field(0, description="File size in bytes")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    document_type: DocumentType = Field(default=DocumentType.UNKNOWN)
    page_count: Optional[int] = Field(None, description="Total pages (PDFs, etc)")
    chunk_count: int = Field(default=0, description="Number of chunks created")
    indexed_chunk_count: int = Field(default=0, description="Number of indexed chunks")
    mime_type: str = Field(default="application/octet-stream")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    ingestion_completed_at: Optional[datetime] = Field(None)
    error_message: Optional[str] = Field(None, description="Error details if status is FAILED")
    metadata: dict = Field(default_factory=dict, description="Custom metadata")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "doc_123",
                "kb_id": "kb_456",
                "filename": "document.pdf",
                "file_path": "/path/to/document.pdf",
                "checksum": "abc123...",
                "size_bytes": 1024000,
                "status": "indexed",
                "document_type": "pdf",
                "page_count": 10,
                "chunk_count": 45,
            }
        }
