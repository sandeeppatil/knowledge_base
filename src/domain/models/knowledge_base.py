"""Knowledge Base model for the knowledge base platform."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class KBStatus(str, Enum):
    """Knowledge base status."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    REBUILDING = "rebuilding"


class KnowledgeBase(BaseModel):
    """Represents a knowledge base — a collection of documents."""

    id: str = Field(..., description="Unique knowledge base ID")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="KB description")
    status: KBStatus = Field(default=KBStatus.ACTIVE)
    embedding_model: str = Field(..., description="Name of the embedding model used")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    vector_store_type: str = Field(..., description="Type of vector store (qdrant, faiss, etc)")
    collection_name: str = Field(..., description="Vector store collection name")
    bm25_index_version: int = Field(default=1, description="Version of BM25 index")
    document_count: int = Field(default=0, description="Total documents ingested")
    chunk_count: int = Field(default=0, description="Total chunks across all documents")
    indexed_chunk_count: int = Field(default=0, description="Chunks in vector store")
    storage_path: Optional[str] = Field(None, description="Local storage path for KB data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_indexed_at: Optional[datetime] = Field(None)
    metadata: dict = Field(default_factory=dict, description="Custom metadata")

    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "id": "kb_123",
                "name": "Company Policies",
                "description": "Internal policies and procedures",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dimension": 1024,
                "vector_store_type": "qdrant",
                "collection_name": "policies",
                "document_count": 25,
                "chunk_count": 500,
            }
        }
