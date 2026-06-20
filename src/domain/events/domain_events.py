"""Domain events — lightweight event objects for cross-module communication.

Events decouple ingestion, indexing, and monitoring concerns.  They are not
persisted; they are emitted in-process and handled synchronously or enqueued
for async processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events."""

    occurred_at: datetime = field(default_factory=_utcnow)
    correlation_id: str = ""


@dataclass(frozen=True)
class KnowledgeBaseCreated(DomainEvent):
    kb_id: str = ""
    kb_name: str = ""


@dataclass(frozen=True)
class KnowledgeBaseDeleted(DomainEvent):
    kb_id: str = ""
    kb_name: str = ""


@dataclass(frozen=True)
class KnowledgeBaseRebuilt(DomainEvent):
    kb_id: str = ""
    kb_name: str = ""
    chunk_count: int = 0


@dataclass(frozen=True)
class DocumentIngested(DomainEvent):
    document_id: str = ""
    document_name: str = ""
    kb_id: str = ""
    chunk_count: int = 0
    parser_used: str = ""
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class DocumentIngestionFailed(DomainEvent):
    document_id: str = ""
    document_name: str = ""
    kb_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class RetrievalPerformed(DomainEvent):
    query: str = ""
    kb_id: str = ""
    answer_found: bool = False
    result_count: int = 0
    latency_ms: float = 0.0
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
