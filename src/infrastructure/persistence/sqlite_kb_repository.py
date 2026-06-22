"""SQLAlchemy async persistence for KnowledgeBase and Document records.

Uses SQLite by default (portable, zero-dependency), configurable to any
SQLAlchemy-supported RDBMS via DATABASE_URL.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    select,
    delete,
    update,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.domain.interfaces import KBRepository
from src.domain.models.document import Document, DocumentStatus, DocumentType
from src.domain.models.knowledge_base import KnowledgeBase
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


# ─── ORM Models ───────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class KnowledgeBaseORM(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    document_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dimension: Mapped[int] = mapped_column(Integer, default=1024)
    vector_store_type: Mapped[str] = mapped_column(String(50), default="qdrant")
    collection_name: Mapped[str] = mapped_column(String(200), nullable=False)
    vector_store: Mapped[str] = mapped_column(String(50))
    version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class DocumentORM(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(500))
    source_path: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(String(20))
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    parser_used: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


# ─── Mapper helpers ───────────────────────────────────────────────────────────


def _kb_from_orm(row: KnowledgeBaseORM) -> KnowledgeBase:
    return KnowledgeBase(
        id=row.id,
        name=row.name,
        description=row.description,
        document_count=row.document_count,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
        embedding_model=row.embedding_model,
        embedding_dimension=row.embedding_dimension,
        vector_store_type=row.vector_store_type,
        collection_name=row.collection_name,
        metadata=json.loads(row.metadata_json),
    )


def _doc_from_orm(row: DocumentORM) -> Document:
    return Document(
        id=row.id,
        kb_id=row.knowledge_base_id,
        filename=row.name,
        file_path=row.source_path,
        document_type=DocumentType(row.document_type),
        size_bytes=row.file_size_bytes,
        page_count=row.page_count,
        chunk_count=row.chunk_count,
        status=DocumentStatus(row.status),
        error_message=row.error_message,
        checksum=row.checksum,
        created_at=row.created_at,
        ingestion_completed_at=row.processed_at,
        metadata=json.loads(row.metadata_json),
    )


# ─── Repository ───────────────────────────────────────────────────────────────


class SQLiteKBRepository(KBRepository):
    """KBRepository implementation backed by SQLite (via SQLAlchemy async).

    Args:
        database_url: SQLAlchemy async connection string.
            Example: "sqlite+aiosqlite:///./data/knowledge_base.db"
    """

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, echo=False)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialise(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialised", url=str(self._engine.url))

    # ── Knowledge Bases ───────────────────────────────────────────────────────

    async def create_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        async with self._session_factory() as session:
            row = KnowledgeBaseORM(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                document_count=kb.document_count,
                chunk_count=kb.chunk_count,
                created_at=kb.created_at,
                updated_at=kb.updated_at,
                embedding_model=kb.embedding_model,
                embedding_dimension=kb.embedding_dimension,
                vector_store_type=kb.vector_store_type,
                collection_name=kb.collection_name,
                vector_store=kb.vector_store_type,
                version=kb.version,
                is_active=kb.is_active,
                metadata_json=json.dumps(kb.metadata),
            )
            session.add(row)
            await session.commit()
        logger.info("KB created", kb_id=kb.id, name=kb.name)
        return kb

    async def get_kb(self, kb_id: str) -> KnowledgeBase | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeBaseORM).where(KnowledgeBaseORM.id == kb_id)
            )
            row = result.scalar_one_or_none()
            return _kb_from_orm(row) if row else None

    async def get_kb_by_name(self, name: str) -> KnowledgeBase | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeBaseORM).where(KnowledgeBaseORM.name == name)
            )
            row = result.scalar_one_or_none()
            return _kb_from_orm(row) if row else None

    async def list_kbs(self) -> list[KnowledgeBase]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgeBaseORM).where(KnowledgeBaseORM.is_active == True)  # noqa: E712
            )
            return [_kb_from_orm(r) for r in result.scalars().all()]

    async def update_kb(self, kb: KnowledgeBase) -> KnowledgeBase:
        async with self._session_factory() as session:
            await session.execute(
                update(KnowledgeBaseORM)
                .where(KnowledgeBaseORM.id == kb.id)
                .values(
                    name=kb.name,
                    description=kb.description,
                    document_count=kb.document_count,
                    chunk_count=kb.chunk_count,
                    updated_at=kb.updated_at,
                    embedding_model=kb.embedding_model,
                    embedding_dimension=kb.embedding_dimension,
                    vector_store_type=kb.vector_store_type,
                    collection_name=kb.collection_name,
                    vector_store=kb.vector_store_type,
                    version=kb.version,
                    is_active=kb.is_active,
                    metadata_json=json.dumps(kb.metadata),
                )
            )
            await session.commit()
        return kb

    async def delete_kb(self, kb_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(KnowledgeBaseORM).where(KnowledgeBaseORM.id == kb_id)
            )
            await session.execute(
                delete(DocumentORM).where(DocumentORM.knowledge_base_id == kb_id)
            )
            await session.commit()
        return result.rowcount > 0  # type: ignore[return-value]

    # ── Documents ────────────────────────────────────────────────────────────

    async def create_document(self, document: Document) -> Document:
        async with self._session_factory() as session:
            row = DocumentORM(
                id=document.id,
                knowledge_base_id=document.kb_id,
                name=document.filename,
                source_path=document.file_path,
                document_type=document.document_type.value,
                file_size_bytes=document.size_bytes,
                page_count=document.page_count,
                chunk_count=document.chunk_count,
                status=document.status.value,
                error_message=document.error_message,
                checksum=document.checksum,
                parser_used=None,  # Not tracked in Document model
                created_at=document.created_at,
                processed_at=document.ingestion_completed_at,
                metadata_json=json.dumps(document.metadata),
            )
            session.add(row)
            await session.commit()
        return document
        return document

    async def get_document(self, document_id: str) -> Document | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentORM).where(DocumentORM.id == document_id)
            )
            row = result.scalar_one_or_none()
            return _doc_from_orm(row) if row else None

    async def list_documents(self, kb_id: str) -> list[Document]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentORM).where(DocumentORM.knowledge_base_id == kb_id)
            )
            return [_doc_from_orm(r) for r in result.scalars().all()]

    async def update_document(self, document: Document) -> Document:
        async with self._session_factory() as session:
            await session.execute(
                update(DocumentORM)
                .where(DocumentORM.id == document.id)
                .values(
                    page_count=document.page_count,
                    chunk_count=document.chunk_count,
                    status=document.status.value,
                    error_message=document.error_message,
                    checksum=document.checksum,
                    processed_at=document.ingestion_completed_at,
                    metadata_json=json.dumps(document.metadata),
                )
            )
            await session.commit()
        return document

    async def delete_document(self, document_id: str) -> bool:
        async with self._session_factory() as session:
            result = await session.execute(
                delete(DocumentORM).where(DocumentORM.id == document_id)
            )
            await session.commit()
        return result.rowcount > 0  # type: ignore[return-value]

    async def get_document_by_checksum(self, kb_id: str, checksum: str) -> Document | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentORM).where(
                    DocumentORM.knowledge_base_id == kb_id,
                    DocumentORM.checksum == checksum,
                )
            )
            row = result.scalar_one_or_none()
            return _doc_from_orm(row) if row else None
