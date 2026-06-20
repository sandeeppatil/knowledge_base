"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.config.settings import Settings, load_settings
from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.domain.models.document import Document, DocumentStatus, DocumentType
from src.domain.models.knowledge_base import KnowledgeBase
from src.domain.models.retrieval import Citation, RetrievalResult


# ─── Settings ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Return test-environment settings."""
    import os

    os.environ["APP_ENV"] = "test"
    return load_settings()


# ─── Domain fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_kb() -> KnowledgeBase:
    return KnowledgeBase(
        id="test-kb-001",
        name="Test KB",
        description="A test knowledge base for unit testing.",
        embedding_model="BAAI/bge-m3",
        vector_store="qdrant",
    )


@pytest.fixture
def sample_document(sample_kb: KnowledgeBase) -> Document:
    return Document(
        id="test-doc-001",
        knowledge_base_id=sample_kb.id,
        name="test_document.pdf",
        source_path="/tmp/test_document.pdf",
        document_type=DocumentType.PDF,
        file_size_bytes=102400,
        page_count=5,
    )


@pytest.fixture
def sample_chunk_metadata(sample_document: Document) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="test-chunk-001",
        document_id=sample_document.id,
        document_name=sample_document.name,
        knowledge_base_id="test-kb-001",
        knowledge_base_name="Test KB",
        page_numbers=[1, 2],
        section_title="Introduction",
        content_type=ContentType.TEXT,
        source_path=sample_document.source_path,
        heading_path=["Introduction"],
    )


@pytest.fixture
def sample_chunk(sample_chunk_metadata: ChunkMetadata) -> Chunk:
    return Chunk(
        id="test-chunk-001",
        content="This is a sample chunk of text for testing purposes. It contains information about quality management systems.",
        metadata=sample_chunk_metadata,
        token_count=22,
        embedding=[0.1] * 768,
    )


@pytest.fixture
def sample_chunk_with_score(sample_chunk: Chunk) -> ChunkWithScore:
    return ChunkWithScore(
        chunk=sample_chunk,
        score=0.92,
        rank=1,
        retrieval_method="dense",
    )


@pytest.fixture
def sample_retrieval_result(
    sample_chunk_with_score: ChunkWithScore,
) -> RetrievalResult:
    return RetrievalResult.from_chunks(
        query="quality management",
        kb_id="test-kb-001",
        kb_name="Test KB",
        chunks=[sample_chunk_with_score],
    )


# ─── Mock infrastructure ──────────────────────────────────────────────────────


@pytest.fixture
def mock_embedding_provider() -> AsyncMock:
    provider = AsyncMock()
    provider.dimension = 768
    provider.model_name = "test-model"
    provider.embed_text.return_value = [0.1] * 768
    provider.embed_batch.return_value = [[0.1] * 768]
    return provider


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.collection_exists.return_value = True
    store.search.return_value = []
    store.upsert.return_value = None
    store.list_collections.return_value = []
    return store


@pytest.fixture
def mock_kb_repository() -> AsyncMock:
    repo = AsyncMock()
    repo.list_kbs.return_value = []
    repo.get_kb.return_value = None
    repo.create_kb.side_effect = lambda kb: kb
    repo.update_kb.side_effect = lambda kb: kb
    return repo


@pytest.fixture
def mock_reranker() -> AsyncMock:
    reranker = AsyncMock()
    reranker.model_name = "test-reranker"

    async def passthrough_rerank(query, candidates, top_k):
        return candidates[:top_k]

    reranker.rerank.side_effect = passthrough_rerank
    return reranker


# ─── Test files ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal valid PDF for parser tests."""
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(pdf_content)
    return pdf_path


# ─── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
async def test_client(
    mock_kb_repository: AsyncMock,
    mock_vector_store: AsyncMock,
    mock_embedding_provider: AsyncMock,
) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with mocked infrastructure."""
    from src.api.app import create_app
    from src.api.dependencies import Container

    app = create_app()

    # Replace container with mocks
    container = app.state.container
    container.kb_repository = mock_kb_repository
    container.vector_store = mock_vector_store
    container.embedding_provider = mock_embedding_provider
    container.kb_service._repo = mock_kb_repository
    container.kb_service._store = mock_vector_store
    container.kb_service._embedder = mock_embedding_provider

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client
