"""Unit tests for domain models."""

from __future__ import annotations

import pytest

from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.domain.models.document import Document, DocumentStatus, DocumentType
from src.domain.models.knowledge_base import KnowledgeBase
from src.domain.models.retrieval import Citation, RetrievalResult


class TestKnowledgeBase:
    def test_collection_name_derivation(self) -> None:
        kb = KnowledgeBase(
            id="abc-123",
            name="Test",
            description="Test KB",
        )
        assert kb.collection_name == "kb_abc_123"

    def test_bump_version(self) -> None:
        kb = KnowledgeBase(id="x", name="X", description="X")
        kb.version = "1.2.3"
        kb.bump_version()
        assert kb.version == "1.2.4"

    def test_touch_updates_timestamp(self) -> None:
        kb = KnowledgeBase(id="x", name="X", description="X")
        original = kb.updated_at
        import time; time.sleep(0.01)
        kb.touch()
        assert kb.updated_at >= original


class TestDocument:
    def test_mark_completed(self, tmp_path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")
        doc = Document.from_path(path, kb_id="kb-1")
        doc.mark_completed(chunk_count=42, page_count=10, parser_used="docling")
        assert doc.status == DocumentStatus.COMPLETED
        assert doc.chunk_count == 42
        assert doc.page_count == 10
        assert doc.parser_used == "docling"
        assert doc.processed_at is not None

    def test_mark_failed(self, tmp_path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF")
        doc = Document.from_path(path, kb_id="kb-1")
        doc.mark_failed("Parse error")
        assert doc.status == DocumentStatus.FAILED
        assert doc.error_message == "Parse error"

    def test_document_type_detection(self, tmp_path) -> None:
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"")
        doc = Document.from_path(pdf_path, "kb-1")
        assert doc.document_type == DocumentType.PDF


class TestChunk:
    def test_has_embedding_true(self) -> None:
        meta = ChunkMetadata(
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            knowledge_base_id="kb1",
            knowledge_base_name="KB",
        )
        chunk = Chunk(id="c1", content="hello", metadata=meta, embedding=[0.1, 0.2])
        assert chunk.has_embedding() is True

    def test_has_embedding_false(self) -> None:
        meta = ChunkMetadata(
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            knowledge_base_id="kb1",
            knowledge_base_name="KB",
        )
        chunk = Chunk(id="c1", content="hello", metadata=meta)
        assert chunk.has_embedding() is False


class TestRetrievalResult:
    def test_not_found_response(self) -> None:
        result = RetrievalResult.not_found("test query")
        assert result.answer_found is False
        assert result.query == "test query"
        assert result.reason is not None
        assert result.chunks == []
        assert result.citations == []

    def test_from_chunks_populates_citations(
        self, sample_chunk_with_score: ChunkWithScore
    ) -> None:
        result = RetrievalResult.from_chunks(
            query="test",
            kb_id="kb-1",
            kb_name="Test KB",
            chunks=[sample_chunk_with_score],
        )
        assert result.answer_found is True
        assert len(result.chunks) == 1
        assert len(result.citations) == 1
        assert result.citations[0].source_document == "test_document.pdf"

    def test_citation_excerpt_truncated(self) -> None:
        meta = ChunkMetadata(
            chunk_id="c",
            document_id="d",
            document_name="long.pdf",
            knowledge_base_id="kb",
            knowledge_base_name="KB",
        )
        long_content = "A" * 500
        chunk = Chunk(id="c", content=long_content, metadata=meta, embedding=[0.1])
        cws = ChunkWithScore(chunk=chunk, score=0.9, rank=1)
        citation = Citation.from_chunk_with_score(cws)
        assert len(citation.excerpt) <= 304  # 300 chars + "…"
        assert citation.excerpt.endswith("…")
