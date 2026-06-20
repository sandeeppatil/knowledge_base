"""Unit tests for the hierarchical chunker."""

from __future__ import annotations

import pytest

from src.chunking.hierarchical_chunker import HierarchicalChunker, _estimate_tokens, _split_with_overlap
from src.domain.models.chunk import ContentType
from src.domain.models.document import Document, DocumentType
from src.parsers.base import ParsedDocumentResult, ParsedFigure, ParsedSection, ParsedTable


def _make_document() -> Document:
    return Document(
        id="doc-1",
        knowledge_base_id="kb-1",
        name="test.pdf",
        source_path="/tmp/test.pdf",
        document_type=DocumentType.PDF,
    )


def _make_parsed_doc(sections=None, tables=None, figures=None) -> ParsedDocumentResult:
    doc = Document(
        id="doc-1",
        knowledge_base_id="kb-1",
        name="test.pdf",
        source_path="/tmp/test.pdf",
    )
    return ParsedDocumentResult(
        document_id="doc-1",
        document_name="test.pdf",
        source_path="/tmp/test.pdf",
        page_count=3,
        sections=sections or [],
        tables=tables or [],
        figures=figures or [],
        raw_text="",
    )


class TestTokenEstimation:
    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 1  # min 1

    def test_approximate_tokens(self) -> None:
        text = "Hello world " * 10  # ~120 chars → ~30 tokens
        tokens = _estimate_tokens(text)
        assert 20 <= tokens <= 40


class TestSplitWithOverlap:
    def test_short_text_returns_single_chunk(self) -> None:
        result = _split_with_overlap("Short text.", chunk_size=500, chunk_overlap=50, min_size=5)
        assert len(result) == 1
        assert result[0] == "Short text."

    def test_long_text_splits_into_multiple(self) -> None:
        long_text = "A paragraph.\n\n" * 100
        result = _split_with_overlap(long_text, chunk_size=200, chunk_overlap=20, min_size=10)
        assert len(result) > 1

    def test_min_size_filters_small_chunks(self) -> None:
        text = "Tiny"
        result = _split_with_overlap(text, chunk_size=500, chunk_overlap=50, min_size=100)
        assert result == []


class TestHierarchicalChunker:
    @pytest.mark.asyncio
    async def test_text_section_produces_chunks(self) -> None:
        chunker = HierarchicalChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        section = ParsedSection(
            title="Introduction",
            level=1,
            content="This is the introduction. " * 20,
            page_numbers=[1],
        )
        parsed = _make_parsed_doc(sections=[section])
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.content.strip()
            assert chunk.metadata.document_id == "doc-1"
            assert chunk.metadata.knowledge_base_id == "kb-1"

    @pytest.mark.asyncio
    async def test_table_produces_table_chunk(self) -> None:
        chunker = HierarchicalChunker()
        table = ParsedTable(
            table_id="t1",
            table_title="Requirements Table",
            headers=["ID", "Requirement"],
            rows=[["REQ-001", "The system shall..."]],
            page_numbers=[2],
        )
        parsed = _make_parsed_doc(tables=[table])
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        table_chunks = [c for c in chunks if c.metadata.content_type == ContentType.TABLE]
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata.table_title == "Requirements Table"

    @pytest.mark.asyncio
    async def test_figure_produces_figure_chunk(self) -> None:
        chunker = HierarchicalChunker()
        figure = ParsedFigure(
            figure_id="f1",
            caption="Figure 1: System Architecture",
            description="A diagram showing system components.",
            page_number=3,
        )
        parsed = _make_parsed_doc(figures=[figure])
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        fig_chunks = [c for c in chunks if c.metadata.content_type == ContentType.FIGURE]
        assert len(fig_chunks) == 1
        assert fig_chunks[0].metadata.figure_caption == "Figure 1: System Architecture"

    @pytest.mark.asyncio
    async def test_empty_document_returns_empty(self) -> None:
        chunker = HierarchicalChunker()
        parsed = _make_parsed_doc()
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_appendix_section_gets_correct_content_type(self) -> None:
        chunker = HierarchicalChunker()
        section = ParsedSection(
            title="Appendix A",
            level=1,
            content="This is appendix content. " * 10,
            page_numbers=[10],
        )
        parsed = _make_parsed_doc(sections=[section])
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        appendix_chunks = [c for c in chunks if c.metadata.content_type == ContentType.APPENDIX]
        assert len(appendix_chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_metadata_populated(self) -> None:
        chunker = HierarchicalChunker()
        section = ParsedSection(
            title="Section 1",
            level=1,
            content="Content here. " * 20,
            page_numbers=[1, 2],
        )
        parsed = _make_parsed_doc(sections=[section])
        doc = _make_document()
        chunks = await chunker.chunk(parsed, doc)
        for chunk in chunks:
            assert chunk.id != ""
            assert chunk.metadata.chunk_id == chunk.id
            assert chunk.metadata.document_name == "test.pdf"
            assert chunk.metadata.source_path == "/tmp/test.pdf"
            assert chunk.token_count > 0
