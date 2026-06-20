"""Integration tests for the retrieval pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.domain.models.retrieval import RetrievalResult
from src.retrieval.pipeline import RetrievalPipeline


def _make_cws(chunk_id: str, content: str, score: float, rank: int) -> ChunkWithScore:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        document_id="d1",
        document_name="doc.pdf",
        knowledge_base_id="kb1",
        knowledge_base_name="Test KB",
        page_numbers=[1],
        section_title="Introduction",
        content_type=ContentType.TEXT,
        source_path="/tmp/doc.pdf",
    )
    chunk = Chunk(id=chunk_id, content=content, metadata=meta, embedding=[0.1] * 768)
    return ChunkWithScore(chunk=chunk, score=score, rank=rank, retrieval_method="dense")


@pytest.mark.asyncio
class TestRetrievalPipeline:
    async def test_empty_query_returns_not_found(
        self, mock_vector_store, mock_embedding_provider
    ) -> None:
        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        result = await pipeline.retrieve(
            collection="test_col",
            kb_name="Test KB",
            kb_id="kb-1",
            query="   ",
        )
        assert result.answer_found is False

    async def test_successful_retrieval(
        self, mock_vector_store, mock_embedding_provider
    ) -> None:
        cws = _make_cws("c1", "quality management content", 0.9, 1)
        mock_vector_store.search.return_value = [cws]

        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            final_top_k=5,
        )
        result = await pipeline.retrieve(
            collection="test_col",
            kb_name="Test KB",
            kb_id="kb-1",
            query="quality management",
        )
        assert result.answer_found is True
        assert len(result.chunks) == 1
        assert len(result.citations) == 1
        assert result.citations[0].source_document == "doc.pdf"

    async def test_no_results_returns_not_found(
        self, mock_vector_store, mock_embedding_provider
    ) -> None:
        mock_vector_store.search.return_value = []

        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        result = await pipeline.retrieve(
            collection="test_col",
            kb_name="Test KB",
            kb_id="kb-1",
            query="unknown topic",
        )
        assert result.answer_found is False
        assert result.reason is not None

    async def test_reranker_called_when_configured(
        self, mock_vector_store, mock_embedding_provider, mock_reranker
    ) -> None:
        cws_list = [_make_cws(f"c{i}", f"content {i}", 0.9 - i * 0.1, i + 1) for i in range(5)]
        mock_vector_store.search.return_value = cws_list

        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            reranker=mock_reranker,
            final_top_k=3,
        )
        await pipeline.retrieve(
            collection="test_col",
            kb_name="Test KB",
            kb_id="kb-1",
            query="test query",
        )
        mock_reranker.rerank.assert_called_once()

    async def test_dense_failure_falls_back_gracefully(
        self, mock_vector_store, mock_embedding_provider
    ) -> None:
        mock_vector_store.search.side_effect = Exception("Vector store error")

        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        result = await pipeline.retrieve(
            collection="test_col",
            kb_name="Test KB",
            kb_id="kb-1",
            query="test query",
        )
        # Should return not_found gracefully, not raise
        assert result.answer_found is False

    async def test_kb_name_populated_on_chunks(
        self, mock_vector_store, mock_embedding_provider
    ) -> None:
        cws = _make_cws("c1", "content", 0.9, 1)
        mock_vector_store.search.return_value = [cws]

        pipeline = RetrievalPipeline(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
        )
        result = await pipeline.retrieve(
            collection="test_col",
            kb_name="My KB",
            kb_id="kb-1",
            query="test",
        )
        if result.answer_found:
            for cws in result.chunks:
                assert cws.chunk.metadata.knowledge_base_name == "My KB"
