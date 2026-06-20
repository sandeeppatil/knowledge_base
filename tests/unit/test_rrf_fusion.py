"""Unit tests for RRF fusion."""

from __future__ import annotations

import pytest

from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore
from src.retrieval.rrf_fusion import reciprocal_rank_fusion


def _make_cws(chunk_id: str, score: float, rank: int) -> ChunkWithScore:
    meta = ChunkMetadata(
        chunk_id=chunk_id,
        document_id="d1",
        document_name="doc.pdf",
        knowledge_base_id="kb1",
        knowledge_base_name="KB",
    )
    chunk = Chunk(id=chunk_id, content=f"content {chunk_id}", metadata=meta)
    return ChunkWithScore(chunk=chunk, score=score, rank=rank)


class TestRRFFusion:
    def test_empty_input_returns_empty(self) -> None:
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_single_list_preserves_order(self) -> None:
        cws_list = [
            _make_cws("c1", 0.9, 1),
            _make_cws("c2", 0.8, 2),
            _make_cws("c3", 0.7, 3),
        ]
        result = reciprocal_rank_fusion([cws_list])
        ids = [r.chunk.id for r in result]
        assert ids == ["c1", "c2", "c3"]

    def test_deduplication_across_lists(self) -> None:
        list1 = [_make_cws("c1", 0.9, 1), _make_cws("c2", 0.8, 2)]
        list2 = [_make_cws("c1", 0.7, 1), _make_cws("c3", 0.6, 2)]
        result = reciprocal_rank_fusion([list1, list2])
        ids = [r.chunk.id for r in result]
        # c1 appears in both lists so should rank highly
        assert ids[0] == "c1"
        # All IDs should be unique
        assert len(ids) == len(set(ids))

    def test_scores_normalised_to_0_1(self) -> None:
        list1 = [_make_cws("c1", 0.9, 1), _make_cws("c2", 0.5, 2)]
        result = reciprocal_rank_fusion([list1])
        for r in result:
            assert 0.0 <= r.score <= 1.0

    def test_retrieval_method_set_to_rrf(self) -> None:
        list1 = [_make_cws("c1", 0.9, 1)]
        result = reciprocal_rank_fusion([list1])
        assert result[0].retrieval_method == "rrf"

    def test_rank_is_sequential(self) -> None:
        list1 = [_make_cws(f"c{i}", 1.0 - i * 0.1, i + 1) for i in range(5)]
        result = reciprocal_rank_fusion([list1])
        for i, r in enumerate(result):
            assert r.rank == i + 1

    def test_weight_influences_ranking(self) -> None:
        # c1 ranks 1st in dense, c2 ranks 1st in BM25 with higher weight
        dense = [_make_cws("c1", 0.9, 1), _make_cws("c2", 0.5, 2)]
        bm25 = [_make_cws("c2", 0.9, 1), _make_cws("c1", 0.3, 2)]
        result = reciprocal_rank_fusion([dense, bm25], weights=[0.1, 0.9])
        # With bm25 weight=0.9, c2 (top BM25) should beat c1 (top dense)
        assert result[0].chunk.id == "c2"

    def test_custom_k_value(self) -> None:
        list1 = [_make_cws("c1", 0.9, 1), _make_cws("c2", 0.8, 2)]
        result_k60 = reciprocal_rank_fusion([list1], k=60)
        result_k1 = reciprocal_rank_fusion([list1], k=1)
        # Both should have same order but different absolute scores
        assert [r.chunk.id for r in result_k60] == [r.chunk.id for r in result_k1]

    def test_weights_length_mismatch_raises(self) -> None:
        list1 = [_make_cws("c1", 0.9, 1)]
        list2 = [_make_cws("c2", 0.8, 1)]
        with pytest.raises(ValueError, match="weights length"):
            reciprocal_rank_fusion([list1, list2], weights=[1.0])
