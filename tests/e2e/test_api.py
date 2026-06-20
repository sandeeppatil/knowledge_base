"""End-to-end API tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from src.domain.models.knowledge_base import KnowledgeBase


@pytest.mark.asyncio
class TestKnowledgeBaseAPI:
    async def test_health_check(self, test_client: AsyncClient) -> None:
        response = await test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_list_knowledge_bases_empty(self, test_client: AsyncClient) -> None:
        response = await test_client.get("/knowledge-bases")
        assert response.status_code == 200
        assert response.json() == []

    async def test_create_knowledge_base(
        self, test_client: AsyncClient, mock_kb_repository, mock_vector_store
    ) -> None:
        kb = KnowledgeBase(
            id="new-kb-id",
            name="ISO Standards",
            description="ISO quality standards",
        )
        mock_kb_repository.get_kb_by_name.return_value = None
        mock_kb_repository.create_kb.return_value = kb
        mock_vector_store.create_collection.return_value = None

        response = await test_client.post(
            "/knowledge-bases",
            json={
                "name": "ISO Standards",
                "description": "ISO quality standards",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "ISO Standards"
        assert "id" in data

    async def test_create_duplicate_kb_returns_409(
        self, test_client: AsyncClient, mock_kb_repository, sample_kb
    ) -> None:
        mock_kb_repository.get_kb_by_name.return_value = sample_kb

        response = await test_client.post(
            "/knowledge-bases",
            json={
                "name": "Test KB",
                "description": "Already exists",
            },
        )
        assert response.status_code == 409

    async def test_get_kb_not_found(self, test_client: AsyncClient) -> None:
        response = await test_client.get("/knowledge-bases/nonexistent-id")
        assert response.status_code == 404

    async def test_delete_kb_not_found(self, test_client: AsyncClient) -> None:
        response = await test_client.delete("/knowledge-bases/nonexistent-id")
        assert response.status_code == 404


@pytest.mark.asyncio
class TestRetrievalAPI:
    async def test_retrieve_returns_not_found_when_no_results(
        self,
        test_client: AsyncClient,
        mock_kb_repository,
        mock_vector_store,
        mock_embedding_provider,
        sample_kb,
    ) -> None:
        mock_kb_repository.get_kb.return_value = sample_kb
        mock_vector_store.search.return_value = []

        response = await test_client.post(
            "/retrieve",
            json={
                "query": "Something that doesn't exist",
                "kb_id": sample_kb.id,
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer_found"] is False

    async def test_retrieve_kb_not_found(self, test_client: AsyncClient) -> None:
        response = await test_client.post(
            "/retrieve",
            json={
                "query": "test query",
                "kb_id": "nonexistent-kb",
                "top_k": 5,
            },
        )
        assert response.status_code == 404
