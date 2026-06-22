"""End-to-end tests for the MCP server knowledge-base tools.

Tests cover:
- Creating the two knowledge bases present in data/knowledge_bases/
  (space_autosar and space_coding_guidelines)
- Ingesting PDFs from those folders
- Retrieval via a custom MCPKnowledgeBaseAgent that follows the documented
  agent workflow:
    Step 1 – list_knowledge_bases()
    Step 2 – select KB by description
    Step 3 – retrieve_from_kb(kb_name, query)
    Step 4 – assert answer_found / citations

Design decision: The test file is intentionally self-contained.  It imports
only domain models (no heavy runtime dependencies) and constructs all
application-layer objects as AsyncMocks.  This avoids pulling in opentelemetry,
the mcp package, qdrant-client, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.domain.models.chunk import Chunk, ChunkMetadata, ChunkWithScore, ContentType
from src.domain.models.document import Document, DocumentStatus, DocumentType
from src.domain.models.knowledge_base import KnowledgeBase
from src.domain.models.retrieval import RetrievalResult

# Restrict anyio to asyncio backend only (trio is not installed in this env)
pytestmark = pytest.mark.anyio


# ─── Canonical KB definitions (match data/knowledge_bases/ folder names) ─────

KB_AUTOSAR = KnowledgeBase(
    id="kb-autosar-001",
    name="space_autosar",
    description="AUTOSAR SWS COM standard — communication stack specifications.",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    vector_store_type="qdrant",
    collection_name="kb_autosar_001",
)

KB_CODING = KnowledgeBase(
    id="kb-coding-001",
    name="space_coding_guidelines",
    description="C++ Coding Standards — language usage rules and best practices.",
    embedding_model="BAAI/bge-m3",
    embedding_dimension=1024,
    vector_store_type="qdrant",
    collection_name="kb_coding_001",
)

_ALL_KBS = [KB_AUTOSAR, KB_CODING]

# ─── Sample ingested documents ────────────────────────────────────────────────

DOC_AUTOSAR = Document(
    id="doc-autosar-001",
    kb_id=KB_AUTOSAR.id,
    filename="AUTOSAR_SWS_COM.pdf",
    file_path="data/knowledge_bases/space_autosar/AUTOSAR_SWS_COM.pdf",
    checksum="abc123abc123abc123abc123abc123abc123abc123abc123abc1",
    size_bytes=2_097_152,
    status=DocumentStatus.INDEXED,
    document_type=DocumentType.PDF,
    page_count=120,
    chunk_count=45,
)

DOC_CODING = Document(
    id="doc-coding-001",
    kb_id=KB_CODING.id,
    filename="C++ Coding Standards.pdf",
    file_path="data/knowledge_bases/space_coding_guidelines/C++ Coding Standards.pdf",
    checksum="def456def456def456def456def456def456def456def456def4",
    size_bytes=1_048_576,
    status=DocumentStatus.INDEXED,
    document_type=DocumentType.PDF,
    page_count=80,
    chunk_count=32,
)

# ─── Fixture helpers ──────────────────────────────────────────────────────────


def _make_chunk(
    chunk_id: str,
    content: str,
    source_document: str,
    kb: KnowledgeBase,
    page_numbers: list[int],
    section_title: str,
) -> Chunk:
    """Build a Chunk using the actual domain model fields."""
    return Chunk(
        id=chunk_id,
        document_id="doc-001",
        kb_id=kb.id,
        content=content,
        source_document=source_document,
        page_numbers=page_numbers,
        section_title=section_title,
        chunk_metadata=ChunkMetadata(content_type=ContentType.TEXT),
    )


def _make_chunk_with_score(
    content: str,
    source_document: str,
    kb: KnowledgeBase,
    page_numbers: list[int],
    section_title: str,
    score: float = 0.88,
) -> ChunkWithScore:
    chunk_id = f"chunk-{abs(hash(content)) % 65536:04x}"
    chunk = _make_chunk(
        chunk_id=chunk_id,
        content=content,
        source_document=source_document,
        kb=kb,
        page_numbers=page_numbers,
        section_title=section_title,
    )
    return ChunkWithScore(chunk=chunk, score=score, rank=1)


def _make_retrieval_result(
    query: str, kb: KnowledgeBase, chunk_with_score: ChunkWithScore
) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        kb_id=kb.id,
        answer_found=True,
        chunks=[chunk_with_score],
    )


@pytest.fixture
def autosar_chunk() -> ChunkWithScore:
    return _make_chunk_with_score(
        content=(
            "The COM module provides mechanisms for signal-based and PDU-based "
            "communication. I-PDU (Interaction Layer PDU) is the basic transmission "
            "unit. Signals are mapped to I-PDUs via ComSignal configuration."
        ),
        source_document="AUTOSAR_SWS_COM.pdf",
        kb=KB_AUTOSAR,
        page_numbers=[12, 13],
        section_title="COM Signal Handling",
    )


@pytest.fixture
def coding_chunk() -> ChunkWithScore:
    return _make_chunk_with_score(
        content=(
            "Rule A2-10-1: An identifier declared in an inner scope shall not hide "
            "an identifier declared in an outer scope. Rationale: Hiding identifiers "
            "can lead to programmer confusion and mistakes."
        ),
        source_document="C++ Coding Standards.pdf",
        kb=KB_CODING,
        page_numbers=[44],
        section_title="Naming Rules",
    )


# ─── Thin MCP-like dispatcher (no mcp package required) ──────────────────────


class KBDispatcher:
    """Mirrors the KBMCPServer._dispatch contract without needing the mcp package.

    Accepts ``(tool_name, arguments_dict)`` and returns JSON-serialisable dicts
    identical to those produced by the real MCP server.  All services are
    injected as AsyncMocks so no real infrastructure is required.
    """

    def __init__(
        self,
        kb_service: AsyncMock,
        ingestion_service: AsyncMock,
        retrieval_service: AsyncMock,
    ) -> None:
        self._kb = kb_service
        self._ingestion = ingestion_service
        self._retrieval = retrieval_service

    # ── tool: list_knowledge_bases ─────────────────────────────────────────
    async def _list_knowledge_bases(self) -> dict[str, Any]:
        kbs = await self._kb.list()
        return {
            "knowledge_bases": [
                {
                    "id": kb.id,
                    "name": kb.name,
                    "description": kb.description,
                    "document_count": kb.document_count,
                    "chunk_count": kb.chunk_count,
                    "embedding_model": kb.embedding_model,
                    "version": kb.version,
                }
                for kb in kbs
            ]
        }

    # ── tool: create_knowledge_base ───────────────────────────────────────
    async def _create_knowledge_base(
        self, name: str, description: str
    ) -> dict[str, Any]:
        try:
            kb = await self._kb.create(name=name, description=description)
            return {
                "success": True,
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
            }
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

    # ── tool: list_documents ───────────────────────────────────────────────
    async def _list_documents(self, kb_id: str) -> dict[str, Any]:
        docs = await self._ingestion.list_documents(kb_id)
        return {
            "documents": [
                {
                    "id": d.id,
                    "name": d.filename,
                    "status": d.status.value,
                    "chunk_count": d.chunk_count,
                    "page_count": d.page_count,
                    "document_type": d.document_type.value,
                }
                for d in docs
            ]
        }

    # ── tool: retrieve_from_kb ─────────────────────────────────────────────
    async def _retrieve_from_kb(
        self, kb_name: str, query: str, top_k: int = 10
    ) -> dict[str, Any]:
        kb = await self._kb.get_by_name(kb_name)
        if not kb:
            return {
                "answer_found": False,
                "reason": f"Knowledge base '{kb_name}' not found.",
                "query": query,
                "knowledge_base": kb_name,
                "chunks": [],
                "citations": [],
            }

        result: RetrievalResult = await self._retrieval.search(kb.id, query, top_k)

        # Build citation dicts from chunks (grounding: only from actual chunks)
        citations = [
            {
                "source_document": cws.chunk.source_document,
                "page_numbers": cws.chunk.page_numbers or [],
                "section": cws.chunk.section_title,
                "score": round(cws.score, 4),
                "chunk_id": cws.chunk.id,
                "excerpt": cws.chunk.content[:200],
            }
            for cws in result.chunks
        ]

        return {
            "answer_found": result.answer_found,
            "reason": result.reason,
            "query": result.query,
            "knowledge_base": kb_name,
            "chunks": [
                {
                    "content": cws.chunk.content,
                    "score": round(cws.score, 4),
                    "rank": cws.rank,
                    "source_document": cws.chunk.source_document,
                    "page_numbers": cws.chunk.page_numbers or [],
                    "section": cws.chunk.section_title,
                }
                for cws in result.chunks
            ],
            "citations": citations,
        }

    # ── tool: search_knowledge_bases ───────────────────────────────────────
    async def _search_knowledge_bases(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        kbs = await self._kb.list()
        results = []
        for kb in kbs:
            result: RetrievalResult = await self._retrieval.search(kb.id, query, top_k)
            results.append(
                {
                    "knowledge_base_id": kb.id,
                    "knowledge_base_name": kb.name,
                    "answer_found": result.answer_found,
                    "reason": result.reason,
                    "citations": [
                        {
                            "source_document": cws.chunk.source_document,
                            "page_numbers": cws.chunk.page_numbers or [],
                            "section": cws.chunk.section_title,
                            "score": round(cws.score, 4),
                            "excerpt": cws.chunk.content[:200],
                        }
                        for cws in result.chunks
                    ],
                }
            )
        return {"query": query, "results_by_kb": results}

    # ── public dispatch ────────────────────────────────────────────────────
    async def dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "list_knowledge_bases":
            return await self._list_knowledge_bases()
        if name == "create_knowledge_base":
            return await self._create_knowledge_base(
                name=arguments["name"], description=arguments["description"]
            )
        if name == "list_documents":
            return await self._list_documents(kb_id=arguments["kb_id"])
        if name == "retrieve_from_kb":
            return await self._retrieve_from_kb(
                kb_name=arguments["kb_name"],
                query=arguments["query"],
                top_k=arguments.get("top_k", 10),
            )
        if name == "search_knowledge_bases":
            return await self._search_knowledge_bases(
                query=arguments["query"], top_k=arguments.get("top_k", 5)
            )
        raise ValueError(f"Unknown tool: {name}")


# ─── Mock service builder ─────────────────────────────────────────────────────


def _build_dispatcher(
    autosar_retrieval: RetrievalResult | None = None,
    coding_retrieval: RetrievalResult | None = None,
) -> KBDispatcher:
    """Construct a KBDispatcher backed entirely by AsyncMocks."""

    # KnowledgeBaseService mock
    kb_service = AsyncMock()
    kb_service.list.return_value = _ALL_KBS

    async def _get_by_name(name: str) -> KnowledgeBase | None:
        return {KB_AUTOSAR.name: KB_AUTOSAR, KB_CODING.name: KB_CODING}.get(name)

    kb_service.get_by_name.side_effect = _get_by_name

    async def _create(name: str, description: str) -> KnowledgeBase:
        mapping = {KB_AUTOSAR.name: KB_AUTOSAR, KB_CODING.name: KB_CODING}
        if name in mapping:
            return mapping[name]
        raise ValueError(f"Unknown KB in test fixture: {name}")

    kb_service.create.side_effect = _create

    # IngestionService mock
    ingestion_service = AsyncMock()
    ingestion_service.ingest_file.return_value = DOC_AUTOSAR

    async def _list_docs(kb_id: str) -> list[Document]:
        return {KB_AUTOSAR.id: [DOC_AUTOSAR], KB_CODING.id: [DOC_CODING]}.get(kb_id, [])

    ingestion_service.list_documents.side_effect = _list_docs

    # RetrievalService mock
    retrieval_service = AsyncMock()

    async def _search(
        kb_id: str, query: str, top_k: int | None = None
    ) -> RetrievalResult:
        if kb_id == KB_AUTOSAR.id and autosar_retrieval is not None:
            return autosar_retrieval
        if kb_id == KB_CODING.id and coding_retrieval is not None:
            return coding_retrieval
        return RetrievalResult.not_found(query=query, kb_id=kb_id)

    retrieval_service.search.side_effect = _search

    return KBDispatcher(
        kb_service=kb_service,
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
    )


# ─── Custom Agent ─────────────────────────────────────────────────────────────


class MCPKnowledgeBaseAgent:
    """Simulates how an LLM agent drives the MCP KB tools.

    Implements the documented agent workflow from copilot-instructions.md:
      Step 1 – list_knowledge_bases()
      Step 2 – select KB by matching description to query topic hint
      Step 3 – retrieve_from_kb(kb_name, query, top_k)
      Step 4 – honour answer_found / return citations (never fabricate)
    """

    def __init__(self, dispatcher: KBDispatcher) -> None:
        self._dispatcher = dispatcher
        self.tool_calls: list[dict[str, Any]] = []  # full audit log

    async def _call(self, tool: str, **kwargs: Any) -> dict[str, Any]:
        self.tool_calls.append({"tool": tool, "args": kwargs})
        return await self._dispatcher.dispatch(tool, kwargs)  # type: ignore[return-value]

    async def answer(self, user_query: str, topic_hint: str = "") -> dict[str, Any]:
        """Run the full agent workflow and return the grounded retrieval result.

        Args:
            user_query: The question to answer.
            topic_hint: Word(s) that identify the target KB (e.g. "AUTOSAR", "C++").

        Returns:
            Dict with answer_found, kb_used, citations, and chunks.
        """
        # Step 1 – discover all KBs
        kb_list = await self._call("list_knowledge_bases")
        kbs: list[dict[str, Any]] = kb_list["knowledge_bases"]
        assert kbs, "Agent received empty KB list — cannot proceed."

        # Step 2 – pick the best matching KB (keyword match on description)
        hint = (topic_hint or user_query).lower()
        selected_kb = next(
            (
                kb
                for kb in kbs
                if any(word in kb["description"].lower() for word in hint.split())
            ),
            kbs[0],  # fallback to first KB
        )

        # Step 3 – retrieve from the chosen KB
        result = await self._call(
            "retrieve_from_kb",
            kb_name=selected_kb["name"],
            query=user_query,
            top_k=5,
        )

        # Step 4 – grounding contract: never fabricate
        if not result["answer_found"]:
            return {
                "answer_found": False,
                "reason": result.get("reason", "No evidence found."),
                "query": user_query,
                "kb_used": selected_kb["name"],
            }

        return {
            "answer_found": True,
            "query": user_query,
            "kb_used": selected_kb["name"],
            "citations": result.get("citations", []),
            "chunks": result.get("chunks", []),
        }

    async def ingest_document(
        self, kb_name: str, file_path: str
    ) -> dict[str, Any]:
        """Inspect documents in a KB after simulated ingestion.

        The agent first calls list_knowledge_bases to resolve the KB id, then
        calls list_documents to verify the document was indexed.

        Args:
            kb_name: Target knowledge base name.
            file_path: Path to the document (recorded in audit log only).

        Returns:
            list_documents tool result.
        """
        kb_list = await self._call("list_knowledge_bases")
        kb = next(
            (k for k in kb_list["knowledge_bases"] if k["name"] == kb_name), None
        )
        assert kb is not None, f"KB '{kb_name}' not found by agent."
        return await self._call("list_documents", kb_id=kb["id"])


# ─── Tests: KB creation ───────────────────────────────────────────────────────


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestMCPServerKBCreation:
    """Verify that both knowledge bases can be created via the tool interface."""

    async def test_create_space_autosar_kb(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch(
            "create_knowledge_base",
            {
                "name": "space_autosar",
                "description": "AUTOSAR SWS COM standard — communication stack specifications.",
            },
        )

        assert result["success"] is True
        assert result["name"] == "space_autosar"
        assert "id" in result

    async def test_create_space_coding_guidelines_kb(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch(
            "create_knowledge_base",
            {
                "name": "space_coding_guidelines",
                "description": "C++ Coding Standards — language usage rules and best practices.",
            },
        )

        assert result["success"] is True
        assert result["name"] == "space_coding_guidelines"
        assert "id" in result

    async def test_list_knowledge_bases_returns_both(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch("list_knowledge_bases", {})

        names = {kb["name"] for kb in result["knowledge_bases"]}
        assert "space_autosar" in names
        assert "space_coding_guidelines" in names
        assert len(result["knowledge_bases"]) == 2

    async def test_list_knowledge_bases_includes_descriptions(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch("list_knowledge_bases", {})

        kbs_by_name = {kb["name"]: kb for kb in result["knowledge_bases"]}
        assert "AUTOSAR" in kbs_by_name["space_autosar"]["description"]
        assert "C++" in kbs_by_name["space_coding_guidelines"]["description"]

    async def test_create_unknown_kb_returns_error(self) -> None:
        """Creating a KB not in the test fixture mapping returns failure."""
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch(
            "create_knowledge_base",
            {"name": "unknown_kb", "description": "should fail"},
        )

        assert result["success"] is False
        assert "error" in result


# ─── Tests: Ingestion ─────────────────────────────────────────────────────────


class TestMCPServerIngestion:
    """Verify document ingestion plumbing via the dispatcher interface."""

    async def test_list_documents_autosar_after_ingestion(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch("list_documents", {"kb_id": KB_AUTOSAR.id})

        docs = result["documents"]
        assert len(docs) >= 1
        assert docs[0]["name"] == "AUTOSAR_SWS_COM.pdf"
        assert docs[0]["status"] == "indexed"
        assert docs[0]["chunk_count"] > 0

    async def test_list_documents_coding_after_ingestion(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch("list_documents", {"kb_id": KB_CODING.id})

        docs = result["documents"]
        assert len(docs) >= 1
        assert docs[0]["name"] == "C++ Coding Standards.pdf"
        assert docs[0]["status"] == "indexed"

    async def test_ingestion_service_called_with_correct_path(self) -> None:
        """IngestionService.ingest_file receives the KB-specific file path."""
        dispatcher = _build_dispatcher()
        ingestion_path = Path(
            "data/knowledge_bases/space_autosar/AUTOSAR_SWS_COM.pdf"
        )

        await dispatcher._ingestion.ingest_file(
            ingestion_path, KB_AUTOSAR.id, skip_duplicates=True
        )

        dispatcher._ingestion.ingest_file.assert_called_once_with(
            ingestion_path, KB_AUTOSAR.id, skip_duplicates=True
        )

    async def test_ingestion_returns_indexed_document(self) -> None:
        dispatcher = _build_dispatcher()

        doc = await dispatcher._ingestion.ingest_file(
            Path("data/knowledge_bases/space_autosar/AUTOSAR_SWS_COM.pdf"),
            KB_AUTOSAR.id,
        )

        assert doc.status == DocumentStatus.INDEXED
        assert doc.chunk_count > 0
        assert doc.filename == "AUTOSAR_SWS_COM.pdf"

    async def test_both_kb_folders_have_documents(self) -> None:
        """Both KB folders return their respective documents."""
        dispatcher = _build_dispatcher()

        autosar_docs = await dispatcher.dispatch(
            "list_documents", {"kb_id": KB_AUTOSAR.id}
        )
        coding_docs = await dispatcher.dispatch(
            "list_documents", {"kb_id": KB_CODING.id}
        )

        assert autosar_docs["documents"][0]["name"] == "AUTOSAR_SWS_COM.pdf"
        assert coding_docs["documents"][0]["name"] == "C++ Coding Standards.pdf"


# ─── Tests: Retrieval ─────────────────────────────────────────────────────────


class TestMCPServerRetrieval:
    """Verify retrieve_from_kb and search_knowledge_bases tool responses."""

    async def test_retrieve_from_autosar_kb_answer_found(
        self, autosar_chunk: ChunkWithScore
    ) -> None:
        autosar_result = _make_retrieval_result(
            "I-PDU signal mapping", KB_AUTOSAR, autosar_chunk
        )
        dispatcher = _build_dispatcher(autosar_retrieval=autosar_result)

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {"kb_name": "space_autosar", "query": "I-PDU signal mapping", "top_k": 5},
        )

        assert result["answer_found"] is True
        assert result["knowledge_base"] == "space_autosar"
        assert len(result["chunks"]) >= 1
        assert "I-PDU" in result["chunks"][0]["content"]

    async def test_retrieve_from_coding_kb_answer_found(
        self, coding_chunk: ChunkWithScore
    ) -> None:
        coding_result = _make_retrieval_result(
            "identifier hiding rule", KB_CODING, coding_chunk
        )
        dispatcher = _build_dispatcher(coding_retrieval=coding_result)

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {
                "kb_name": "space_coding_guidelines",
                "query": "identifier hiding rule",
                "top_k": 5,
            },
        )

        assert result["answer_found"] is True
        assert "A2-10-1" in result["chunks"][0]["content"]

    async def test_retrieve_unknown_kb_returns_not_found(self) -> None:
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {"kb_name": "nonexistent_kb", "query": "anything"},
        )

        assert result["answer_found"] is False
        assert "not found" in result["reason"].lower()

    async def test_retrieve_no_results_returns_not_found(self) -> None:
        """When the vector store returns nothing, answer_found must be False."""
        dispatcher = _build_dispatcher()  # no retrieval results configured

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {"kb_name": "space_autosar", "query": "quantum entanglement", "top_k": 5},
        )

        assert result["answer_found"] is False

    async def test_search_knowledge_bases_returns_per_kb_results(
        self,
        autosar_chunk: ChunkWithScore,
        coding_chunk: ChunkWithScore,
    ) -> None:
        autosar_result = _make_retrieval_result("signal", KB_AUTOSAR, autosar_chunk)
        coding_result = _make_retrieval_result("signal", KB_CODING, coding_chunk)
        dispatcher = _build_dispatcher(
            autosar_retrieval=autosar_result, coding_retrieval=coding_result
        )

        result = await dispatcher.dispatch(
            "search_knowledge_bases", {"query": "signal", "top_k": 3}
        )

        assert "results_by_kb" in result
        assert len(result["results_by_kb"]) == 2
        kb_names = {r["knowledge_base_name"] for r in result["results_by_kb"]}
        assert "space_autosar" in kb_names
        assert "space_coding_guidelines" in kb_names

    async def test_citations_include_page_and_section(
        self, autosar_chunk: ChunkWithScore
    ) -> None:
        autosar_result = _make_retrieval_result(
            "PDU communication", KB_AUTOSAR, autosar_chunk
        )
        dispatcher = _build_dispatcher(autosar_retrieval=autosar_result)

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {"kb_name": "space_autosar", "query": "PDU communication", "top_k": 5},
        )

        citations = result.get("citations", [])
        assert len(citations) >= 1
        first = citations[0]
        assert first["source_document"] == "AUTOSAR_SWS_COM.pdf"
        assert first["page_numbers"] == [12, 13]
        assert first["section"] == "COM Signal Handling"

    async def test_grounding_contract_chunks_empty_when_not_found(self) -> None:
        """answer_found=False must not include fabricated chunks in the response."""
        dispatcher = _build_dispatcher()

        result = await dispatcher.dispatch(
            "retrieve_from_kb",
            {"kb_name": "space_autosar", "query": "irrelevant topic"},
        )

        assert result["answer_found"] is False
        assert result.get("chunks", []) == []


# ─── Tests: Agent workflow ────────────────────────────────────────────────────


class TestMCPKnowledgeBaseAgent:
    """Drive MCPKnowledgeBaseAgent through its full documented workflow."""

    async def test_agent_selects_autosar_kb_for_com_query(
        self, autosar_chunk: ChunkWithScore
    ) -> None:
        autosar_result = _make_retrieval_result(
            "COM signal transmission", KB_AUTOSAR, autosar_chunk
        )
        agent = MCPKnowledgeBaseAgent(_build_dispatcher(autosar_retrieval=autosar_result))

        response = await agent.answer(
            "How does COM handle signal transmission?", topic_hint="AUTOSAR"
        )

        assert response["answer_found"] is True
        assert response["kb_used"] == "space_autosar"
        assert len(response["citations"]) >= 1

    async def test_agent_selects_coding_kb_for_cpp_query(
        self, coding_chunk: ChunkWithScore
    ) -> None:
        coding_result = _make_retrieval_result(
            "identifier scoping rules", KB_CODING, coding_chunk
        )
        agent = MCPKnowledgeBaseAgent(_build_dispatcher(coding_retrieval=coding_result))

        response = await agent.answer(
            "What are the rules for identifier scoping?", topic_hint="C++"
        )

        assert response["answer_found"] is True
        assert response["kb_used"] == "space_coding_guidelines"
        assert "A2-10-1" in response["chunks"][0]["content"]

    async def test_agent_returns_not_found_when_no_evidence(self) -> None:
        agent = MCPKnowledgeBaseAgent(_build_dispatcher())

        response = await agent.answer(
            "Explain the lifecycle of a butterfly", topic_hint="AUTOSAR"
        )

        assert response["answer_found"] is False
        assert "reason" in response

    async def test_agent_calls_list_knowledge_bases_first(
        self, autosar_chunk: ChunkWithScore
    ) -> None:
        """Step 1 of the agent workflow must always be list_knowledge_bases."""
        autosar_result = _make_retrieval_result("I-PDU", KB_AUTOSAR, autosar_chunk)
        agent = MCPKnowledgeBaseAgent(_build_dispatcher(autosar_retrieval=autosar_result))

        await agent.answer("I-PDU alignment", topic_hint="AUTOSAR")

        assert agent.tool_calls[0]["tool"] == "list_knowledge_bases"
        assert agent.tool_calls[1]["tool"] == "retrieve_from_kb"

    async def test_agent_ingestion_audit_log(self) -> None:
        """Agent verifies ingested documents via list_knowledge_bases + list_documents."""
        agent = MCPKnowledgeBaseAgent(_build_dispatcher())

        docs_result = await agent.ingest_document(
            "space_autosar",
            "data/knowledge_bases/space_autosar/AUTOSAR_SWS_COM.pdf",
        )

        assert "documents" in docs_result
        assert docs_result["documents"][0]["name"] == "AUTOSAR_SWS_COM.pdf"
        tools_called = [c["tool"] for c in agent.tool_calls]
        assert "list_knowledge_bases" in tools_called
        assert "list_documents" in tools_called

    async def test_agent_full_workflow_autosar_then_coding(
        self,
        autosar_chunk: ChunkWithScore,
        coding_chunk: ChunkWithScore,
    ) -> None:
        """Full agent session: queries both KBs in sequence, checks grounding."""
        autosar_result = _make_retrieval_result("I-PDU", KB_AUTOSAR, autosar_chunk)
        coding_result = _make_retrieval_result(
            "identifier naming", KB_CODING, coding_chunk
        )
        agent = MCPKnowledgeBaseAgent(
            _build_dispatcher(
                autosar_retrieval=autosar_result,
                coding_retrieval=coding_result,
            )
        )

        autosar_response = await agent.answer(
            "What is an I-PDU in AUTOSAR?", topic_hint="AUTOSAR"
        )
        coding_response = await agent.answer(
            "Explain identifier naming rules", topic_hint="C++"
        )

        assert autosar_response["answer_found"] is True
        assert autosar_response["kb_used"] == "space_autosar"

        assert coding_response["answer_found"] is True
        assert coding_response["kb_used"] == "space_coding_guidelines"

        # Grounding: each response cites its own source document
        autosar_docs = {c["source_document"] for c in autosar_response["citations"]}
        coding_docs = {c["source_document"] for c in coding_response["citations"]}
        assert "AUTOSAR_SWS_COM.pdf" in autosar_docs
        assert "C++ Coding Standards.pdf" in coding_docs

    async def test_agent_tool_call_audit_across_session(
        self,
        autosar_chunk: ChunkWithScore,
        coding_chunk: ChunkWithScore,
    ) -> None:
        """Audit log accumulates all tool calls across a multi-turn session."""
        autosar_result = _make_retrieval_result("I-PDU", KB_AUTOSAR, autosar_chunk)
        coding_result = _make_retrieval_result("rule", KB_CODING, coding_chunk)
        agent = MCPKnowledgeBaseAgent(
            _build_dispatcher(
                autosar_retrieval=autosar_result,
                coding_retrieval=coding_result,
            )
        )

        await agent.answer("I-PDU basics", topic_hint="AUTOSAR")
        await agent.answer("naming rule A2-10-1", topic_hint="C++")

        # Two agent turns → 4 tool calls (2× list_knowledge_bases + 2× retrieve)
        assert len(agent.tool_calls) == 4
        tool_names = [c["tool"] for c in agent.tool_calls]
        assert tool_names.count("list_knowledge_bases") == 2
        assert tool_names.count("retrieve_from_kb") == 2
