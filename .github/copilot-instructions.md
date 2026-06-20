# Knowledge Base Platform — GitHub Copilot Instructions

## Project Overview

This is a **production-grade local RAG (Retrieval-Augmented Generation) platform**
built in Python.  It provides:

- **Multi-knowledge-base management** — create, delete, rebuild isolated KBs.
- **Document ingestion** — PDF parsing with Docling (Tier 1), PyMuPDF (Tier 2), OCR fallback.
- **Hierarchical chunking** — structure-aware, table/figure-preserving.
- **Hybrid retrieval** — dense (ANN) + BM25 + RRF fusion + cross-encoder reranking.
- **MCP tool server** — callable by GitHub Copilot Agent Mode.
- **FastAPI REST API** — OpenAPI-documented.
- **Docker deployment** — fully containerised with Qdrant.

---

## Architecture Rules

This project uses **Clean Architecture** with strict separation of concerns.

### Layer Dependency Order (must not be violated)
```
API  →  Application  →  Domain  →  (nothing)
                ↓
         Infrastructure  →  Domain interfaces
```

- **`src/domain/`** — models, interfaces, events.  Zero external imports.
- **`src/application/`** — use cases and services.  Depends on domain interfaces only.
- **`src/infrastructure/`** — concrete implementations (Qdrant, SQLite, SentenceTransformers).
- **`src/api/`** — FastAPI routes.  No business logic — delegates to services only.
- **`src/parsers/`**, **`src/chunking/`**, **`src/retrieval/`**, **`src/reranking/`** — pluggable domain modules.

### Critical Rules

1. **No business logic in API routes.** Routes call services; services contain logic.
2. **Depend on interfaces, not implementations.** `VectorStore`, `EmbeddingProvider`, `DocumentParser`, etc.
3. **Factories create implementations.** `src/embeddings/factory.py`, `src/vectorstores/factory.py`, etc.
4. **Configuration drives behaviour.** All component selection is YAML + env-var driven.

---

## Grounding Contract — NEVER VIOLATE

The system must never hallucinate.

```python
# CORRECT: answer_found=False when no evidence
return RetrievalResult.not_found(query)

# WRONG: fabricating citations or content
return RetrievalResult(answer_found=True, chunks=[fake_chunk])
```

**Rules:**
- Return `answer_found: false` when no supporting evidence exists.
- Never inject unsupported context into the LLM.
- Every returned chunk must cite `source_document`, `page_numbers`, and `section`.
- Never generate fake citations.

---

## Coding Standards

- **Python 3.11+** — use `from __future__ import annotations` in all files.
- **Type hints** — full type annotations on all public functions and methods.
- **Pydantic v2** — use `model_validate()`, not `parse_obj()`.
- **Async** — all I/O operations must be `async`/`await`.  Use `run_in_executor` for CPU-bound work.
- **Loguru** — use `get_logger(__name__)` from `src.monitoring.logging`.
- **Google-style docstrings** on all public classes and methods.
- **Line length** — 100 characters (Ruff enforced).

```python
# CORRECT
from src.monitoring.logging import get_logger
logger = get_logger(__name__)
logger.info("Processing document", doc_id=doc.id, kb=kb.name)

# WRONG
import logging
logging.info(f"Processing {doc.id}")
```

---

## Adding a New Parser

1. Create `src/parsers/<type>/<name>_parser.py`
2. Implement `DocumentParser` interface:
   ```python
   class MyParser(DocumentParser):
       @property
       def name(self) -> str: return "my_parser"
       def supports(self, path: Path) -> bool: ...
       async def parse(self, path: Path, document: Document) -> ParsedDocument: ...
   ```
3. Return a `ParsedDocumentResult` — the canonical intermediate format.
4. Register in `src/parsers/registry.py`.
5. Write unit tests in `tests/unit/test_parsers.py`.

---

## Adding a New Vector Store

1. Create `src/infrastructure/vector_stores/<name>_store.py`
2. Implement `VectorStore` interface (all 7 abstract methods).
3. Add a factory branch in `src/vectorstores/factory.py`.
4. Add YAML config block in `config/dev.yaml`.
5. Write integration tests.

---

## Adding a New Embedding Provider

1. Create `src/infrastructure/embeddings/<name>_provider.py`
2. Implement `EmbeddingProvider` interface (`embed_text`, `embed_batch`, `dimension`, `model_name`).
3. Add factory branch in `src/embeddings/factory.py`.
4. Write unit tests.

---

## Ingestion Rules

- **Always compute SHA-256 checksums** for deduplication.
- **Never split a table across chunks.** Tables are atomic units.
- **Preserve structural metadata** — every chunk must have `page_numbers`, `section_title`, `heading_path`.
- **Parser fallback order** — Docling → PyMuPDF → OCR. Never skip fallbacks.
- **Validate files** before ingestion — check path traversal, size, type.

---

## Retrieval Rules

- **Hybrid always wins** — use dense + BM25 + RRF, not just ANN.
- **Reranker is optional but recommended** for production.
- **BM25 corpus** is in-memory; invalidate cache after new ingestion.
- **Top-k defaults**: 20 candidates → 10 final (configurable).

---

## Testing Rules

- **Minimum coverage: 90%** (enforced by pytest-cov).
- **Unit tests** — use mocks for all I/O.  No network calls.
- **Integration tests** — use mocked infrastructure; test pipeline logic.
- **E2E tests** — use `httpx.AsyncClient` with `ASGITransport`.
- **All tests are async** — `@pytest.mark.asyncio`.
- **Fixtures in `tests/conftest.py`** — shared across all test modules.

```python
# CORRECT
@pytest.mark.asyncio
async def test_retrieval(mock_vector_store, mock_embedding_provider):
    ...

# WRONG — synchronous test for async code
def test_retrieval():
    result = asyncio.run(retrieve(...))
```

---

## MCP Tool Contract

Every MCP tool must:
- Have a JSON Schema input spec.
- Return JSON-serialisable output.
- Be **stateless** — no session state between calls.
- Return `{"answer_found": false}` when no evidence, not an error.
- Never raise unhandled exceptions — wrap in try/except and return error dict.

```python
# Tool response shape when no evidence found:
{
    "answer_found": false,
    "reason": "No supporting evidence found in selected knowledge base.",
    "query": "..."
}
```

---

## Docker Rules

- **Docker-first development** — everything must run in Docker.
- Use **multi-stage builds** to keep images lean.
- **Never run as root** in production containers.
- **Named volumes** for all persistent state (`kb_data`, `qdrant_data`, `kb_models`).
- **Health checks** on all services.
- Use **profiles** to control which services start (`dev`, `production`, `monitoring`).

---

## Agent Workflow

When an agent calls the retrieval tools, it **must** follow this workflow:

```
Step 1: list_knowledge_bases()
        → Read name + description of each KB

Step 2: Select the most relevant KB based on description

Step 3: retrieve_from_kb(kb_name="...", query="...", top_k=10)
        → Check answer_found

Step 4: If answer_found=True:
          Inject chunks into LLM context with citations
        If answer_found=False:
          Return {answer_found: false} to the user
          DO NOT fabricate an answer
```

---

## File Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Modules | `snake_case.py` | `qdrant_store.py` |
| Classes | `PascalCase` | `QdrantVectorStore` |
| Functions | `snake_case` | `create_vector_store()` |
| Constants | `UPPER_SNAKE` | `MAX_FILE_SIZE_BYTES` |
| Test files | `test_<module>.py` | `test_rrf_fusion.py` |

---

## Observability Checklist

Every significant operation must:
- [ ] Log with `logger.info/debug/warning/error` (structured fields)
- [ ] Record Prometheus metric (duration, count, or gauge)
- [ ] Have an OpenTelemetry span (via `get_tracer(__name__)`)

```python
tracer = get_tracer(__name__)

async def my_operation():
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("kb_id", kb.id)
        # ... work ...
        MY_COUNTER.inc()
        logger.info("Operation complete", kb_id=kb.id)
```
