# Extending the Knowledge Base Platform

## Adding a New Document Parser

1. **Create the parser file:**

```python
# src/parsers/docx/docx_parser.py
from __future__ import annotations
from pathlib import Path
from src.domain.interfaces import DocumentParser, ParsedDocument
from src.domain.models.document import Document
from src.parsers.base import ParsedDocumentResult

class DocxParser(DocumentParser):
    @property
    def name(self) -> str:
        return "docx"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    async def parse(self, path: Path, document: Document) -> ParsedDocument:
        # ... implementation ...
        return ParsedDocumentResult(...)
```

2. **Register in the registry:**

```python
# src/parsers/registry.py — add to build_pdf_parser_registry or create new builder
from src.parsers.docx.docx_parser import DocxParser
registry.register(DocxParser(), priority=5)
```

3. **Write tests** in `tests/unit/test_parsers.py`.

---

## Adding a New Embedding Provider

```python
# src/infrastructure/embeddings/my_provider.py
from src.domain.interfaces import EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
    @property
    def dimension(self) -> int: return 768
    @property
    def model_name(self) -> str: return "my-model"
    async def embed_text(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Add factory branch in `src/embeddings/factory.py`.

---

## Adding a New Vector Store

```python
# src/infrastructure/vector_stores/weaviate_store.py
from src.domain.interfaces import VectorStore

class WeaviateVectorStore(VectorStore):
    # Implement all 7 abstract methods
    ...
```

Add factory branch in `src/vectorstores/factory.py` and config block in YAML files.

---

## Adding a New Reranker

```python
# src/reranking/my_reranker.py
from src.domain.interfaces import Reranker
from src.domain.models.chunk import ChunkWithScore

class MyReranker(Reranker):
    @property
    def model_name(self) -> str: return "my-reranker"

    async def rerank(
        self, query: str, candidates: list[ChunkWithScore], top_k: int
    ) -> list[ChunkWithScore]: ...
```

Add factory branch in `src/reranking/factory.py`.

---

## Knowledge Graph Extension Points

The platform is designed for future knowledge graph integration.  The following extension points exist:

1. **Entity extraction** — add a post-ingestion step after chunking
2. **Graph store** — implement a `GraphStore` interface alongside `VectorStore`
3. **Hybrid retrieval** — add a `GraphRetriever` to the retrieval pipeline
4. **RRF fusion** — extend to fuse graph + vector + BM25 results

No core modules need to be modified. Add new components and wire them in via factories and the DI container.
