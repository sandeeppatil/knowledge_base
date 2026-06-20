# Architecture

## Overview

The Knowledge Base platform follows **Clean Architecture** principles with strict layer separation and dependency inversion.

## Layer Diagram

```mermaid
graph TB
    subgraph "Entry Points"
        REST["REST API\n(FastAPI)"]
        MCP_S["MCP Server\n(stdio)"]
        CLI["CLI Tools"]
    end

    subgraph "Application Layer"
        KBS["KnowledgeBaseService"]
        IS["IngestionService"]
        RS["RetrievalService"]
    end

    subgraph "Domain Layer"
        DM["Domain Models\n(KnowledgeBase, Document, Chunk)"]
        DI["Domain Interfaces\n(VectorStore, EmbeddingProvider, ...)"]
        DE["Domain Events"]
    end

    subgraph "Infrastructure Layer"
        VS["Vector Stores\n(Qdrant, Chroma, FAISS)"]
        EM_P["Embedding Providers\n(SentenceTransformers, Ollama)"]
        DB_P["Persistence\n(SQLite)"]
    end

    subgraph "Pipeline Modules"
        PARSE["Parser Registry\n(Docling → PyMuPDF → OCR)"]
        CHUNK["Hierarchical Chunker"]
        RETR["Retrieval Pipeline\n(Dense + BM25 + RRF)"]
        RERANK["Reranker\n(BGE Cross-Encoder)"]
        ROUTE["KB Router\n(Semantic Routing)"]
    end

    REST & MCP_S & CLI --> KBS & IS & RS
    KBS & IS & RS --> DM & DI
    DI -.-> VS & EM_P & DB_P
    IS --> PARSE --> CHUNK
    RS --> RETR --> RERANK
    KBS --> ROUTE
```

## Component Responsibilities

### Domain Layer (`src/domain/`)

Zero external dependencies. Contains:

| Module | Purpose |
|--------|---------|
| `models/knowledge_base.py` | KB entity |
| `models/document.py` | Document entity |
| `models/chunk.py` | Chunk entity + metadata |
| `models/retrieval.py` | RetrievalResult + Citation |
| `interfaces/` | Abstract contracts for all pluggable components |
| `events/` | Domain events for loose coupling |

### Application Layer (`src/application/`)

Depends only on domain interfaces. Contains:

| Module | Purpose |
|--------|---------|
| `services/services.py` | KB CRUD, ingestion, retrieval orchestration |

### Infrastructure Layer (`src/infrastructure/`)

Concrete implementations:

| Module | Purpose |
|--------|---------|
| `persistence/sqlite_kb_repository.py` | SQLite-backed KB/Doc persistence |
| `vector_stores/qdrant_store.py` | Qdrant vector store |
| `vector_stores/chroma_store.py` | ChromaDB vector store |
| `vector_stores/faiss_store.py` | FAISS local vector store |
| `embeddings/sentence_transformers_provider.py` | Local dense embeddings |
| `embeddings/ollama_provider.py` | Ollama embedding API |
| `embeddings/openai_compatible_provider.py` | OpenAI-compatible API |

## Ingestion Pipeline

```mermaid
sequenceDiagram
    participant C as Client
    participant IP as IngestionPipeline
    participant PR as ParserRegistry
    participant CH as HierarchicalChunker
    participant EP as EmbeddingProvider
    participant VS as VectorStore
    participant DB as KBRepository

    C->>IP: ingest(file_path, kb)
    IP->>IP: validate_file() + checksum()
    IP->>DB: create_document()
    IP->>PR: select(file_path)
    PR-->>IP: parser (Docling/PyMuPDF/OCR)
    IP->>PR: parse(file, document)
    PR-->>IP: ParsedDocumentResult
    IP->>CH: chunk(parsed_doc, document)
    CH-->>IP: [Chunk, ...]
    IP->>EP: embed_batch([chunk.content, ...])
    EP-->>IP: [[float, ...], ...]
    IP->>VS: upsert(collection, chunks_with_embeddings)
    IP->>DB: update_document(status=COMPLETED)
    IP->>DB: update_kb(document_count++, chunk_count+=N)
    IP-->>C: Document(status=COMPLETED)
```

## Retrieval Pipeline

```mermaid
graph LR
    Q[Query] --> DE[Dense Embed]
    DE --> ANN[ANN Search\nQdrant]
    Q --> BM[BM25\nKeyword Search]
    ANN --> RRF[RRF Fusion]
    BM --> RRF
    RRF --> RR[BGE Reranker]
    RR --> GC[Grounded Context\nwith Citations]
```

## Storage Architecture

```
data/
├── knowledge_base.db          # SQLite: KB and Document metadata
├── knowledge_bases/           # Raw source files (optional)
│   └── {kb_id}/
│       └── {doc_id}/
│           └── original.pdf
├── uploads/                   # Temporary upload staging
├── models/                    # Cached embedding/reranker models
│   ├── BAAI_bge-m3/
│   └── BAAI_bge-reranker-v2-m3/
├── faiss_indices/             # FAISS index files (if using FAISS)
│   ├── {collection}.index
│   └── {collection}.meta.pkl
└── logs/
    └── app_2024-01-01.log.gz
```

Qdrant stores vectors externally in its own storage volume.

## Configuration Strategy

```
config/dev.yaml     → development defaults
config/test.yaml    → test overrides (fast, no OCR, no reranker)
config/prod.yaml    → production settings (GPU, full reranker, gRPC)

.env                → secrets and environment-specific overrides
```

Environment variables with `APP_` prefix or nested `__` delimiter override YAML.
