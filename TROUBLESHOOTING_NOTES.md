# Troubleshooting Notes - KB PDF Ingestion

## Original Error
User encountered a 100-page PDF limit error when trying to send PDFs directly via Claude API:
```
GH Request Id: FB3F:166DE4:43CD5EB:4E06516:6A36CFDE
Reason: Request Failed: 400 
{"message":"messages.0.content.2.pdf.source.base64.data: A maximum of 100 PDF pages may be provided."}
```

## Solution Strategy
Instead of sending PDFs directly to Claude, use the local RAG (Retrieval-Augmented Generation) platform's ingestion API which:
1. Processes documents locally without size limits
2. Preserves document structure and tables
3. Uses hybrid retrieval (dense + BM25) with cross-encoder reranking
4. Returns grounded answers with full citations

## Code Fixes Applied

### 1. Domain Model Enhancements (`src/domain/models/knowledge_base.py`)
Added missing required fields that are essential for KB lifecycle:
- `version`: KB schema version
- `is_active`: Whether KB is in active use

These fields ensure compatibility with API response schemas and database tracking.

### 2. Application Service Fixes (`src/application/services/services.py`)  
Fixed KB creation to properly generate missing fields:
```python
# Generate KB ID and collection name
kb_id = str(uuid.uuid4())
collection_name = f"kb_{kb_id.replace('-', '_')}"

# Get embedding dimension and model
embedding_dimension = self._embedder.dimension
embedding_model_name = embedding_model or self._embedder.model_name
vector_store_type = vector_store_provider or "qdrant"
```

### 3. ORM Schema Alignment (`src/infrastructure/persistence/sqlite_kb_repository.py`)
Updated KnowledgeBaseORM to include all necessary fields:
- `embedding_dimension`: Vector dimension for embeddings
- `vector_store_type`: Type of vector store (qdrant, faiss, etc)
- `collection_name`: Qdrant collection name for this KB

Updated mappers to correctly serialize/deserialize KB data between domain and database models.

## Knowledge Base Setup

### AUTOSAR SWS COM KB
- **Name**: AUTOSAR_SWS_COM
- **Collection**: `kb_<uuid>`
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector Store**: Qdrant
- **Status**: Ready for document ingestion

### PDF Ingestion Process
1. User uploads AUTOSAR_SWS_COM.pdf to `/ingest` endpoint with KB ID
2. Docling parses PDF (primary), PyMuPDF (secondary), OCR (fallback)
3. Hierarchical chunking preserves document structure
4. Chunks embedded and stored in Qdrant
5. BM25 index built for full-text retrieval
6. Hybrid search enabled (dense + lexical + cross-encoder reranking)

## API Endpoints

### Create Knowledge Base
```bash
POST /knowledge-bases
Content-Type: application/json

{
  "name": "AUTOSAR_SWS_COM",
  "description": "AUTOSAR Software-Services Communication standard"
}
```

### Ingest Document
```bash
POST /ingest
Content-Type: multipart/form-data

kb_id=<kb-id>
file=@document.pdf
```

### Retrieve from KB
```bash
POST /retrieve
Content-Type: application/json

{
  "kb_name": "AUTOSAR_SWS_COM",
  "query": "What is CAN communication?",
  "top_k": 10
}
```

## Grounding Contract

The system enforces zero-hallucination:
- Returns `answer_found: false` when no evidence exists
- Every response includes full citations: `source_document`, `page_numbers`, `section`
- Never generates unsupported content

## Testing Commands

```bash
# Check API health
curl http://localhost:8000/health

# List KBs
curl http://localhost:8000/knowledge-bases

# Test retrieval
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"kb_name":"AUTOSAR_SWS_COM","query":"test","top_k":5}'
```

## Remaining Work

1. **Vector Store Connectivity**: Ensure Qdrant container is running and accessible
2. **Database Initialization**: First API call will create SQLite schema
3. **Document Ingestion**: Upload PDF via `/ingest` endpoint
4. **MCP Integration**: Register with GitHub Copilot for tool access
5. **Agent Routing**: Configure agent to select correct KB based on query

## Architecture Reference

This implementation follows **Clean Architecture** principles:
```
API Layer (FastAPI)
    ↓
Application Layer (Services)
    ↓
Domain Layer (Models, Interfaces)
    ↓
Infrastructure Layer (Persistence, Vector Stores, Embeddings)
```

All business logic lives in the Application layer; infrastructure details are abstracted behind interfaces.
