# Operations Runbooks

## Create a Knowledge Base

```bash
curl -X POST http://localhost:8000/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "ISO Standards", "description": "ISO quality management standards."}'
```

Save the returned `id` for subsequent operations.

---

## Ingest Documents

### Single file
```bash
curl -X POST http://localhost:8000/ingest \
  -F "kb_id=<kb-id>" \
  -F "file=@/path/to/document.pdf"
```

### Folder (server-side)
```bash
curl -X POST http://localhost:8000/ingest/folder \
  -H "Content-Type: application/json" \
  -d '{"kb_id": "<kb-id>", "folder_path": "/data/docs/"}'
```

---

## Rebuild a Knowledge Base

Rebuilds all embeddings using the current model. Use after upgrading the embedding model.

```python
# Via API (call rebuild endpoint when implemented)
# Or directly via the service:
from src.application.services.services import KnowledgeBaseService
await kb_service.rebuild(kb_id, ingestion_pipeline)
```

---

## Backup a Knowledge Base

```bash
# Backup Qdrant data
docker exec kb-qdrant qdrant snapshot create <collection-name>

# Backup SQLite metadata
cp data/knowledge_base.db data/knowledge_base.db.backup

# Backup uploaded files
tar -czf kb_backup_$(date +%Y%m%d).tar.gz data/knowledge_bases/
```

---

## Restore a Knowledge Base

```bash
# Restore SQLite
cp data/knowledge_base.db.backup data/knowledge_base.db

# Restore files
tar -xzf kb_backup_20240101.tar.gz

# Rebuild vector index from files
# Call rebuild endpoint for each KB
```

---

## Upgrade Embedding Models

1. Update `EMBEDDING_MODEL` in `.env` or `config/prod.yaml`
2. Restart the API server
3. Call rebuild for each KB (rebuilds embeddings with new model)

> **Warning**: Different models have incompatible vector spaces.
> Always rebuild after changing models.

---

## Vector Store Migration (Qdrant → FAISS)

1. Update `VECTOR_STORE_PROVIDER=faiss` in config
2. Rebuild all KBs: existing vectors are not portable between stores
3. Verify with retrieval queries

---

## Disaster Recovery

1. Restore SQLite backup (document/KB metadata)
2. Restore uploaded files
3. Rebuild each KB (re-embed all documents)

Estimated recovery time: depends on document volume and hardware.
~100 pages/minute on CPU, ~1000 pages/minute on GPU.

---

## Health Checks

```bash
# API liveness
curl http://localhost:8000/health

# API readiness (checks vector store)
curl http://localhost:8000/ready

# Qdrant health
curl http://localhost:6333/healthz

# Prometheus metrics
curl http://localhost:8000/metrics
```
