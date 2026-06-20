"""Prometheus metrics for the Knowledge Base platform.

All metrics are defined here and imported by the components that update them.
The /metrics endpoint is mounted by the FastAPI app.
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    CollectorRegistry,
    REGISTRY,
)

# ─── Ingestion metrics ────────────────────────────────────────────────────────

DOCUMENTS_INGESTED_TOTAL = Counter(
    "kb_documents_ingested_total",
    "Total number of documents successfully ingested.",
    ["knowledge_base", "document_type", "parser"],
)

DOCUMENTS_FAILED_TOTAL = Counter(
    "kb_documents_failed_total",
    "Total number of documents that failed ingestion.",
    ["knowledge_base", "error_type"],
)

CHUNKS_CREATED_TOTAL = Counter(
    "kb_chunks_created_total",
    "Total number of chunks created during ingestion.",
    ["knowledge_base", "content_type"],
)

INGESTION_DURATION_SECONDS = Histogram(
    "kb_ingestion_duration_seconds",
    "Time to ingest a single document end-to-end.",
    ["knowledge_base", "parser"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# ─── Retrieval metrics ────────────────────────────────────────────────────────

RETRIEVAL_REQUESTS_TOTAL = Counter(
    "kb_retrieval_requests_total",
    "Total retrieval requests.",
    ["knowledge_base", "answer_found"],
)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "kb_retrieval_latency_seconds",
    "End-to-end retrieval latency.",
    ["knowledge_base"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RETRIEVAL_CHUNKS_RETURNED = Histogram(
    "kb_retrieval_chunks_returned",
    "Number of chunks returned per retrieval request.",
    ["knowledge_base"],
    buckets=[1, 2, 5, 10, 15, 20],
)

# ─── Embedding metrics ────────────────────────────────────────────────────────

EMBEDDING_REQUESTS_TOTAL = Counter(
    "kb_embedding_requests_total",
    "Total embedding requests.",
    ["provider", "model"],
)

EMBEDDING_LATENCY_SECONDS = Histogram(
    "kb_embedding_latency_seconds",
    "Batch embedding latency.",
    ["provider"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

# ─── Knowledge base metrics ───────────────────────────────────────────────────

KNOWLEDGE_BASES_ACTIVE = Gauge(
    "kb_knowledge_bases_active",
    "Number of active knowledge bases.",
)

KB_CHUNK_COUNT = Gauge(
    "kb_chunk_count",
    "Number of chunks in a knowledge base.",
    ["knowledge_base"],
)

KB_DOCUMENT_COUNT = Gauge(
    "kb_document_count",
    "Number of documents in a knowledge base.",
    ["knowledge_base"],
)

# ─── API metrics ──────────────────────────────────────────────────────────────

API_REQUESTS_TOTAL = Counter(
    "kb_api_requests_total",
    "Total HTTP requests to the API.",
    ["method", "endpoint", "status_code"],
)

API_REQUEST_DURATION_SECONDS = Histogram(
    "kb_api_request_duration_seconds",
    "HTTP request processing latency.",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
