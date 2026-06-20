"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import uvicorn
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.dependencies import get_container, Container
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import knowledge_bases, ingestion, retrieval
from src.config.settings import settings
from src.monitoring.logging import configure_logging, get_logger
from src.monitoring.tracing import configure_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup and shutdown."""
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info(
        "Starting Knowledge Base platform",
        env=settings.app_env,
        version="1.0.0",
    )
    settings.ensure_dirs()

    container: Container = app.state.container  # type: ignore[attr-defined]
    await container.initialise()

    logger.info("Knowledge Base platform ready")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("Shutting down Knowledge Base platform")
    await container.shutdown()


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns:
        Fully configured FastAPI application instance.
    """
    configure_logging(
        log_level=settings.observability.log_level,
        structured=settings.observability.structured_logging,
        log_dir=settings.paths.logs_dir,
    )
    configure_tracing(
        enabled=settings.observability.otel_enabled,
        otlp_endpoint=settings.observability.otel_endpoint,
    )

    app = FastAPI(
        title="Knowledge Base RAG Platform",
        description=(
            "Production-grade local RAG platform for enterprise document retrieval. "
            "Supports multiple knowledge bases, hybrid search, and MCP tool calling."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── Dependency injection container ────────────────────────────────────
    app.state.container = Container(settings)

    # ── Middleware ────────────────────────────────────────────────────────
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────
    app.include_router(knowledge_bases.router, prefix="/knowledge-bases", tags=["knowledge-bases"])
    app.include_router(ingestion.router, prefix="/ingest", tags=["ingestion"])
    app.include_router(retrieval.router, prefix="", tags=["retrieval"])

    # ── Prometheus metrics endpoint ───────────────────────────────────────
    if settings.observability.prometheus_enabled:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    @app.get("/health", tags=["ops"])
    async def health_check() -> dict:
        """Liveness probe."""
        return {"status": "ok", "env": settings.app_env}

    @app.get("/ready", tags=["ops"])
    async def readiness_check() -> dict:
        """Readiness probe — checks vector store connectivity."""
        try:
            container = app.state.container
            collections = await container.vector_store.list_collections()
            return {"status": "ready", "collections": len(collections)}
        except Exception as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=f"Not ready: {exc}") from exc

    return app


def main() -> None:
    """Entry point for the API server."""
    app = create_app()
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers,
        log_config=None,  # Loguru handles logging
        access_log=False,
    )


# Module-level app instance for ASGI servers (gunicorn, etc.)
app = create_app()
