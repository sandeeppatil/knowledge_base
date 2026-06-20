# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base Platform — Production Dockerfile
# Multi-stage build: builder + slim runtime image
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests
COPY pyproject.toml .

# Install into a virtual environment for clean copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies (no dev extras)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[vlm]" || \
    pip install --no-cache-dir -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

LABEL maintainer="Knowledge Base Platform"
LABEL description="Production-grade local RAG platform"

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgomp1 \
    libopenblas0 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r kbuser && useradd -r -g kbuser kbuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .

# Create data directories and set ownership
RUN mkdir -p \
    /app/data/knowledge_bases \
    /app/data/models \
    /app/data/uploads \
    /app/data/logs \
    /app/data/faiss_indices && \
    chown -R kbuser:kbuser /app/data

# Switch to non-root user
USER kbuser

# Environment defaults (overridden by docker-compose or env vars)
ENV APP_ENV=production
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entry point
CMD ["python", "-m", "uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-config", "/dev/null"]
