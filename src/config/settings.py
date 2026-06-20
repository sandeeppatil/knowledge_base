"""Configuration settings for the Knowledge Base platform.

Loads YAML config from config/{env}.yaml and merges with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Sub-config models ────────────────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: str = "sentence_transformers"
    model: str = "BAAI/bge-m3"
    batch_size: int = 64
    device: str = "cpu"
    normalize: bool = True
    cache_dir: str = "./data/models"


class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    prefer_grpc: bool = False
    timeout: int = 30


class ChromaConfig(BaseModel):
    host: str = "localhost"
    port: int = 8001


class FaissConfig(BaseModel):
    index_path: str = "./data/faiss_indices"


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    provider: str = "qdrant"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    faiss: FaissConfig = Field(default_factory=FaissConfig)


class RerankerConfig(BaseModel):
    """Reranker configuration."""

    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    batch_size: int = 16
    top_k: int = 10


class OcrConfig(BaseModel):
    enabled: bool = True
    engine: str = "tesseract"


class ParserPdfConfig(BaseModel):
    primary: str = "docling"
    fallback: str = "pymupdf"
    ocr_fallback: bool = True


class ParsersConfig(BaseModel):
    """Document parser configuration."""

    pdf: ParserPdfConfig = Field(default_factory=ParserPdfConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)


class ChunkingConfig(BaseModel):
    """Chunking configuration."""

    strategy: str = "hierarchical"
    chunk_size: int = 750
    chunk_overlap: int = 100
    min_chunk_size: int = 100


class RetrievalConfig(BaseModel):
    """Retrieval pipeline configuration."""

    top_k: int = 20
    dense_weight: float = 0.7
    bm25_weight: float = 0.3
    final_top_k: int = 10
    rrf_k: int = 60


class VlmConfig(BaseModel):
    """Visual Language Model configuration for figure descriptions."""

    enabled: bool = False
    provider: str = "ollama"
    model: str = "qwen2.5-vl:7b"
    ollama_url: str = "http://localhost:11434"


class PathsConfig(BaseModel):
    """Filesystem paths configuration."""

    data_dir: str = "./data"
    knowledge_bases_dir: str = "./data/knowledge_bases"
    uploads_dir: str = "./data/uploads"
    logs_dir: str = "./data/logs"
    models_dir: str = "./data/models"


class ObservabilityConfig(BaseModel):
    """Observability configuration."""

    log_level: str = "INFO"
    structured_logging: bool = True
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:4317"
    prometheus_enabled: bool = True


class ApiConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_size_mb: int = 100


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite+aiosqlite:///./data/knowledge_base.db"


# ─── Root settings ────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Root application settings.

    Configuration is loaded in this order (later wins):
    1. config/{env}.yaml defaults
    2. Environment variables (APP_ prefix or direct names)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development", alias="APP_ENV")

    # ── Sub-configs populated from YAML ──────────────────────────────────────
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    parsers: ParsersConfig = Field(default_factory=ParsersConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        allowed = {"development", "test", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    def ensure_dirs(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.paths.data_dir,
            self.paths.knowledge_bases_dir,
            self.paths.uploads_dir,
            self.paths.logs_dir,
            self.paths.models_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def _load_yaml_config(env: str) -> dict[str, Any]:
    """Load YAML config file for the given environment."""
    config_dir = Path(__file__).parent.parent.parent / "config"
    yaml_path = config_dir / f"{env}.yaml"

    if not yaml_path.exists():
        yaml_path = config_dir / "dev.yaml"

    with yaml_path.open("r") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> Settings:
    """Load and return fully-merged application settings.

    Reads the APP_ENV environment variable (default: development), loads
    the corresponding YAML file, then overlays any environment variables.

    Returns:
        Settings: Fully initialised settings instance.
    """
    env = os.getenv("APP_ENV", "development")
    yaml_data = _load_yaml_config(env)

    # Build nested config objects from YAML, then let pydantic-settings
    # overlay env-var overrides on top.
    return Settings.model_validate({**yaml_data, "app_env": env})


# Module-level singleton – import this everywhere.
settings: Settings = load_settings()
