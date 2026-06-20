"""Ingestion routes — upload and manage documents."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.api.dependencies import IngestionService, get_ingestion_service
from src.config.settings import settings
from src.domain.models.document import Document, DocumentStatus

router = APIRouter()


# ─── Response schemas ─────────────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    id: str
    name: str
    knowledge_base_id: str
    document_type: str
    status: str
    chunk_count: int
    page_count: int
    file_size_bytes: int
    parser_used: str | None
    error_message: str | None
    created_at: str
    processed_at: str | None

    @classmethod
    def from_domain(cls, doc: Document) -> DocumentResponse:
        return cls(
            id=doc.id,
            name=doc.name,
            knowledge_base_id=doc.knowledge_base_id,
            document_type=doc.document_type.value,
            status=doc.status.value,
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            file_size_bytes=doc.file_size_bytes,
            parser_used=doc.parser_used,
            error_message=doc.error_message,
            created_at=doc.created_at.isoformat(),
            processed_at=doc.processed_at.isoformat() if doc.processed_at else None,
        )


class IngestFolderRequest(BaseModel):
    kb_id: str
    folder_path: str
    skip_duplicates: bool = True


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest a document",
)
async def ingest_document(
    kb_id: Annotated[str, Form(description="Target knowledge base ID")],
    file: Annotated[UploadFile, File(description="Document file to ingest")],
    skip_duplicates: Annotated[bool, Form()] = True,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentResponse:
    """Upload and ingest a document into a knowledge base.

    Supported formats: PDF (more coming soon).

    The document is parsed, chunked, embedded, and indexed asynchronously.
    Poll the returned document ID to check processing status.

    **curl example:**
    ```bash
    curl -X POST http://localhost:8000/ingest \\
      -F "kb_id=<your-kb-id>" \\
      -F "file=@document.pdf"
    ```
    """
    # Validate file type
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is required.")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".txt", ".md"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Validate size
    max_bytes = settings.api.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {settings.api.max_upload_size_mb} MB.",
        )

    # Write to temp file and ingest
    uploads_dir = Path(settings.paths.uploads_dir)
    uploads_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = uploads_dir / file.filename
    tmp_path.write_bytes(content)

    try:
        doc = await service.ingest_file(tmp_path, kb_id, skip_duplicates)
        return DocumentResponse.from_domain(doc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # Keep file for re-ingestion; clean up only on error
        pass


@router.post(
    "/folder",
    response_model=list[DocumentResponse],
    summary="Ingest a folder of documents",
)
async def ingest_folder(
    body: IngestFolderRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentResponse]:
    """Ingest all supported documents from a server-side folder path.

    The folder must be accessible to the server process.

    **Example:**
    ```json
    {
      "kb_id": "...",
      "folder_path": "/data/iso_standards/"
    }
    ```
    """
    folder = Path(body.folder_path)
    try:
        docs = await service.ingest_folder(folder, body.kb_id, body.skip_duplicates)
        return [DocumentResponse.from_domain(d) for d in docs]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/{kb_id}/documents",
    response_model=list[DocumentResponse],
    summary="List documents in a knowledge base",
)
async def list_documents(
    kb_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentResponse]:
    """List all documents belonging to a knowledge base."""
    docs = await service.list_documents(kb_id)
    return [DocumentResponse.from_domain(d) for d in docs]
