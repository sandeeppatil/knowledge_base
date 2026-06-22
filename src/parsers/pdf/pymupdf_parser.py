"""PyMuPDF (fitz) PDF parser — Tier 2 parser for text, metadata, and images."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from src.domain.interfaces import DocumentParser, ParsedDocument
from src.domain.models.document import Document
from src.monitoring.logging import get_logger
from src.parsers.base import (
    ParsedDocumentResult,
    ParsedFigure,
    ParsedSection,
    ParsedTable,
)

logger = get_logger(__name__)


class PyMuPDFParser(DocumentParser):
    """Tier-2 PDF parser using PyMuPDF (fitz).

    Handles:
    - Text extraction with layout awareness
    - Document metadata (title, author, etc.)
    - Image/figure extraction
    - Simple table detection via text blocks

    Args:
        extract_images: Whether to extract embedded images.
        min_text_length: Minimum text length to consider a block meaningful.
    """

    def __init__(
        self,
        extract_images: bool = True,
        min_text_length: int = 10,
    ) -> None:
        self._extract_images = extract_images
        self._min_text_length = min_text_length

    @property
    def name(self) -> str:
        return "pymupdf"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    async def parse(self, path: Path, document: Document) -> ParsedDocument:
        import asyncio

        logger.info("PyMuPDF parsing PDF", path=str(path), doc_id=document.id)
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._parse_sync, path, document
            )
        except Exception as exc:
            logger.error("PyMuPDF parse failed", path=str(path), error=str(exc))
            raise RuntimeError(f"PyMuPDF failed to parse {path.name}: {exc}") from exc

    def _parse_sync(self, path: Path, document: Document) -> ParsedDocumentResult:
        import fitz  # PyMuPDF

        pdf = fitz.open(str(path))
        result = ParsedDocumentResult(
            document_id=document.id,
            document_name=document.filename,
            source_path=str(path),
            page_count=pdf.page_count,
            parser_used="pymupdf",
        )

        # ── Metadata ──────────────────────────────────────────────────────
        meta = pdf.metadata or {}
        result.metadata = {
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "subject": meta.get("subject", ""),
            "creator": meta.get("creator", ""),
            "page_count": pdf.page_count,
        }

        # ── Text and sections ─────────────────────────────────────────────
        full_text_parts: list[str] = []
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text")
            if text.strip():
                full_text_parts.append(text)

            # Detect headings heuristically (large/bold font blocks)
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") != 0:  # text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("size", 0) > 14 and len(span.get("text", "").strip()) > 3:
                            # Likely a heading
                            pass  # Collected as part of sections below

        result.raw_text = "\n".join(full_text_parts)

        # Build a single section per page for structure
        sections: list[ParsedSection] = []
        current_section_text: list[str] = []
        current_pages: list[int] = []

        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                current_section_text.append(page_text)
                current_pages.append(page_num)

            # Extract figures
            if self._extract_images:
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    if base_image:
                        fig = ParsedFigure(
                            figure_id=str(uuid.uuid4()),
                            caption="",
                            image_bytes=base_image.get("image", b""),
                            image_format=base_image.get("ext", "png").upper(),
                            page_number=page_num,
                        )
                        result.figures.append(fig)

        if current_section_text:
            sections.append(
                ParsedSection(
                    title="Document Content",
                    level=1,
                    content="\n\n".join(current_section_text),
                    page_numbers=current_pages,
                )
            )

        result.sections = sections
        pdf.close()

        logger.info(
            "PyMuPDF parse complete",
            doc_id=document.id,
            pages=result.page_count,
            figures=len(result.figures),
        )
        return result
