"""OCR fallback parser for scanned PDFs.

Uses Tesseract (via pytesseract) to extract text from scanned pages.
PaddleOCR is supported as an alternative engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.domain.interfaces import DocumentParser, ParsedDocument
from src.domain.models.document import Document
from src.monitoring.logging import get_logger
from src.parsers.base import ParsedDocumentResult, ParsedSection

logger = get_logger(__name__)


class OCRParser(DocumentParser):
    """Tier-3 OCR fallback parser for scanned or image-heavy PDFs.

    Args:
        engine: "tesseract" or "paddle".
        language: OCR language string (Tesseract format, e.g. "eng").
        dpi: Rendering DPI for page-to-image conversion.
    """

    def __init__(
        self,
        engine: str = "tesseract",
        language: str = "eng",
        dpi: int = 300,
    ) -> None:
        self._engine = engine
        self._language = language
        self._dpi = dpi

    @property
    def name(self) -> str:
        return f"ocr_{self._engine}"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    async def parse(self, path: Path, document: Document) -> ParsedDocument:
        import asyncio

        logger.info("OCR parsing PDF", engine=self._engine, path=str(path))
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._parse_sync, path, document
            )
        except Exception as exc:
            logger.error("OCR parse failed", path=str(path), error=str(exc))
            raise RuntimeError(f"OCR failed to parse {path.name}: {exc}") from exc

    def _parse_sync(self, path: Path, document: Document) -> ParsedDocumentResult:
        import fitz  # PyMuPDF — used for page→image conversion

        pdf = fitz.open(str(path))
        result = ParsedDocumentResult(
            document_id=document.id,
            document_name=document.name,
            source_path=str(path),
            page_count=pdf.page_count,
            parser_used=self.name,
        )

        all_text_parts: list[str] = []
        for page_num, page in enumerate(pdf, start=1):
            page_text = self._ocr_page(page, page_num)
            if page_text.strip():
                all_text_parts.append(page_text)

        result.raw_text = "\n\n".join(all_text_parts)
        result.sections = [
            ParsedSection(
                title="OCR Extracted Content",
                level=1,
                content=result.raw_text,
                page_numbers=list(range(1, pdf.page_count + 1)),
            )
        ]
        pdf.close()

        logger.info(
            "OCR parse complete",
            doc_id=document.id,
            pages=result.page_count,
            chars=len(result.raw_text),
        )
        return result

    def _ocr_page(self, page: Any, page_num: int) -> str:
        """OCR a single PDF page, returning plain text."""
        import fitz

        # Render page to PIL Image
        mat = fitz.Matrix(self._dpi / 72, self._dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")

        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_bytes))

        if self._engine == "tesseract":
            return self._tesseract_ocr(img)
        elif self._engine == "paddle":
            return self._paddle_ocr(img_bytes)
        else:
            raise ValueError(f"Unknown OCR engine: {self._engine}")

    def _tesseract_ocr(self, image: Any) -> str:
        import pytesseract

        return pytesseract.image_to_string(image, lang=self._language)

    def _paddle_ocr(self, image_bytes: bytes) -> str:
        from paddleocr import PaddleOCR

        if not hasattr(self, "_paddle"):
            self._paddle = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        import numpy as np
        from PIL import Image
        import io

        img = np.array(Image.open(io.BytesIO(image_bytes)))
        result = self._paddle.ocr(img, cls=True)
        lines = []
        if result:
            for line_group in result:
                if line_group:
                    for line in line_group:
                        if line and len(line) > 1:
                            lines.append(line[1][0])
        return "\n".join(lines)
