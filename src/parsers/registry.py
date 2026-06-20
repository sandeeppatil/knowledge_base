"""Parser registry — manages parser plugin discovery and selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.domain.interfaces import DocumentParser
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


class ParserRegistry:
    """Registry that selects the best parser for a given file.

    Parsers are tried in priority order.  The first parser that declares
    ``supports(path) == True`` is used.  If it fails, the registry falls back
    to the next compatible parser.

    Example:
        >>> registry = ParserRegistry()
        >>> registry.register(DoclingPDFParser(), priority=1)
        >>> registry.register(PyMuPDFParser(), priority=2)
        >>> parser = registry.select("document.pdf")
    """

    def __init__(self) -> None:
        self._parsers: list[tuple[int, DocumentParser]] = []

    def register(self, parser: DocumentParser, priority: int = 10) -> None:
        """Register a parser at the given priority (lower = higher priority).

        Args:
            parser: DocumentParser implementation to register.
            priority: Sort key — lower values are tried first.
        """
        self._parsers.append((priority, parser))
        self._parsers.sort(key=lambda x: x[0])
        logger.debug("Parser registered", name=parser.name, priority=priority)

    def select(self, path: Path) -> DocumentParser | None:
        """Return the highest-priority parser that supports the file.

        Args:
            path: Path to the file to parse.

        Returns:
            The first compatible DocumentParser, or None if none found.
        """
        for _, parser in self._parsers:
            if parser.supports(path):
                return parser
        return None

    def list_parsers(self) -> list[str]:
        """Return list of registered parser names in priority order."""
        return [p.name for _, p in self._parsers]


def build_pdf_parser_registry(
    primary: str = "docling",
    ocr_enabled: bool = True,
    ocr_engine: str = "tesseract",
) -> ParserRegistry:
    """Build a pre-configured ParserRegistry for PDF files.

    Args:
        primary: Primary parser name ("docling" or "pymupdf").
        ocr_enabled: Whether to register the OCR fallback parser.
        ocr_engine: OCR engine for the fallback parser.

    Returns:
        Configured ParserRegistry with parsers in priority order.
    """
    registry = ParserRegistry()

    if primary == "docling":
        from src.parsers.pdf.docling_parser import DoclingPDFParser

        registry.register(DoclingPDFParser(ocr_enabled=ocr_enabled), priority=1)

        from src.parsers.pdf.pymupdf_parser import PyMuPDFParser

        registry.register(PyMuPDFParser(), priority=2)
    else:
        from src.parsers.pdf.pymupdf_parser import PyMuPDFParser

        registry.register(PyMuPDFParser(), priority=1)

    if ocr_enabled:
        from src.parsers.pdf.ocr_parser import OCRParser

        registry.register(OCRParser(engine=ocr_engine), priority=99)

    return registry
