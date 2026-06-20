"""Docling PDF parser — Tier 1 parser for structured document understanding.

Docling provides best-in-class structure preservation including tables,
figures, headings, sections, and multi-page table reconstruction.
"""

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


class DoclingPDFParser(DocumentParser):
    """Primary PDF parser using Docling for full structural understanding.

    Handles:
    - Hierarchical sections and headings
    - Complex tables (nested, merged cells, multi-page)
    - Figures and captions
    - References, appendices, footnotes
    - Research papers, standards, technical documents

    Args:
        ocr_enabled: Enable OCR for scanned pages.
        extract_figures: Extract figure images.
    """

    def __init__(
        self,
        ocr_enabled: bool = True,
        extract_figures: bool = True,
    ) -> None:
        self._ocr_enabled = ocr_enabled
        self._extract_figures = extract_figures
        self._converter: Any = None

    @property
    def name(self) -> str:
        return "docling"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def _get_converter(self) -> Any:
        """Lazily initialise the Docling DocumentConverter."""
        if self._converter is None:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions

            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_ocr = self._ocr_enabled
            pipeline_opts.do_table_structure = True
            pipeline_opts.table_structure_options.do_cell_matching = True

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
                }
            )
        return self._converter

    async def parse(self, path: Path, document: Document) -> ParsedDocument:
        """Parse a PDF using Docling.

        Args:
            path: Absolute path to the PDF file.
            document: Domain Document (provides IDs, metadata).

        Returns:
            ParsedDocumentResult with full structural information.

        Raises:
            RuntimeError: If Docling fails to convert the document.
        """
        import asyncio

        logger.info("Docling parsing PDF", path=str(path), doc_id=document.id)

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._parse_sync, path, document
            )
            return result
        except Exception as exc:
            logger.error("Docling parse failed", path=str(path), error=str(exc))
            raise RuntimeError(f"Docling failed to parse {path.name}: {exc}") from exc

    def _parse_sync(self, path: Path, document: Document) -> ParsedDocumentResult:
        """Synchronous Docling parsing — runs in a thread executor."""
        converter = self._get_converter()
        conv_result = converter.convert(str(path))
        doc = conv_result.document

        result = ParsedDocumentResult(
            document_id=document.id,
            document_name=document.name,
            source_path=str(path),
            parser_used="docling",
        )

        # ── Extract metadata ──────────────────────────────────────────────
        result.metadata = {
            "title": getattr(doc, "description", {}).get("title", "") if hasattr(doc, "description") else "",
            "docling_version": "2.x",
        }

        # ── Export to markdown for section/text extraction ────────────────
        try:
            markdown_text = doc.export_to_markdown()
        except Exception:
            markdown_text = ""

        result.raw_text = markdown_text

        # ── Extract page count ────────────────────────────────────────────
        try:
            result.page_count = len(doc.pages) if hasattr(doc, "pages") else 0
        except Exception:
            result.page_count = 0

        # ── Extract tables ────────────────────────────────────────────────
        result.tables = self._extract_tables(doc, document)

        # ── Extract figures ───────────────────────────────────────────────
        if self._extract_figures:
            result.figures = self._extract_figures_from_doc(doc, document)

        # ── Extract sections from docling items ───────────────────────────
        result.sections = self._extract_sections(doc)

        logger.info(
            "Docling parse complete",
            doc_id=document.id,
            pages=result.page_count,
            tables=len(result.tables),
            figures=len(result.figures),
            sections=len(result.sections),
        )
        return result

    def _extract_tables(self, doc: Any, document: Document) -> list[ParsedTable]:
        """Extract all tables from the Docling document."""
        tables: list[ParsedTable] = []
        try:
            for item in doc.iterate_items():
                from docling.datamodel.base_models import TableItem
                if not isinstance(item[0], TableItem):
                    continue
                table_item = item[0]
                table_id = str(uuid.uuid4())

                # Get table title from nearest heading/caption
                caption = ""
                if hasattr(table_item, "caption_text"):
                    caption = table_item.caption_text(doc) or ""

                # Build markdown and JSON representations
                markdown = ""
                headers: list[str] = []
                rows: list[list[str]] = []
                page_numbers: list[int] = []

                if hasattr(table_item, "export_to_dataframe"):
                    try:
                        df = table_item.export_to_dataframe()
                        if df is not None and not df.empty:
                            headers = [str(c) for c in df.columns.tolist()]
                            rows = [[str(v) for v in row] for row in df.values.tolist()]
                            markdown = df.to_markdown(index=False)
                    except Exception:
                        pass

                if hasattr(table_item, "prov") and table_item.prov:
                    page_numbers = list({p.page_no for p in table_item.prov})

                table = ParsedTable(
                    table_id=table_id,
                    table_title=caption,
                    headers=headers,
                    rows=rows,
                    markdown=markdown,
                    source_json={"headers": headers, "rows": rows},
                    page_numbers=page_numbers,
                    is_multipage=len(page_numbers) > 1,
                )
                tables.append(table)
        except Exception as exc:
            logger.warning("Table extraction partial failure", error=str(exc))
        return tables

    def _extract_figures_from_doc(self, doc: Any, document: Document) -> list[ParsedFigure]:
        """Extract figures from the Docling document."""
        figures: list[ParsedFigure] = []
        try:
            for item in doc.iterate_items():
                from docling.datamodel.base_models import PictureItem
                if not isinstance(item[0], PictureItem):
                    continue
                fig_item = item[0]
                figure_id = str(uuid.uuid4())

                caption = ""
                if hasattr(fig_item, "caption_text"):
                    caption = fig_item.caption_text(doc) or ""

                page_no = 0
                if hasattr(fig_item, "prov") and fig_item.prov:
                    page_no = fig_item.prov[0].page_no

                figure = ParsedFigure(
                    figure_id=figure_id,
                    caption=caption,
                    page_number=page_no,
                )
                figures.append(figure)
        except Exception as exc:
            logger.warning("Figure extraction partial failure", error=str(exc))
        return figures

    def _extract_sections(self, doc: Any) -> list[ParsedSection]:
        """Extract hierarchical sections from the Docling document."""
        sections: list[ParsedSection] = []
        current_heading = "Document"
        current_level = 1
        current_content_lines: list[str] = []
        current_pages: set[int] = set()

        try:
            for item, _ in doc.iterate_items():
                from docling.datamodel.base_models import SectionHeaderItem, TextItem

                if isinstance(item, SectionHeaderItem):
                    # Save previous section
                    if current_content_lines:
                        sections.append(
                            ParsedSection(
                                title=current_heading,
                                level=current_level,
                                content="\n".join(current_content_lines).strip(),
                                page_numbers=sorted(current_pages),
                            )
                        )
                    current_heading = item.text
                    current_level = getattr(item, "level", 1)
                    current_content_lines = []
                    current_pages = set()

                elif isinstance(item, TextItem):
                    current_content_lines.append(item.text)
                    if hasattr(item, "prov") and item.prov:
                        for p in item.prov:
                            current_pages.add(p.page_no)

            # Flush last section
            if current_content_lines:
                sections.append(
                    ParsedSection(
                        title=current_heading,
                        level=current_level,
                        content="\n".join(current_content_lines).strip(),
                        page_numbers=sorted(current_pages),
                    )
                )
        except Exception as exc:
            logger.warning("Section extraction partial failure", error=str(exc))

        return sections
