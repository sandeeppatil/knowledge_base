"""Hierarchical chunker — the primary chunking strategy.

Produces semantic chunks that respect document structure: sections,
subsections, tables, and figures are treated as atomic units.

Design principles:
- Never split a table across multiple chunks.
- Never split a figure from its caption.
- Respect heading boundaries.
- Target 500–1000 tokens per chunk with 50–150 token overlap.
- Maintain full provenance metadata on every chunk.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from src.domain.interfaces import Chunker, ParsedDocument
from src.domain.models.chunk import Chunk, ChunkMetadata, ContentType
from src.domain.models.document import Document
from src.monitoring.logging import get_logger
from src.parsers.base import ParsedDocumentResult, ParsedFigure, ParsedSection, ParsedTable

logger = get_logger(__name__)

# Rough token estimation (characters / 4 ≈ tokens)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_with_overlap(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_size: int,
) -> list[str]:
    """Split text into overlapping windows based on approximate token count.

    Args:
        text: Input text to split.
        chunk_size: Target size in tokens.
        chunk_overlap: Overlap in tokens between consecutive chunks.
        min_size: Minimum token count to emit a chunk.

    Returns:
        List of text chunk strings.
    """
    chunk_chars = chunk_size * _CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * _CHARS_PER_TOKEN
    min_chars = min_size * _CHARS_PER_TOKEN

    if len(text) <= chunk_chars:
        return [text] if len(text) >= min_chars else []

    # Split on paragraph or sentence boundaries first
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        if current_len + para_len > chunk_chars and current:
            chunk_text = "\n\n".join(current).strip()
            if len(chunk_text) >= min_chars:
                chunks.append(chunk_text)
            # Keep overlap: retain last few paragraphs
            overlap_len = 0
            overlap_parts: list[str] = []
            for p in reversed(current):
                if overlap_len + len(p) <= overlap_chars:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_parts
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    if current:
        chunk_text = "\n\n".join(current).strip()
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)

    return chunks if chunks else [text[:chunk_chars]]


class HierarchicalChunker(Chunker):
    """Produces semantically-aware, hierarchically-structured chunks.

    Args:
        chunk_size: Target token count per text chunk.
        chunk_overlap: Overlap token count between adjacent text chunks.
        min_chunk_size: Minimum tokens to emit a chunk.
        include_tables: Emit table chunks.
        include_figures: Emit figure chunks.
    """

    def __init__(
        self,
        chunk_size: int = 750,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        include_tables: bool = True,
        include_figures: bool = True,
    ) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._include_tables = include_tables
        self._include_figures = include_figures

    async def chunk(self, parsed_doc: ParsedDocument, document: Document) -> list[Chunk]:
        """Chunk a parsed document into vector-store-ready Chunk objects.

        Args:
            parsed_doc: Output from any DocumentParser.
            document: Source Document (provides IDs, KB references).

        Returns:
            List of Chunk objects without embeddings.
        """
        if not isinstance(parsed_doc, ParsedDocumentResult):
            raise TypeError(f"Expected ParsedDocumentResult, got {type(parsed_doc)}")

        chunks: list[Chunk] = []

        # ── Text sections ───────────────────────────────────────────────
        for section in parsed_doc.sections:
            chunks.extend(self._chunk_section(section, document, parsed_doc))

        # ── Top-level tables (not nested in sections) ───────────────────
        if self._include_tables:
            section_table_ids = {
                t.table_id for s in parsed_doc.sections for t in s.tables
            }
            for table in parsed_doc.tables:
                if table.table_id not in section_table_ids:
                    chunk = self._make_table_chunk(table, document, parsed_doc, heading_path=[])
                    if chunk:
                        chunks.append(chunk)

        # ── Top-level figures ───────────────────────────────────────────
        if self._include_figures:
            section_fig_ids = {
                f.figure_id for s in parsed_doc.sections for f in s.figures
            }
            for figure in parsed_doc.figures:
                if figure.figure_id not in section_fig_ids:
                    chunk = self._make_figure_chunk(figure, document, parsed_doc, heading_path=[])
                    if chunk:
                        chunks.append(chunk)

        # Assign sequential token counts
        for c in chunks:
            c.token_count = _estimate_tokens(c.content)

        logger.debug(
            "Chunking complete",
            doc_id=document.id,
            total_chunks=len(chunks),
        )
        return chunks

    def _chunk_section(
        self,
        section: ParsedSection,
        document: Document,
        doc_result: ParsedDocumentResult,
        parent_heading: list[str] | None = None,
    ) -> list[Chunk]:
        """Recursively chunk a section and its children."""
        heading_path = (parent_heading or []) + ([section.title] if section.title else [])
        chunks: list[Chunk] = []

        # ── Chunk the section's text content ────────────────────────────
        if section.content.strip():
            text_splits = _split_with_overlap(
                section.content,
                self._chunk_size,
                self._chunk_overlap,
                self._min_chunk_size,
            )
            for i, text in enumerate(text_splits):
                chunk = self._make_text_chunk(
                    text=text,
                    section=section,
                    document=document,
                    doc_result=doc_result,
                    heading_path=heading_path,
                    part_index=i,
                )
                chunks.append(chunk)

        # ── Tables within this section ───────────────────────────────────
        if self._include_tables:
            for table in section.tables:
                chunk = self._make_table_chunk(
                    table, document, doc_result, heading_path=heading_path
                )
                if chunk:
                    chunks.append(chunk)

        # ── Figures within this section ──────────────────────────────────
        if self._include_figures:
            for figure in section.figures:
                chunk = self._make_figure_chunk(
                    figure, document, doc_result, heading_path=heading_path
                )
                if chunk:
                    chunks.append(chunk)

        # ── Recurse into child sections ──────────────────────────────────
        for child in section.children:
            chunks.extend(
                self._chunk_section(child, document, doc_result, heading_path)
            )

        return chunks

    def _make_text_chunk(
        self,
        text: str,
        section: ParsedSection,
        document: Document,
        doc_result: ParsedDocumentResult,
        heading_path: list[str],
        part_index: int = 0,
    ) -> Chunk:
        chunk_id = str(uuid.uuid4())
        # Determine content type
        content_type = ContentType.TEXT
        lower = section.title.lower()
        if "appendix" in lower:
            content_type = ContentType.APPENDIX
        elif "reference" in lower or "bibliography" in lower:
            content_type = ContentType.REFERENCE

        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document.id,
            document_name=document.name,
            knowledge_base_id=document.knowledge_base_id,
            knowledge_base_name="",  # Populated by ingestion pipeline
            page_numbers=section.page_numbers,
            section_title=section.title,
            subsection_title=None,
            content_type=content_type,
            source_path=document.source_path,
            heading_path=heading_path,
        )
        return Chunk(id=chunk_id, content=text, metadata=meta)

    def _make_table_chunk(
        self,
        table: ParsedTable,
        document: Document,
        doc_result: ParsedDocumentResult,
        heading_path: list[str],
    ) -> Chunk | None:
        content = table.to_plain_text()
        if not content.strip():
            return None

        chunk_id = str(uuid.uuid4())
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document.id,
            document_name=document.name,
            knowledge_base_id=document.knowledge_base_id,
            knowledge_base_name="",
            page_numbers=table.page_numbers,
            section_title=table.table_title or "Table",
            content_type=ContentType.TABLE,
            source_path=document.source_path,
            table_id=table.table_id,
            table_title=table.table_title,
            heading_path=heading_path,
        )
        return Chunk(
            id=chunk_id,
            content=content,
            content_markdown=table.markdown or None,
            content_json=table.source_json or None,
            metadata=meta,
        )

    def _make_figure_chunk(
        self,
        figure: ParsedFigure,
        document: Document,
        doc_result: ParsedDocumentResult,
        heading_path: list[str],
    ) -> Chunk | None:
        content = figure.to_plain_text()
        if not content.strip():
            return None

        chunk_id = str(uuid.uuid4())
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document.id,
            document_name=document.name,
            knowledge_base_id=document.knowledge_base_id,
            knowledge_base_name="",
            page_numbers=[figure.page_number] if figure.page_number else [],
            section_title=figure.caption or "Figure",
            content_type=ContentType.FIGURE,
            source_path=document.source_path,
            figure_id=figure.figure_id,
            figure_caption=figure.caption,
            heading_path=heading_path,
        )
        return Chunk(id=chunk_id, content=content, metadata=meta)
