"""Base classes and shared data structures for document parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ParsedTable:
    """A table extracted from a document.

    Attributes:
        table_id: Unique identifier within the document.
        table_title: Caption / title found near the table.
        headers: List of column header strings.
        rows: List of row lists (each row is a list of cell values).
        markdown: Markdown representation of the table.
        source_json: Structured JSON representation.
        page_numbers: Pages this table spans.
        is_multipage: True if the table spans multiple pages.
    """

    table_id: str
    table_title: str = ""
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    markdown: str = ""
    source_json: dict[str, Any] = field(default_factory=dict)
    page_numbers: list[int] = field(default_factory=list)
    is_multipage: bool = False

    def to_plain_text(self) -> str:
        """Generate plain-text representation for embedding."""
        lines: list[str] = []
        if self.table_title:
            lines.append(f"Table: {self.table_title}")
        if self.headers:
            lines.append(" | ".join(self.headers))
        for row in self.rows:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)


@dataclass
class ParsedFigure:
    """A figure, chart, or diagram extracted from a document.

    Attributes:
        figure_id: Unique identifier within the document.
        caption: Caption found near the figure.
        description: AI-generated description from a VLM.
        image_bytes: Raw image bytes.
        image_format: Format string (e.g. 'PNG', 'JPEG').
        page_number: Page number where the figure appears.
    """

    figure_id: str
    caption: str = ""
    description: str = ""
    image_bytes: bytes = b""
    image_format: str = "PNG"
    page_number: int = 0

    def to_plain_text(self) -> str:
        """Generate plain-text for embedding (description > caption > id)."""
        parts = []
        if self.caption:
            parts.append(f"Figure caption: {self.caption}")
        if self.description:
            parts.append(f"Figure description: {self.description}")
        if not parts:
            parts.append(f"Figure {self.figure_id}")
        return "\n".join(parts)


@dataclass
class ParsedSection:
    """A hierarchical section of text within a document."""

    title: str
    level: int  # 1 = H1, 2 = H2, etc.
    content: str
    page_numbers: list[int] = field(default_factory=list)
    children: list[ParsedSection] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)


@dataclass
class ParsedDocumentResult:
    """The complete structured output produced by a DocumentParser.

    This is the canonical intermediate representation passed from parsers to
    chunkers.  Every parser must produce a ParsedDocumentResult regardless of
    the underlying library used.
    """

    document_id: str
    document_name: str
    source_path: str
    page_count: int = 0
    sections: list[ParsedSection] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    figures: list[ParsedFigure] = field(default_factory=list)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    parser_used: str = ""

    @property
    def pages(self) -> list[Any]:  # satisfies ParsedDocument interface
        return list(range(self.page_count))
