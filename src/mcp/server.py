"""MCP server — exposes KB tools via the Model Context Protocol.

Run with:
    python -m src.mcp.server

Or via the CLI entry point:
    kb-mcp
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.api.dependencies import Container
from src.config.settings import settings
from src.monitoring.logging import configure_logging, get_logger
from src.tools.tools import KBTools, RetrievalTools

logger = get_logger(__name__)

# ─── Tool schemas ─────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[types.Tool] = [
    types.Tool(
        name="list_knowledge_bases",
        description=(
            "List all available knowledge bases with their names and descriptions. "
            "Call this FIRST to understand which KB to query. "
            "The description field helps you select the right KB."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="retrieve_from_kb",
        description=(
            "Retrieve grounded information from a specific knowledge base. "
            "Returns answer_found=false when no evidence exists — NEVER fabricate. "
            "Always call list_knowledge_bases first to get the correct kb_name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "kb_name": {
                    "type": "string",
                    "description": "Name of the knowledge base to search.",
                },
                "query": {
                    "type": "string",
                    "description": "The user question or search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–50).",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["kb_name", "query"],
        },
    ),
    types.Tool(
        name="search_knowledge_bases",
        description=(
            "Search across ALL knowledge bases simultaneously. "
            "Use when unsure which KB contains the answer. "
            "Returns results per KB — check answer_found for each."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Max results per KB.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="list_documents",
        description="List documents in a specific knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {
                "kb_id": {"type": "string", "description": "Knowledge base ID."}
            },
            "required": ["kb_id"],
        },
    ),
    types.Tool(
        name="create_knowledge_base",
        description="Create a new knowledge base.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Unique name for the KB."},
                "description": {
                    "type": "string",
                    "description": "Natural-language description for routing.",
                },
            },
            "required": ["name", "description"],
        },
    ),
]


# ─── MCP Server ───────────────────────────────────────────────────────────────


class KBMCPServer:
    """MCP server for the Knowledge Base platform.

    Args:
        container: Application IoC container.
    """

    def __init__(self, container: Container) -> None:
        self._server = Server("knowledge-base")
        self._kb_tools = KBTools(
            kb_service=container.kb_service,
            ingestion_service=container.ingestion_service,
        )
        self._retrieval_tools = RetrievalTools(
            retrieval_service=container.retrieval_service,
            kb_service=container.kb_service,
        )
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register MCP protocol handlers."""

        @self._server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return TOOL_SCHEMAS

        @self._server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[types.TextContent]:
            try:
                result = await self._dispatch(name, arguments)
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as exc:
                logger.error("Tool call failed", tool=name, error=str(exc))
                error_result = {
                    "error": str(exc),
                    "tool": name,
                    "answer_found": False,
                }
                return [
                    types.TextContent(type="text", text=json.dumps(error_result, indent=2))
                ]

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> Any:
        """Dispatch a tool call to the appropriate handler."""
        if name == "list_knowledge_bases":
            return await self._kb_tools.list_knowledge_bases()

        if name == "retrieve_from_kb":
            return await self._retrieval_tools.retrieve_from_kb(
                kb_name=arguments["kb_name"],
                query=arguments["query"],
                top_k=arguments.get("top_k", 10),
            )

        if name == "search_knowledge_bases":
            return await self._retrieval_tools.search_knowledge_bases(
                query=arguments["query"],
                top_k=arguments.get("top_k", 5),
            )

        if name == "list_documents":
            return await self._kb_tools.list_documents(kb_id=arguments["kb_id"])

        if name == "create_knowledge_base":
            return await self._kb_tools.create_knowledge_base(
                name=arguments["name"],
                description=arguments["description"],
            )

        raise ValueError(f"Unknown tool: {name}")

    async def run(self) -> None:
        """Start the MCP server on stdio."""
        logger.info("Starting MCP server")
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


def main() -> None:
    """Entry point for the MCP server."""
    configure_logging(
        log_level=settings.observability.log_level,
        structured=False,  # MCP uses stdio; keep logs on stderr
    )
    settings.ensure_dirs()
    container = Container(settings)

    async def run() -> None:
        await container.initialise()
        server = KBMCPServer(container)
        await server.run()

    asyncio.run(run())


if __name__ == "__main__":
    main()
