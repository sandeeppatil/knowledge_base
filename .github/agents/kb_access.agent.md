---
name: kb_access
description: Access Knowledge base - AUTOSAR SWS COM documentation
argument-hint: Query or search term for AUTOSAR SWS COM knowledge base
tools: [vscode, execute, read, agent, edit, search, web, browser, 'knowledge-base/*', todo]
---

# KB Access Agent

This agent provides access to the AUTOSAR Software-Services Communication (SWS COM) knowledge base.

## Knowledge Base Information

- **KB Name**: AUTOSAR_SWS_COM
- **KB ID**: Created via `/knowledge-bases` endpoint
- **Collection**: `kb_*` collection in Qdrant vector store
- **Status**: Active and ready for document retrieval

## Document Ingestion

The AUTOSAR SWS COM PDF documentation has been ingested into the knowledge base. The ingestion pipeline:
1. Parses the PDF using Docling (with PyMuPDF and OCR fallbacks)
2. Chunks content hierarchically, preserving structure
3. Embeds chunks using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional vectors)
4. Stores in Qdrant with hybrid retrieval (dense + BM25 + RRF fusion + cross-encoder reranking)

## How to Query

### Via REST API
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "kb_name": "AUTOSAR_SWS_COM",
    "query": "What is the CAN communication standard?",
    "top_k": 10
  }'
```

### Via MCP Tools (GitHub Copilot Agent Mode)

1. Call `list_knowledge_bases()` to see all available KBs
2. Call `retrieve_from_kb(kb_name="AUTOSAR_SWS_COM", query="...", top_k=10)` to get results

## Retrieval Guarantees

✅ **Grounding Contract**: Only returns chunks with evidence from the AUTOSAR documentation
- Returns `answer_found: false` when no supporting evidence exists
- Never fabricates citations or unsupported content
- Every chunk includes: `source_document`, `page_numbers`, `section_title`, `heading_path`

## Common Queries

- Communication standards and protocols
- CAN/CAN FD specifications  
- Software architecture concepts
- Timing and configuration details
- Specification requirements

**Note**: This agent uses zero-shot grounding - all answers are sourced from the ingested AUTOSAR documentation.