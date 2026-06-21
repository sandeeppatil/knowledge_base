#!/usr/bin/env python3
"""
Script to set up 2 knowledge bases with document ingestion.
Runs steps 1 and 2:
  Step 1: Create space_autosar and space_coding_guidelines KBs
  Step 2: Ingest PDFs into both KBs
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from src.api.dependencies import Container
from src.config.settings import settings
from src.monitoring.logging import get_logger

logger = get_logger(__name__)


async def main() -> int:
    """Create KBs and ingest documents."""
    container = Container(settings)
    
    try:
        # Step 1: Create or get space_autosar KB
        logger.info("Setting up space_autosar knowledge base...")
        existing_kb1 = await container.kb_repository.get_kb_by_name("space_autosar")
        if existing_kb1:
            kb1 = existing_kb1
            logger.info(f"✓ Using existing space_autosar KB (id={kb1.id})")
        else:
            kb1 = await container.kb_service.create(
                name="space_autosar",
                description="AUTOSAR SWS COM standard — communication stack specifications.",
            )
            logger.info(f"✓ Created space_autosar KB with id={kb1.id}")
        
        # Step 1: Create or get space_coding_guidelines KB
        logger.info("Setting up space_coding_guidelines knowledge base...")
        existing_kb2 = await container.kb_repository.get_kb_by_name("space_coding_guidelines")
        if existing_kb2:
            kb2 = existing_kb2
            logger.info(f"✓ Using existing space_coding_guidelines KB (id={kb2.id})")
        else:
            kb2 = await container.kb_service.create(
                name="space_coding_guidelines",
                description="C++ Coding Standards — language usage rules and best practices.",
            )
            logger.info(f"✓ Created space_coding_guidelines KB with id={kb2.id}")
        
        # Step 2: Ingest AUTOSAR PDF
        pdf1_path = Path("/home/sandeep/workspace/kb/data/knowledge_bases/space_autosar/AUTOSAR_SWS_COM.pdf")
        logger.info(f"PDF1 path: {pdf1_path}")
        logger.info(f"PDF1 exists: {pdf1_path.exists()}")
        if pdf1_path.exists():
            logger.info(f"Ingesting {pdf1_path.name}...")
            try:
                doc1 = await container.ingestion_service.ingest_file(pdf1_path, kb1.id)
                logger.info(f"✓ Ingested {pdf1_path.name} (doc_id={doc1.id}, chunks={doc1.chunk_count})")
            except Exception as e:
                logger.error(f"Error ingesting PDF1: {type(e).__name__}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"File not found: {pdf1_path}")
        
        # Step 2: Ingest C++ PDF
        pdf2_path = Path("/home/sandeep/workspace/kb/data/knowledge_bases/space_coding_guidelines/C++ Coding Standards.pdf")
        logger.info(f"PDF2 path: {pdf2_path}")
        logger.info(f"PDF2 exists: {pdf2_path.exists()}")
        if pdf2_path.exists():
            logger.info(f"Ingesting {pdf2_path.name}...")
            try:
                doc2 = await container.ingestion_service.ingest_file(pdf2_path, kb2.id)
                logger.info(f"✓ Ingested {pdf2_path.name} (doc_id={doc2.id}, chunks={doc2.chunk_count})")
            except Exception as e:
                logger.error(f"Error ingesting PDF2: {type(e).__name__}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"File not found: {pdf2_path}")
        
        # Verify
        logger.info("Verifying setup...")
        kbs = await container.kb_repository.list_kbs()
        logger.info(f"✓ Total KBs: {len(kbs)}")
        for kb in kbs:
            logger.info(f"  - {kb.name}: {kb.document_count} documents, {kb.chunk_count} chunks")
        
        logger.info("✓ Setup complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
