"""Structured logging configuration using Loguru.

Usage:
    from src.monitoring.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing document", document_id="abc", kb="ISO Standards")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


def configure_logging(
    log_level: str = "INFO",
    structured: bool = True,
    log_dir: str | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        log_level: Minimum log level (DEBUG | INFO | WARNING | ERROR | CRITICAL).
        structured: When True, emit JSON-structured logs (for production).
        log_dir: Optional directory for log file output.
    """
    logger.remove()  # Remove default handler

    if structured:
        fmt = (
            "{time:YYYY-MM-DDTHH:mm:ss.SSSZ} | {level} | {name}:{function}:{line} | "
            "{message} | {extra}"
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stdout,
        format=fmt,
        level=log_level,
        colorize=not structured,
        serialize=structured,
        enqueue=True,
    )

    if log_dir:
        log_path = Path(log_dir) / "app_{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_path),
            format=fmt,
            level=log_level,
            rotation="00:00",
            retention="30 days",
            compression="gz",
            serialize=structured,
            enqueue=True,
        )


def get_logger(name: str) -> Any:
    """Return a loguru logger bound with the given module name.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A loguru logger instance with ``name`` bound to all records.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("chunk ingested", chunk_id="abc", kb="Research")
    """
    return logger.bind(module=name)
