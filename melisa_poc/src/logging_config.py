"""Logging setup for CLI and scripts."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def configure_logging(level: str = "WARNING", *, structured: bool = False) -> None:
    """Configure root logging for the POC.

    When ``structured=True``, emit key=value fields suitable for log aggregation.
    """
    numeric = getattr(logging, level.upper(), logging.WARNING)
    if structured:
        fmt = "%(levelname)s %(name)s event=%(message)s"
    else:
        fmt = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=numeric, format=fmt, stream=sys.stderr, force=True)


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Emit a single structured log line without external dependencies."""
    if not fields:
        logger.info(event)
        return
    parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logger.info("%s %s", event, parts)
