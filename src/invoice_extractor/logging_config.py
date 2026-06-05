"""Logging setup helper, shared by the CLI and the Streamlit app."""

from __future__ import annotations

import logging

_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once with a concise, timestamped format.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        from invoice_extractor.config import settings

        level = settings.log_level

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
