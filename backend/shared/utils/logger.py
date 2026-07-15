"""
Shared logger factory.

Usage:
    from backend.shared.utils.logger import get_logger
    logger = get_logger(__name__)
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

_CONFIGURED: set[str] = set()
_ROOT_CONFIGURED = False


def _configure_root() -> None:
    global _ROOT_CONFIGURED
    if _ROOT_CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    use_colors = os.getenv("LOG_COLORS", "true").lower() == "true"
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    if use_colors:
        try:
            import colorlog  # type: ignore

            formatter = colorlog.ColoredFormatter(
                "%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )
        except ImportError:
            formatter = logging.Formatter(fmt, datefmt=datefmt)
    else:
        formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Optional file handler
    if os.getenv("LOG_FILE_ENABLED", "false").lower() == "true":
        log_dir = os.getenv("LOG_DIR", "/tmp/thea_logs")
        os.makedirs(log_dir, exist_ok=True)
        max_bytes = int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
        backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
        fh = RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)

    _ROOT_CONFIGURED = True


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger, configuring the root logger on first call.

    Args:
        name:  Typically __name__ of the calling module.
        level: Optional override log level for this specific logger.
    """
    _configure_root()
    logger = logging.getLogger(name)
    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
