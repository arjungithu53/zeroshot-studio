"""
Shared API exception handler for FastAPI endpoints.

Usage:
    from backend.shared.utils.error_handlers import handle_api_exception
    except Exception as exc:
        return handle_api_exception(exc, logger)
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Optional

from fastapi.responses import JSONResponse


def handle_api_exception(
    exc: Exception,
    logger: Optional[logging.Logger] = None,
    status_code: int = 500,
    context: Optional[str] = None,
) -> JSONResponse:
    """
    Log an exception and return a standard JSON error response.

    Args:
        exc:         The caught exception.
        logger:      Logger to use; falls back to module-level logger.
        status_code: HTTP status code (default 500).
        context:     Optional string describing where the error occurred.

    Returns:
        JSONResponse with error detail.
    """
    _log = logger or logging.getLogger(__name__)
    prefix = f"[{context}] " if context else ""
    _log.error("%sUnhandled exception: %s", prefix, exc, exc_info=True)

    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": str(exc),
            "detail": traceback.format_exc() if logging.getLogger().level <= logging.DEBUG else None,
        },
    )
