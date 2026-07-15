"""Idempotency helpers for production phase workflows."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional, Tuple

from app.config import get_idempotency_keys_collection
from app.services.idempotency_service import (
    IdempotencyConflictError,
    IdempotencyService,
)

logger = logging.getLogger(__name__)


def get_idempotency_service() -> IdempotencyService:
    """Get idempotency service instance."""
    _, collection = get_idempotency_keys_collection()
    return IdempotencyService(collection)


def generate_idempotency_key(
    user_id: Optional[str] = None,
    scene_id: Optional[str] = None,
    phase_number: int = 1,
    idempotency_key_header: Optional[str] = None,
) -> str:
    """
    Generate an idempotency key for a phase workflow.
    
    Priority:
    1. Use idempotency_key_header if provided (from frontend)
    2. Generate fallback key using user_id + scene_id + phase_number
    
    Args:
        user_id: User identifier (for fallback key generation)
        scene_id: Scene/project identifier (for fallback key generation)
        phase_number: Phase number (1, 2, or 3)
        idempotency_key_header: Optional key from Idempotency-Key header
        
    Returns:
        Idempotency key string
    """
    if idempotency_key_header:
        return idempotency_key_header.strip()
    
    # Generate fallback key
    if not scene_id:
        # If we don't have scene_id, use a hash of available params
        key_parts = [f"phase{phase_number}"]
        if user_id:
            key_parts.append(user_id)
        key_string = ":".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]
    
    # Standard fallback: user_id:scene_id:phase_number (or scene_id:phase_number if no user_id)
    if user_id:
        return f"{user_id}:{scene_id}:phase{phase_number}"
    else:
        return f"{scene_id}:phase{phase_number}"


def check_idempotency(
    endpoint: str,
    idempotency_key: str,
    payload: Any,
    service: Optional[IdempotencyService] = None,
    ttl_minutes: int = 30,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Check if an idempotency key exists and return cached result if available.

    Args:
        endpoint: Endpoint identifier (e.g., "phase1.start")
        idempotency_key: Idempotency key
        payload: Request payload (for hash validation)
        service: Optional idempotency service instance
        ttl_minutes: TTL for idempotency key in minutes (default: 30 minutes)
                     Prevents permanent locks from stuck jobs

    Returns:
        Tuple of (is_duplicate, cached_response)
        - is_duplicate: True if this is a duplicate request
        - cached_response: Cached response if duplicate, None otherwise
    """
    if service is None:
        service = get_idempotency_service()

    from datetime import timedelta

    try:
        record = service.reserve(
            endpoint=endpoint,
            key=idempotency_key,
            payload=payload,
            ttl=timedelta(minutes=ttl_minutes),
        )
        
        # If record is completed, return cached response
        if record.is_completed:
            logger.info(
                f"Idempotency key {idempotency_key} already completed. "
                f"Returning cached response."
            )
            return True, record.response_payload
        
        # If record is failed, allow retry (don't treat as duplicate)
        if record.is_failed:
            logger.info(
                f"Idempotency key {idempotency_key} previously failed. "
                f"Allowing retry."
            )
            return False, None
        
        # If record is processing, this is a duplicate request
        if record.is_processing:
            logger.warning(
                f"Idempotency key {idempotency_key} is already processing. "
                f"This is a duplicate request."
            )
            # Return None for cached response - caller should handle this case
            return True, None
        
        # New record created - not a duplicate
        return False, None
        
    except IdempotencyConflictError as e:
        logger.error(f"Idempotency conflict for key {idempotency_key}: {e}")
        # Conflict means different payload with same key - treat as error
        raise


def mark_idempotency_completed(
    endpoint: str,
    idempotency_key: str,
    workflow_id: Optional[str],
    task_id: Optional[str],
    response_payload: Any,
    service: Optional[IdempotencyService] = None,
) -> None:
    """Mark an idempotency record as completed."""
    if service is None:
        service = get_idempotency_service()
    
    try:
        service.mark_completed(
            endpoint=endpoint,
            key=idempotency_key,
            workflow_id=workflow_id,
            task_id=task_id,
            response_payload=response_payload,
        )
        logger.info(f"Marked idempotency key {idempotency_key} as completed")
    except Exception as e:
        logger.error(f"Failed to mark idempotency as completed: {e}")


def mark_idempotency_failed(
    endpoint: str,
    idempotency_key: str,
    error_message: str,
    service: Optional[IdempotencyService] = None,
) -> None:
    """Mark an idempotency record as failed."""
    if service is None:
        service = get_idempotency_service()
    
    try:
        service.mark_failed(
            endpoint=endpoint,
            key=idempotency_key,
            error_message=error_message,
        )
        logger.info(f"Marked idempotency key {idempotency_key} as failed")
    except Exception as e:
        logger.error(f"Failed to mark idempotency as failed: {e}")

