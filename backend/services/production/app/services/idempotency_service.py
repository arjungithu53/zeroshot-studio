"""Utilities for idempotent workflow execution in production."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Dict

from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

logger = logging.getLogger(__name__)


class IdempotencyConflictError(Exception):
    """Raised when the same key is reused with a different payload."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_payload(payload: Any) -> str:
    """Create a deterministic hash for the request payload."""
    normalized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,  # fallback for non-JSON-serializable (e.g., ObjectId)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class IdempotencyRecord:
    """In-memory representation of an idempotency record."""

    key: str
    endpoint: str
    status: str
    request_hash: str
    workflow_id: Optional[str]
    task_id: Optional[str]
    response_payload: Optional[Any]
    project_id: Optional[str]
    movie_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]

    raw: dict[str, Any]

    @property
    def is_processing(self) -> bool:
        return self.status == "processing"

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"


class IdempotencyService:
    """Service that stores and retrieves idempotency state for production phases."""

    def __init__(self, collection: Collection):
        self.collection = collection
        # Ensure TTL index exists for automatic cleanup of expired records
        self._ensure_ttl_index()

    def reserve(
        self,
        *,
        endpoint: str,
        key: str,
        payload: Any,
        ttl: Optional[timedelta] = timedelta(days=7),
    ) -> IdempotencyRecord:
        """Reserve an idempotency key if available, otherwise return existing.
        
        This method uses atomic operations to prevent race conditions:
        1. Attempts to insert a new record with status "processing"
        2. If record exists (DuplicateKeyError), atomically retrieves it
        3. Checks if existing record has same payload hash
        4. Returns existing record regardless of status (completed/processing/failed)
        """

        request_hash = _hash_payload(payload)
        now = _utc_now()
        expires_at = now + ttl if ttl else None

        # First, check if a record already exists (atomic read)
        existing = self.collection.find_one({"endpoint": endpoint, "key": key})
        
        if existing:
            # Record exists - validate payload hash matches
            existing_hash = existing.get("request_hash")
            if existing_hash != request_hash:
                # Log the payload mismatch for debugging
                logger.error(
                    f"Idempotency conflict - key: {key}, endpoint: {endpoint}\n"
                    f"Existing hash: {existing_hash}\n"
                    f"New hash: {request_hash}\n"
                    f"New payload: {payload}"
                )
                raise IdempotencyConflictError(
                    "Idempotency key reused with different payload."
                )
            # Return existing record (could be processing, completed, or failed)
            return self._to_record(existing)
        
        # Record doesn't exist - try to atomically insert with "processing" status
        doc = {
            "key": key,
            "endpoint": endpoint,
            "request_hash": request_hash,
            "status": "processing",
            "workflow_id": None,
            "task_id": None,
            "response_payload": None,
            "created_at": now,
            "updated_at": now,
            "expires_at": expires_at,
        }

        try:
            self.collection.insert_one(doc)
            return self._to_record(doc)
        except DuplicateKeyError:
            # Race condition: another request inserted between find_one and insert_one
            # Atomically retrieve the record that was just inserted
            stored = self.collection.find_one({"endpoint": endpoint, "key": key})
            
            if not stored:
                # Extremely unlikely: record was inserted then deleted
                # Try one more time to insert
                try:
                    self.collection.insert_one(doc)
                    return self._to_record(doc)
                except DuplicateKeyError:
                    # Another request got it - retrieve again
                    stored = self.collection.find_one({"endpoint": endpoint, "key": key})
                    if not stored:
                        raise RuntimeError(
                            f"Unable to reserve idempotency key {endpoint}:{key} "
                            "due to concurrent access"
                        )
            
            # Validate the concurrently inserted record has same payload
            if stored.get("request_hash") != request_hash:
                raise IdempotencyConflictError(
                    "Idempotency key reused with different payload."
                )
            
            return self._to_record(stored)

    def mark_completed(
        self,
        *,
        endpoint: str,
        key: str,
        workflow_id: Optional[str],
        task_id: Optional[str],
        response_payload: Any,
    ) -> IdempotencyRecord:
        """Mark an idempotency record as completed.
        
        Only updates if record is in "processing" state to prevent race conditions
        where multiple concurrent operations try to update the same record.
        """
        now = _utc_now()
        # Only update if status is "processing" - prevents overwriting terminal states
        stored = self.collection.find_one_and_update(
            {
                "endpoint": endpoint,
                "key": key,
                "status": "processing"  # Only update if still processing
            },
            {
                "$set": {
                    "status": "completed",
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "response_payload": response_payload,
                    "updated_at": now,
                }
            },
            return_document=True,
        )
        if stored is None:
            # Check if record exists but is in a different state
            existing = self.collection.find_one({"endpoint": endpoint, "key": key})
            if existing:
                # Record exists but already in terminal state
                record = self._to_record(existing)
                if record.is_completed:
                    # Already completed - return existing record
                    return record
                elif record.is_failed:
                    # Already failed - don't overwrite
                    raise RuntimeError(
                        f"Idempotency record {endpoint}:{key} is already marked as failed"
                    )
            raise KeyError(f"Idempotency record not found for {endpoint}:{key}")
        return self._to_record(stored)

    def mark_failed(
        self,
        *,
        endpoint: str,
        key: str,
        error_message: str,
    ) -> IdempotencyRecord:
        """Mark an idempotency record as failed.
        
        Only updates if record is in "processing" state to prevent race conditions
        where multiple concurrent operations try to update the same record.
        """
        now = _utc_now()
        # Only update if status is "processing" - prevents overwriting terminal states
        stored = self.collection.find_one_and_update(
            {
                "endpoint": endpoint,
                "key": key,
                "status": "processing"  # Only update if still processing
            },
            {
                "$set": {
                    "status": "failed",
                    "error": error_message,
                    "updated_at": now,
                }
            },
            return_document=True,
        )
        if stored is None:
            # Check if record exists but is in a different state
            existing = self.collection.find_one({"endpoint": endpoint, "key": key})
            if existing:
                # Record exists but already in terminal state
                record = self._to_record(existing)
                if record.is_failed:
                    # Already failed - return existing record
                    return record
                elif record.is_completed:
                    # Already completed - don't overwrite
                    raise RuntimeError(
                        f"Idempotency record {endpoint}:{key} is already marked as completed"
                    )
            raise KeyError(f"Idempotency record not found for {endpoint}:{key}")
        return self._to_record(stored)

    def attach_task_reference(
        self,
        *,
        endpoint: str,
        key: str,
        workflow_id: Optional[str],
        task_id: Optional[str],
        response_payload: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach task and workflow references to an idempotency record.
        
        Uses atomic update that only succeeds if record is still in "processing" state.
        This prevents race conditions where a task completes before task_reference is attached.
        """
        now = _utc_now()
        update_doc: dict[str, Any] = {
            "workflow_id": workflow_id,
            "task_id": task_id,
            "updated_at": now,
        }
        if response_payload is not None:
            update_doc["response_payload"] = response_payload
        if metadata:
            # Store any provided metadata (e.g., project_id, movie_id) to aid debugging/traceability
            update_doc.update(metadata)
        
        # Use find_one_and_update to atomically check and update
        # Only update if status is "processing" - prevents updating terminal states
        result = self.collection.find_one_and_update(
            {
                "endpoint": endpoint,
                "key": key,
                "status": "processing"  # Only update if still processing
            },
            {
                "$set": update_doc
            },
            return_document=False,  # We don't need the document back
        )
        
        # If update didn't match, it means record is already completed/failed with all fields set
        # This is okay - the task may have completed very quickly
        if result is None:
            # Check if record exists and is in a terminal state
            existing = self.collection.find_one({"endpoint": endpoint, "key": key})
            if existing:
                record = self._to_record(existing)
                # If already completed with task_id set, that's fine - task completed quickly
                if record.is_completed and record.task_id:
                    return
                # Log warning for other cases but don't fail
                logger.warning(
                    f"Could not attach task reference to {endpoint}:{key} - "
                    f"record is in state: {record.status}"
                )

    def get_record(self, *, endpoint: str, key: str) -> Optional[IdempotencyRecord]:
        doc = self.collection.find_one({"endpoint": endpoint, "key": key})
        return self._to_record(doc) if doc else None

    def get_record_by_task_id(self, *, task_id: str) -> Optional[IdempotencyRecord]:
        """Find idempotency record by task_id."""
        doc = self.collection.find_one({"task_id": task_id})
        return self._to_record(doc) if doc else None

    def get_record_by_workflow_id(self, *, workflow_id: str) -> Optional[IdempotencyRecord]:
        """Find idempotency record by workflow_id."""
        doc = self.collection.find_one({"workflow_id": workflow_id})
        return self._to_record(doc) if doc else None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _ensure_ttl_index(self) -> None:
        """
        Ensure TTL index exists on expires_at field for automatic cleanup.

        MongoDB TTL indexes automatically delete documents when the current time
        exceeds the value in the indexed field (expires_at).

        This prevents permanent locks from stuck/failed jobs and keeps the
        collection size manageable.
        """
        try:
            # Check if TTL index already exists
            existing_indexes = self.collection.list_indexes()
            ttl_index_exists = False

            for index in existing_indexes:
                if index.get("name") == "expires_at_ttl":
                    ttl_index_exists = True
                    break

            if not ttl_index_exists:
                # Create TTL index: expireAfterSeconds=0 means delete when expires_at < current time
                self.collection.create_index(
                    "expires_at",
                    name="expires_at_ttl",
                    expireAfterSeconds=0,
                )
                logger.info("Created TTL index on idempotency_keys.expires_at")
            else:
                logger.debug("TTL index already exists on idempotency_keys.expires_at")

        except Exception as e:
            # Don't fail initialization if index creation fails
            # The service can still work without automatic cleanup
            logger.warning(f"Failed to create TTL index on idempotency_keys: {e}")

    @staticmethod
    def _to_record(doc: dict[str, Any]) -> IdempotencyRecord:
        return IdempotencyRecord(
            key=doc["key"],
            endpoint=doc["endpoint"],
            status=doc["status"],
            request_hash=doc["request_hash"],
            workflow_id=doc.get("workflow_id"),
            task_id=doc.get("task_id"),
            response_payload=doc.get("response_payload"),
            project_id=doc.get("project_id"),
            movie_id=doc.get("movie_id"),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            expires_at=doc.get("expires_at"),
            raw=doc,
        )

