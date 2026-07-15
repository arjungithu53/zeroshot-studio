"""
MongoDB client factory for infrastructure layer.
Provides MongoConfig, MongoClientFactory, connection pooling, and reset utilities.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

logger = logging.getLogger(__name__)

# Registry of all active MongoClientFactory instances (for reset_all_connections)
_active_connections: List["MongoClientFactory"] = []
_registry_lock = threading.Lock()


@dataclass
class MongoConfig:
    uri: str
    database_name: str
    ssl_verify: bool = False
    max_pool_size: int = 50
    server_selection_timeout_ms: int = 30000
    connect_timeout_ms: int = 30000
    socket_timeout_ms: int = 30000


class MongoClientFactory:
    """
    Thread-safe singleton-friendly MongoDB client factory with connection pooling.
    """

    def __init__(self, config: MongoConfig) -> None:
        self.config = config
        self._client: Optional[MongoClient] = None
        self._lock = threading.Lock()

        with _registry_lock:
            _active_connections.append(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self) -> MongoClient:
        uri = self.config.uri

        # Append TLS option for Atlas URIs that don't already have it
        if not self.config.ssl_verify and (
            "mongodb+srv://" in uri or "mongodb://" in uri
        ):
            if "tlsAllowInvalidCertificates" not in uri:
                sep = "&" if "?" in uri else "?"
                uri = f"{uri}{sep}tlsAllowInvalidCertificates=true"

        client: MongoClient = MongoClient(
            uri,
            maxPoolSize=self.config.max_pool_size,
            serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
            connectTimeoutMS=self.config.connect_timeout_ms,
            socketTimeoutMS=self.config.socket_timeout_ms,
        )
        logger.info(
            "MongoDB client created for database '%s'", self.config.database_name
        )
        return client

    def _ensure_client(self) -> MongoClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_client(self) -> MongoClient:
        return self._ensure_client()

    def get_database(self) -> Database:
        return self._ensure_client()[self.config.database_name]

    def get_collection(self, name: str) -> Tuple[MongoClient, Collection]:
        """Return (client, collection) tuple — matches production's expected interface."""
        client = self._ensure_client()
        collection: Collection = client[self.config.database_name][name]
        return client, collection

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                    logger.info("MongoDB client closed.")
                except Exception:
                    logger.exception("Error closing MongoDB client")
                finally:
                    self._client = None


# ------------------------------------------------------------------
# Module-level helpers used by production config.py
# ------------------------------------------------------------------

def get_connection_stats(factories: Optional[List[MongoClientFactory]] = None) -> Dict[str, Any]:
    """Return basic stats about active factory instances."""
    targets = factories if factories is not None else _active_connections
    return {
        "total_factories": len(targets),
        "connected": sum(1 for f in targets if f._client is not None),
    }


def reset_all_connections(
    factories: Optional[List[MongoClientFactory]] = None,
) -> Dict[str, Any]:
    """
    Close and reset all (or supplied) MongoClientFactory instances.
    Returns a summary dict.
    """
    targets = factories if factories is not None else list(_active_connections)
    closed = 0
    for factory in targets:
        try:
            factory.close()
            closed += 1
        except Exception:
            logger.exception("Failed to close factory %s", factory)

    return {"factories_reset": closed, "status": "ok"}
