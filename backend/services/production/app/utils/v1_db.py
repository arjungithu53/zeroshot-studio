"""Read-only utility for fetching data from the v1 MongoDB database.

The v1 database is a separate database from the production database.
This module is the ONLY place in the production service that connects to v1.
It must never write to v1 — only read.
"""
import os
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.errors import PyMongoError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

logger = get_logger(__name__)

# Environment variables — same ones used by the v1 / pre-production services
_V1_MONGO_URI = None
_V1_DB_NAME = None
_V1_PROJECTS_COLLECTION = None


def _get_v1_config():
    """Lazily read v1 connection config from environment."""
    global _V1_MONGO_URI, _V1_DB_NAME, _V1_PROJECTS_COLLECTION
    if _V1_MONGO_URI is None:
        _V1_MONGO_URI = os.getenv("MongoDB")
        _V1_DB_NAME = os.getenv("DB_NAME", "v1")
        _V1_PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
    return _V1_MONGO_URI, _V1_DB_NAME, _V1_PROJECTS_COLLECTION


def fetch_product_image_url(v1_project_id: str) -> Optional[str]:
    """
    Fetch the product image S3 URL from the v1 projects collection.

    This is the only function that should be used to read from v1.
    It connects, reads one document, and closes the connection immediately.

    Args:
        v1_project_id: The _id (as a string) of the document in the v1
                       projects collection.

    Returns:
        The product_image.s3_url string if found, otherwise None.
    """
    uri, db_name, collection_name = _get_v1_config()

    if not uri:
        logger.warning("v1_db: MongoDB env var not set — cannot fetch product image URL")
        return None

    if not v1_project_id:
        return None

    try:
        object_id = ObjectId(v1_project_id)
    except InvalidId:
        logger.warning(f"v1_db: Invalid v1_project_id format: {v1_project_id!r}")
        return None

    client = None
    try:
        client = MongoClient(uri, tlsAllowInvalidCertificates=True, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        doc = db[collection_name].find_one(
            {"_id": object_id},
            {"product_image": 1}  # projection — only fetch the field we need
        )
        if not doc:
            logger.warning(f"v1_db: No project found for id={v1_project_id}")
            return None

        product_image = doc.get("product_image") or {}
        s3_url = product_image.get("s3_url")

        if s3_url:
            logger.info(f"v1_db: Found product_image.s3_url for project {v1_project_id}")
        else:
            logger.warning(
                f"v1_db: Project {v1_project_id} found but product_image.s3_url is empty"
            )
        return s3_url or None

    except PyMongoError as exc:
        logger.error(f"v1_db: MongoDB error fetching product image: {exc}")
        return None
    finally:
        if client:
            client.close()
