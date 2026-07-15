"""
MongoDB validation helpers shared across services.

Usage:
    from backend.shared.utils.mongodb_validators import validate_object_id
    oid = validate_object_id(some_string)
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import HTTPException


def validate_object_id(value: str, field_name: str = "id") -> ObjectId:
    """
    Validate and convert a string to a MongoDB ObjectId.

    Args:
        value:      String to validate.
        field_name: Human-readable field name for error messages.

    Returns:
        bson.ObjectId instance.

    Raises:
        HTTPException(400): If the string is not a valid ObjectId.
    """
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ObjectId for field '{field_name}': '{value}'",
        )
