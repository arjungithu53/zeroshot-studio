"""
Admin authentication dependency for FastAPI endpoints.

Reads ADMIN_API_KEY from environment. If not set, passes through (dev mode).
In production, set ADMIN_API_KEY and pass it as the X-Admin-Key header.
"""
import os
from fastapi import Header, HTTPException
from pydantic import BaseModel
from typing import Optional


class AdminUser(BaseModel):
    key: str
    user_id: str
    is_dev: bool = False


async def validate_admin_from_header(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key")
) -> AdminUser:
    """
    FastAPI dependency that validates the X-Admin-Key header.

    - If ADMIN_API_KEY env var is set: requires the header to match.
    - If ADMIN_API_KEY is not set: passes through (development mode).
    """
    expected_key = os.getenv("ADMIN_API_KEY", "")

    if not expected_key:
        # Dev mode — no key configured, allow all requests
        return AdminUser(key="dev", user_id="dev", is_dev=True)

    if not x_admin_key or x_admin_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Key header")

    return AdminUser(key=x_admin_key, user_id=x_admin_key)
