"""
Shared Pydantic response models used across services.
"""
from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    Generic API response envelope.

    Attributes:
        success:  Whether the request succeeded.
        data:     Response payload (type-parameterised).
        message:  Optional human-readable message.
        error:    Error description when success=False.
    """

    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any = None, message: Optional[str] = None) -> "ApiResponse":
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, error: str, message: Optional[str] = None) -> "ApiResponse":
        return cls(success=False, error=error, message=message)
