"""Response models for production API endpoints."""
from pydantic import BaseModel, Field
from typing import Optional


class CreateProjectResponse(BaseModel):
    """Response model for project creation."""
    success: bool = Field(..., description="Whether the operation succeeded")
    project_id: str = Field(..., description="MongoDB ObjectId of created project")
    name: str = Field(..., description="Project name")
    status: str = Field(..., description="Current project status")
    message: str = Field(..., description="Response message")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "project_id": "507f1f77bcf86cd799439011",
                "name": "My Awesome Movie",
                "status": "extracting",
                "message": "Project created successfully. Pipeline started.",
                "created_at": "2025-10-15T16:45:00Z"
            }
        }
