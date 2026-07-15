"""Request models for production API endpoints."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class CreateProjectNameRequest(BaseModel):
    """Request model for creating a new project with just a name."""
    name: str = Field(..., min_length=1, description="Project name")
    user_id: Optional[str] = Field(None, description="User ID (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Awesome Movie",
                "user_id": "user123"
            }
        }


class CreateProjectRequest(BaseModel):
    """Request model for creating a new project."""
    name: str = Field(..., min_length=1, description="Project name")
    script: str = Field(..., min_length=1, description="Full script text content")
    user_id: Optional[str] = Field(None, description="User ID (optional)")
    shotlist: Optional[str] = Field(None, description="Shotlist or scene breakdown (optional)")
    visual_style: Optional[str] = Field(None, description="Visual style for image generation (pixar, realistic, anime, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "My Awesome Movie",
                "script": "INT. COFFEE SHOP - DAY\n\nJOHN enters the coffee shop, looking around nervously...",
                "user_id": "user123",
                "shotlist": "Scene 1: Wide shot of coffee shop exterior...",
                "visual_style": "pixar"
            }
        }


# ============================================================================
# Movie-related Request Models
# ============================================================================

class CreateMovieRequest(BaseModel):
    """Request model for creating a new movie."""
    title: str = Field(..., min_length=1, description="Movie title")
    description: Optional[str] = Field(None, description="Movie description/synopsis")
    genre: Optional[str] = Field(None, description="Movie genre")
    user_id: Optional[str] = Field(None, description="User ID (optional)")
    global_settings: Optional[Dict[str, Any]] = Field(None, description="Global movie settings")
    start_phase1: bool = Field(True, description="Automatically start Phase 1 after creation")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "The Great Adventure",
                "description": "An epic tale of heroism and discovery",
                "genre": "Action/Adventure",
                "user_id": "user123",
                "global_settings": {
                    "aspect_ratio": "9:16",
                    "visual_style": "Cinematic",
                    "color_palette": "Warm tones"
                },
                "start_phase1": True
            }
        }


class StartMoviePhase1Request(BaseModel):
    """Request model for starting Phase 1 for a movie."""
    movie_id: str = Field(..., description="Movie ID")

    class Config:
        json_schema_extra = {
            "example": {
                "movie_id": "6554f3a7b3e4c12345678901"
            }
        }


class MovieResponse(BaseModel):
    """Response model for movie operations."""
    success: bool
    movie_id: str
    title: Optional[str] = None
    total_scenes: Optional[int] = None
    assets_collection_id: Optional[str] = None
    project_ids: Optional[List[str]] = None
    phase1_job_id: Optional[str] = None
    phase1_status: Optional[str] = None
    overall_status: Optional[str] = None
    created_at: Optional[str] = None
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "movie_id": "6554f3a7b3e4c12345678901",
                "title": "The Great Adventure",
                "total_scenes": 5,
                "assets_collection_id": "6554f3a7b3e4c12345678902",
                "project_ids": ["6554f3a7b3e4c12345678903", "6554f3a7b3e4c12345678904"],
                "phase1_job_id": "job_123456",
                "phase1_status": "running",
                "overall_status": "created",
                "created_at": "2024-11-19T10:30:00Z"
            }
        }
