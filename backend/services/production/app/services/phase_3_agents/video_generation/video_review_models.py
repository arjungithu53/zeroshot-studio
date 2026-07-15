"""
Pydantic Models for Video Review Agent (Phase 3)
=================================================
Defines structured output models for video review using Pydantic.
Follows the same pattern as phase_1_agents for consistency.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class VideoQualityScores(BaseModel):
    """Quality scores for video review (simplified)"""
    prompt_accuracy: int = Field(
        ge=0, le=40,
        description="How well the video matches the prompt (0-40)"
    )
    visual_quality: int = Field(
        ge=0, le=30,
        description="Technical quality, clarity, smoothness (0-30)"
    )
    motion_quality: int = Field(
        ge=0, le=20,
        description="Quality of movement and animation (0-20)"
    )
    production_readiness: int = Field(
        ge=0, le=10,
        description="Ready for use in production (0-10)"
    )


class VideoAssessment(BaseModel):
    """Assessment details for video"""
    strengths: List[str] = Field(
        default_factory=list,
        description="What works well in this video"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Problems identified in the video"
    )
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Elements from prompt that are missing"
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="AI artifacts detected (glitches, warping, etc.)"
    )


class PromptSuggestions(BaseModel):
    """Prompt improvement suggestions (aligned with existing prompt review pattern)"""
    original_prompt: str = Field(
        description="The original video generation prompt"
    )
    suggested_prompt: str = Field(
        description="Improved prompt for better results (if changes needed, else same as original)"
    )
    changes_made: str = Field(
        description="Description of changes made to the prompt"
    )
    reasoning: str = Field(
        description="Why these changes would improve the video"
    )


class VideoReviewResult(BaseModel):
    """Complete video review result - streamlined version"""
    shot_id: str = Field(
        description="ID of the shot being reviewed"
    )
    decision: str = Field(
        description="Decision: approved/refine_prompt/regenerate"
    )
    overall_score: int = Field(
        ge=0, le=100,
        description="Overall video quality score (0-100)"
    )
    scores: VideoQualityScores = Field(
        description="Detailed scoring breakdown"
    )
    assessment: VideoAssessment = Field(
        description="Strengths, issues, and artifacts"
    )
    prompt_suggestions: PromptSuggestions = Field(
        description="Prompt refinement suggestions"
    )
    production_notes: str = Field(
        description="Additional notes for production use"
    )
    timestamp: str = Field(
        description="ISO timestamp of review"
    )
