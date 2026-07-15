"""
Agent-specific data schemas for prompt review agent.

Contains Pydantic models for structured output with LangChain Gemini.
"""

from typing import List
from pydantic import BaseModel, Field


class PromptReviewItem(BaseModel):
    """Structured output for a single shot's prompt review."""

    shot_id: str = Field(default="", description="Unique identifier for the shot")
    original_prompt: str = Field(default="", description="The original prompt from Agent 2")
    reviewed_prompt: str = Field(default="", description="The refined version (or same if no changes needed)")
    changes_made: List[str] = Field(
        default_factory=list,
        description="List of specific modifications made to the prompt (empty if none)"
    )
    shot_modified: bool = Field(
        default=False,
        description="Boolean indicating if the prompt was changed"
    )
    reason_for_modification: str = Field(
        default="",
        description="Explanation for changes made (empty if none)"
    )
    continuity_observations: List[str] = Field(
        default_factory=list,
        description="List of continuity notes about this shot"
    )
    continuity_status: str = Field(
        default="Pass",
        description="Continuity status: 'Pass' if no changes needed, 'Fixed' if modified"
    )


class PromptReviewResponse(BaseModel):
    """Structured output for the complete prompt review analysis."""
    
    reviews: List[PromptReviewItem] = Field(
        ...,
        description="List of reviewed prompts, one for each shot"
    )

