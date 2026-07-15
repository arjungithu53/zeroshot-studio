"""
Agent-specific data schemas for shot strategy agent.

Contains enums and types specific to the shot strategy agent logic.
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

# Generation strategy enum for shot strategy agent
GenerationStrategy = Literal["multi_shot", "last_frame_seed", "generate_new"]


# Pydantic models for structured output with LangChain Gemini
class AnnotatedShotStrategy(BaseModel):
    """Structured output for a single shot's generation strategy."""
    
    shot_id: str = Field(..., description="Unique identifier for the shot")
    generation_strategy: GenerationStrategy = Field(
        ..., 
        description="Recommended generation strategy: generate_new, last_frame_seed, or multi_shot"
    )
    reasoning: str = Field(
        ..., 
        description="Brief 1-2 sentence explanation of why this strategy was chosen"
    )
    continuity_notes: Optional[str] = Field(
        None, 
        description="Brief notes about visual/action continuity with previous shots"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence level in the strategy choice (0.0 to 1.0)"
    )
    seed_shot_id: Optional[str] = Field(
        None, 
        description="ID of the shot to use as seed (for last_frame_seed or multi_shot strategies)"
    )


class ShotStrategyResponse(BaseModel):
    """Structured output for the complete shot strategy analysis."""
    
    annotated_shots: List[AnnotatedShotStrategy] = Field(
        ..., 
        description="List of annotated shots with generation strategies"
    )
