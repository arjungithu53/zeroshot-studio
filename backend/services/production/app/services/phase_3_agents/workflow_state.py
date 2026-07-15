#!/usr/bin/env python3
"""
LangGraph State Schema for Phase 3 Agent Pipeline
==================================================
Defines the shared state structure that flows through all video generation agents.
"""

from typing import TypedDict, Dict, List, Any, Optional


class Phase3State(TypedDict):
    """
    Shared state for the Phase 3 video generation pipeline.
    Each agent reads from and writes to this state.
    """

    # Input
    shot_id: str  # Shot ID to process
    show_id: str  # Show ID for MongoDB queries
    episode_number: int  # Episode number for MongoDB queries
    shot_data: Dict[str, Any]  # MongoDB shot document
    image_version: Optional[str]  # Optional version of the shot image to use for video generation (e.g., "v0", "v1", "v2")
    job_id: Optional[str]  # Job ID for tracking in production_pipelines collection
    scene_number: Optional[int]  # Scene number from shot document (used for S3 naming)
    sequence_number: Optional[int]  # Shot sequence number within scene (used for S3 naming)

    # Prompt Generation (conditional based on strategy)
    generation_strategy: str  # generate_new, last_frame_seed, multi_shot
    video_prompt: str  # Generated video prompt (from prompt_A or prompt_B)
    prompt_version: str  # "A" or "B" to track which prompt agent was used

    # Video Generation
    start_image_url: str  # S3 URL of the start image
    video_generation_task_id: Optional[str]  # Freepik API task ID
    generated_video_url: Optional[str]  # S3 URL of generated video
    video_generation_status: str  # pending/processing/completed/failed
    video_generation_attempt: int  # Track regeneration attempts
    max_video_generation_attempts: int  # Max attempts (default: 3)

    # AI Video Review
    review_result: Optional[Dict[str, Any]]  # VideoReviewResult from AI review
    review_decision: Optional[str]  # approved/refine_prompt/regenerate
    review_score: Optional[int]  # Overall score (0-100)
    suggested_prompt: Optional[str]  # AI-suggested improved prompt
    suggested_prompt_reasoning: Optional[str]  # Reasoning behind the suggested prompt
    ai_review_status: str  # pending/completed/failed

    # Human Checkpoint
    human_decision: Optional[str]  # approved/needs_changes
    human_updated_prompt: Optional[str]  # Human-provided updated prompt
    human_feedback: Optional[str]  # Human comments/feedback
    human_checkpoint_status: str  # pending/completed
    human_regeneration_attempt: int  # Track human-requested regenerations
    max_human_regeneration_attempts: int  # Max human regeneration attempts (default: 3)

    # Pipeline Control
    current_node: str  # Current node in the workflow
    pipeline_status: str  # running/completed/failed/waiting_for_human
    error_message: Optional[str]

    # MongoDB tracking
    mongodb_save_status: str  # Track if results were saved to MongoDB

    # Versioning for iterative improvements
    video_versions: List[Dict[str, Any]]  # Track all video versions generated
    current_version: int  # Current version number (v0, v1, v2, etc.)
