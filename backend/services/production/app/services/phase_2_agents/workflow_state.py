#!/usr/bin/env python3
"""
LangGraph State Schema for Phase 2 Agent Pipeline
==================================================
Defines the shared state structure that flows through all phase 2 agents.
"""

from typing import TypedDict, Dict, List, Any, Optional
from typing_extensions import Annotated
import operator

from backend.services.production.app.models.mongodb.shots import ShotList, AnnotatedShotList, AnnotatedShotItem


class Phase2State(TypedDict):
    """
    Shared state for the 3-agent phase 2 pipeline.
    Each agent reads from and writes to this state.
    """

    # Input
    shot_list_request: Dict[str, Any]  # ShotListRequest from API
    show_id: str
    episode_number: int
    project_id: Optional[str]  # Phase 1 project_id for asset loading
    scene_description: Optional[str]
    job_id: Optional[str]  # Pipeline job identifier for tracking
    movie_id: str  # Movie ID to fetch visual_style from movies collection

    # Agent 1: Shot Strategy Agent
    annotated_shot_list: Optional[AnnotatedShotList]  # Strategy analysis results
    strategy_analysis_results: Dict[str, Any]
    agent1_status: str  # pending/completed/failed
    agent1_human_feedback: Optional[Dict[str, Any]]

    # Human Approval Checkpoint (after Agent 1)
    strategy_approval_decision: Optional[bool]  # True/False for approval
    strategy_approval_feedback: Optional[Dict[str, Any]]  # Optional human comments

    # Agent 2: Image Prompt Generator Agent
    image_prompts_generated: Dict[str, Any]  # Generated prompts for each shot
    agent2_status: str
    agent2_human_feedback: Optional[Dict[str, Any]]

    # Agent 3: Prompt Review Agent
    reviewed_prompts: Dict[str, Any]  # Reviewed and refined prompts
    agent3_status: str
    agent3_human_feedback: Optional[Dict[str, Any]]

    # Agent 12: Shot Design Agent
    shot_designs: Dict[str, Any]  # Shot design analysis and asset selection
    agent12_status: str
    agent12_human_feedback: Optional[Dict[str, Any]]

    # Agent 13: Prompt Modifier Agent
    modified_prompts: Dict[str, Any]  # Modified prompts with corrected assets
    agent13_status: str
    agent13_human_feedback: Optional[Dict[str, Any]]
    
    # Human Approval Checkpoint (after Agent 13) - for prompt approval
    prompt_approval_decision: Optional[bool]  # True/False for prompt approval
    prompt_approval_feedback: Optional[Dict[str, Any]]  # Optional human comments

    # Agent 14: Imagen Generator Agent
    generated_images: Dict[str, Any]  # Generated images with S3 URLs
    agent14_status: str
    agent14_human_feedback: Optional[Dict[str, Any]]

    # Agent 15: Image Reviewer Agent
    image_reviews: Dict[str, Any]  # Image review results
    agent15_status: str
    agent15_human_feedback: Optional[Dict[str, Any]]

    # Agent 15A: Prompt Regeneration Agent (for regeneration loop)
    regenerated_prompts: Dict[str, Any]  # Regenerated prompts from Agent 15A
    agent15A_status: str
    agent15A_human_feedback: Optional[Dict[str, Any]]

    # Agent 7: Shot Editor Agent (for edit-review loop)
    edited_shots: Dict[str, Any]  # Edited shot images with versioning
    agent7_status: str
    agent7_human_feedback: Optional[Dict[str, Any]]

    # Edit-Review Loop Tracking
    edit_loop_iterations: Dict[str, int]  # Track iteration count per shot (shot_id -> iteration)
    shots_needing_edit: List[str]  # List of shot_ids that need editing
    shots_approved: List[str]  # List of shot_ids that are approved
    shots_max_retries: List[str]  # List of shot_ids that exceeded max edit attempts
    
    # Regeneration Loop Tracking (15-15A-14-15 loop)
    regenerate_loop_iterations: Dict[str, int]  # Track regeneration iteration count per shot (shot_id -> iteration)
    shots_needing_regeneration: List[str]  # List of shot_ids that need regeneration
    shots_edit_instructions: Dict[str, Any]  # Store edit instructions per shot for Agent 7
    
    # Product Fidelity Review Loop (Agents 16, 17, 18)
    product_review_results: Dict[str, Any]       # shot_id → ProductReviewResult dict
    product_review_iterations: Dict[str, int]    # shot_id → fix attempts so far (max 3)
    product_fix_prompts: Dict[str, str]          # shot_id → Nano Banana replacement prompt (Agent 17)
    product_corrected_images: Dict[str, str]     # shot_id → latest corrected S3 URL (Agent 18)
    shots_needing_product_fix: List[str]         # shot_ids currently failing product review
    shots_product_approved: List[str]            # shot_ids that passed product review (or force-passed)

    # Final Human Approval
    final_approval_decision: Optional[bool]  # Final human approval after all edits
    final_approval_feedback: Optional[Dict[str, Any]]

    # MongoDB integration
    mongodb_client: Optional[Any]  # MongoDB client instance
    mongodb_operations: Dict[str, Any]  # Track MongoDB save/update operations

    # Pipeline control
    current_agent: str  # Which agent is currently executing
    pipeline_status: str  # running/completed/failed/waiting_for_approval/waiting_for_final_approval
    error_message: Optional[str]

    # Output files tracking
    output_files: List[str]  # All output files

    # Human feedback tracking
    requires_human_feedback: bool
    feedback_agent: Optional[str]  # Which agent is waiting for feedback

    # Episode metadata
    episode_id: str
    title: str

    # v1 database linkage (read-only — only used to fetch product image URL)
    v1_project_id: Optional[str]
    product_image_url: Optional[str]  # Fetched from v1 projects.product_image.s3_url
