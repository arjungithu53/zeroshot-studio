#!/usr/bin/env python3
"""
LangGraph State Schema for Phase 1 Agent Pipeline
==================================================
De
es the shared state structure that flows through all agents.
"""

from typing import TypedDict, Dict, List, Any, Optional
from typing_extensions import Annotated
import operator


class Phase1State(TypedDict):
    """
    Shared state for the 4-agent pipeline.
    Each agent reads from and writes to this state.
    """

    # Input
    script_path: str
    script_content: str
    project_id: Optional[str]  # MongoDB project ID for saving agent outputs (legacy mode)

    # Movie workflow fields (NEW)
    movie_id: Optional[str]  # MongoDB movie ID (for movie workflow)
    assets_collection_id: Optional[str]  # MongoDB assets collection ID (for movie workflow)
    job_id: Optional[str]  # Pipeline job ID for status tracking
    visual_style: Optional[str]  # Visual style for image generation (realistic, pixar, etc.)
    csv_entity_mapping: Optional[Dict[str, Any]]  # Pre-extracted CSV entities from shotlist (characters and locations)
    product_prop: Optional[Dict]  # Product prop fetched from DB when product_present=Yes in shotlist
    product_image_s3_url: Optional[str]  # S3 URL of the uploaded product image (from production MongoDB)

    # Agent 1: Asset Generator
    # Each asset now has an 'id' field (UUID)
    # Structure: {characters: [{id: "uuid", name: "...", ...}], locations: [...], props: [...]}
    extracted_assets: Dict[str, List[Dict]]
    agent1_status: str  # pending/completed/failed
    agent1_human_feedback: Optional[Dict[str, Any]]

    # Agent 2: Asset Reviewer
    review_results: Dict[str, Any]
    # Enhanced assets maintain the id field from Agent 1
    enhanced_assets: Dict[str, List[Dict]]
    agent2_status: str
    agent2_human_feedback: Optional[Dict[str, Any]]

    # Agent 3: Prompt Generator
    # Changed from Dict[str, Dict] to Dict[str, List[Dict]] to use list with asset_id
    # Structure: {characters: [{id: "uuid", name: "...", prompt: {...}, ...}], locations: [...], props: [...]}
    generated_prompts: Dict[str, List[Dict]]
    agent3_status: str
    agent3_human_feedback: Optional[Dict[str, Any]]

    # Agent 4: Prompt Optimizer
    # Changed from Dict[str, Dict] to Dict[str, List[Dict]] to use list with asset_id
    # Structure: {characters: [{id: "uuid", name: "...", final_prompt: {...}, ...}], locations: [...], props: [...]}
    optimized_prompts: Dict[str, List[Dict]]
    agent4_status: str
    agent4_human_feedback: Optional[Dict[str, Any]]

    # Agent 5: Image Generator
    # Changed to list structure with asset_id
    # Structure: {characters: [{id: "uuid", name: "...", images: [...], task_id: "...", ...}], locations: [...], props: [...]}
    generated_images: Dict[str, List[Dict]]
    failed_generations: List[Dict[str, Any]]  # List of failed asset generations with retry info (includes asset_id)
    agent5_status: str

    # Agent 6: Image Reviewer (AI Critic)
    # Structure: {characters: [{id: "uuid", name: "...", review: {...}, decision: "...", ...}], locations: [...], props: [...]}
    image_reviews: Dict[str, List[Dict]]
    agent6_status: str
    needs_regeneration_assets: List[str]  # List of asset_ids that need regeneration (Agent 6 decision)
    regenerated_prompts: Optional[Dict[str, List[Dict]]]  # Rewritten prompts for regeneration
    auto_regeneration_count: int  # Track auto-regeneration attempts (max 3)

    # Agent 7: Image Editor (Auto-Fix)
    # Structure: {characters: [{id: "uuid", name: "...", edited_images: [...], edits_applied: [...], ...}], locations: [...], props: [...]}
    edited_images: Dict[str, List[Dict]]
    agent7_status: str
    needs_editing_assets: List[str]  # List of asset_ids (UUIDs) that need editing
    recently_edited_asset_ids: Optional[List[str]]  # Track recently edited assets for selective re-review
    auto_edit_count: int  # Track auto-edit loops (max 3)

    # Agent 8: Variation Generator
    # Structure: {characters: [{id: "uuid", name: "...", variations: {...}, ...}], locations: [...], props: [...]}
    variation_images: Dict[str, List[Dict]]
    agent8_status: str

    # Human Approval Checkpoint (after Agent 7)
    human_approval_decision: Optional[str]  # "approve_all", "partial_approve", "reject_all", or "edit_prompts"
    human_approval_feedback: Optional[Dict[str, Any]]  # Optional human comments
    regeneration_count: int  # Track how many times we've regenerated
    max_regenerations: int  # Safety limit (default: 5)

    # Asset-level approval tracking (for partial approvals)
    approved_asset_ids: Optional[List[str]]  # List of asset UUIDs that are approved
    rejected_asset_ids: Optional[List[str]]  # List of asset UUIDs that need regeneration

    # Pipeline control
    current_agent: str  # Which agent is currently executing
    pipeline_status: str  # running/completed/failed/waiting_for_human_approval
    error_message: Optional[str]

    # Output files tracking
    output_files: List[str]  # All output files

    # Human feedback tracking
    requires_human_feedback: bool
    feedback_agent: Optional[str]  # Which agent is waiting for feedback
