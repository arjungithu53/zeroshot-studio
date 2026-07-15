"""
Phase 1 Workflow API Endpoints
================================
FastAPI endpoints for managing Phase 1 workflow jobs.

MAJOR CHANGE: Celery + SQS Integration
---------------------------------------
This file has been updated to use Celery with Amazon SQS instead of FastAPI BackgroundTasks.

What changed:
- Removed: BackgroundTasks dependency
- Added: Celery task dispatching via apply_async()
- Added: Task ID tracking in job records
- Added: Task status monitoring endpoint

Why this change?
- Reliability: Tasks survive server restarts (stored in SQS)
- Scalability: Multiple workers can process tasks in parallel
- Monitoring: Full task progress tracking and error handling
- Resource separation: Heavy processing doesn't block API server
"""

import os
import sys
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Request, Depends
from fastapi.responses import JSONResponse
from typing import Dict, Any
from pathlib import Path
from shared.auth.dependencies import validate_admin_from_header, AdminUser

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception

# Initialize logger for this module
logger = get_logger(__name__)

from app.models.mongodb.pipelines import (
    PipelineJobCreate,
    PipelineJobResponse,
    HumanApprovalRequest,
    AssetPromptEdit
)
from app.models.requests import CreateProjectRequest
from app.services.pipeline_service import PipelineService
from app.services.project_service import ProjectService
from app.services.phase_1_agents.langgraph_workflow import create_phase1_workflow
from app.config import get_workflow_queue_name

# Import Celery tasks (replacing background functions)
from app.tasks.phase1_tasks import (
    run_phase1_workflow_task,
    resume_phase1_workflow_task,
    retry_failed_asset_task
)

# Import Celery app for task status queries
from app.celery_app import celery_app
from celery.result import AsyncResult

# Import rate limiter from main app
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/phase1", tags=["Phase 1 Workflow"])
pipeline_service = PipelineService()
project_service = ProjectService()

# Initialize limiter (will use the one from app.state in practice)
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# Background Task Functions - REMOVED
# ============================================================================
# The following functions have been moved to app/tasks/phase1_tasks.py as Celery tasks:
# - run_workflow_background → run_phase1_workflow_task
# - resume_workflow_background → resume_phase1_workflow_task
#
# Why moved to Celery?
# -------------------
# - Tasks persist in SQS (survive server restarts)
# - Distributed execution across multiple workers
# - Built-in retry logic and error handling
# - Real-time progress tracking
# - Separate resource allocation from API server
# ============================================================================


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/start", response_model=PipelineJobResponse, status_code=202)
@limiter.limit("10/minute")
async def start_workflow(
    request: Request,
    project_request: CreateProjectRequest = None,
    project_id: str = None,
    visual_style: str = None,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Start a new Phase 1 workflow job using Celery + SQS

    This endpoint supports TWO workflows:

    OPTION 1 - Create new project with script:
    1. Pass CreateProjectRequest with name, script, shotlist
    2. Creates a new project with the script
    3. Creates a pipeline job to track execution
    4. Dispatches Celery task to SQS queue (non-blocking)

    OPTION 2 - Use existing project (with uploaded files):
    1. Pass project_id of existing project (created via /projects/create-name)
    2. Project must already have script uploaded via /projects/{project_id}/upload-files
    3. Creates a pipeline job to track execution
    4. Dispatches Celery task to SQS queue (non-blocking)

    The workflow will pause at the human checkpoint after Agent 6/7 completes.

    OLD (BackgroundTasks):
    ----------------------
    - Task runs in FastAPI process
    - Lost if server restarts
    - No retry mechanism
    - Hard to monitor

    NEW (Celery + SQS):
    -------------------
    - Task queued in SQS, processed by separate worker
    - Persists through restarts
    - Automatic retry on failure (3 attempts)
    - Real-time progress tracking via celery_task_id
    - Scalable: multiple workers can process tasks in parallel

    Response includes celery_task_id for monitoring:
    - Poll: GET /api/v1/phase1/task-status/{celery_task_id}
    """
    try:
        # DEBUG LOGGING: Log all incoming parameters
        logger.info("="*80)
        logger.info("PHASE 1 START - INCOMING PARAMETERS")
        logger.info("="*80)
        logger.info(f"project_id: {project_id}")
        logger.info(f"visual_style (query param): {visual_style}")
        if project_request:
            logger.info(f"project_request.name: {project_request.name}")
            logger.info(f"project_request.visual_style: {getattr(project_request, 'visual_style', 'ATTRIBUTE NOT PRESENT')}")
        logger.info("="*80)

        # Determine which workflow option to use
        if project_id:
            # OPTION 2: Use existing project
            logger.info(f"Starting workflow with existing project: {project_id}")

            # Get existing project
            project = project_service.get_project(project_id)
            if not project:
                raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

            # Verify project has script
            script_content = project.get("script")
            if not script_content:
                raise HTTPException(
                    status_code=400,
                    detail="Project does not have a script. Please upload script first via /projects/{project_id}/upload-files"
                )

            project_result = {
                "project_id": project_id,
                "name": project.get("name", "Unknown Project")
            }

        elif project_request:
            # OPTION 1: Create new project with script
            logger.info(f"Creating new project with script: {project_request.name}")

            project_result = project_service.create_project(
                name=project_request.name,
                script=project_request.script,
                user_id=project_request.user_id,
                shotlist=project_request.shotlist
            )
            script_content = project_request.script

            # Extract visual_style from project_request if not provided as query param
            if not visual_style and project_request.visual_style:
                visual_style = project_request.visual_style
                logger.info(f"Using visual_style from project_request: {visual_style}")

        else:
            raise HTTPException(
                status_code=400,
                detail="Either project_request or project_id must be provided"
            )

        # DEBUG LOGGING: Log final visual_style value
        logger.info("="*80)
        logger.info(f"FINAL visual_style VALUE TO BE SENT TO CELERY: '{visual_style}'")
        logger.info("="*80)

        # STEP 2: Create pipeline job (lightweight tracking)
        job_data = PipelineJobCreate(project_id=project_result["project_id"])
        job_result = pipeline_service.create_job(job_data)
        job_id = job_result["job_id"]

        # STEP 3: Dispatch Celery task to SQS queue
        # apply_async() sends task to queue and returns immediately
        # Use kwargs to be explicit about parameter mapping
        queue_name = get_workflow_queue_name()
        task = run_phase1_workflow_task.apply_async(
            kwargs={
                "job_id": job_id,
                "project_id": project_result["project_id"],
                "script_content": script_content,
                "visual_style": visual_style
            },
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched Phase 1 workflow to Celery")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Project ID: {project_result['project_id']}")
        logger.info(f"Visual Style: {visual_style or 'not specified (will use default)'}")
        logger.info(f"Celery Task ID: {task.id}")
        logger.info(f"Queue: {queue_name}")

        # STEP 4: Update job with Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)

        # STEP 5: Return job info immediately (includes celery_task_id)
        job = pipeline_service.get_job(job_id)
        return pipeline_service.to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "start_workflow")


@router.post("/upload-script", response_model=PipelineJobResponse, status_code=202)
@limiter.limit("10/minute")
async def upload_script(
    request: Request,
    file: UploadFile = File(...),
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Upload a script file and start Phase 1 workflow using Celery + SQS

    Accepts text files (.txt, .fountain, .fdx) containing screenplay content.

    Updated to use Celery for processing the uploaded script.
    """
    try:
        # Read file content
        content = await file.read()
        script_content = content.decode('utf-8')

        # Create project
        project_result = project_service.create_project(
            name=file.filename or "Uploaded Script",
            script=script_content
        )

        # Create pipeline job
        job_data = PipelineJobCreate(project_id=project_result["project_id"])
        job_result = pipeline_service.create_job(job_data)
        job_id = job_result["job_id"]

        # Dispatch Celery task
        # Use kwargs to be explicit about parameter mapping
        queue_name = get_workflow_queue_name()
        task = run_phase1_workflow_task.apply_async(
            kwargs={
                "job_id": job_id,
                "project_id": project_result["project_id"],
                "script_content": script_content,
                "visual_style": None  # visual_style not available in upload endpoint
            },
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched uploaded script to Celery")
        logger.info(f"File: {file.filename}")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Celery Task ID: {task.id}")

        # Update job with Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)

        # Return job info
        job = pipeline_service.get_job(job_id)
        return pipeline_service.to_response(job)

    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be a valid text file")
    except Exception as e:
        raise handle_api_exception(e, "upload_script")


@router.get("/status/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("100/minute")
async def get_job_status(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get current status of a Phase 1 workflow job

    Returns job status, current agent, and completion info.
    Poll this endpoint to track workflow progress.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return pipeline_service.to_response(job)


@router.get("/results/{job_id}")
@limiter.limit("100/minute")
async def get_job_results(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get complete results of a Phase 1 workflow job

    Returns pipeline tracking info + project data.
    Project data contains all actual workflow outputs (assets, images, etc.)
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get project data (contains actual assets and outputs)
    project = project_service.get_project(job.project_id)

    # Return combined data
    return {
        "job_id": job.job_id,
        "project_id": job.project_id,
        "status": job.status,
        "pipeline_status": job.pipeline_status,
        "current_agent": job.current_agent,
        "agent_statuses": {
            "agent_1": job.agent1_status,
            "agent_2": job.agent2_status,
            "agent_3": job.agent3_status,
            "agent_4": job.agent4_status,
            "agent_5": job.agent5_status,
            "agent_6": job.agent6_status,
            "agent_7": job.agent7_status,
            "agent_8": job.agent8_status,
        },
        "project": project,  # Contains all agent outputs and data
        "output_files": job.output_files,
        "regeneration_count": job.regeneration_count,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at
    }


@router.get("/results/by-project/{project_id}")
@limiter.limit("100/minute")
async def get_results_by_project(request: Request, project_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get complete results of the most recent Phase 1 workflow job for a project

    This endpoint is useful when you only have the project_id and want to fetch
    the results without needing to know the job_id.

    Returns pipeline tracking info + project data.
    Project data contains all actual workflow outputs (assets, images, etc.)
    """
    # Get most recent job for this project
    job = pipeline_service.get_job_by_project_id(project_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"No jobs found for project {project_id}")

    # Get project data (contains actual assets and outputs)
    project = project_service.get_project(job.project_id)

    # Reconstruct approved_assets from human_approval_feedback
    approved_assets = []
    if job.human_approval_feedback:
        approved_assets = job.human_approval_feedback.get("approved_assets", [])

    # Return combined data
    return {
        "job_id": job.job_id,
        "project_id": job.project_id,
        "status": job.status,
        "pipeline_status": job.pipeline_status,
        "current_agent": job.current_agent,
        "checkpoint_approved": job.checkpoint_approved,
        "approved_assets": approved_assets,
        "agent_statuses": {
            "agent_1": job.agent1_status,
            "agent_2": job.agent2_status,
            "agent_3": job.agent3_status,
            "agent_4": job.agent4_status,
            "agent_5": job.agent5_status,
            "agent_6": job.agent6_status,
            "agent_7": job.agent7_status,
            "agent_8": job.agent8_status,
        },
        "project": project,  # Contains all agent outputs and data
        "output_files": job.output_files,
        "regeneration_count": job.regeneration_count,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at
    }


@router.post("/checkpoint/approve/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def approve_checkpoint(
    request: Request,
    job_id: str,
    approval: HumanApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Approve specific assets at human checkpoint (does NOT proceed to Agent 8)

    This endpoint marks assets as approved and stores them in the job state.
    The job remains at the checkpoint so you can:
    - Approve more assets
    - Edit prompts for other assets
    - Use /checkpoint/finalize when ready to proceed to Agent 8

    User provides a list of approved asset IDs with individual feedback.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "waiting_for_human_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not waiting for approval. Current status: {job.status}"
        )

    # Extract approved asset IDs
    approved_asset_ids = [asset.asset_id for asset in approval.approved_assets]

    # Get existing approved assets and merge
    existing_approved = job.approved_assets_list or []
    all_approved = list(set(existing_approved + approved_asset_ids))

    logger.info(f"{len(approved_asset_ids)} new assets approved for job {job_id}")
    logger.info(f"Total approved assets: {len(all_approved)}")
    for asset in approval.approved_assets:
        logger.info(f"Approved {asset.asset_type}: {asset.asset_id}")
        if asset.feedback:
            logger.info(f"Feedback: {asset.feedback}")

    # Update approved assets list in database (but keep status as waiting_for_human_approval)
    pipeline_service.update_approved_assets(
        job_id,
        all_approved,
        {
            "global_feedback": approval.global_feedback,
            "approved_assets": [a.dict() for a in approval.approved_assets]
        }
    )

    logger.info("Assets marked as approved. Job remains at checkpoint.")
    logger.info("Use /checkpoint/finalize to proceed to Agent 8 when ready.")

    # Return updated job (still at checkpoint)
    job = pipeline_service.get_job(job_id)
    return pipeline_service.to_response(job)


@router.post("/checkpoint/finalize/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def finalize_checkpoint(
    request: Request,
    job_id: str,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Finalize the checkpoint and proceed to Agent 8 (Variation Generator) using Celery

    This endpoint should be called after all assets are approved.
    Only the approved assets will be processed by Agent 8 for variation generation.

    Requirements:
    - Job must be at waiting_for_human_approval status
    - At least one asset must be approved (via /checkpoint/approve)

    Updated to use Celery: Dispatches resume task to SQS instead of background task.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "waiting_for_human_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not waiting for approval. Current status: {job.status}"
        )

    # Check if any assets are approved
    approved_assets = job.approved_assets_list or []
    if not approved_assets:
        raise HTTPException(
            status_code=400,
            detail="No assets have been approved yet. Please approve at least one asset before finalizing."
        )

    logger.info(f"Finalizing checkpoint for job {job_id}")
    logger.info(f"{len(approved_assets)} approved assets will proceed to Agent 8")

    # Get project data to reconstruct full workflow state
    project = project_service.get_project(job.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {job.project_id} not found")

    # Helper function to safely extract nested agent output
    def safe_get_agent_output(agent_outputs, agent_key, output_key, default=None):
        """Safely extract agent output, handling None values"""
        agent_data = agent_outputs.get(agent_key)
        if not agent_data or not isinstance(agent_data, dict):
            return default if default is not None else {}

        output = agent_data.get("output")
        if not output or not isinstance(output, dict):
            return default if default is not None else {}

        return output.get(output_key, default if default is not None else {})

    # Get agent_outputs dictionary
    agent_outputs = project.get("agent_outputs", {})

    # Debug: Check what's in agent5 output
    logger.debug("Checking Agent 5 output structure...")
    agent5_output = safe_get_agent_output(agent_outputs, "agent5", "generated_images")
    if agent5_output:
        for asset_type in ["characters", "locations", "props"]:
            assets = agent5_output.get(asset_type, [])
            logger.debug(f"Agent 5 {asset_type}: {len(assets)} items")
            if isinstance(assets, list):
                for idx, asset in enumerate(assets[:3]):  # Show first 3
                    logger.debug(f"[{idx}] id={asset.get('id')}, name={asset.get('name')}")
    else:
        logger.warning("Agent 5 generated_images is empty or None")

    # Debug: Check what's in agent7 output
    logger.debug("Checking Agent 7 output structure...")
    agent7_output = safe_get_agent_output(agent_outputs, "agent7", "edited_images")
    if agent7_output:
        logger.debug(f"Agent 7 edited_images keys: {list(agent7_output.keys())[:10]}")  # Show first 10 keys
    else:
        logger.warning("Agent 7 edited_images is empty or None")

    # Create minimal workflow state for Celery task (to avoid SQS 256KB message size limit)
    # Agent 8 will load full data from project when needed
    current_state = {
        # Job tracking fields
        "job_id": job.job_id,
        "project_id": job.project_id,  # Critical: Agent 8 uses this to load data
        "current_agent": "agent_8",
        "pipeline_status": "generating_variations",
        "agent1_status": job.agent1_status,
        "agent2_status": job.agent2_status,
        "agent3_status": job.agent3_status,
        "agent4_status": job.agent4_status,
        "agent5_status": job.agent5_status,
        "agent6_status": job.agent6_status,
        "agent7_status": job.agent7_status,
        "agent8_status": "pending",
        "regeneration_count": job.regeneration_count,
        "output_files": job.output_files,

        # Human approval
        "human_approval_decision": "approve",
        "human_approval_feedback": job.human_approval_feedback,

        # Asset-level approval tracking (use all approved assets)
        "approved_asset_ids": approved_assets,

        # Script content (small, safe to include)
        "script_content": project.get("script", ""),
    }

    # Resume workflow using Celery (Agent 8 variation generation)
    # Dispatch to SQS queue for processing by Celery worker
    queue_name = get_workflow_queue_name()
    task = resume_phase1_workflow_task.apply_async(
        args=[job_id, current_state],
        queue=queue_name,
        routing_key=queue_name,
    )

    logger.info(f"Proceeding to Agent 8 with {len(approved_assets)} approved assets")
    logger.info(f"Celery Task ID: {task.id}")

    # Update job with new Celery task ID (for resume operation)
    pipeline_service.update_job_celery_task_id(job_id, task.id)

    # Mark checkpoint as finalized
    pipeline_service.mark_checkpoint_finalized(job_id)

    # Return updated job (includes new celery_task_id for monitoring)
    job = pipeline_service.get_job(job_id)
    return pipeline_service.to_response(job)


@router.post("/checkpoint/edit-prompt/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def edit_asset_prompt(
    request: Request,
    job_id: str,
    edit_request: AssetPromptEdit,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Edit the prompt for a specific asset and re-run Agent 7 (Image Editor) using Celery

    This endpoint allows modifying the prompt for a single asset.
    The edited prompt and current image are sent to Agent 7 for re-processing.
    After Agent 7 completes, the workflow returns to the human checkpoint.

    Updated to use Celery: Dispatches resume task for Agent 7 re-processing.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status != "waiting_for_human_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not waiting for approval. Current status: {job.status}"
        )

    # Check regeneration limit
    if job.regeneration_count >= job.max_regenerations:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum regeneration attempts ({job.max_regenerations}) reached"
        )

    logger.info(f"Editing prompt for asset {edit_request.asset_id} ({edit_request.asset_type})")
    logger.info(f"New prompt: {edit_request.edited_prompt}")
    if edit_request.feedback:
        logger.info(f"Feedback: {edit_request.feedback}")

    # Get project data to reconstruct full workflow state
    project = project_service.get_project(job.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {job.project_id} not found")

    # Helper function to safely extract nested agent output
    def safe_get_agent_output(agent_outputs, agent_key, output_key, default=None):
        """Safely extract agent output, handling None values"""
        agent_data = agent_outputs.get(agent_key)
        if not agent_data or not isinstance(agent_data, dict):
            return default if default is not None else {}

        output = agent_data.get("output")
        if not output or not isinstance(output, dict):
            return default if default is not None else {}

        return output.get(output_key, default if default is not None else {})

    # Get agent_outputs dictionary
    agent_outputs = project.get("agent_outputs", {})

    # Get current optimized_prompts and update the specific asset
    optimized_prompts = safe_get_agent_output(agent_outputs, "agent4", "optimized_prompts")

    # Store the original prompt before updating (for Agent 7's context)
    original_prompt_text = ""
    asset_type_key = f"{edit_request.asset_type}s"  # Convert 'character' -> 'characters'
    if asset_type_key in optimized_prompts:
        for asset in optimized_prompts[asset_type_key]:
            if asset.get("id") == edit_request.asset_id:
                # Get the original prompt BEFORE updating
                final_prompt = asset.get("final_prompt", {})
                if isinstance(final_prompt, dict):
                    original_prompt_text = final_prompt.get("prompt", "")

                # Update the prompt (only the 'prompt' key, preserve negative_prompt and technical_specs)
                if "final_prompt" not in asset or not isinstance(asset["final_prompt"], dict):
                    asset["final_prompt"] = {}
                asset["final_prompt"]["prompt"] = edit_request.edited_prompt
                logger.info(f"Updated prompt for asset {edit_request.asset_id}")
                break
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Asset {edit_request.asset_id} not found in {asset_type_key}"
            )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Asset type {asset_type_key} not found in optimized_prompts"
        )

    # Set human approval in database
    pipeline_service.set_human_approval(
        job_id,
        "edit_prompt",
        {
            "asset_id": edit_request.asset_id,
            "asset_type": edit_request.asset_type,
            "edited_prompt": edit_request.edited_prompt,
            "feedback": edit_request.feedback
        }
    )

    # Reconstruct full workflow state from project data
    current_state = {
        # Job tracking fields
        "job_id": job.job_id,
        "project_id": job.project_id,
        "current_agent": "agent_5",  # Route to agent_5 for regeneration with new prompt
        "pipeline_status": "regenerating_asset_image",
        "agent1_status": job.agent1_status,
        "agent2_status": job.agent2_status,
        "agent3_status": job.agent3_status,
        "agent4_status": job.agent4_status,
        "agent5_status": "pending",  # Will regenerate image with new prompt
        "agent6_status": "pending",  # Will review the regenerated image
        "agent7_status": "pending",  # May edit after review if needed
        "agent8_status": "pending",
        "regeneration_count": job.regeneration_count + 1,
        "output_files": job.output_files,

        # Human approval
        "human_approval_decision": "edit_prompt",
        "human_approval_feedback": {
            "asset_id": edit_request.asset_id,
            "asset_type": edit_request.asset_type,
            "edited_prompt": edit_request.edited_prompt,
            "feedback": edit_request.feedback
        },

        # Track which asset needs regeneration with the new prompt
        "needs_regeneration_assets": [f"{edit_request.asset_type}s:{edit_request.asset_id}"],

        # Enable selective review - only review the edited asset
        "recently_edited_asset_ids": [edit_request.asset_id],

        # Script content (small, safe to include)
        "script_content": project.get("script", ""),

        # Optimized prompts with the updated prompt (needed for regeneration)
        "optimized_prompts": optimized_prompts,
    }

    # Note: We don't need to create synthetic reviews since agent_5 will regenerate the image
    # and agent_6 will create fresh reviews after regeneration

    # Resume workflow using Celery (regenerate with agent_5)
    # Dispatch to SQS queue for processing by Celery worker
    queue_name = get_workflow_queue_name()
    task = resume_phase1_workflow_task.apply_async(
        args=[job_id, current_state],
        queue=queue_name,
        routing_key=queue_name,
    )

    logger.info("Dispatched image regeneration task to Celery (prompt was edited)")
    logger.info(f"Asset: {edit_request.asset_id} ({edit_request.asset_type})")
    logger.info(f"New prompt: {edit_request.edited_prompt[:100]}...")
    logger.info(f"Celery Task ID: {task.id}")

    # Update job with new Celery task ID (for regeneration operation)
    pipeline_service.update_job_celery_task_id(job_id, task.id)

    # Return updated job (includes new celery_task_id for monitoring)
    job = pipeline_service.get_job(job_id)
    return pipeline_service.to_response(job)


@router.get("/outputs/{job_id}")
@limiter.limit("100/minute")
async def get_output_files(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get list of output files generated by the workflow

    Returns paths to all JSON output files from agents 1-6.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {
        "job_id": job.job_id,
        "output_files": job.output_files,
        "total_files": len(job.output_files)
    }


@router.get("/failed-assets/{job_id}")
@limiter.limit("100/minute")
async def get_failed_assets(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get list of failed asset generations for a job

    Args:
        job_id: The pipeline job ID

    Returns:
        List of failed generations with retry information
    """
    try:
        job = pipeline_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get failed generations from state
        state = job.state if hasattr(job, 'state') else {}
        failed_generations = state.get("failed_generations", [])

        return JSONResponse(content={
            "job_id": job_id,
            "failed_count": len(failed_generations),
            "failed_assets": failed_generations
        })

    except Exception as e:
        raise handle_api_exception(e, "get_failed_assets")


@router.post("/retry-asset/{job_id}")
@limiter.limit("10/minute")
async def retry_failed_asset(
    request: Request,
    job_id: str,
    asset_id: str,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Retry a single failed asset generation

    Args:
        job_id: The pipeline job ID
        asset_id: UUID of the failed asset

    Returns:
        Result of the retry attempt with success status and details
    """
    try:
        # Get the job
        job = pipeline_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        # Get project to access optimized prompts
        project = project_service.get_project(job.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get state
        state = job.state if hasattr(job, 'state') else {}

        # Verify job has completed agent 5
        agent5_status = state.get("agent5_status")
        if agent5_status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"Agent 5 hasn't run yet or is still running. Status: {agent5_status}"
            )

        # Look up the asset in optimized_prompts to get name and type
        agent_outputs = project.get("agent_outputs", {})
        agent4_output = agent_outputs.get("agent4", {}).get("output", {})
        optimized_prompts = agent4_output.get("optimized_prompts", {})

        asset_name = None
        asset_type = None

        # Search for asset by ID
        for asset_type_key in ["characters", "locations", "props"]:
            assets = optimized_prompts.get(asset_type_key, [])
            for asset in assets:
                if asset.get("id") == asset_id:
                    asset_name = asset.get("name")
                    # Convert plural to singular (characters -> character)
                    asset_type = asset_type_key.rstrip('s')
                    break
            if asset_name:
                break

        if not asset_name or not asset_type:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with ID {asset_id} not found in project"
            )

        # Initialize Agent 5 with the saved state
        from app.services.phase_1_agents.agent_5_image_generator import ImageGeneratorAgent

        api_key = os.getenv("FREEPIK_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="FREEPIK_API_KEY not configured")

        agent = ImageGeneratorAgent(api_key=api_key)

        # Restore agent state from job
        agent.final_prompts = state.get("optimized_prompts", {})
        agent.generated_images = state.get("generated_images", {
            "characters": {}, "locations": {}, "props": {}
        })
        agent.failed_generations = state.get("failed_generations", [])

        logger.info(f"Retrying asset: {asset_name} ({asset_type}) [ID: {asset_id}]")
        logger.info(f"Current failed count: {len(agent.failed_generations)}")

        # Retry the asset
        result = agent.retry_single_asset(
            asset_name=asset_name,
            asset_type=asset_type
        )

        if result['success']:
            # Update job state with new generated images and updated failed list
            state["generated_images"] = agent.generated_images
            state["failed_generations"] = agent.failed_generations

            # Save updated state
            pipeline_service.update_job_state(job_id, state)

            # Also update project agent output
            project_id = state.get("project_id")
            if project_id:
                try:
                    project_service.update_agent_output(
                        project_id=project_id,
                        agent_number=5,
                        status="completed",
                        output={
                            "generated_images": agent.generated_images,
                            "failed_generations": agent.failed_generations,
                            "metadata_file": state.get("output_files", [])[-1] if state.get("output_files") else None
                        }
                    )
                    logger.info("Updated project agent output in MongoDB")
                except Exception as db_error:
                    logger.warning(f"Failed to update MongoDB: {db_error}")

            logger.info(f"Retry successful! Remaining failed: {len(agent.failed_generations)}")
        else:
            logger.error(f"Retry failed: {result.get('error')}")

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying failed asset: {type(e).__name__}", exc_info=True)
        raise handle_api_exception(e, "retry_failed_asset")


# ============================================================================
# Celery Task Monitoring Endpoints
# ============================================================================

@router.get("/task-status/{task_id}")
@limiter.limit("100/minute")
async def get_task_status(request: Request, task_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get real-time status of a Celery task

    This endpoint allows monitoring of task progress, errors, and completion.
    Frontend should poll this endpoint while task is running.

    Args:
        task_id: Celery task ID (returned in celery_task_id field from /start)

    Returns:
        {
            "task_id": "abc-123",
            "status": "PENDING" | "STARTED" | "PROGRESS" | "SUCCESS" | "FAILURE",
            "result": {...},  # Only present when status = SUCCESS
            "error": "...",   # Only present when status = FAILURE
            "progress": {     # Only present when status = PROGRESS
                "current_agent": "agent_5",
                "progress": 62.5,
                "message": "Generating images...",
                "job_id": "...",
                "project_id": "..."
            }
        }

    Why this endpoint?
    -----------------
    - Real-time progress updates (shows which agent is running)
    - Better UX than polling job status alone
    - See errors immediately without waiting for job to fail
    - Frontend can show progress bars, loading indicators

    Example usage (frontend):
    -------------------------
    ```javascript
    // 1. Start workflow
    const response = await fetch('/api/v1/phase1/start', {...})
    const { job_id, celery_task_id } = await response.json()

    // 2. Poll task status every 2 seconds
    const interval = setInterval(async () => {
        const taskStatus = await fetch(`/api/v1/phase1/task-status/${celery_task_id}`)
        const data = await taskStatus.json()

        if (data.status === 'PROGRESS') {
            updateProgressBar(data.progress.progress)
            showCurrentAgent(data.progress.current_agent)
        } else if (data.status === 'SUCCESS') {
            clearInterval(interval)
            showSuccess()
        } else if (data.status === 'FAILURE') {
            clearInterval(interval)
            showError(data.error)
        }
    }, 2000)
    ```
    """
    try:
        # Query Celery result backend for task status
        task_result = AsyncResult(task_id, app=celery_app)

        response = {
            "task_id": task_id,
            "status": task_result.state,  # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE
        }

        # Add status-specific information
        if task_result.state == 'PENDING':
            response["message"] = "Task is queued in SQS, waiting for worker"

        elif task_result.state == 'STARTED':
            response["message"] = "Task picked up by worker, starting execution"

        elif task_result.state == 'PROGRESS':
            # Task is running and reporting progress
            response["progress"] = task_result.info  # Dict from self.update_state()
            response["message"] = task_result.info.get('message', 'Processing...')

        elif task_result.state == 'SUCCESS':
            # Task completed successfully
            response["result"] = task_result.result  # Return value from task
            response["completed_at"] = str(datetime.utcnow())

        elif task_result.state == 'FAILURE':
            # Task failed
            response["error"] = str(task_result.info)  # Exception message
            response["traceback"] = task_result.traceback if hasattr(task_result, 'traceback') else None

        elif task_result.state == 'RETRY':
            # Task is being retried
            response["message"] = f"Task failed, retrying... (attempt {task_result.info.get('retries', 0)})"
            response["error"] = str(task_result.info.get('exc', ''))

        elif task_result.state == 'REVOKED':
            # Task was cancelled
            response["message"] = "Task was cancelled"

        return JSONResponse(content=response)

    except Exception as e:
        raise handle_api_exception(e, "get_task_status")


@router.post("/cancel-task/{task_id}")
@limiter.limit("10/minute")
async def cancel_task(request: Request, task_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Cancel a running Celery task

    This endpoint allows cancelling a task that is currently queued or running.
    The task will be revoked and marked as cancelled.

    Args:
        task_id: Celery task ID to cancel

    Returns:
        {
            "task_id": "abc-123",
            "status": "cancelled",
            "message": "Task has been cancelled"
        }

    Why this endpoint?
    -----------------
    - Allow users to stop long-running tasks
    - Free up worker resources
    - Prevent unnecessary processing

    Note: Tasks that have already started may take a moment to stop gracefully.
    """
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')

        logger.info(f"Cancelled task: {task_id}")

        return JSONResponse(content={
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task has been cancelled. May take a moment to stop if already running."
        })

    except Exception as e:
        raise handle_api_exception(e, "cancel_task")


@router.get("/task-info/{job_id}")
@limiter.limit("100/minute")
async def get_task_info(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get Celery task information for a specific job

    Convenience endpoint that looks up the celery_task_id from job_id
    and returns task status. Useful when you only have the job_id.

    Args:
        job_id: Pipeline job ID

    Returns:
        Same as /task-status/{task_id}

    Why this endpoint?
    -----------------
    - Convenience: no need to store celery_task_id separately
    - Can query task status using just job_id
    - Matches existing /status/{job_id} pattern
    """
    try:
        # Get job to retrieve celery_task_id
        job = pipeline_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if not job.celery_task_id:
            return JSONResponse(content={
                "job_id": job_id,
                "message": "No Celery task associated with this job (may be old job created before Celery migration)"
            })

        # Forward to task status endpoint
        task_result = AsyncResult(job.celery_task_id, app=celery_app)

        response = {
            "job_id": job_id,
            "task_id": job.celery_task_id,
            "status": task_result.state,
        }

        if task_result.state == 'PROGRESS':
            response["progress"] = task_result.info
        elif task_result.state == 'SUCCESS':
            response["result"] = task_result.result
        elif task_result.state == 'FAILURE':
            response["error"] = str(task_result.info)

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "get_task_info_by_job")
