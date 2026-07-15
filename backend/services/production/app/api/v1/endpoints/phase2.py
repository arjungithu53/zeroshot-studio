"""
Phase 2 Workflow API Endpoints
================================
FastAPI endpoints for managing Phase 2 workflow jobs.

MAJOR CHANGE: Celery + SQS Integration
---------------------------------------
This file has been updated to use Celery with Amazon SQS instead of FastAPI BackgroundTasks.

What changed:
- Removed: BackgroundTasks dependency
- Added: Celery task dispatching via apply_async()
- Added: Task ID tracking in job records
- Added: Task status monitoring endpoint
- Added: Phase 1 dependency checking

Why this change?
- Reliability: Tasks survive server restarts (stored in SQS)
- Scalability: Multiple workers can process tasks in parallel
- Monitoring: Full task progress tracking and error handling
- Resource separation: Heavy processing doesn't block API server

Available endpoints:
- /start: Start Phase 2 workflow (requires completed Phase 1)
- /approve-strategy/{job_id}: Approve/reject strategies
- /approve-prompts/{job_id}: Approve/reject corrected prompts
- /final-approve/{job_id}: Final approval after edit loop
- /status/{job_id}: Get job status
- /results/{job_id}: Get job results
- /outputs/{job_id}: Get output files
- /task-status/{task_id}: Get Celery task status
- /task-info/{job_id}: Get Celery task info by job_id
- /cancel-task/{task_id}: Cancel running Celery task
- /mongodb/stats: Get MongoDB statistics
- /mongodb/shots/{show_id}/{episode_number}: Get shots from MongoDB
"""

import io
import os
import sys
import zipfile
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from pathlib import Path
from shared.auth.dependencies import validate_admin_from_header, AdminUser

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception
from backend.shared.models.responses import ApiResponse

# Initialize logger for this module
logger = get_logger(__name__)

from app.models.mongodb.pipelines import (
    PipelineJobCreate,
    PipelineJobResponse,
    HumanApprovalRequest
)
from app.services.pipeline_service import PipelineService
from app.models.mongodb.shots import MongoDBAtlasClient
from app.core.quota import QuotaManager, get_quota_manager

# Import Celery tasks (replacing background functions)
from app.tasks.phase2_tasks import (
    run_phase2_workflow_task,
    resume_phase2_after_strategy_approval_task,
    resume_phase2_after_prompt_approval_task,
    resume_phase2_after_final_approval_task,
    check_phase1_completion
)
from app.services.phase_2_agents.imagen_generator_agent import ImagenGeneratorAgent

# Import Celery app for task status queries
from app.celery_app import celery_app
from celery.result import AsyncResult

# Import config for queue name
from app.config import get_workflow_queue_name

# Import rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from infrastructure.s3.streaming import parse_s3_url


# ============================================================================
# Request/Response Models
# ============================================================================

class ShotItemRequest(BaseModel):
    """Individual shot item request model."""
    shot_id: str = Field(..., description="Unique identifier for the shot")
    description: str = Field(..., description="Detailed description of the shot content")
    duration: Optional[float] = Field(None, description="Shot duration in seconds")
    scene_number: Optional[int] = Field(None, description="Scene number this shot belongs to")
    sequence_number: Optional[int] = Field(None, description="Sequence number within the scene")
    shot_style: Optional[str] = Field(None, description="Shot style (e.g., close_up, wide_shot, medium_shot)")
    camera_movement: Optional[str] = Field(None, description="Camera movement (e.g., push_in, pan, zoom)")
    source_type: str = Field("generated", description="Source type: generated or uploaded")
    uploaded_image_id: Optional[str] = Field(None, description="ObjectId from storyboard_images if source_type is 'uploaded'")
    generated_image_id: Optional[str] = Field(None, description="ObjectId of generated image if source_type is 'generated'")
    generated_video_id: Optional[str] = Field(None, description="ObjectId of generated video")
    optimized_ai_notes: Optional[str] = Field(None, description="Optimized AI notes for image/video generation")
    characters: Optional[list[str]] = Field(None, description="List of character names appearing in this shot (from CSV)")
    locations: Optional[str] = Field(None, description="Location name for this shot (from CSV)")
    product_present: bool = Field(False, description="Whether the product appears in this shot (from CSV product_present yes/no column)")


class ShotListRequest(BaseModel):
    """Complete episode shot list request."""
    episode_id: str = Field(..., description="Unique identifier for the episode")
    title: Optional[str] = Field(None, description="Episode title")
    shots: list[ShotItemRequest] = Field(..., description="List of all shots in the episode")
    scene_description: Optional[str] = Field(None, description="Overall scene description")


class StrategyApprovalRequest(BaseModel):
    """Strategy approval request model."""
    show_id: str = Field(..., description="Show identifier")
    episode_number: int = Field(..., description="Episode number")
    approval_status: bool = Field(..., description="Approval status (true/false)")
    feedback: Optional[Union[str, Dict[str, Any]]] = Field(
        None, 
        description="Optional feedback comments. Can be a string or a dictionary."
    )


class PromptApprovalRequest(BaseModel):
    """Prompt approval request model (for Agent 13 corrected prompts)."""
    show_id: str = Field(..., description="Show identifier (MongoDB project_id)")
    approval_status: bool = Field(..., description="Approval status (true/false)")
    feedback: Optional[Union[str, Dict[str, Any]]] = Field(
        None, 
        description="Optional feedback comments. Can be a string or a dictionary."
    )


class FinalApprovalRequest(BaseModel):
    """Final image approval request model (for all completed images after edit loop)."""
    show_id: str = Field(..., description="Show identifier")
    episode_number: int = Field(..., description="Episode number")
    approval_status: bool = Field(..., description="Final approval status (true/false)")
    feedback: Optional[Union[str, Dict[str, Any]]] = Field(
        None, 
        description="Optional feedback comments. Can be a string or a dictionary."
    )


class Phase2StartRequest(BaseModel):
    """Request model for starting Phase 2 workflow with Celery."""
    shot_list: ShotListRequest = Field(..., description="Shot list request")
    show_id: str = Field(..., description="Show identifier (MongoDB project_id)")
    episode_number: int = Field(..., description="Episode number")
    project_id: str = Field(..., description="Phase 1 project_id (REQUIRED - must be completed)")
    scene_description: Optional[str] = Field(None, description="Overall scene description")
    movie_id: str = Field(..., description="Movie ID to fetch visual_style from movies collection")


class ShotStrategyUpdateRequest(BaseModel):
    """Request payload for updating a single shot's generation strategy."""

    generation_strategy: Literal["multi_shot", "last_frame_seed", "generate_new"] = Field(
        ..., description="Updated generation strategy"
    )
    optimized_ai_notes: Optional[str] = Field(
        None, description="Optional optimized AI notes override"
    )
    confidence_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score override (0.0 - 1.0)"
    )
    seed_shot_id: Optional[str] = Field(
        None, description="Seed shot identifier when using last_frame_seed strategy"
    )
    continuity_notes: Optional[str] = Field(
        None, description="Continuity notes for the updated strategy"
    )


class ManualShotImageGenerationRequest(BaseModel):
    """Manual request to trigger Agent 14 with the latest prompt."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Latest or edited prompt fetched from the shots collection image field"
    )
    storyboard_s3_link: Optional[str] = Field(
        None,
        description="Optional S3 link to storyboard image to use as reference during generation"
    )


class ImageSelectionRequest(BaseModel):
    """Request body for saving a human's image selection for a shot."""

    version: str = Field(..., description="Version key, e.g. 'v1'")
    index: int = Field(..., ge=0, description="Index within that version's generated_images_s3 array")
    url: str = Field(..., description="The chosen S3 image URL")


async def start_phase2_pipeline_job(
    phase2_request: Phase2StartRequest,
) -> PipelineJobResponse:
    """
    Shared implementation that wires up Phase 2 jobs without going through the HTTP layer.

    Used by both the public /phase2/start endpoint and internal callers (e.g., movie bootstrapper).
    """
    try:
        # CRITICAL: Verify Phase 1 completion BEFORE creating job
        logger.info(f"Verifying Phase 1 completion for show: {phase2_request.show_id}")
        try:
            check_phase1_completion(
                project_id=phase2_request.project_id,
                show_id=phase2_request.show_id
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Phase 1 dependency check failed: {str(e)}"
            )

        # Create pipeline job
        job_data = PipelineJobCreate(project_id=phase2_request.project_id)
        job_result = pipeline_service.create_job(job_data)
        job_id = job_result["job_id"]

        # Dispatch Celery task
        queue_name = get_workflow_queue_name()
        task = run_phase2_workflow_task.apply_async(
            args=[
                job_id,
                phase2_request.shot_list.model_dump(),
                phase2_request.show_id,
                phase2_request.episode_number,
                phase2_request.project_id,
                phase2_request.scene_description,
                phase2_request.movie_id,
            ],
            kwargs={},
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched Phase 2 workflow to Celery")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Celery Task ID: {task.id}")
        logger.info(f"Queue: {queue_name}")

        # Update job with Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)

        # Return job info
        job = pipeline_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Failed to load pipeline job")
        return pipeline_service.to_response(job)

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, logger, context="start_phase2_pipeline_job")


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/phase2", tags=["Phase 2 Workflow"])
pipeline_service = PipelineService()

# Initialize limiter (will use the one from app.state in practice)
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# MongoDB Client Dependency
# ============================================================================

def get_mongodb_client() -> Optional[MongoDBAtlasClient]:
    """
    Get the MongoDB Atlas client singleton.

    This function returns the singleton instance from config.py to ensure
    only ONE MongoDB connection is used across the entire application,
    preventing connection pool exhaustion.

    Returns:
        MongoDBAtlasClient singleton instance or None if not configured
    """
    try:
        from app.config import get_mongodb_atlas_client
        return get_mongodb_atlas_client()
    except (ValueError, ConnectionError) as e:
        logger.warning(f"MongoDB Atlas client not configured or connection failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to get MongoDB Atlas client: {str(e)}")
        return None


# ============================================================================
# Background Task Functions - REMOVED
# ============================================================================
# The following functions have been moved to app/tasks/phase2_tasks.py as Celery tasks:
# - run_phase2_background → run_phase2_workflow_task
# - run_strategy_agent_background → run_phase2_workflow_task
#
# Why moved to Celery?
# -------------------
# - Tasks persist in SQS (survive server restarts)
# - Distributed execution across multiple workers
# - Built-in retry logic and error handling
# - Real-time progress tracking
# - Separate resource allocation from API server
# - Phase 1 dependency checking
# ============================================================================


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/start", response_model=PipelineJobResponse, status_code=202)
@limiter.limit("10/minute")
async def start_phase2_workflow(
    request: Request,
    phase2_request: Phase2StartRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager)
):
    """
    Start a new Phase 2 workflow job using Celery + SQS

    **Quota Enforcement:** This endpoint consumes 1 quota unit

    This endpoint:
    1. Verifies Phase 1 is complete (CRITICAL DEPENDENCY)
    2. Creates a new pipeline job to track execution
    3. Dispatches Celery task to SQS queue (non-blocking)
    4. Returns immediately with job_id and celery_task_id

    The workflow will pause at the first human checkpoint (after Agent 1).

    CRITICAL DEPENDENCY: Phase 1 must be completed
    ------------------------------------------------
    Phase 2 depends on Phase 1 because:
    - Agent 12 (Shot Design) needs asset library from Phase 1's Agent 8
    - Phase 2 agents reference character/location/prop assets

    Response includes celery_task_id for monitoring:
    - Poll: GET /api/v1/phase2/task-status/{celery_task_id}
    """
    # Enforce quota before starting expensive Phase 2 workflow
    quota_manager.consume(
        user_id=admin_user.user_id,
        pipeline_name="production_workflow"
    )
    logger.info(f"Quota consumed for user {admin_user.user_id} (start Phase 2)")
    try:
        return await start_phase2_pipeline_job(phase2_request)
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, logger, context="start_phase2_workflow")


# ============================================================================
# DEPRECATED ENDPOINTS (kept for backward compatibility)
# ============================================================================
# The following endpoints have been deprecated in favor of Celery-based endpoints.
# They are kept for backward compatibility but will be removed in a future version.
# Please use the new endpoints instead:
# - /start (replaces /run-strategy-agent)
# - /approve-strategy/{job_id} (updated to use Celery)
# ============================================================================

@router.post("/run-strategy-agent", status_code=410)
async def run_strategy_agent_deprecated():
    """
    DEPRECATED: This endpoint has been replaced by /start

    Please use POST /phase2/start instead, which:
    - Uses Celery for reliable task execution
    - Requires Phase 1 project_id
    - Provides full workflow support with human checkpoints

    This endpoint will be removed in a future version.
    """
    raise HTTPException(
        status_code=410,
        detail={
            "error": "Endpoint deprecated",
            "message": "This endpoint has been replaced. Please use POST /phase2/start instead.",
            "new_endpoint": "/api/v1/phase2/start",
            "documentation": "See API docs for new request format"
        }
    )


@router.post("/approve-strategy/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def approve_strategy(
    request: Request,
    job_id: str,
    approval: StrategyApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Approve/reject strategies and resume workflow using Celery (Checkpoint 1)

    This endpoint should be called after Agent 1 (Shot Strategy) completes.

    If approved: Continues to Agent 2 (Image Prompt Generator)
    If rejected: Ends workflow with status "rejected"

    Updated to use Celery: Dispatches resume task to SQS instead of background task.
    Includes idempotency to prevent duplicate approvals.
    """
    try:
        # ===== STEP 0: Idempotency Check =====
        from app.core.idempotency import (
            get_idempotency_service,
            generate_idempotency_key,
            check_idempotency,
            mark_idempotency_completed,
            mark_idempotency_failed,
        )
        
        # Generate idempotency key: job_id + checkpoint identifier
        idempotency_key_value = generate_idempotency_key(
            user_id=admin_user.user_id if admin_user else None,
            scene_id=job_id,
            phase_number=2,
            idempotency_key_header=request.headers.get("Idempotency-Key"),
        )
        idempotency_key_value = f"{idempotency_key_value}:strategy_approval"
        
        # Build payload (exclude feedback to allow different feedback with same approval)
        payload = {
            "job_id": job_id,
            "approval_status": approval.approval_status,
            "show_id": approval.show_id,
            "episode_number": approval.episode_number,
        }
        
        # Check idempotency
        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase2.approve_strategy",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )
        
        if is_duplicate:
            if cached_response:
                logger.info(f"Returning cached response for strategy approval: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase2.approve_strategy",
                    key=idempotency_key_value
                )
                
                if record and record.task_id:
                    logger.warning(f"Strategy approval already processing with task ID: {record.task_id}")
                    job = pipeline_service.get_job(job_id)
                    if job:
                        return pipeline_service.to_response(job)
                else:
                    logger.info(f"No task_id in record yet - proceeding with approval")
        
        job = pipeline_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.status != "waiting_for_human_approval":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not waiting for approval. Current status: {job.status}"
            )

        logger.info(f"Strategy approval for job {job_id}: {'APPROVED' if approval.approval_status else 'REJECTED'}")

        # Get show_id and episode_number from job state
        state = job.state if (hasattr(job, 'state') and job.state is not None) else {}
        show_id = state.get('show_id', approval.show_id)  # Fall back to request data
        episode_number = state.get('episode_number', approval.episode_number)

        # Convert string feedback to dict format if needed
        feedback = approval.feedback
        if isinstance(feedback, str):
            feedback = {"comment": feedback}
        elif feedback is None:
            feedback = {}

        # Dispatch Celery resume task
        queue_name = get_workflow_queue_name()
        task = resume_phase2_after_strategy_approval_task.apply_async(
            args=[
                job_id,
                show_id,
                episode_number,
                approval.approval_status,
                feedback
            ],
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched strategy approval resume task")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Decision: {'APPROVED' if approval.approval_status else 'REJECTED'}")
        logger.info(f"Celery Task ID: {task.id}")

        # Update job with new Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)
        
        # Attach task reference to idempotency record
        try:
            idempotency_service.attach_task_reference(
                endpoint="phase2.approve_strategy",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
            )
        except Exception as e:
            logger.warning(f"Failed to attach task reference to idempotency record: {e}")

        # Return updated job
        job = pipeline_service.get_job(job_id)
        response = pipeline_service.to_response(job)
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase2.approve_strategy",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
                response_payload=response.dict() if hasattr(response, 'dict') else response,
            )
        except Exception as e:
            logger.warning(f"Failed to mark idempotency as completed: {e}")
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        # Mark idempotency as failed
        try:
            from app.core.idempotency import mark_idempotency_failed, generate_idempotency_key
            
            idempotency_key_value = generate_idempotency_key(
                user_id=admin_user.user_id if admin_user else None,
                scene_id=job_id,
                phase_number=2,
                idempotency_key_header=request.headers.get("Idempotency-Key"),
            )
            idempotency_key_value = f"{idempotency_key_value}:strategy_approval"
            
            mark_idempotency_failed(
                endpoint="phase2.approve_strategy",
                idempotency_key=idempotency_key_value,
                error_message=f"Strategy approval failed: {str(e)}",
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")
        
        raise handle_api_exception(e, logger, context="approve_strategy")



@router.get("/status/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("100/minute")
async def get_job_status(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get current status of a Phase 2 workflow job

    Returns job status, current agent, and completion info.
    Poll this endpoint to track workflow progress.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return pipeline_service.to_response(job)


@router.get("/results/{job_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_job_results(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get complete results of a Phase 2 workflow job

    Returns pipeline tracking info + all workflow outputs wrapped in ApiResponse.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Return combined data wrapped in ApiResponse
    return ApiResponse(
        success=True,
        data={
            "job_id": job.job_id,
            "project_id": job.project_id,
            "status": job.status,
            "pipeline_status": job.pipeline_status,
            "current_agent": job.current_agent,
            "agent_statuses": {
                "agent_1_strategy": job.agent1_status,
                "agent_2_prompt_generator": job.agent2_status,
                "agent_3_prompt_review": job.agent3_status,
            },
            "workflow_state": job.workflow_state,  # Contains all agent outputs
            "output_files": job.output_files,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at
        },
        error=None
    )


@router.get("/outputs/{job_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_output_files(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get list of output files generated by the workflow

    Returns paths to all JSON output files from agents wrapped in ApiResponse.
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return ApiResponse(
        success=True,
        data={
            "job_id": job.job_id,
            "output_files": job.output_files,
            "total_files": len(job.output_files)
        },
        error=None
    )


@router.get("/jobs/by-show/{show_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_jobs_by_show(request: Request, show_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get all Phase 2 jobs for a specific show/project

    Returns all jobs for this show, useful for:
    - Monitoring overall progress
    - Dashboard views
    - Discovering which episodes are being processed

    Args:
        show_id: Show/Project ID

    Returns:
        List of all Phase 2 jobs for the show wrapped in ApiResponse, grouped by episode_number
    """
    try:
        from backend.services.production.app.config import get_mongo_factory
        mongo_factory = get_mongo_factory()
        client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

        # Query for all Phase 2 jobs in this project
        # Phase2 jobs store show_id in state.show_id, and project_id should match show_id
        query = {
            "$or": [
                {"project_id": show_id},  # Direct project_id match
                {"state.show_id": show_id}  # Or in state field
            ]
        }

        # Find all matching jobs, sorted by creation time
        job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

        # Group jobs by episode_number
        jobs_by_episode = {}
        all_jobs = []

        for job_doc in job_docs:
            jid = job_doc.get("job_id")
            if not jid:
                continue

            job = pipeline_service.get_job(jid)
            if not job:
                continue

            # Get episode_number from state
            state = job.state if (hasattr(job, 'state') and job.state is not None) else {}
            episode_number = state.get("episode_number", 0)

            job_data = {
                "job_id": job.job_id,
                "project_id": job.project_id,
                "show_id": state.get("show_id", show_id),
                "episode_number": episode_number,
                "status": job.status,
                "pipeline_status": job.pipeline_status,
                "current_agent": job.current_agent,
                "agent_statuses": {
                    "agent_1_strategy": job.agent1_status,
                    "agent_2_prompt_generator": job.agent2_status,
                    "agent_3_prompt_review": job.agent3_status,
                    "agent_12_shot_design": state.get("agent12_status", "pending"),
                    "agent_13_prompt_modifier": state.get("agent13_status", "pending"),
                },
                "waiting_for_approval": job.status == "waiting_for_human_approval",
                "error_message": job.error_message,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "celery_task_id": job.celery_task_id
            }

            all_jobs.append(job_data)

            # Group by episode_number
            if episode_number not in jobs_by_episode:
                jobs_by_episode[episode_number] = []
            jobs_by_episode[episode_number].append(job_data)

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "total_jobs": len(all_jobs),
                "total_episodes": len(jobs_by_episode),
                "jobs": all_jobs,
                "jobs_by_episode": jobs_by_episode
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_jobs_by_show")


@router.get("/jobs/by-show/{show_id}/{episode_number}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_jobs_by_show_and_episode(request: Request, show_id: str, episode_number: int, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get all Phase 2 jobs for a specific show and episode

    Useful for:
    - Finding jobs for a specific scene/episode
    - Getting job_id for approval endpoints
    - Tracking episode-specific progress

    Args:
        show_id: Show/Project ID
        episode_number: Episode number

    Returns:
        List of Phase 2 jobs for the show/episode wrapped in ApiResponse
    """
    try:
        from backend.services.production.app.config import get_mongo_factory
        mongo_factory = get_mongo_factory()
        client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

        # Query for Phase 2 jobs matching show_id and episode_number
        query = {
            "$or": [
                {"project_id": show_id},
                {"state.show_id": show_id}
            ],
            "state.episode_number": episode_number
        }

        # Find all matching jobs, sorted by creation time (newest first)
        job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

        jobs = []
        for job_doc in job_docs:
            jid = job_doc.get("job_id")
            if not jid:
                continue

            job = pipeline_service.get_job(jid)
            if not job:
                continue

            state = job.state if (hasattr(job, 'state') and job.state is not None) else {}

            job_data = {
                "job_id": job.job_id,
                "project_id": job.project_id,
                "show_id": state.get("show_id", show_id),
                "episode_number": episode_number,
                "status": job.status,
                "pipeline_status": job.pipeline_status,
                "current_agent": job.current_agent,
                "agent_statuses": {
                    "agent_1_strategy": job.agent1_status,
                    "agent_2_prompt_generator": job.agent2_status,
                    "agent_3_prompt_review": job.agent3_status,
                    "agent_12_shot_design": state.get("agent12_status", "pending"),
                    "agent_13_prompt_modifier": state.get("agent13_status", "pending"),
                },
                "waiting_for_approval": job.status == "waiting_for_human_approval",
                "error_message": job.error_message,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "celery_task_id": job.celery_task_id
            }

            jobs.append(job_data)

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "episode_number": episode_number,
                "total_jobs": len(jobs),
                "jobs": jobs
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_jobs_by_show_and_episode")


@router.get("/jobs/waiting-for-approval/{show_id}/{episode_number}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_jobs_waiting_for_approval(request: Request, show_id: str, episode_number: int, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get Phase 2 jobs waiting for human approval for a specific show and episode

    This endpoint is specifically designed for the frontend approval button.
    Returns jobs that are in "waiting_for_human_approval" status, making it easy
    to find which jobs need approval.

    Args:
        show_id: Show/Project ID
        episode_number: Episode number

    Returns:
        List of jobs waiting for approval wrapped in ApiResponse
    """
    try:
        from backend.services.production.app.config import get_mongo_factory
        mongo_factory = get_mongo_factory()
        client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

        # Query for jobs waiting for approval
        query = {
            "$or": [
                {"project_id": show_id},
                {"state.show_id": show_id}
            ],
            "state.episode_number": episode_number,
            "status": "waiting_for_human_approval"
        }

        # Find all matching jobs, sorted by creation time (newest first)
        job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

        jobs = []
        for job_doc in job_docs:
            jid = job_doc.get("job_id")
            if not jid:
                continue

            job = pipeline_service.get_job(jid)
            if not job:
                continue

            state = job.state if (hasattr(job, 'state') and job.state is not None) else {}

            # Determine which checkpoint this job is waiting at
            checkpoint_type = "unknown"
            agent13_status = state.get("agent13_status", "pending")
            if job.current_agent == "human_checkpoint" or job.agent1_status == "completed":
                checkpoint_type = "strategy_approval"  # Waiting for strategy approval (after Agent 1)
            elif agent13_status == "completed":
                checkpoint_type = "prompt_approval"  # Waiting for prompt approval (after Agent 13)
            elif job.pipeline_status == "waiting_for_final_approval":
                checkpoint_type = "final_approval"  # Waiting for final approval

            job_data = {
                "job_id": job.job_id,
                "project_id": job.project_id,
                "show_id": state.get("show_id", show_id),
                "episode_number": episode_number,
                "status": job.status,
                "pipeline_status": job.pipeline_status,
                "current_agent": job.current_agent,
                "checkpoint_type": checkpoint_type,
                "agent_statuses": {
                    "agent_1_strategy": job.agent1_status,
                    "agent_2_prompt_generator": job.agent2_status,
                    "agent_3_prompt_review": job.agent3_status,
                    "agent_12_shot_design": state.get("agent12_status", "pending"),
                    "agent_13_prompt_modifier": agent13_status,
                },
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "celery_task_id": job.celery_task_id
            }

            jobs.append(job_data)

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "episode_number": episode_number,
                "total_jobs_waiting": len(jobs),
                "jobs": jobs
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_jobs_waiting_for_approval")


@router.get("/mongodb/stats", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_mongodb_stats(request: Request, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """Get MongoDB Atlas collection statistics wrapped in ApiResponse."""
    try:
        from app.config import get_shots_service
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB Atlas client not configured: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get shots service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize MongoDB client"
        )

    try:
        db_stats = shots_service.get_database_stats()
        shots_count = shots_service.get_shots_count()

        return ApiResponse(
            success=True,
            data={
                "database_stats": db_stats,
                "shots_count": shots_count
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_mongodb_stats")


@router.get("/mongodb/shots/{show_id}/{episode_number}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_shots_from_mongodb(request: Request, show_id: str, episode_number: int, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """Retrieve shots from MongoDB Atlas for a specific episode wrapped in ApiResponse."""
    try:
        from app.config import get_shots_service
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB Atlas client not configured: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get shots service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize MongoDB client"
        )

    try:
        shot_collection = shots_service.get_shots_from_atlas(show_id, episode_number)

        # Return 200 with empty data instead of 404 when no shots found
        if not shot_collection:
            return ApiResponse(
                success=True,
                data={
                    "show_id": show_id,
                    "episode_number": episode_number,
                    "total_shots": 0,
                    "shot_collection": None
                },
                error=None
            )

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "episode_number": episode_number,
                "total_shots": len(shot_collection.annotated_shots) if shot_collection.annotated_shots else 0,
                "shot_collection": shot_collection.model_dump() if hasattr(shot_collection, 'model_dump') else shot_collection
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_shots_from_mongodb")


@router.get(
    "/image-review/{movie_id}",
    response_model=ApiResponse[dict]
)
@limiter.limit("60/minute")
async def get_image_review_gallery(
    request: Request,
    movie_id: str,
    scene_number: Optional[int] = None,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Return all shots for a movie with their per-version image galleries and human selections.
    Walks movies → project_ids → shots to collect every scene's shots in order.
    Optionally filter by scene_number.
    """
    try:
        from app.config import get_shots_service, get_mongo_factory
        shots_service = get_shots_service()
        mongo_factory = get_mongo_factory()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"MongoDB Atlas client not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail="Failed to initialize MongoDB client")

    try:
        from bson import ObjectId

        _, movies_col = mongo_factory.get_collection("movies")
        movie_doc = movies_col.find_one({"_id": ObjectId(movie_id)})
        if not movie_doc:
            return ApiResponse(
                success=True,
                data={"movie_id": movie_id, "total_shots": 0, "reviewed_count": 0, "shots": []},
                error=None
            )

        # project_ids are ObjectIds; show_id in shots is stored as the string form
        project_ids = [str(pid) for pid in movie_doc.get("project_ids", [])]

        # Load all scene docs in episode_number order
        raw_shots = []
        scene_docs = list(
            shots_service.shots_collection.find(
                {"show_id": {"$in": project_ids}}
            ).sort("episode_number", 1)
        )
        for doc in scene_docs:
            raw_shots.extend(doc.get("annotated_shots") or doc.get("shots") or [])

        shots_out = []
        reviewed_count = 0

        for shot in raw_shots:
            if scene_number is not None and shot.get("scene_number") != scene_number:
                continue

            image_obj = shot.get("image") or {}

            versions = {}
            if isinstance(image_obj, dict):
                for k, v in image_obj.items():
                    if k.startswith("v") and k[1:].isdigit() and isinstance(v, dict):
                        versions[k] = v.get("generated_images_s3", [])

            selected = image_obj.get("selected") if isinstance(image_obj, dict) else None
            if selected:
                reviewed_count += 1

            shots_out.append({
                "shot_id": shot.get("shot_id"),
                "scene_number": shot.get("scene_number"),
                "sequence_number": shot.get("sequence_number"),
                "description": shot.get("description"),
                "generation_strategy": shot.get("generation_strategy"),
                "versions": versions,
                "selected": selected
            })

        return ApiResponse(
            success=True,
            data={
                "movie_id": movie_id,
                "total_shots": len(shots_out),
                "reviewed_count": reviewed_count,
                "shots": shots_out
            },
            error=None
        )
    except Exception as e:
        raise handle_api_exception(e, logger, context="get_image_review_gallery")


@router.post(
    "/image-review/{movie_id}/{shot_id}/select",
    response_model=ApiResponse[dict]
)
@limiter.limit("120/minute")
async def save_image_selection(
    request: Request,
    movie_id: str,
    shot_id: str,
    body: ImageSelectionRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Persist the human's chosen image for a shot.
    Finds the shot by shot_id across all scene documents for this movie.
    """
    try:
        from app.config import get_mongo_factory, get_shots_service
        mongo_factory = get_mongo_factory()
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"MongoDB Atlas client not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail="Failed to initialize MongoDB client")

    try:
        from bson import ObjectId

        # Resolve the correct show_id from the movie's project_ids so we never update
        # a shot belonging to a different movie (same shot_id can exist across movies).
        _, movies_col = mongo_factory.get_collection("movies")
        movie_doc = movies_col.find_one({"_id": ObjectId(movie_id)})
        if not movie_doc:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

        project_ids = [str(pid) for pid in movie_doc.get("project_ids", [])]

        # Find which scene document (show_id) actually contains this shot
        scene_doc = shots_service.shots_collection.find_one({
            "show_id": {"$in": project_ids},
            "annotated_shots.shot_id": shot_id,
        })
        if not scene_doc:
            # Try legacy shots field
            scene_doc = shots_service.shots_collection.find_one({
                "show_id": {"$in": project_ids},
                "shots.shot_id": shot_id,
            })
        if not scene_doc:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id} not found in movie {movie_id}")

        correct_show_id = scene_doc["show_id"]

        success = shots_service.set_shot_image_selection(
            show_id=correct_show_id,
            shot_id=shot_id,
            version=body.version,
            index=body.index,
            url=body.url,
        )
        if not success:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id} not found")

        return ApiResponse(
            success=True,
            data={"success": True, "shot_id": shot_id, "selected_url": body.url},
            error=None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, logger, context="save_image_selection")


@router.patch(
    "/shots/{show_id}/{episode_number}/{shot_id}",
    response_model=ApiResponse[dict]
)
@limiter.limit("60/minute")
async def update_shot_generation_strategy(
    request: Request,
    show_id: str,
    episode_number: int,
    shot_id: str,
    update_request: ShotStrategyUpdateRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Update generation strategy fields for a single shot within the shots collection.

    Parameters mirror the scene UI controls so users can switch between
    multi_shot, last_frame_seed, or generate_new strategies directly from the frontend.
    """
    try:
        from app.config import get_shots_service
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB Atlas client not configured: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get shots service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize MongoDB client"
        )

    try:
        updated_shot = shots_service.update_shot_generation_strategy(
            show_id=show_id,
            episode_number=episode_number,
            shot_id=shot_id,
            generation_strategy=update_request.generation_strategy,
            optimized_ai_notes=update_request.optimized_ai_notes,
            confidence_score=update_request.confidence_score,
            seed_shot_id=update_request.seed_shot_id,
            continuity_notes=update_request.continuity_notes
        )

        if not updated_shot:
            raise HTTPException(
                status_code=404,
                detail=f"Shot {shot_id} not found for show_id={show_id}, episode={episode_number}"
            )

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "episode_number": episode_number,
                "shot_id": shot_id,
                "shot": updated_shot
            },
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, logger, context="update_shot_generation_strategy")


@router.post(
    "/shots/{show_id}/{episode_number}/{shot_id}/upload-storyboard",
    response_model=ApiResponse[dict]
)
@limiter.limit("20/minute")
async def upload_storyboard_image(
    request: Request,
    show_id: str,
    episode_number: int,
    shot_id: str,
    storyboard: UploadFile = File(...),
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Upload a storyboard image for a specific shot to S3.

    This endpoint accepts a storyboard image file, uploads it to S3 using production credentials,
    and returns the S3 URL. The URL can then be used in the regenerate-image endpoint.

    S3 Configuration (from environment):
    - Bucket: production_S3_BUCKET_NAME (productionvideos)
    - Region: production_AWS_REGION (eu-north-1)
    - Credentials: production_AWS_ACCESS_KEY_ID, production_AWS_SECRET_ACCESS_KEY

    Args:
        show_id: Show identifier
        episode_number: Episode number
        shot_id: Shot identifier
        storyboard: Image file to upload (jpg, jpeg, png)

    Returns:
        ApiResponse with s3_url of the uploaded storyboard
    """
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        if storyboard.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
            )

        # Validate file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        content = await storyboard.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 10MB limit"
            )

        # Reset file pointer
        await storyboard.seek(0)

        # Generate S3 key with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = storyboard.filename.split('.')[-1] if storyboard.filename else 'jpg'
        safe_shot_id = shot_id.replace('.', '_').replace('/', '_')
        s3_key = f"storyboards/{show_id}/{episode_number}/{safe_shot_id}/storyboard_{timestamp}.{file_extension}"

        logger.info(f"Uploading storyboard for shot {shot_id} to S3: {s3_key}")

        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Upload to S3 using production credentials (automatically loaded from env)
            from app.config import upload_file_wrapper

            s3_url = upload_file_wrapper(
                file_path=temp_file_path,
                s3_key=s3_key,
                content_type=storyboard.content_type,
                use_presigned_url=False  # Return permanent S3 URL
            )

            logger.info(f"✓ Storyboard uploaded successfully to S3: {s3_url}")

            # Save storyboard S3 link to MongoDB immediately after upload
            try:
                from app.config import get_shots_service
                shots_service = get_shots_service()

                # Find the shot in MongoDB
                episode_doc = shots_service.shots_collection.find_one(
                    {"show_id": show_id, "episode_number": episode_number}
                )

                if episode_doc:
                    # Determine which field to use (annotated_shots or shots)
                    shots_field = None
                    for field_name in ("annotated_shots", "shots"):
                        field_value = episode_doc.get(field_name)
                        if isinstance(field_value, list):
                            for shot in field_value:
                                if shot.get("shot_id") == shot_id:
                                    shots_field = field_name
                                    break
                        if shots_field:
                            break

                    if shots_field:
                        # Update MongoDB with storyboard S3 link
                        query = {
                            "show_id": show_id,
                            "episode_number": episode_number,
                            f"{shots_field}.shot_id": shot_id
                        }
                        shots_service.shots_collection.update_one(
                            query,
                            {
                                "$set": {
                                    f"{shots_field}.$.storyboard_s3_link": s3_url,
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                            }
                        )
                        logger.info(f"✓ Saved storyboard S3 link to MongoDB ({shots_field}) for shot {shot_id}")
                    else:
                        logger.warning(f"Shot {shot_id} not found in MongoDB, skipping MongoDB save")
                else:
                    logger.warning(f"Episode not found in MongoDB (show_id={show_id}, episode={episode_number})")

            except Exception as mongo_error:
                # Don't fail the upload if MongoDB save fails
                logger.error(f"Failed to save storyboard to MongoDB: {mongo_error}")
                logger.info("Continuing with S3 URL response despite MongoDB error")

            return ApiResponse(
                success=True,
                data={
                    "s3_url": s3_url,
                    "show_id": show_id,
                    "episode_number": episode_number,
                    "shot_id": shot_id
                },
                error=None
            )

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload storyboard: {str(e)}")
        raise handle_api_exception(e, logger, context="upload_storyboard_image")


@router.post(
    "/shots/{show_id}/{episode_number}/{shot_id}/generate-image",
    response_model=ApiResponse[dict]
)
@limiter.limit("10/minute")
async def generate_shot_image_with_prompt(
    request: Request,
    show_id: str,
    episode_number: int,
    shot_id: str,
    generation_request: ManualShotImageGenerationRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Generate a new image version for a specific shot using a manually edited prompt.

    This endpoint bypasses the LangGraph flow and directly invokes Agent 14 to
    regenerate a single shot's image. The resulting S3 URL is stored under the next
    available image version in MongoDB with updated_prompt populated and all other
    metadata left null, per UI requirements.
    """
    try:
        from app.config import get_shots_service
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB Atlas client not configured: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get shots service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize MongoDB client"
        )

    prompt_text = generation_request.prompt.strip()
    if not prompt_text:
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    try:
        episode_doc = shots_service.shots_collection.find_one(
            {"show_id": show_id, "episode_number": episode_number}
        )

        if not episode_doc:
            raise HTTPException(
                status_code=404,
                detail=f"No shots found for show_id={show_id}, episode={episode_number}"
            )

        target_shot = None
        shots_field = None
        for field_name in ("annotated_shots", "shots"):
            field_value = episode_doc.get(field_name)
            if isinstance(field_value, list):
                for shot in field_value:
                    if shot.get("shot_id") == shot_id:
                        target_shot = shot
                        shots_field = field_name
                        break
            if target_shot:
                break

        if not target_shot or not shots_field:
            raise HTTPException(
                status_code=404,
                detail=f"Shot {shot_id} not found for show_id={show_id}, episode={episode_number}"
            )

        image_versions = target_shot.get("image") if isinstance(target_shot.get("image"), dict) else {}
        version_numbers = []
        if isinstance(image_versions, dict):
            for key in image_versions.keys():
                if key.startswith("v") and key[1:].isdigit():
                    version_numbers.append(int(key[1:]))
        next_version = (max(version_numbers) + 1) if version_numbers else 0
        version_key = f"v{next_version}"

        query = {
            "show_id": show_id,
            "episode_number": episode_number,
            f"{shots_field}.shot_id": shot_id
        }

        if target_shot.get("image") is None:
            shots_service.shots_collection.update_one(
                query,
                {
                    "$set": {
                        f"{shots_field}.$.image": {},
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )

        # Fetch corrected_assets from prompt_modifications (stored by Agent 13)
        prompt_modifications = target_shot.get("prompt_modifications", {})
        corrected_assets = prompt_modifications.get("corrected_assets", [])

        logger.info(f"Fetched {len(corrected_assets)} corrected_assets from prompt_modifications for shot {shot_id}")

        # Handle storyboard: save new one if provided, or fetch existing one from MongoDB
        storyboard_s3_link = generation_request.storyboard_s3_link

        if storyboard_s3_link:
            # New storyboard provided - save it to MongoDB in annotated_shots
            logger.info(f"New storyboard S3 link provided for shot {shot_id}: {storyboard_s3_link}")
            shots_service.shots_collection.update_one(
                query,
                {
                    "$set": {
                        f"{shots_field}.$.storyboard_s3_link": storyboard_s3_link,
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
            )
            logger.info(f"Saved storyboard S3 link to MongoDB for shot {shot_id}")
        else:
            # No storyboard in request - check if one exists in MongoDB from previous upload
            existing_storyboard = target_shot.get("storyboard_s3_link")
            if existing_storyboard:
                storyboard_s3_link = existing_storyboard
                logger.info(f"Using existing storyboard from MongoDB (annotated_shots) for shot {shot_id}: {storyboard_s3_link}")
            else:
                logger.info(f"No storyboard found for shot {shot_id} (neither in request nor in MongoDB)")

        agent = ImagenGeneratorAgent()
        generation_results = agent.generate_images_for_shots(
            modified_shots=[
                {
                    "shot_id": shot_id,
                    "corrected_prompt": prompt_text,
                    "corrected_assets": corrected_assets,
                    "storyboard_s3_link": storyboard_s3_link
                }
            ],
            movie_id=show_id
        )

        generated_images = generation_results.get("generated_images", [])
        if not generated_images:
            raise HTTPException(status_code=500, detail="Agent 14 did not return any generated images")

        shot_image = next(
            (img for img in generated_images if img.get("shot_id") == shot_id),
            generated_images[0]
        )
        s3_url = shot_image.get("s3_url")
        if not s3_url:
            raise HTTPException(status_code=502, detail="Image generated but S3 upload failed")

        version_payload = {
            "updated_prompt": prompt_text,
            "changes_made": None,
            "reasoning": None,
            "generated_images_s3": [s3_url]
        }

        update_result = shots_service.shots_collection.update_one(
            query,
            {
                "$set": {
                    f"{shots_field}.$.image.{version_key}": version_payload,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )

        if update_result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to persist image metadata in MongoDB")

        return ApiResponse(
            success=True,
            data={
                "show_id": show_id,
                "episode_number": episode_number,
                "shot_id": shot_id,
                "version": version_key,
                "image": version_payload
            },
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, logger, context="generate_shot_image_with_prompt")


@router.get(
    "/shots/{show_id}/{episode_number}/{shot_id}/download",
    response_class=StreamingResponse
)
@limiter.limit("30/minute")
async def download_shot_versions(
    request: Request,
    show_id: str,
    episode_number: int,
    shot_id: str,
    media: Literal["image", "video"] = "image",
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Download every generated version of a shot (images or videos) as a single ZIP.
    """
    try:
        from app.config import get_shots_service, get_s3_client, get_bucket_name
    except Exception as e:
        logger.error(f"Failed to import required services: {e}")
        raise HTTPException(status_code=500, detail="Server configuration error")

    try:
        shots_service = get_shots_service()
    except (ValueError, ConnectionError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"MongoDB Atlas client not configured: {e}"
        )
    except Exception as e:
        logger.error(f"Failed to get shots service: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize MongoDB client"
        )

    try:
        s3_client = get_s3_client()
        default_bucket = get_bucket_name()
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise HTTPException(
            status_code=503,
            detail="S3 client not available"
        )

    shot_collection = shots_service.get_shots_from_atlas(show_id, episode_number)
    if not shot_collection or not shot_collection.annotated_shots:
        raise HTTPException(status_code=404, detail="Shot list not found for episode")

    target_shot = next(
        (shot for shot in shot_collection.annotated_shots if shot.shot_id == shot_id),
        None
    )
    if not target_shot:
        raise HTTPException(
            status_code=404,
            detail=f"Shot {shot_id} not found for show_id={show_id}, episode={episode_number}"
        )

    version_data = getattr(target_shot, media, None)
    if not version_data:
        raise HTTPException(
            status_code=404,
            detail=f"No {media} versions available for shot {shot_id}"
        )

    url_field = "generated_images_s3" if media == "image" else "generated_videos_s3"
    endpoint_url = os.getenv("production_AWS_ENDPOINT_URL")

    zip_buffer = io.BytesIO()
    files_added = 0

    try:
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for version_key in sorted(version_data.keys()):
                urls = version_data[version_key].get(url_field) or []
                if not urls:
                    continue

                for index, s3_url in enumerate(urls, start=1):
                    try:
                        bucket_name, object_key = parse_s3_url(
                            s3_url,
                            endpoint_url=endpoint_url,
                            default_bucket=default_bucket
                        )
                    except ValueError as parse_error:
                        logger.error(f"Invalid S3 URL for shot {shot_id}: {parse_error}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Invalid S3 URL stored for shot {shot_id}"
                        )

                    try:
                        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
                        payload = response["Body"].read()
                    except Exception as download_error:
                        logger.error(
                            f"Failed to download {s3_url} for shot {shot_id}: {download_error}"
                        )
                        raise HTTPException(
                            status_code=502,
                            detail="Failed to fetch assets from storage"
                        )

                    filename = Path(object_key).name or f"{media}_{index}"
                    zip_path = f"{version_key}/{filename}"
                    archive.writestr(zip_path, payload)
                    files_added += 1

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to build ZIP archive for shot {shot_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare download")

    if files_added == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No {media} files available for shot {shot_id}"
        )

    zip_buffer.seek(0)
    filename = f"{shot_id}-{media}-versions.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers=headers
    )


@router.post("/test-agents-14-15", status_code=410)
async def test_agents_14_15_deprecated():
    """
    DEPRECATED: This test endpoint has been replaced by the main workflow

    Please use the main Phase 2 workflow instead:
    1. POST /phase2/start - Start full workflow
    2. POST /phase2/approve-strategy/{job_id} - Approve after Agent 1
    3. POST /phase2/approve-prompts/{job_id} - Approve after Agent 13
    4. POST /phase2/final-approve/{job_id} - Final approval

    This endpoint will be removed in a future version.
    """
    raise HTTPException(
        status_code=410,
        detail={
            "error": "Endpoint deprecated",
            "message": "This endpoint has been replaced. Please use the main Phase 2 workflow instead.",
            "new_workflow": [
                "POST /phase2/start",
                "POST /phase2/approve-strategy/{job_id}",
                "POST /phase2/approve-prompts/{job_id}",
                "POST /phase2/final-approve/{job_id}"
            ]
        }
    )


# Remove old test endpoint code - everything below until next @ router
"""
OLD CODE REMOVED - test endpoint deprecated in favor of main workflow
"""


@router.post("/approve-prompts/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def approve_prompts(
    request: Request,
    job_id: str,
    approval: PromptApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Approve/reject corrected prompts and resume workflow using Celery (Checkpoint 2)

    This endpoint should be called after Agent 13 (Prompt Modifier) completes.

    If approved: Continues to Agent 14 (Imagen Generator)
    If rejected: Ends workflow with status "rejected"

    Updated to use Celery: Dispatches resume task to SQS instead of background task.
    Includes idempotency to prevent duplicate approvals.
    """
    try:
        # ===== STEP 0: Idempotency Check =====
        from app.core.idempotency import (
            get_idempotency_service,
            generate_idempotency_key,
            check_idempotency,
            mark_idempotency_completed,
            mark_idempotency_failed,
        )
        
        # Generate idempotency key: job_id + checkpoint identifier
        idempotency_key_value = generate_idempotency_key(
            user_id=admin_user.user_id if admin_user else None,
            scene_id=job_id,
            phase_number=2,
            idempotency_key_header=request.headers.get("Idempotency-Key"),
        )
        idempotency_key_value = f"{idempotency_key_value}:prompt_approval"
        
        # Build payload (exclude feedback to allow different feedback with same approval)
        payload = {
            "job_id": job_id,
            "approval_status": approval.approval_status,
        }
        
        # Check idempotency
        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase2.approve_prompts",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )
        
        if is_duplicate:
            if cached_response:
                logger.info(f"Returning cached response for prompt approval: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase2.approve_prompts",
                    key=idempotency_key_value
                )
                
                if record and record.task_id:
                    logger.warning(f"Prompt approval already processing with task ID: {record.task_id}")
                    job = pipeline_service.get_job(job_id)
                    if job:
                        return pipeline_service.to_response(job)
                else:
                    logger.info(f"No task_id in record yet - proceeding with approval")
        
        job = pipeline_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.pipeline_status != "waiting_for_prompt_approval":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not waiting for prompt approval. Current status: {job.pipeline_status}"
            )

        logger.info(f"Prompt approval for job {job_id}: {'APPROVED' if approval.approval_status else 'REJECTED'}")

        # Convert string feedback to dict format if needed
        feedback = approval.feedback
        if isinstance(feedback, str):
            feedback = {"comment": feedback}
        elif feedback is None:
            feedback = {}

        # Dispatch Celery resume task
        queue_name = get_workflow_queue_name()
        task = resume_phase2_after_prompt_approval_task.apply_async(
            args=[
                job_id,
                approval.approval_status,
                feedback
            ],
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched prompt approval resume task")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Decision: {'APPROVED' if approval.approval_status else 'REJECTED'}")
        logger.info(f"Celery Task ID: {task.id}")

        # Update job with new Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)
        
        # Attach task reference to idempotency record
        try:
            idempotency_service.attach_task_reference(
                endpoint="phase2.approve_prompts",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
            )
        except Exception as e:
            logger.warning(f"Failed to attach task reference to idempotency record: {e}")

        # Return updated job
        job = pipeline_service.get_job(job_id)
        response = pipeline_service.to_response(job)
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase2.approve_prompts",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
                response_payload=response.dict() if hasattr(response, 'dict') else response,
            )
        except Exception as e:
            logger.warning(f"Failed to mark idempotency as completed: {e}")
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        # Mark idempotency as failed
        try:
            from app.core.idempotency import mark_idempotency_failed, generate_idempotency_key
            
            idempotency_key_value = generate_idempotency_key(
                user_id=admin_user.user_id if admin_user else None,
                scene_id=job_id,
                phase_number=2,
                idempotency_key_header=request.headers.get("Idempotency-Key"),
            )
            idempotency_key_value = f"{idempotency_key_value}:prompt_approval"
            
            mark_idempotency_failed(
                endpoint="phase2.approve_prompts",
                idempotency_key=idempotency_key_value,
                error_message=f"Prompt approval failed: {str(e)}",
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")
        
        raise handle_api_exception(e, logger, context="approve_prompts")


@router.post("/final-approve/{job_id}", response_model=PipelineJobResponse)
@limiter.limit("20/minute")
async def final_approve(
    request: Request,
    job_id: str,
    approval: FinalApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Final approval after all images are generated and edited using Celery (Checkpoint 3)

    This endpoint should be called after the edit-review loop completes.

    If approved: Marks workflow as completed
    If rejected: Ends workflow with status "rejected"

    Updated to use Celery: Dispatches resume task to SQS instead of background task.
    Includes idempotency to prevent duplicate approvals.
    """
    try:
        # ===== STEP 0: Idempotency Check =====
        from app.core.idempotency import (
            get_idempotency_service,
            generate_idempotency_key,
            check_idempotency,
            mark_idempotency_completed,
            mark_idempotency_failed,
        )
        
        # Generate idempotency key: job_id + checkpoint identifier
        idempotency_key_value = generate_idempotency_key(
            user_id=admin_user.user_id if admin_user else None,
            scene_id=job_id,
            phase_number=2,
            idempotency_key_header=request.headers.get("Idempotency-Key"),
        )
        idempotency_key_value = f"{idempotency_key_value}:final_approval"
        
        # Build payload (exclude feedback to allow different feedback with same approval)
        payload = {
            "job_id": job_id,
            "approval_status": approval.approval_status,
        }
        
        # Check idempotency
        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase2.final_approve",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )
        
        if is_duplicate:
            if cached_response:
                logger.info(f"Returning cached response for final approval: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase2.final_approve",
                    key=idempotency_key_value
                )
                
                if record and record.task_id:
                    logger.warning(f"Final approval already processing with task ID: {record.task_id}")
                    job = pipeline_service.get_job(job_id)
                    if job:
                        return pipeline_service.to_response(job)
                else:
                    logger.info(f"No task_id in record yet - proceeding with approval")
        
        job = pipeline_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.pipeline_status != "waiting_for_final_approval":
            raise HTTPException(
                status_code=400,
                detail=f"Job is not waiting for final approval. Current status: {job.pipeline_status}"
            )

        logger.info(f"Final approval for job {job_id}: {'APPROVED' if approval.approval_status else 'REJECTED'}")

        # Convert string feedback to dict format if needed
        feedback = approval.feedback
        if isinstance(feedback, str):
            feedback = {"comment": feedback}
        elif feedback is None:
            feedback = {}

        # Dispatch Celery resume task
        queue_name = get_workflow_queue_name()
        task = resume_phase2_after_final_approval_task.apply_async(
            args=[
                job_id,
                approval.approval_status,
                feedback
            ],
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("Dispatched final approval task")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Decision: {'APPROVED' if approval.approval_status else 'REJECTED'}")
        logger.info(f"Celery Task ID: {task.id}")

        # Update job with new Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)
        
        # Attach task reference to idempotency record
        try:
            idempotency_service.attach_task_reference(
                endpoint="phase2.final_approve",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
            )
        except Exception as e:
            logger.warning(f"Failed to attach task reference to idempotency record: {e}")

        # Return updated job
        job = pipeline_service.get_job(job_id)
        response = pipeline_service.to_response(job)
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase2.final_approve",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=task.id,
                response_payload=response.dict() if hasattr(response, 'dict') else response,
            )
        except Exception as e:
            logger.warning(f"Failed to mark idempotency as completed: {e}")
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        # Mark idempotency as failed
        try:
            from app.core.idempotency import mark_idempotency_failed, generate_idempotency_key
            
            idempotency_key_value = generate_idempotency_key(
                user_id=admin_user.user_id if admin_user else None,
                scene_id=job_id,
                phase_number=2,
                idempotency_key_header=request.headers.get("Idempotency-Key"),
            )
            idempotency_key_value = f"{idempotency_key_value}:final_approval"
            
            mark_idempotency_failed(
                endpoint="phase2.final_approve",
                idempotency_key=idempotency_key_value,
                error_message=f"Final approval failed: {str(e)}",
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")
        
        raise handle_api_exception(e, logger, context="final_approve")


# ============================================================================
# Celery Task Monitoring Endpoints (Same as Phase 1)
# ============================================================================

@router.get("/task-status/{task_id}")
@limiter.limit("100/minute")
async def get_task_status(request: Request, task_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get real-time status of a Celery task

    This endpoint allows monitoring of task progress, errors, and completion.
    Frontend should poll this endpoint while task is running.

    Same implementation as Phase 1 - reused for consistency.
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
        raise handle_api_exception(e, logger, context="get_task_status")


@router.post("/cancel-task/{task_id}")
@limiter.limit("20/minute")
async def cancel_task(request: Request, task_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Cancel a running Celery task

    Same implementation as Phase 1 - reused for consistency.
    """
    try:
        # Revoke the task
        celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')

        logger.info(f"Cancelled Phase 2 task: {task_id}")

        return JSONResponse(content={
            "task_id": task_id,
            "status": "cancelled",
            "message": "Task has been cancelled. May take a moment to stop if already running."
        })

    except Exception as e:
        raise handle_api_exception(e, logger, context="cancel_task")


@router.get("/task-info/{job_id}")
@limiter.limit("100/minute")
async def get_task_info_by_job(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get Celery task information for a specific job

    Convenience endpoint that looks up the celery_task_id from job_id
    and returns task status. Useful when you only have the job_id.

    Same implementation as Phase 1 - reused for consistency.
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
        raise handle_api_exception(e, logger, context="get_task_info_by_job")
