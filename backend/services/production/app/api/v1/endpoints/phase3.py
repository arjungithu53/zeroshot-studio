"""
Phase 3 Workflow API Endpoints
================================
FastAPI endpoints for managing Phase 3 video generation workflow jobs.
"""

import os
import sys
from datetime import datetime
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pathlib import Path
from shared.auth.dependencies import validate_admin_from_header, AdminUser
import io
import zipfile
import tempfile

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception
from backend.shared.models.responses import ApiResponse

# Initialize logger for this module
logger = get_logger(__name__)

from backend.services.production.app.services.phase_3_agents.langgraph_workflow import (
    run_phase3_pipeline,
    create_phase3_workflow
)
from backend.services.production.app.config import get_mongo_factory, get_workflow_queue_name
from backend.services.production.app.services.pipeline_service import PipelineService
from backend.services.production.app.models.mongodb.pipelines import PipelineJobCreate
from backend.services.production.app.core.quota import QuotaManager, get_quota_manager
from backend.services.production.app.tasks.phase3_tasks import run_phase3_workflow_task
from backend.services.production.app.core.idempotency import (
    get_idempotency_service,
    generate_idempotency_key,
    check_idempotency,
    mark_idempotency_failed,
    mark_idempotency_completed,
)
from backend.services.production.app.services.idempotency_service import (
    IdempotencyConflictError,
    _hash_payload,
)

# Import rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/phase3", tags=["Phase 3 Video Generation"])

# Initialize limiter (will use the one from app.state in practice)
limiter = Limiter(key_func=get_remote_address)

# Pipeline service for persistent job tracking
pipeline_service = PipelineService()

# In-memory storage for workflow state (temporary until state is persisted in DB)
workflow_states: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def save_approval_status_to_mongodb(
    shot_id: str,
    show_id: str,
    versions: list[str],
    decision: str,
    feedback: Optional[str] = None
) -> bool:
    """
    Save approval status for video versions to MongoDB shots collection

    Args:
        shot_id: Shot ID
        show_id: Show/Project ID
        versions: List of version strings to mark as approved (e.g., ['v0', 'v1'])
        decision: 'approved' or 'needs_changes'
        feedback: Optional human feedback

    Returns:
        True if successful, False otherwise
    """
    try:
        from backend.services.production.app.config import get_mongo_factory
        mongo_factory = get_mongo_factory()
        client, shots_collection = mongo_factory.get_collection("shots")

        # Find the document containing the shot
        shot_doc = shots_collection.find_one({
            "show_id": show_id,
            "annotated_shots.shot_id": shot_id
        })

        if not shot_doc:
            # Try alternate query pattern
            shot_doc = shots_collection.find_one({"shot_id": shot_id})

        if not shot_doc:
            logger.warning(f"Shot {shot_id} not found in MongoDB, cannot save approval status")
            return False

        # Prepare approval metadata
        approval_data = {
            "approval_status": "approved" if decision == "approved" else "pending",
            "approved_at": datetime.utcnow().isoformat(),
            "approval_feedback": feedback or ""
        }

        # Update each version with approval status
        updates_successful = 0

        if "annotated_shots" in shot_doc:
            # Find shot index
            shot_index = None
            for i, shot_item in enumerate(shot_doc["annotated_shots"]):
                if shot_item.get("shot_id") == shot_id:
                    shot_index = i
                    break

            if shot_index is not None:
                for version in versions:
                    result = shots_collection.update_one(
                        {
                            "_id": shot_doc["_id"],
                            f"annotated_shots.{shot_index}.shot_id": shot_id
                        },
                        {
                            "$set": {
                                f"annotated_shots.{shot_index}.video.{version}.approval_status": approval_data["approval_status"],
                                f"annotated_shots.{shot_index}.video.{version}.approved_at": approval_data["approved_at"],
                                f"annotated_shots.{shot_index}.video.{version}.approval_feedback": approval_data["approval_feedback"]
                            }
                        }
                    )
                    if result.modified_count > 0:
                        updates_successful += 1
                        logger.info(f"✓ Saved approval status for {version} of shot {shot_id}")
        else:
            # Fallback: individual shot document
            for version in versions:
                result = shots_collection.update_one(
                    {"shot_id": shot_id},
                    {
                        "$set": {
                            f"video.{version}.approval_status": approval_data["approval_status"],
                            f"video.{version}.approved_at": approval_data["approved_at"],
                            f"video.{version}.approval_feedback": approval_data["approval_feedback"]
                        }
                    }
                )
                if result.modified_count > 0:
                    updates_successful += 1
                    logger.info(f"✓ Saved approval status for {version} of shot {shot_id}")

        logger.info(f"Saved approval status for {updates_successful}/{len(versions)} versions")
        return updates_successful > 0

    except Exception as e:
        logger.error(f"Failed to save approval status to MongoDB: {e}")
        return False


# ============================================================================
# Request/Response Models
# ============================================================================

class Phase3StartRequest(BaseModel):
    """Request to start Phase 3 workflow for a shot"""
    shot_id: str = Field(..., description="Shot ID to process")
    show_id: str = Field(..., description="Show/Project ID to uniquely identify the shot (required to avoid shot_id conflicts across shows)")
    version_number: Optional[str] = Field(None, description="Optional version of the shot image to use. Accepts either integer (0, 1, 2) or string format (v0, v1, v2). If not provided, uses the latest version (v0).")


class Phase3HumanApprovalRequest(BaseModel):
    """Request for human approval/changes"""
    job_id: str = Field(..., description="Job ID")
    decision: str = Field(..., description="approved or needs_changes")
    version: Optional[str] = Field(None, description="Specific version to approve (e.g., 'v1', 'v2'). If not provided, uses current/latest version. Mutually exclusive with versions.")
    versions: Optional[list[str]] = Field(None, description="Multiple versions to approve (e.g., ['v0', 'v1', 'v2']). Use this to approve multiple versions at once. Mutually exclusive with version.")
    updated_prompt: Optional[str] = Field(None, description="Updated video prompt if needs_changes")
    feedback: Optional[str] = Field(None, description="Human feedback/comments")


class Phase3EditPromptRequest(BaseModel):
    """Request to edit prompt at human checkpoint"""
    job_id: str = Field(..., description="Job ID")
    version: Optional[str] = Field(None, description="Specific version to edit (e.g., 'v1', 'v2'). If not provided, creates new version based on latest")
    updated_prompt: str = Field(..., description="Updated video prompt")
    feedback: Optional[str] = Field(None, description="Reason for prompt change")


class Phase3JobResponse(BaseModel):
    """Response with job status"""
    job_id: str
    shot_id: str
    status: str  # pending/running/waiting_for_human/completed/failed
    pipeline_status: str
    current_node: str
    generated_video_url: Optional[str] = None
    ai_review_score: Optional[int] = None
    ai_review_decision: Optional[str] = None
    video_generation_attempt: int = 0
    human_regeneration_attempt: int = 0
    current_version: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Background Task Functions
# ============================================================================

def run_phase3_workflow_background(job_id: str, shot_id: str, show_id: str, version_number: Optional[str] = None) -> None:
    """
    Run Phase 3 workflow in background and update job status

    Args:
        job_id: Job identifier
        shot_id: Shot ID to process
        show_id: Show/Project ID to uniquely identify the shot
        version_number: Optional version of the shot image (accepts "v0", "v1", "v2" or "0", "1", "2")
    """
    try:
        # Normalize version_number to string format (v0, v1, v2)
        normalized_version = None
        if version_number is not None:
            version_str = str(version_number).strip()
            # If it's a plain integer like "0", "1", "2", convert to "v0", "v1", "v2"
            if version_str.isdigit():
                normalized_version = f"v{version_str}"
            # If it already starts with "v", use as-is
            elif version_str.startswith("v") and version_str[1:].isdigit():
                normalized_version = version_str
            else:
                logger.warning(f"Invalid version_number format: {version_number}. Expected format: 0, 1, 2, v0, v1, v2")

        # Update status to running in production_pipelines
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            current_agent="agent_17",
            started_at=datetime.utcnow()
        )

        # Run the workflow
        final_state = run_phase3_pipeline(shot_id=shot_id, show_id=show_id, image_version=normalized_version, job_id=job_id)

        # Map pipeline status to proper status field
        status_mapping = {
            "completed": "completed",
            "waiting_for_human": "phase_3_checkpoint",
            "failed": "failed"
        }
        final_status = status_mapping.get(final_state.get("pipeline_status", "completed"), "completed")

        # Update job with final state in production_pipelines
        update_data = {
            "agent17_status": "completed" if final_state.get("video_prompt") else "pending",
            "agent18_status": "completed" if final_state.get("generated_video_url") else "pending",
            "agent19_status": "completed" if final_state.get("review_result") else "pending",
            "phase_3_checkpoint_status": "waiting" if final_status == "phase_3_checkpoint" else "completed",
            "current_agent": final_state.get("current_node", "end"),
            "pipeline_status": final_state.get("pipeline_status", "completed"),
            "error_message": final_state.get("error_message"),
        }
        pipeline_service.update_job_state(job_id, update_data)

        # Store workflow state for resumption
        workflow_states[job_id] = final_state

        logger.info(f"Phase 3 Job {job_id} completed with status: {final_state.get('pipeline_status')}")

    except Exception as e:
        # Update job with error (use safe error message)
        logger.error(f"Phase 3 Job {job_id} failed: {type(e).__name__}", exc_info=True)
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow execution failed"
        )


def resume_phase3_workflow_background(job_id: str, current_state: Dict[str, Any]) -> None:
    """
    Resume Phase 3 workflow from checkpoint in background

    Args:
        job_id: Job identifier
        current_state: Current workflow state with human decision
    """
    try:
        # Update status to running
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running"
        )

        # Create workflow and resume from current state
        app = create_phase3_workflow()
        final_state = app.invoke(current_state)

        # Map pipeline status
        status_mapping = {
            "completed": "completed",
            "waiting_for_human": "phase_3_checkpoint",
            "failed": "failed"
        }
        final_status = status_mapping.get(final_state.get("pipeline_status", "completed"), "completed")

        # Update job with final state
        update_data = {
            "agent17_status": "completed" if final_state.get("video_prompt") else "pending",
            "agent18_status": "completed" if final_state.get("generated_video_url") else "pending",
            "agent19_status": "completed" if final_state.get("review_result") else "pending",
            "phase_3_checkpoint_status": "waiting" if final_status == "phase_3_checkpoint" else "completed",
            "current_agent": final_state.get("current_node", "end"),
            "pipeline_status": final_state.get("pipeline_status", "completed"),
            "error_message": final_state.get("error_message"),
        }
        pipeline_service.update_job_state(job_id, update_data)

        # Store workflow state
        workflow_states[job_id] = final_state

        logger.info(f"Phase 3 Job {job_id} resumed and completed with status: {final_state.get('pipeline_status')}")

    except Exception as e:
        # Update job with error (use safe error message)
        logger.error(f"Phase 3 Job {job_id} resume failed: {type(e).__name__}", exc_info=True)
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow resume failed"
        )


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/start", response_model=Phase3JobResponse, status_code=202)
@limiter.limit("10/minute")
async def start_phase3_workflow(
    request: Request,
    phase3_request: Phase3StartRequest,
    background_tasks: BackgroundTasks,
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager)
):
    """
    Start a new Phase 3 video generation workflow for a shot

    **Quota Enforcement:** This endpoint consumes 1 quota unit

    This endpoint:
    1. Creates a job to track execution
    2. Starts the video generation workflow in the background
    3. Returns immediately with job_id for tracking

    The workflow will:
    - Fetch shot data from MongoDB (shots collection with production_projects fallback)
    - Use the specified version of the shot image if version_number is provided
    - Route to appropriate prompt agent (A or B) based on generation_strategy
    - Generate video using Gemini Veo 3.1
    - Run AI review with Gemini
    - Pause at human checkpoint if AI review is approved or max retries reached

    Shot Data Lookup Priority:
    1. shots collection - annotated_shots array (episode-based structure)
    2. shots collection - individual shot document
    3. production_projects collection - agent14/agent12 outputs (fallback for testing)

    Image Version Selection:
    - If version_number is provided, uses that specific version from the shots collection
    - Accepts both integer format (0, 1, 2) and string format (v0, v1, v2)
    - Examples: version_number=0 or version_number="v0" both refer to the same version
    - If not provided, defaults to the latest version (v0)
    - The version corresponds to the image versions stored in shots.image.{version}.generated_images_s3

    Note: show_id is required to avoid shot_id conflicts across different shows.

    Args:
        request: Shot ID, Show ID, and optional version_number
        background_tasks: FastAPI background tasks

    Returns:
        Job information with job_id for status tracking
    """
    try:
        # ===== STEP 0: Idempotency Check (BEFORE quota consumption) =====
        # IMPORTANT: Check idempotency BEFORE consuming quota to avoid
        # charging users multiple times for duplicate requests
        idempotency_service = get_idempotency_service()

        # Normalize version_number FIRST for consistent key generation
        # This must happen before idempotency key generation
        # Convert None -> "v0", 0 -> "v0", "0" -> "v0", 1 -> "v1", "v1" -> "v1", etc.
        # This ensures that requests with different representations of the same version
        # (e.g., None, 0, "0", "v0") all hash to the same payload
        normalized_version = "v0"  # Default when None or empty string
        if phase3_request.version_number is not None and str(phase3_request.version_number).strip():
            version_str = str(phase3_request.version_number).strip()
            if version_str.isdigit():
                # "0", "1", "2" -> "v0", "v1", "v2"
                normalized_version = f"v{version_str}"
            elif version_str.startswith("v") and len(version_str) > 1 and version_str[1:].isdigit():
                # "v0", "v1", "v2" -> keep as-is
                normalized_version = version_str
            # Invalid formats default to "v0" (already set above)

        # Generate idempotency key AFTER normalization
        # Include version in the scene_id to ensure different versions get different keys
        # IMPORTANT: Do NOT use frontend-provided Idempotency-Key header directly
        # because frontend may reuse the same key with different payloads
        scene_id_with_version = f"{phase3_request.show_id}:{phase3_request.shot_id}:{normalized_version}"
        idempotency_key_value = generate_idempotency_key(
            user_id=admin_user.user_id if admin_user else None,
            scene_id=scene_id_with_version,
            phase_number=3,
            idempotency_key_header=None,  # Don't use frontend key - generate our own
        )

        # Create payload with normalized fields to ensure consistent hashing
        # Sort keys and use only normalized values to avoid hash mismatches
        payload = {
            "shot_id": phase3_request.shot_id.strip() if phase3_request.shot_id else "",
            "show_id": phase3_request.show_id.strip() if phase3_request.show_id else "",
            "version_number": normalized_version,  # Always normalized to "v0", "v1", etc.
        }

        # Debug logging to track payload differences
        logger.info(f"Phase 3 start request - idempotency_key: {idempotency_key_value}")
        logger.info(f"Phase 3 start request - normalized payload: {payload}")

        # ===== STEP 1: Check for existing idempotency record =====
        # CRITICAL: Check BEFORE reserving to avoid creating phantom PROCESSING state
        existing_record = idempotency_service.get_record(
            endpoint="phase3.start",
            key=idempotency_key_value,
        )

        if existing_record:
            # Validate payload hash matches
            request_hash = _hash_payload(payload)

            if existing_record.request_hash != request_hash:
                # Different payload with same key - this is an error
                logger.error(
                    f"Idempotency conflict - key: {idempotency_key_value}\n"
                    f"Existing hash: {existing_record.request_hash}\n"
                    f"New hash: {request_hash}"
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "Idempotency conflict",
                        "message": "The Idempotency-Key header was reused with a different request payload. "
                                   "Please use a new idempotency key for a different request, or retry with the same payload."
                    }
                )

            # Same payload - handle based on status
            if existing_record.is_completed:
                # Job completed - return cached response
                logger.info(f"Returning cached response for completed job: {existing_record.workflow_id}")
                return existing_record.response_payload

            if existing_record.is_processing:
                # ===== AUTO-HEAL ORPHANED PROCESSING STATE =====
                # SAFETY INVARIANT: PROCESSING without job_id is illegal
                if not existing_record.workflow_id:
                    logger.warning(
                        f"⚠️  ILLEGAL STATE DETECTED: Idempotency key {idempotency_key_value} "
                        f"has status=PROCESSING but workflow_id=NULL. This is a phantom lock. "
                        f"Auto-healing by deleting the corrupted record and proceeding as new request."
                    )
                    # Delete the corrupted record
                    idempotency_service.collection.delete_one({
                        "endpoint": "phase3.start",
                        "key": idempotency_key_value,
                    })
                    # Continue to create new job (don't return - fall through)
                else:
                    # Valid PROCESSING state with job_id - return existing job
                    job = pipeline_service.get_job(existing_record.workflow_id)
                    if job:
                        logger.info(
                            f"Duplicate request detected. Returning existing job: {existing_record.workflow_id}"
                        )
                        return pipeline_service.to_response(job)
                    else:
                        # Job not found in DB - orphaned idempotency record
                        logger.warning(
                            f"⚠️  Idempotency record references non-existent job {existing_record.workflow_id}. "
                            f"Auto-healing by deleting the record and proceeding as new request."
                        )
                        idempotency_service.collection.delete_one({
                            "endpoint": "phase3.start",
                            "key": idempotency_key_value,
                        })
                        # Continue to create new job

            if existing_record.is_failed:
                # Previous attempt failed - allow retry
                logger.info(f"Previous attempt failed. Allowing retry for key: {idempotency_key_value}")
                # Continue to create new job

        # ===== STEP 2: Consume Quota (BEFORE job creation) =====
        # Only consume quota for new (non-duplicate) requests
        quota_manager.consume(
            user_id=admin_user.user_id,
            pipeline_name="production_workflow"
        )
        logger.info(f"Quota consumed for user {admin_user.user_id} (start Phase 3)")

        # ===== STEP 3: Create Phase 3 job SYNCHRONOUSLY =====
        # CRITICAL: Job MUST exist in DB before marking idempotency key as PROCESSING
        # This prevents phantom PROCESSING state where no job actually exists
        job_create = PipelineJobCreate(
            project_id=phase3_request.show_id,
            shot_id=phase3_request.shot_id,  # Include shot_id for tracking
            max_regenerations=3
        )
        job_result = pipeline_service.create_job(job_create)
        job_id = job_result["job_id"]  # Extract job_id from dict response

        logger.info(f"✓ Phase 3 job created synchronously: {job_id}")

        # ===== STEP 4: Reserve idempotency key NOW (AFTER job creation) =====
        # CRITICAL: Only mark as PROCESSING after job_id exists
        # This ensures PROCESSING state always has a valid job_id attached
        from datetime import timedelta
        try:
            request_hash = _hash_payload(payload)
            now = datetime.utcnow()
            ttl = timedelta(minutes=30)

            idempotency_doc = {
                "key": idempotency_key_value,
                "endpoint": "phase3.start",
                "request_hash": request_hash,
                "status": "processing",
                "workflow_id": job_id,  # CRITICAL: job_id is set IMMEDIATELY
                "task_id": None,  # Will be set after Celery dispatch
                "response_payload": None,
                "created_at": now,
                "updated_at": now,
                "expires_at": now + ttl,
            }

            # Atomically insert (will fail if another request already created it)
            from pymongo.errors import DuplicateKeyError
            try:
                idempotency_service.collection.insert_one(idempotency_doc)
                logger.info(f"✓ Idempotency key reserved with job_id: {job_id}")
            except DuplicateKeyError:
                # Another concurrent request already created the record
                # This is extremely rare but possible in high-concurrency scenarios
                # Just log and continue - the job is already created
                logger.warning(
                    f"Race condition: Another request already created idempotency record for {idempotency_key_value}. "
                    f"Continuing with job {job_id} already created."
                )

        except Exception as idemp_error:
            # Don't fail the entire request if idempotency write fails
            # The job is already created and will execute
            logger.error(f"Failed to write idempotency record: {idemp_error}", exc_info=True)

        # ===== STEP 5: Dispatch async execution to Celery =====
        queue_name = get_workflow_queue_name()
        task = run_phase3_workflow_task.apply_async(
            args=[
                job_id,
                phase3_request.shot_id,
                phase3_request.show_id,
                request.headers.get("Idempotency-Key"),
                admin_user.user_id if admin_user else None,
            ],
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info("✓ Dispatched Phase 3 workflow to Celery")
        logger.info(f"  Job ID: {job_id}")
        logger.info(f"  Celery Task ID: {task.id}")
        logger.info(f"  Queue: {queue_name}")

        # Update idempotency record with task_id
        try:
            idempotency_service.collection.update_one(
                {"endpoint": "phase3.start", "key": idempotency_key_value},
                {"$set": {"task_id": task.id, "updated_at": datetime.utcnow()}}
            )
        except Exception as task_update_error:
            logger.warning(f"Failed to update task_id in idempotency record: {task_update_error}")

        # Update job with Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)

        # ===== STEP 6: Return response with job metadata =====
        job = pipeline_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=500, detail="Failed to load pipeline job")

        response_payload = pipeline_service.to_response(job)

        # Store response in idempotency record for future duplicates
        try:
            idempotency_service.collection.update_one(
                {"endpoint": "phase3.start", "key": idempotency_key_value},
                {"$set": {"response_payload": response_payload, "updated_at": datetime.utcnow()}}
            )
        except Exception as response_update_error:
            logger.warning(f"Failed to update response_payload in idempotency record: {response_update_error}")

        return response_payload

    except IdempotencyConflictError as conflict_error:
        # Client error: same idempotency key used with different payload
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Idempotency conflict",
                "message": str(conflict_error),
                "hint": "The Idempotency-Key header was reused with a different request payload. "
                        "Please use a new idempotency key for a different request, or retry with the same payload."
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        try:
            mark_idempotency_failed(
                endpoint="phase3.start",
                idempotency_key=idempotency_key_value,
                error_message=str(e),
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")
        raise handle_api_exception(e, "start_phase3_workflow")


@router.get("/status/{job_id}", response_model=Phase3JobResponse)
@limiter.limit("100/minute")
async def get_job_status(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get current status of a Phase 3 workflow job

    Returns job status, current node, video URL, and AI review results.
    Poll this endpoint to track workflow progress.

    When status is "waiting_for_human", use the /human-approval endpoint
    to submit approval decision or /human-edit-prompt to modify the prompt.

    Args:
        job_id: Job identifier

    Returns:
        Current job status and results
    """
    job = pipeline_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get workflow state if available
    workflow_state = workflow_states.get(job_id, {})

    return Phase3JobResponse(
        job_id=job_id,
        shot_id=workflow_state.get("shot_id", ""),
        status=job.status,
        pipeline_status=job.pipeline_status,
        current_node=job.current_agent,
        generated_video_url=workflow_state.get("generated_video_url"),
        ai_review_score=workflow_state.get("review_score"),
        ai_review_decision=workflow_state.get("review_decision"),
        video_generation_attempt=workflow_state.get("video_generation_attempt", 0),
        human_regeneration_attempt=workflow_state.get("human_regeneration_attempt", 0),
        current_version=workflow_state.get("current_version", 0),
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.post("/human-approval", response_model=Phase3JobResponse, status_code=202)
@limiter.limit("20/minute")
async def submit_human_approval(
    request: Request,
    approval_request: Phase3HumanApprovalRequest,
    background_tasks: BackgroundTasks,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Submit human approval decision for a video at checkpoint

    This endpoint is called when the workflow reaches the human checkpoint
    (after AI review approves or max AI retries reached).

    Human Decisions:
    - "approved": Accept the video and complete the workflow
    - "needs_changes": Provide updated prompt and regenerate video

    Version Selection:
    - Specify a version parameter (e.g., "v1", "v2") to approve a specific version
    - If version is not provided, uses the current/latest version
    - Version must exist in either workflow state or MongoDB shots collection

    Args:
        request: Human approval decision with optional version and updated prompt
        background_tasks: FastAPI background tasks

    Returns:
        Updated job status
    """
    job = pipeline_service.get_job(approval_request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {approval_request.job_id} not found")

    # Validate job is waiting for human approval
    if job.status not in ["waiting_for_human_approval", "phase_3_checkpoint"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {approval_request.job_id} is not waiting for human approval (current status: {job.status})"
        )

    # Validate that version and versions are mutually exclusive
    if approval_request.version and approval_request.versions:
        raise HTTPException(
            status_code=400,
            detail="Cannot specify both 'version' and 'versions'. Use 'version' for single version or 'versions' for multiple."
        )

    # Validate decision
    if approval_request.decision not in ["approved", "needs_changes"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision: {approval_request.decision}. Must be 'approved' or 'needs_changes'"
        )

    # Validate updated_prompt if needs_changes
    if approval_request.decision == "needs_changes" and not approval_request.updated_prompt:
        raise HTTPException(
            status_code=400,
            detail="updated_prompt is required when decision is 'needs_changes'"
        )

    try:
        # Get current workflow state
        current_state = workflow_states.get(approval_request.job_id)

        # If not in memory, reconstruct from MongoDB
        if not current_state:
            logger.info(f"Workflow state not in memory for job {approval_request.job_id}, reconstructing from MongoDB...")

            from backend.services.production.app.config import get_mongo_factory
            mongo_factory = get_mongo_factory()
            client, shots_collection = mongo_factory.get_collection("shots")

            # Get shot_id from job
            if not job.shot_id:
                raise HTTPException(
                    status_code=400,
                    detail="Job has no associated shot_id, cannot reconstruct workflow state"
                )

            # Try to find shot data
            shot_doc = shots_collection.find_one({
                "show_id": job.project_id,
                "annotated_shots.shot_id": job.shot_id
            })

            if not shot_doc:
                # Try alternate query pattern (individual shot document)
                shot_doc = shots_collection.find_one({
                    "shot_id": job.shot_id
                })

            if not shot_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"Shot {job.shot_id} not found in MongoDB, cannot reconstruct workflow state"
                )

            # Reconstruct minimal state from MongoDB
            current_state = {
                "shot_id": job.shot_id,
                "show_id": job.project_id,
                "video_versions": [],
                "current_version": 0,
                "pipeline_status": job.pipeline_status,
            }

            # Extract video versions from MongoDB
            video_data = {}

            # Check if shot data is in annotated_shots array
            if "annotated_shots" in shot_doc:
                for shot_item in shot_doc.get("annotated_shots", []):
                    if shot_item.get("shot_id") == job.shot_id:
                        video_data = shot_item.get("video", {})
                        break
            else:
                # Shot document directly
                video_data = shot_doc.get("video", {})

            # Build version list
            for version_key in sorted(video_data.keys()):
                version_info = video_data[version_key]
                video_urls = version_info.get("generated_videos_s3", [])
                if video_urls:
                    current_state["video_versions"].append({
                        "version": version_key,
                        "generated_videos_s3": video_urls,
                        "prompt": version_info.get("updated_prompt", ""),
                        "source": "mongodb_reconstruction"
                    })

            # Set current version to latest
            if current_state["video_versions"]:
                current_state["current_version"] = len(current_state["video_versions"]) - 1
                current_state["generated_video_url"] = current_state["video_versions"][-1]["generated_videos_s3"][0]
                logger.info(f"Reconstructed {len(current_state['video_versions'])} versions from MongoDB")
            else:
                logger.warning(f"No video versions found in MongoDB for shot {job.shot_id}")

            # Store reconstructed state
            workflow_states[approval_request.job_id] = current_state

        # Handle version selection (single or multiple)
        versions_to_approve = []
        if approval_request.versions:
            versions_to_approve = approval_request.versions
        elif approval_request.version:
            versions_to_approve = [approval_request.version]

        approved_versions = []
        if versions_to_approve:
            # Fetch MongoDB data once if needed
            from backend.services.production.app.config import get_mongo_factory
            mongo_factory = get_mongo_factory()
            client, shots_collection = mongo_factory.get_collection("shots")

            shot_doc = shots_collection.find_one({
                "show_id": job.project_id,
                "annotated_shots.shot_id": current_state.get("shot_id")
            })

            video_data_from_db = {}
            if shot_doc:
                for shot_item in shot_doc.get("annotated_shots", []):
                    if shot_item.get("shot_id") == current_state.get("shot_id"):
                        video_data_from_db = shot_item.get("video") or {}
                        break

            # Validate and collect all requested versions
            video_versions = current_state.get("video_versions", [])

            for selected_version in versions_to_approve:
                version_exists = any(v.get("version") == selected_version for v in video_versions)
                version_info = None

                if not version_exists:
                    # Try MongoDB
                    if selected_version in video_data_from_db:
                        version_exists = True
                        selected_video_urls = video_data_from_db[selected_version].get("generated_videos_s3", [])
                        if selected_video_urls:
                            version_info = {
                                "version": selected_version,
                                "video_url": selected_video_urls[0],
                                "prompt": video_data_from_db[selected_version].get("updated_prompt", ""),
                                "source": "mongodb"
                            }

                if not version_exists:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Version '{selected_version}' not found for this job"
                    )

                if not version_info:
                    # Get from workflow state
                    for version_data in video_versions:
                        if version_data.get("version") == selected_version:
                            video_url = version_data.get("generated_videos_s3", [None])[0]
                            if video_url:
                                version_info = {
                                    "version": selected_version,
                                    "video_url": video_url,
                                    "prompt": version_data.get("prompt", ""),
                                    "source": "workflow_state"
                                }
                            break

                if version_info:
                    approved_versions.append(version_info)

            # Update state with approved versions
            if approved_versions:
                current_state["approved_versions"] = approved_versions
                # Set the first approved version as the primary one
                current_state["generated_video_url"] = approved_versions[0]["video_url"]
                current_state["selected_version"] = approved_versions[0]["version"]

        # Update state with human decision
        current_state["human_decision"] = approval_request.decision
        current_state["human_updated_prompt"] = approval_request.updated_prompt
        current_state["human_feedback"] = approval_request.feedback

        # Save approval status to MongoDB
        if versions_to_approve and current_state.get("shot_id") and current_state.get("show_id"):
            logger.info(f"Saving approval status for versions {versions_to_approve} to MongoDB...")
            save_success = save_approval_status_to_mongodb(
                shot_id=current_state["shot_id"],
                show_id=current_state["show_id"],
                versions=versions_to_approve,
                decision=approval_request.decision,
                feedback=approval_request.feedback
            )
            if save_success:
                logger.info("✓ Approval status saved to MongoDB")
            else:
                logger.warning("⚠ Failed to save approval status to MongoDB")

        # Resume workflow in background
        background_tasks.add_task(
            resume_phase3_workflow_background,
            approval_request.job_id,
            current_state
        )

        # Get updated job and return response
        job = pipeline_service.get_job(approval_request.job_id)
        workflow_state = workflow_states.get(approval_request.job_id, {})

        return Phase3JobResponse(
            job_id=approval_request.job_id,
            shot_id=workflow_state.get("shot_id", ""),
            status=job.status,
            pipeline_status=job.pipeline_status,
            current_node=job.current_agent,
            generated_video_url=workflow_state.get("generated_video_url"),
            ai_review_score=workflow_state.get("review_score"),
            ai_review_decision=workflow_state.get("review_decision"),
            video_generation_attempt=workflow_state.get("video_generation_attempt", 0),
            human_regeneration_attempt=workflow_state.get("human_regeneration_attempt", 0),
            current_version=workflow_state.get("current_version", 0),
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "submit_human_approval")


@router.post("/human-edit-prompt", response_model=Phase3JobResponse, status_code=202)
@limiter.limit("20/minute")
async def human_edit_prompt(
    request: Request,
    edit_request: Phase3EditPromptRequest,
    background_tasks: BackgroundTasks,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Edit prompt at human checkpoint and regenerate video

    This endpoint allows humans to modify the video generation prompt
    and trigger a new video generation attempt.

    This is equivalent to calling /human-approval with decision="needs_changes"
    but provides a more explicit API for prompt editing.

    Version Selection:
    - Specify a version parameter (e.g., "v1", "v2") to edit a specific version's prompt
    - If version is not provided, creates new version based on latest
    - Version must exist in either workflow state or MongoDB shots collection
    - The selected version's video will be used as the base for regeneration

    Args:
        request: Updated prompt with optional version and feedback
        background_tasks: FastAPI background tasks

    Returns:
        Updated job status
    """
    job = pipeline_service.get_job(edit_request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {edit_request.job_id} not found")

    # Validate job is waiting for human approval
    if job.status not in ["waiting_for_human_approval", "phase_3_checkpoint"]:
        raise HTTPException(
            status_code=400,
            detail=f"Job {edit_request.job_id} is not waiting for human input (current status: {job.status})"
        )

    # Validate updated_prompt is not empty
    if not edit_request.updated_prompt or not edit_request.updated_prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="updated_prompt cannot be empty"
        )

    try:
        # Get current workflow state
        current_state = workflow_states.get(edit_request.job_id)

        # If not in memory, reconstruct from MongoDB
        if not current_state:
            logger.info(f"Workflow state not in memory for job {edit_request.job_id}, reconstructing from MongoDB...")

            from backend.services.production.app.config import get_mongo_factory
            mongo_factory = get_mongo_factory()
            client, shots_collection = mongo_factory.get_collection("shots")

            # Get shot_id from job
            if not job.shot_id:
                raise HTTPException(
                    status_code=400,
                    detail="Job has no associated shot_id, cannot reconstruct workflow state"
                )

            # Try to find shot data
            shot_doc = shots_collection.find_one({
                "show_id": job.project_id,
                "annotated_shots.shot_id": job.shot_id
            })

            if not shot_doc:
                # Try alternate query pattern (individual shot document)
                shot_doc = shots_collection.find_one({
                    "shot_id": job.shot_id
                })

            if not shot_doc:
                raise HTTPException(
                    status_code=404,
                    detail=f"Shot {job.shot_id} not found in MongoDB, cannot reconstruct workflow state"
                )

            # Reconstruct minimal state from MongoDB
            current_state = {
                "shot_id": job.shot_id,
                "show_id": job.project_id,
                "video_versions": [],
                "current_version": 0,
                "pipeline_status": job.pipeline_status,
            }

            # Extract video versions from MongoDB
            video_data = {}

            # Check if shot data is in annotated_shots array
            if "annotated_shots" in shot_doc:
                for shot_item in shot_doc.get("annotated_shots", []):
                    if shot_item.get("shot_id") == job.shot_id:
                        video_data = shot_item.get("video", {})
                        break
            else:
                # Shot document directly
                video_data = shot_doc.get("video", {})

            # Build version list
            for version_key in sorted(video_data.keys()):
                version_info = video_data[version_key]
                video_urls = version_info.get("generated_videos_s3", [])
                if video_urls:
                    current_state["video_versions"].append({
                        "version": version_key,
                        "generated_videos_s3": video_urls,
                        "prompt": version_info.get("updated_prompt", ""),
                        "source": "mongodb_reconstruction"
                    })

            # Set current version to latest
            if current_state["video_versions"]:
                current_state["current_version"] = len(current_state["video_versions"]) - 1
                current_state["generated_video_url"] = current_state["video_versions"][-1]["generated_videos_s3"][0]
                logger.info(f"Reconstructed {len(current_state['video_versions'])} versions from MongoDB")
            else:
                logger.warning(f"No video versions found in MongoDB for shot {job.shot_id}")

            # Store reconstructed state
            workflow_states[edit_request.job_id] = current_state

        # Handle version selection if provided
        selected_version = edit_request.version
        if selected_version:
            # Validate version exists
            video_versions = current_state.get("video_versions", [])
            version_exists = any(v.get("version") == selected_version for v in video_versions)

            if not version_exists:
                # Try to fetch from MongoDB shots collection
                from backend.services.production.app.config import get_mongo_factory
                mongo_factory = get_mongo_factory()
                client, shots_collection = mongo_factory.get_collection("shots")

                shot_doc = shots_collection.find_one({
                    "show_id": job.project_id,
                    "annotated_shots.shot_id": current_state.get("shot_id")
                })

                if shot_doc:
                    for shot_item in shot_doc.get("annotated_shots", []):
                        if shot_item.get("shot_id") == current_state.get("shot_id"):
                            video_data = shot_item.get("video", {})
                            if selected_version in video_data:
                                version_exists = True
                                # Get the video URL and prompt from the selected version
                                selected_video_urls = video_data[selected_version].get("generated_videos_s3", [])
                                if selected_video_urls:
                                    current_state["generated_video_url"] = selected_video_urls[0]
                                    current_state["selected_version"] = selected_version
                                # If no updated_prompt provided in request, use the version's prompt as base
                                if not edit_request.updated_prompt.strip():
                                    existing_prompt = video_data[selected_version].get("updated_prompt", "")
                                    if existing_prompt:
                                        current_state["base_prompt_for_edit"] = existing_prompt
                            break

                if not version_exists:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Version '{selected_version}' not found for this job"
                    )
            else:
                # Version exists in workflow state - use it
                for version_data in video_versions:
                    if version_data.get("version") == selected_version:
                        video_url = version_data.get("generated_videos_s3", [None])[0]
                        if video_url:
                            current_state["generated_video_url"] = video_url
                            current_state["selected_version"] = selected_version
                        break

        # Update state with human decision to regenerate
        current_state["human_decision"] = "needs_changes"
        current_state["human_updated_prompt"] = edit_request.updated_prompt
        current_state["human_feedback"] = edit_request.feedback or f"Human edited prompt{f' for {selected_version}' if selected_version else ''}"

        # Save edit metadata to MongoDB if a specific version was edited
        if selected_version and current_state.get("shot_id") and current_state.get("show_id"):
            logger.info(f"Saving edit metadata for version {selected_version} to MongoDB...")

            # Mark the version as pending since it needs changes
            save_success = save_approval_status_to_mongodb(
                shot_id=current_state["shot_id"],
                show_id=current_state["show_id"],
                versions=[selected_version],
                decision="needs_changes",
                feedback=edit_request.feedback or f"Prompt edited for {selected_version}"
            )
            if save_success:
                logger.info("✓ Edit metadata saved to MongoDB")
            else:
                logger.warning("⚠ Failed to save edit metadata to MongoDB")

        # Resume workflow in background
        background_tasks.add_task(
            resume_phase3_workflow_background,
            edit_request.job_id,
            current_state
        )

        # Get updated job and return response
        job = pipeline_service.get_job(edit_request.job_id)
        workflow_state = workflow_states.get(edit_request.job_id, {})

        return Phase3JobResponse(
            job_id=edit_request.job_id,
            shot_id=workflow_state.get("shot_id", ""),
            status=job.status,
            pipeline_status=job.pipeline_status,
            current_node=job.current_agent,
            generated_video_url=workflow_state.get("generated_video_url"),
            ai_review_score=workflow_state.get("review_score"),
            ai_review_decision=workflow_state.get("review_decision"),
            video_generation_attempt=workflow_state.get("video_generation_attempt", 0),
            human_regeneration_attempt=workflow_state.get("human_regeneration_attempt", 0),
            current_version=workflow_state.get("current_version", 0),
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "human_edit_prompt")


@router.get("/results/{job_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_job_results(request: Request, job_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get complete results of a Phase 3 workflow job

    Returns full job data including:
    - Job status and timeline
    - Generated video URLs (all versions)
    - AI review results
    - Human feedback
    - Complete workflow state

    All wrapped in standardized ApiResponse format.

    Args:
        job_id: Job identifier

    Returns:
        Complete job results and workflow state wrapped in ApiResponse
    """
    job = pipeline_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Get the full workflow state if available
    final_state = workflow_states.get(job_id, {})

    return ApiResponse(
        success=True,
        data={
            "job_id": job_id,
            "shot_id": final_state.get("shot_id", ""),
            "status": job.status,
            "pipeline_status": job.pipeline_status,
            "current_node": job.current_agent,
            "generated_video_url": final_state.get("generated_video_url"),
            "ai_review": {
                "score": final_state.get("review_score"),
                "decision": final_state.get("review_decision"),
                "full_result": final_state.get("review_result")
            },
            "video_versions": final_state.get("video_versions", []),
            "current_version": final_state.get("current_version", 0),
            "attempts": {
                "video_generation": final_state.get("video_generation_attempt", 0),
                "human_regeneration": final_state.get("human_regeneration_attempt", 0)
            },
            "error_message": job.error_message,
            "timestamps": {
                "created_at": job.created_at,
                "updated_at": job.updated_at
            }
        },
        error=None
    )


@router.get("/jobs/by-shot/{show_id}/{shot_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_jobs_by_shot(request: Request, show_id: str, shot_id: str):
    """
    Get all jobs for a specific shot in a specific show

    IMPORTANT: Both show_id and shot_id are required because shot_ids
    are not unique across different shows.

    Useful for tracking multiple runs/regenerations of the same shot.

    Args:
        show_id: Show/Project ID (required to uniquely identify the shot)
        shot_id: Shot identifier

    Returns:
        List of all jobs for the shot wrapped in ApiResponse, sorted by creation time (newest first)
    """
    # Query pipeline_service for jobs with this shot_id and show_id
    from backend.services.production.app.config import get_mongo_factory
    mongo_factory = get_mongo_factory()
    client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

    # Build query - BOTH shot_id and show_id required
    query = {
        "shot_id": shot_id,
        "project_id": show_id
    }

    # Find all matching jobs
    job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

    matching_jobs = []
    for job_doc in job_docs:
        jid = job_doc.get("job_id")
        if jid:
            job = pipeline_service.get_job(jid)
            if job:
                # Get workflow state if available
                state = workflow_states.get(jid, {})

                matching_jobs.append(Phase3JobResponse(
                    job_id=jid,
                    shot_id=job_doc.get("shot_id", shot_id),
                    status=job.status,
                    pipeline_status=job.pipeline_status,
                    current_node=job.current_agent,
                    generated_video_url=state.get("generated_video_url"),
                    ai_review_score=state.get("review_score"),
                    ai_review_decision=state.get("review_decision"),
                    video_generation_attempt=state.get("video_generation_attempt", 0),
                    human_regeneration_attempt=state.get("human_regeneration_attempt", 0),
                    current_version=state.get("current_version", 0),
                    error_message=job.error_message,
                    created_at=job.created_at,
                    updated_at=job.updated_at
                ))

    return ApiResponse(
        success=True,
        data={
            "shot_id": shot_id,
            "show_id": show_id,
            "jobs": [job.dict() for job in matching_jobs],
            "count": len(matching_jobs)
        },
        error=None
    )


@router.get("/jobs/by-project/{show_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_jobs_by_project(request: Request, show_id: str):
    """
    Get all Phase 3 jobs for a specific show/project

    Returns all jobs (across all shots) for this show, useful for:
    - Monitoring overall progress
    - Dashboard views
    - Discovering which shots are being processed

    Args:
        show_id: Show/Project ID

    Returns:
        List of all Phase 3 jobs for the show wrapped in ApiResponse, grouped by shot_id
    """
    from backend.services.production.app.config import get_mongo_factory
    mongo_factory = get_mongo_factory()
    client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

    # Query for all jobs in this project
    query = {"project_id": show_id}

    # Find all matching jobs, sorted by creation time
    job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

    # Group jobs by shot_id
    jobs_by_shot = {}
    all_jobs = []

    for job_doc in job_docs:
        jid = job_doc.get("job_id")
        shot_id = job_doc.get("shot_id")

        if jid:
            job = pipeline_service.get_job(jid)
            if job:
                # Get workflow state if available
                state = workflow_states.get(jid, {})

                job_response = Phase3JobResponse(
                    job_id=jid,
                    shot_id=shot_id or "unknown",
                    status=job.status,
                    pipeline_status=job.pipeline_status,
                    current_node=job.current_agent,
                    generated_video_url=state.get("generated_video_url"),
                    ai_review_score=state.get("review_score"),
                    ai_review_decision=state.get("review_decision"),
                    video_generation_attempt=state.get("video_generation_attempt", 0),
                    human_regeneration_attempt=state.get("human_regeneration_attempt", 0),
                    current_version=state.get("current_version", 0),
                    error_message=job.error_message,
                    created_at=job.created_at,
                    updated_at=job.updated_at
                )

                all_jobs.append(job_response)

                # Group by shot_id
                if shot_id:
                    if shot_id not in jobs_by_shot:
                        jobs_by_shot[shot_id] = []
                    jobs_by_shot[shot_id].append(job_response)

    # Convert Pydantic models to dicts for JSON serialization
    all_jobs_dict = [job.dict() for job in all_jobs]
    jobs_by_shot_dict = {
        shot_id: [job.dict() for job in jobs]
        for shot_id, jobs in jobs_by_shot.items()
    }

    return ApiResponse(
        success=True,
        data={
            "show_id": show_id,
            "total_jobs": len(all_jobs),
            "total_shots": len(jobs_by_shot),
            "jobs": all_jobs_dict,
            "jobs_by_shot": jobs_by_shot_dict
        },
        error=None
    )


@router.get("/results/by-project/{show_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_results_by_project(request: Request, show_id: str, latest_only: bool = False):
    """
    Get complete results for all Phase 3 jobs in a project (one-stop endpoint for frontend)

    This endpoint combines job listing and results retrieval into a single call.
    Instead of calling /by-project to get job IDs and then /results for each job,
    frontends can call this single endpoint to get everything.

    Returns comprehensive data including:
    - All jobs for the project
    - Complete results for each job (video URLs, AI reviews, workflow state)
    - Shot metadata (shot_description) from MongoDB shots collection
    - Video version details with approval fields (approval_status, approval_feedback, approved_at)
    - Organized by shot_id for easy navigation
    - Summary statistics

    Args:
        show_id: Show/Project ID
        latest_only: If True, returns only the most recent job per shot (default: False)

    Returns:
        Complete results for all jobs in the project, grouped by shot_id
    """
    from backend.services.production.app.config import get_mongo_factory
    mongo_factory = get_mongo_factory()
    client, pipelines_collection = mongo_factory.get_collection("production_pipelines")

    # Query for all jobs in this project
    query = {"project_id": show_id}

    # Find all matching jobs, sorted by creation time
    job_docs = list(pipelines_collection.find(query).sort("created_at", -1))

    # Build comprehensive results
    results_by_shot = {}
    all_results = []
    summary = {
        "total_jobs": 0,
        "completed": 0,
        "running": 0,
        "waiting_for_human": 0,
        "failed": 0
    }

    # If latest_only, we'll track the first (most recent) job per shot
    seen_shots = set()

    # Get shots collection to fetch video data
    client, shots_collection = mongo_factory.get_collection("shots")

    for job_doc in job_docs:
        jid = job_doc.get("job_id")
        shot_id = job_doc.get("shot_id")

        # Skip if latest_only and we've already seen this shot
        if latest_only and shot_id in seen_shots:
            continue

        if jid:
            job = pipeline_service.get_job(jid)
            if job:
                # Get workflow state if available
                state = workflow_states.get(jid, {})

                # Try to fetch video data and shot metadata from shots collection
                generated_video_url = state.get("generated_video_url")
                video_versions = state.get("video_versions", [])
                video_details = []  # Store detailed info for each version
                shot_description = None  # Store shot description

                # Always try to fetch shot metadata (including description) from MongoDB
                if shot_id:
                    # Query shots collection for this shot
                    shot_doc = shots_collection.find_one({
                        "show_id": show_id,
                        "annotated_shots.shot_id": shot_id
                    })

                    if shot_doc:
                        # Find the specific shot in annotated_shots array
                        for shot_item in shot_doc.get("annotated_shots", []):
                            if shot_item.get("shot_id") == shot_id:
                                # Get shot description
                                shot_description = shot_item.get("description")

                                # Get video data from shot (only if not already in state)
                                if not generated_video_url:
                                    video_data = shot_item.get("video", {})

                                    # Extract all video versions with full details
                                    if video_data:
                                        for version_key in sorted(video_data.keys()):
                                            version_data = video_data[version_key]
                                            video_urls = version_data.get("generated_videos_s3", [])

                                            # Build detailed version info
                                            version_info = {
                                                "version": version_key,
                                                "prompt": version_data.get("updated_prompt"),
                                                "changes_made": version_data.get("changes_made"),
                                                "reasoning": version_data.get("reasoning"),
                                                "video_urls": video_urls,
                                                "primary_video_url": video_urls[0] if video_urls else None,
                                                "approval_status": version_data.get("approval_status"),
                                                "approval_feedback": version_data.get("approval_feedback"),
                                                "approved_at": version_data.get("approved_at")
                                            }
                                            video_details.append(version_info)

                                            # Also add URLs to simple list for backward compatibility
                                            if video_urls:
                                                video_versions.extend(video_urls)

                                        # Use the latest video as the main URL
                                        if video_versions:
                                            generated_video_url = video_versions[-1]

                                break

                # Build complete result object
                job_result = {
                    "job_id": jid,
                    "shot_id": shot_id or "unknown",
                    "shot_description": shot_description,  # Shot description from MongoDB
                    "status": job.status,
                    "pipeline_status": job.pipeline_status,
                    "current_node": job.current_agent,
                    "generated_video_url": generated_video_url,
                    "ai_review": {
                        "score": state.get("review_score"),
                        "decision": state.get("review_decision"),
                        "full_result": state.get("review_result")
                    },
                    "video_versions": video_versions,  # Simple list of URLs for backward compatibility
                    "video_details": video_details,  # Detailed info with prompts, reasoning, etc.
                    "current_version": state.get("current_version", len(video_versions) - 1 if video_versions else 0),
                    "attempts": {
                        "video_generation": state.get("video_generation_attempt", 0),
                        "human_regeneration": state.get("human_regeneration_attempt", 0)
                    },
                    "error_message": job.error_message,
                    "timestamps": {
                        "created_at": job.created_at.isoformat() if job.created_at else None,
                        "updated_at": job.updated_at.isoformat() if job.updated_at else None
                    }
                }

                all_results.append(job_result)

                # Update summary statistics
                summary["total_jobs"] += 1
                if job.status == "completed":
                    summary["completed"] += 1
                elif job.status == "running":
                    summary["running"] += 1
                elif job.status in ["waiting_for_human_approval", "phase_3_checkpoint"]:
                    summary["waiting_for_human"] += 1
                elif job.status == "failed":
                    summary["failed"] += 1

                # Group by shot_id
                if shot_id:
                    if shot_id not in results_by_shot:
                        results_by_shot[shot_id] = []
                    results_by_shot[shot_id].append(job_result)

                    # Mark this shot as seen
                    if latest_only:
                        seen_shots.add(shot_id)

    return ApiResponse(
        success=True,
        data={
            "show_id": show_id,
            "summary": summary,
            "results": all_results,
            "results_by_shot": results_by_shot
        },
        error=None
    )


@router.get("/shots/{show_id}/{shot_id}/download-approved")
@limiter.limit("20/minute")
async def download_approved_shots(
    request: Request,
    show_id: str,
    shot_id: str,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Download all approved video versions for a specific shot as a ZIP file

    This endpoint:
    1. Fetches shot data from MongoDB shots collection
    2. Filters for video versions with approval_status = "approved"
    3. Downloads approved videos from S3
    4. Creates a ZIP file containing all approved videos
    5. Returns the ZIP as a StreamingResponse

    Args:
        show_id: Show/Project ID (required to uniquely identify the shot)
        shot_id: Shot identifier
        admin_user: Admin user from authorization header

    Returns:
        StreamingResponse with ZIP file containing approved videos

    Raises:
        HTTPException: If shot not found, no approved videos, or download fails
    """
    try:
        # Get MongoDB shots collection
        from backend.services.production.app.config import get_mongo_factory, get_s3_client, get_bucket_name
        from infrastructure.s3.streaming import parse_s3_url, stream_bytes_from_s3

        mongo_factory = get_mongo_factory()
        client, shots_collection = mongo_factory.get_collection("shots")

        # Find the shot document
        shot_doc = shots_collection.find_one({
            "show_id": show_id,
            "annotated_shots.shot_id": shot_id
        })

        if not shot_doc:
            # Try alternate query pattern (individual shot document)
            shot_doc = shots_collection.find_one({
                "shot_id": shot_id,
                "show_id": show_id
            })

        if not shot_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Shot {shot_id} not found in show {show_id}"
            )

        # Extract video data based on document structure
        video_data = {}
        shot_description = None

        if "annotated_shots" in shot_doc:
            # Find the specific shot in annotated_shots array
            for shot_item in shot_doc.get("annotated_shots", []):
                if shot_item.get("shot_id") == shot_id:
                    video_data = shot_item.get("video", {})
                    shot_description = shot_item.get("description", "")
                    break
        else:
            # Individual shot document
            video_data = shot_doc.get("video", {})
            shot_description = shot_doc.get("description", "")

        if not video_data:
            raise HTTPException(
                status_code=404,
                detail=f"No video data found for shot {shot_id}"
            )

        # Filter for approved versions
        approved_videos = []

        for version_key in sorted(video_data.keys()):
            version_info = video_data[version_key]
            approval_status = version_info.get("approval_status")

            # Check if this version is approved
            if approval_status == "approved":
                video_urls = version_info.get("generated_videos_s3", [])

                if video_urls:
                    for idx, video_url in enumerate(video_urls):
                        approved_videos.append({
                            "version": version_key,
                            "url": video_url,
                            "index": idx,
                            "prompt": version_info.get("updated_prompt", ""),
                            "approved_at": version_info.get("approved_at", "")
                        })

        if not approved_videos:
            raise HTTPException(
                status_code=404,
                detail=f"No approved videos found for shot {shot_id}. Please approve videos first using the /human-approval endpoint."
            )

        logger.info(f"Found {len(approved_videos)} approved video(s) for shot {shot_id}")

        # Get S3 client for downloading
        s3_client = get_s3_client()
        bucket_name = get_bucket_name()

        # Create ZIP file in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add a metadata file with shot information
            metadata = {
                "show_id": show_id,
                "shot_id": shot_id,
                "shot_description": shot_description,
                "approved_videos": [
                    {
                        "version": v["version"],
                        "filename": f"{shot_id}_{v['version']}_{'video' if v['index'] == 0 else 'video_' + str(v['index'])}.mp4",
                        "prompt": v["prompt"],
                        "approved_at": v["approved_at"]
                    }
                    for v in approved_videos
                ]
            }

            import json
            zip_file.writestr(
                "metadata.json",
                json.dumps(metadata, indent=2)
            )

            # Download each approved video from S3 and add to ZIP
            for video_info in approved_videos:
                try:
                    video_url = video_info["url"]
                    version = video_info["version"]
                    index = video_info["index"]

                    # Parse S3 URL to get bucket and key
                    try:
                        # Try to parse as S3 URL
                        bucket, s3_key = parse_s3_url(video_url, default_bucket=bucket_name)
                    except:
                        # If parsing fails, assume it's just a key
                        bucket = bucket_name
                        s3_key = video_url.split(f"{bucket_name}/")[-1] if f"{bucket_name}/" in video_url else video_url

                    logger.info(f"Downloading {version} from S3: {s3_key}")

                    # Download video bytes from S3
                    video_bytes = stream_bytes_from_s3(
                        video_url,
                        s3_client,
                        default_bucket=bucket_name
                    )

                    # Generate filename for the video in the ZIP
                    if index == 0:
                        filename = f"{shot_id}_{version}_video.mp4"
                    else:
                        filename = f"{shot_id}_{version}_video_{index}.mp4"

                    # Add video to ZIP
                    zip_file.writestr(filename, video_bytes)
                    logger.info(f"Added {filename} to ZIP ({len(video_bytes)} bytes)")

                except Exception as e:
                    logger.error(f"Failed to download video {video_info['url']}: {e}")
                    # Continue with other videos instead of failing completely
                    # Add error info to a separate file
                    error_msg = f"Failed to download {version}: {str(e)}\n"
                    try:
                        existing_errors = zip_file.read("errors.txt").decode('utf-8')
                        error_msg = existing_errors + error_msg
                    except:
                        pass
                    zip_file.writestr("errors.txt", error_msg)

        # Prepare ZIP for streaming
        zip_buffer.seek(0)

        # Generate filename for the download
        zip_filename = f"{show_id}_{shot_id}_approved_videos.zip"

        logger.info(f"Streaming ZIP file with {len(approved_videos)} approved video(s): {zip_filename}")

        # Return as streaming response
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download approved shots: {e}", exc_info=True)
        raise handle_api_exception(e, "download_approved_shots")
