"""
Phase 2 Celery Tasks
===================

This module contains Celery tasks for the Phase 2 workflow (7-agent shot generation pipeline).

Tasks:
------
1. run_phase2_workflow_task: Start a new Phase 2 workflow (from Agent 1)
2. resume_phase2_after_strategy_approval_task: Resume after human approves strategies (Agent 1 → Agent 2)
3. resume_phase2_after_prompt_approval_task: Resume after human approves prompts (Agent 13 → Agent 14)
4. resume_phase2_after_final_approval_task: Resume after final human approval (complete workflow)

Why Celery vs BackgroundTasks?
------------------------------
OLD (BackgroundTasks):
- Runs in FastAPI process
- Lost if server restarts
- No retry mechanism
- Hard to monitor
- Competes with API for resources

NEW (Celery + SQS):
- Runs in separate worker processes
- Persisted in SQS (survives restarts)
- Built-in retry logic
- Full monitoring via task_id
- Dedicated resources for heavy processing

Phase 2 Specifics:
------------------
- REQUIRES Phase 1 completion (depends on Agent 8 asset library)
- Has 3 human checkpoints (strategy, prompt, final approval)
- Contains edit-review loop between Agent 7 and Agent 15
- Tracks image versions (v0, v1, v2, v3) across edits
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Import Celery app
from app.celery_app import celery_app

# Import services
from app.services.pipeline_service import PipelineService
from app.services.project_service import ProjectService
from app.services.phase_2_agents.langgraph_workflow import (
    run_phase2_pipeline,
    create_phase2_workflow
)
from app.models.mongodb.shots import MongoDBAtlasClient
from app.core.idempotency import (
    get_idempotency_service,
    generate_idempotency_key,
    check_idempotency,
    mark_idempotency_completed,
    mark_idempotency_failed,
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_mongodb_client() -> Optional[MongoDBAtlasClient]:
    """
    Get the MongoDB Atlas client singleton for Phase 2.

    This function returns the singleton instance from config.py to ensure
    only ONE MongoDB connection is used across the entire application,
    preventing connection pool exhaustion.

    Returns:
        MongoDBAtlasClient instance or None if not configured
    """
    try:
        from app.config import get_mongodb_atlas_client
        return get_mongodb_atlas_client()
    except (ValueError, ConnectionError) as e:
        logger.warning(f"MongoDB Atlas client not configured or connection failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to get MongoDB Atlas client: {type(e).__name__}", exc_info=True)
        return None


def _state_has_product_shots(state: Dict[str, Any]) -> bool:
    shots = state.get("shot_list_request", {}).get("shots", [])
    return any(bool(shot.get("product_present")) for shot in shots if isinstance(shot, dict))


def _fetch_project_product_image_url(project_id: Optional[str]) -> Optional[str]:
    if not project_id:
        return None

    try:
        project_doc = ProjectService().get_project(project_id)
        return project_doc.get("product_image_s3_url") if project_doc else None
    except Exception as exc:
        logger.warning(f"Failed to fetch product_image_s3_url from project {project_id}: {exc}")
        return None


def _restore_product_image_url_if_needed(state: Dict[str, Any], *, job_id: str) -> None:
    if state.get("product_image_url") or not _state_has_product_shots(state):
        return

    show_id = state.get("show_id")
    project_id = state.get("project_id")

    product_image_url = _fetch_project_product_image_url(show_id)
    if product_image_url:
        state["product_image_url"] = product_image_url
        logger.info(f"Restored product_image_url for Phase 2 job {job_id} from show_id {show_id}")
        return

    if project_id and project_id != show_id:
        product_image_url = _fetch_project_product_image_url(project_id)
        if product_image_url:
            state["product_image_url"] = product_image_url
            logger.info(f"Restored product_image_url for Phase 2 job {job_id} from fallback project_id {project_id}")
            return

    logger.warning(
        f"Phase 2 job {job_id} has product_present shots but no product_image_s3_url "
        f"on show_id={show_id} or fallback project_id={project_id}"
    )


def check_phase1_completion(project_id: Optional[str] = None, show_id: Optional[str] = None) -> bool:
    """
    Verify Phase 1 is complete before starting Phase 2

    Phase 2 DEPENDS on Phase 1 because:
    - Agent 12 (Shot Design) needs asset library from Phase 1's Agent 8
    - Phase 2 agents reference character/location/prop assets

    Args:
        project_id: Phase 1 project identifier / assets_collection_id (optional)
        show_id: Show identifier (production_projects._id) used to resolve assets_collection_id

    Returns:
        True if Phase 1 is complete

    Raises:
        ValueError: If Phase 1 is not complete or project not found
    """
    from app.config import get_database
    from bson import ObjectId
    from backend.shared.utils.mongodb_validators import validate_object_id

    if not project_id and not show_id:
        raise ValueError("project_id or show_id is required to verify Phase 1 completion")

    # Look up assets_collection by _id (project_id is the assets_collection_id)
    client, db = get_database()
    assets_collection_col = db["assets_collections"]
    projects_collection = db["production_projects"]

    assets_collection = None
    assets_collection_id = None

    def _find_assets_collection(collection_id):
        try:
            collection_obj_id = (
                collection_id if isinstance(collection_id, ObjectId) else validate_object_id(str(collection_id))
            )
            return assets_collection_col.find_one({"_id": collection_obj_id})
        except Exception:
            return None

    # First attempt: project_id provided (legacy behavior)
    if project_id:
        assets_collection_id = project_id
        assets_collection = _find_assets_collection(assets_collection_id)

    # Fallback: resolve via show_id -> production_projects.assets_collection_id
    if not assets_collection and show_id:
        try:
            show_obj_id = validate_object_id(show_id)
        except ValueError:
            raise ValueError(f"Invalid show_id: {show_id}")

        project_doc = projects_collection.find_one({"_id": show_obj_id})
        if not project_doc:
            raise ValueError(f"No project found for show_id {show_id}")

        assets_collection_id = project_doc.get("assets_collection_id")
        if not assets_collection_id:
            raise ValueError(
                f"Project {show_id} does not have an assets_collection_id. Phase 1 must be completed first."
            )

        assets_collection = _find_assets_collection(assets_collection_id)

    if not assets_collection:
        identifier = assets_collection_id or project_id
        raise ValueError(f"No assets collection found for assets_collection_id {identifier}")

    # Check agent8_output status in assets collection
    agent8_output = assets_collection.get("agent8_output", {})
    agent8_status = agent8_output.get("status")

    if agent8_status != "completed":
        raise ValueError(
            f"Phase 1 not complete. Agent 8 status: {agent8_status or 'not started'}. "
            f"Please complete Phase 1 before starting Phase 2."
        )

    # Verify variation images exist (needed by Phase 2 Agent 12)
    agent8_data = agent8_output.get("output", {})
    variation_images = agent8_data.get("variation_images", {})
    if not variation_images:
        raise ValueError(
            "No variation images found from Phase 1 Agent 8. "
            "Phase 2 requires completed asset library."
        )

    logger.info(f"Phase 1 verified complete for movie_id {project_id}")
    return True


# ============================================================================
# Custom Task Base Class (for common functionality)
# ============================================================================

class Phase2Task(Task):
    """
    Custom base task class for Phase 2 tasks

    Why custom base class?
    ----------------------
    - Shared error handling logic
    - Automatic pipeline_service initialization
    - Common logging patterns
    - Progress reporting utilities
    """

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """
        Called when task fails

        Args:
            exc: Exception that caused failure
            task_id: Unique task ID
            args: Task positional arguments
            kwargs: Task keyword arguments
            einfo: Exception info object
        """
        logger.error(f"Phase 2 Task {task_id} failed: {exc}")
        logger.error(f"Exception info: {einfo}")

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when task succeeds"""
        logger.info(f"Phase 2 Task {task_id} completed successfully")

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """Called when task is retried"""
        logger.warning(f"Phase 2 Task {task_id} is being retried. Reason: {exc}")


# ============================================================================
# Task 1: Run Phase 2 Workflow (Full Pipeline from Agent 1)
# ============================================================================

@celery_app.task(
    bind=True,  # Pass task instance as first argument
    base=Phase2Task,  # Use custom base class
    name='phase2.run_workflow',  # Explicit task name
    max_retries=3,  # Retry up to 3 times on failure
    default_retry_delay=60,  # Wait 60 seconds between retries
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,  # Track when task starts
    time_limit=10800,  # 3 hours hard limit (Phase 2 is longer due to image gen)
    soft_time_limit=10500,  # 2h 55min soft limit
)
def run_phase2_workflow_task(
    self,
    job_id: str,
    shot_list_request: Dict[str, Any],
    show_id: str,
    episode_number: int,
    project_id: str,  # Phase 1 project_id (REQUIRED)
    scene_description: Optional[str] = None,
    movie_id: str = None,
    idempotency_key: Optional[str] = None,
    user_id: Optional[str] = None,
    v1_project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 2 workflow (7 agents) as a Celery task

    This task:
    1. Verifies Phase 1 is complete (CRITICAL DEPENDENCY)
    2. Updates job status to "running"
    3. Executes Agent 1 (Shot Strategy Agent)
    4. Pauses at human checkpoint (after Agent 1)
    5. Updates job with final state
    6. Reports progress throughout execution

    Phase 2 Flow:
    Agent 1 (Strategy) → [CHECKPOINT 1] → Agent 2 (Prompts) → Agent 3 (Review) →
    Agent 12 (Design) → Agent 13 (Modifier) → [CHECKPOINT 2] → Agent 14 (Imagen) →
    Agent 15 (Review) → Agent 7 (Editor) ↔ Agent 15 (loop) → [CHECKPOINT 3: Final]

    Args:
        self: Celery task instance (injected by bind=True)
        job_id: Pipeline job identifier
        shot_list_request: Shot list request data (episode_id, shots, etc.)
        show_id: Show identifier (MongoDB project_id)
        episode_number: Episode number
        project_id: Phase 1 project_id (REQUIRED - must be completed)
        scene_description: Optional scene description

    Returns:
        Dict with status, job_id, and pipeline_status

    Raises:
        ValueError: If Phase 1 is not complete
        SoftTimeLimitExceeded: If task exceeds soft time limit (cleanup time)
        Exception: Any other errors (will trigger retry)

    Why this approach?
    -----------------
    - Celery task_id stored in job record for status queries
    - Progress updates allow UI to show real-time status
    - Errors automatically trigger retries (resilience)
    - Task persists in SQS if worker crashes
    - Verifies dependencies before starting
    """

    pipeline_service = PipelineService()

    try:
        if not movie_id:
            raise ValueError("movie_id is required for Phase 2 pipeline")

        # ===== STEP 0: Idempotency Check =====
        scene_id = show_id or project_id or job_id
        idempotency_key_value = generate_idempotency_key(
            user_id=user_id,
            scene_id=scene_id,
            phase_number=2,
            idempotency_key_header=idempotency_key,
        )

        # Build payload for idempotency check
        # Note: job_id is NOT included because it's generated fresh for each request
        # and doesn't represent the semantic intent (which is based on show/episode/shots)
        payload = {
            "show_id": show_id,
            "episode_number": episode_number,
            "project_id": project_id,
            "shot_count": len(shot_list_request.get("shots", [])) if isinstance(shot_list_request, dict) else 0,
            "scene_description": scene_description,
            "movie_id": movie_id,
        }

        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase2.run_workflow",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )

        if is_duplicate:
            if cached_response:
                logger.info(f"Returning cached response for idempotency key: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if this is a true duplicate or first attempt
                logger.warning(f"Duplicate request detected for idempotency key: {idempotency_key_value}")
                
                # Get the idempotency record to check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase2.run_workflow",
                    key=idempotency_key_value
                )
                
                # If record has no task_id, this might be the first attempt
                # (record was created but task not yet attached)
                # Only return early if task_id is already set (true duplicate)
                if record and record.task_id:
                    logger.warning(f"Task already running with ID: {record.task_id}")
                    job = pipeline_service.get_job(job_id)
                    if job:
                        return {
                            'status': job.status,
                            'job_id': job_id,
                            'show_id': show_id,
                            'episode_number': episode_number,
                            'pipeline_status': job.pipeline_status,
                            'current_agent': getattr(job, 'current_agent', None),
                            'celery_task_id': record.task_id,
                            'message': 'Workflow already in progress',
                        }
                else:
                    # No task_id set yet - this is the first attempt
                    # Continue with execution (will attach task_id below)
                    logger.info(f"No task_id in record yet - proceeding with workflow execution")
                    # Continue to next step (don't return early)

        try:
            idempotency_service.attach_task_reference(
                endpoint="phase2.run_workflow",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                metadata={
                    "project_id": project_id,
                    "show_id": show_id,
                    "episode_number": episode_number,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to attach task reference to idempotency record: {e}")

        # ===== STEP 1: Verify Phase 1 Completion (CRITICAL) =====
        logger.info("="*80)
        logger.info("Verifying Phase 1 Completion")
        logger.info(f"Phase 1 Project ID: {project_id}")
        logger.info("="*80)

        check_phase1_completion(project_id=project_id, show_id=show_id)  # Raises ValueError if not complete

        # ===== STEP 2: Initialize Task =====
        logger.info("="*80)
        logger.info("Starting Phase 2 Workflow")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Idempotency Key: {idempotency_key_value}")
        logger.info(f"Show ID: {show_id}")
        logger.info(f"Episode Number: {episode_number}")
        logger.info(f"Phase 1 Project ID: {project_id}")
        logger.info(f"Celery Task ID: {self.request.id}")
        logger.info("="*80)

        # Update job status to running and store Celery task_id
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            started_at=datetime.utcnow(),
            celery_task_id=self.request.id  # Store for status queries
        )

        # ===== STEP 2: Report Initial Progress =====
        # This allows frontend to show "Task started, initializing agents..."
        self.update_state(
            state='PROGRESS',
            meta={
                'current_agent': 'agent_1',
                'progress': 0,
                'message': 'Initializing Phase 2 pipeline (Agent 1: Shot Strategy)...',
                'job_id': job_id,
                'show_id': show_id,
                'episode_number': episode_number,
            }
        )

        # ===== STEP 3: Get MongoDB Client =====
        mongodb_client = get_mongodb_client()
        if not mongodb_client:
            logger.warning("MongoDB client not available - will run without MongoDB persistence")

        # ===== STEP 4: Run the Phase 2 Pipeline =====
        # This runs Agent 1 and pauses at human checkpoint
        logger.info(f"Processing {len(shot_list_request.get('shots', []))} shots")

        final_state = run_phase2_pipeline(
            shot_list_request=shot_list_request,
            show_id=show_id,
            episode_number=episode_number,
            scene_description=scene_description,
            mongodb_client=mongodb_client,
            strategy_approval=None,  # Start from beginning (Agent 1)
            project_id=project_id,  # Pass Phase 1 project_id
            job_id=job_id,  # Pass job_id for tracking
            movie_id=movie_id,  # Pass movie_id to fetch visual_style
            v1_project_id=v1_project_id,  # Pass v1 project id to fetch product image URL
        )

        # ===== STEP 5: Update Job with Final State =====
        pipeline_service.update_job_state(job_id, final_state)

        # ===== STEP 6: Report Completion =====
        final_status = final_state.get('pipeline_status', 'completed')
        logger.info("="*80)
        logger.info("Phase 2 Workflow Task Completed")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final Status: {final_status}")
        logger.info(f"Current Agent: {final_state.get('current_agent', 'N/A')}")
        logger.info(f"Requires Feedback: {final_state.get('requires_human_feedback', False)}")
        logger.info("="*80)

        # Return result (stored in MongoDB result backend)
        result = {
            'status': 'completed',
            'job_id': job_id,
            'show_id': show_id,
            'episode_number': episode_number,
            'pipeline_status': final_status,
            'current_agent': final_state.get('current_agent'),
            'requires_human_feedback': final_state.get('requires_human_feedback', False),
            'feedback_agent': final_state.get('feedback_agent'),
            'celery_task_id': self.request.id,
        }
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase2.run_workflow",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                response_payload=result,
            )
        except Exception as e:
            logger.warning(f"Failed to mark idempotency as completed: {e}")
        
        return result

    except ValueError as ve:
        # Phase 1 dependency error - don't retry
        logger.error(f"Phase 1 dependency error: {ve}")

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="dependency_error",
            error_message=str(ve)
        )

        # Don't retry for dependency errors
        raise

    except SoftTimeLimitExceeded:
        # Soft time limit reached - cleanup time before hard kill
        logger.warning(f"Task {self.request.id} exceeded soft time limit")
        logger.warning("Performing cleanup before termination...")

        # Check if job is at a checkpoint - don't overwrite if waiting for approval
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status != "waiting_for_human_approval":
            pipeline_service.update_job_status(
                job_id,
                status="failed",
                pipeline_status="timeout",
                error_message="Task exceeded time limit (3 hours)"
            )

        # Re-raise to mark task as failed
        raise

    except Exception as e:
        # Any other error - check if workflow reached a checkpoint before failing
        logger.error(f"Error in Phase 2 workflow: {type(e).__name__}", exc_info=True)

        # Check current job status - don't overwrite if already at a checkpoint
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status == "waiting_for_human_approval":
            logger.warning(f"Job {job_id} is at checkpoint (waiting for approval). Not overwriting status due to exception.")
            logger.warning(f"Exception was: {e}")
            # Don't retry or set to failed - job is successfully waiting for approval
            return {
                'status': 'completed',
                'job_id': job_id,
                'pipeline_status': 'waiting_for_approval',
                'message': 'Workflow paused at checkpoint (Agent 1 completed, waiting for strategy approval)',
                'celery_task_id': self.request.id,
            }
        
        # Mark idempotency as failed
        try:
            scene_id = show_id or project_id or job_id
            idempotency_key_value = generate_idempotency_key(
                user_id=user_id,
                scene_id=scene_id,
                phase_number=2,
                idempotency_key_header=idempotency_key,
            )
            mark_idempotency_failed(
                endpoint="phase2.run_workflow",
                idempotency_key=idempotency_key_value,
                error_message=f"Workflow execution failed: {str(e)}",
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")
        
        # Otherwise, update job with error
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow execution failed"
        )

        # Retry task (Celery will handle retry logic)
        # countdown=60 means wait 60 seconds before retry
        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# Task 2: Resume Phase 2 After Strategy Approval (Checkpoint 1)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase2Task,
    name='phase2.resume_after_strategy_approval',
    max_retries=3,
    default_retry_delay=60,
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,
    time_limit=10800,  # 3 hours
    soft_time_limit=10500,
)
def resume_phase2_after_strategy_approval_task(
    self,
    job_id: str,
    show_id: str,
    episode_number: int,
    approval_decision: bool,
    feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Resume Phase 2 workflow after human approves/rejects strategies (Agent 1)

    This task is used after:
    1. User approves strategies (/approve-strategy)
    2. User rejects strategies (workflow ends)

    If approved: Continues to Agent 2 (Image Prompt Generator)
    If rejected: Ends workflow with status "rejected"

    Args:
        self: Celery task instance
        job_id: Pipeline job identifier
        show_id: Show identifier
        episode_number: Episode number
        approval_decision: True = approve and continue, False = reject and end
        feedback: Optional human feedback/comments

    Returns:
        Dict with status and completion info

    Why separate task?
    -----------------
    - Resume operations have different characteristics than new runs
    - Easier to track which tasks are continuations vs new starts
    - Can apply different retry policies if needed
    """

    pipeline_service = PipelineService()

    try:
        logger.info("="*80)
        logger.info("Resuming Phase 2 After Strategy Approval")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Show ID: {show_id}")
        logger.info(f"Approval Decision: {'APPROVED' if approval_decision else 'REJECTED'}")
        logger.info(f"Celery Task ID: {self.request.id}")
        logger.info("="*80)

        # Get job to retrieve current state
        job = pipeline_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Verify job is in a valid state for resuming
        # Accept: waiting_for_human_approval (expected) or running (already resumed)
        valid_statuses = ["waiting_for_human_approval", "running"]
        if job.status not in valid_statuses:
            raise ValueError(
                f"Job cannot be resumed. Current status: {job.status}. "
                f"Expected one of: {', '.join(valid_statuses)}"
            )

        # If job is already running, check if it's a duplicate task
        if job.status == "running":
            logger.warning(f"Job {job_id} is already running. Checking if this is a duplicate task...")
            if job.celery_task_id and job.celery_task_id != self.request.id:
                logger.warning(f"Another task {job.celery_task_id} is already processing this job. Exiting.")
                return {
                    'status': 'duplicate',
                    'job_id': job_id,
                    'message': 'Job is already being processed by another task',
                    'active_task_id': job.celery_task_id,
                    'celery_task_id': self.request.id,
                }
            logger.info("This is a retry of the same task. Continuing...")

        # Get current state from job
        current_state = job.state if hasattr(job, 'state') else {}

        # Load annotated shots from MongoDB
        mongodb_client = get_mongodb_client()
        if mongodb_client:
            try:
                logger.info("Loading annotated shots from MongoDB...")
                shot_collection = mongodb_client.get_shots_from_atlas(show_id, episode_number)
                if shot_collection:
                    from app.models.mongodb.shots import AnnotatedShotList, AnnotatedShotItem

                    annotated_shots = []
                    for shot_data in shot_collection.annotated_shots:
                        shot_item = AnnotatedShotItem(**shot_data.model_dump())
                        annotated_shots.append(shot_item)

                    annotated_shot_list = AnnotatedShotList(
                        episode_id=shot_collection.episode_id,
                        title=shot_collection.title,
                        annotated_shots=annotated_shots,
                        overall_continuity_notes=shot_collection.overall_continuity_notes,
                        strategy_summary=shot_collection.strategy_summary
                    )

                    current_state["annotated_shot_list"] = annotated_shot_list
                    current_state["strategy_analysis_results"] = annotated_shot_list.model_dump()
                    logger.info(f"Loaded {len(annotated_shots)} annotated shots from MongoDB")
            except Exception as e:
                logger.warning(f"Failed to load shots from MongoDB: {e}")

        # Ensure critical state fields are preserved/restored
        # MongoDB client is needed for asset library and database operations
        if mongodb_client:
            current_state["mongodb_client"] = mongodb_client

        # Ensure show_id and episode_number are in state (fallback to function parameters)
        if "show_id" not in current_state:
            current_state["show_id"] = show_id
        if "episode_number" not in current_state:
            current_state["episode_number"] = episode_number

        _restore_product_image_url_if_needed(current_state, job_id=job_id)

        # Log warnings if critical fields are missing
        if "movie_id" not in current_state:
            logger.warning(f"movie_id missing from saved state for job {job_id}")
        if "project_id" not in current_state:
            logger.warning(f"project_id missing from saved state for job {job_id}")

        # Update state with approval decision
        current_state["strategy_approval_decision"] = approval_decision
        current_state["strategy_approval_feedback"] = feedback

        # If rejected, end workflow
        if not approval_decision:
            logger.info("Strategies rejected by user - ending workflow")

            pipeline_service.update_job_status(
                job_id,
                status="rejected",
                pipeline_status="rejected",
                celery_task_id=self.request.id
            )

            return {
                'status': 'rejected',
                'job_id': job_id,
                'pipeline_status': 'rejected',
                'message': 'Strategies rejected by user',
                'celery_task_id': self.request.id,
            }

        # If approved, continue workflow
        logger.info("Strategies approved - continuing to Agent 2 (Image Prompt Generator)")

        # Update status to running
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            celery_task_id=self.request.id
        )

        # Report progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_agent': 'agent_2',
                'progress': 20,  # 20% through pipeline
                'message': 'Resuming from Agent 2 (Image Prompt Generator)...',
                'job_id': job_id,
            }
        )

        # Create workflow and resume from checkpoint
        app = create_phase2_workflow("agent_2")

        # Set current agent to agent_2 to continue
        current_state["current_agent"] = "agent_2"
        current_state["agent1_status"] = "completed"
        current_state["pipeline_status"] = "running"
        # Ensure job_id is preserved
        if "job_id" not in current_state:
            current_state["job_id"] = job_id

        final_state = app.invoke(current_state)

        # Update job with final state
        pipeline_service.update_job_state(job_id, final_state)

        logger.info("="*80)
        logger.info("Phase 2 Workflow Resumed and Continued")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final Status: {final_state.get('pipeline_status')}")
        logger.info("="*80)

        return {
            'status': 'completed',
            'job_id': job_id,
            'pipeline_status': final_state.get('pipeline_status'),
            'current_agent': final_state.get('current_agent'),
            'requires_human_feedback': final_state.get('requires_human_feedback', False),
            'celery_task_id': self.request.id,
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Resume task {self.request.id} exceeded soft time limit")

        # Check if job reached next checkpoint - don't overwrite if waiting for approval
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status not in ["waiting_for_human_approval", "phase_3_checkpoint"]:
            pipeline_service.update_job_status(
                job_id,
                status="failed",
                pipeline_status="timeout",
                error_message="Resume task exceeded time limit"
            )

        raise

    except Exception as e:
        logger.error(f"Error resuming Phase 2 workflow: {type(e).__name__}", exc_info=True)

        # Check if workflow reached next checkpoint before failing
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status in ["waiting_for_human_approval", "phase_3_checkpoint"]:
            logger.warning(f"Job {job_id} reached next checkpoint. Not overwriting status due to exception.")
            logger.warning(f"Exception was: {e}")
            # Job successfully reached next checkpoint
            return {
                'status': 'completed',
                'job_id': job_id,
                'pipeline_status': current_job.pipeline_status,
                'message': 'Workflow paused at next checkpoint',
                'celery_task_id': self.request.id,
            }

        # Otherwise, update job with error
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow resume failed"
        )

        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# Task 3: Resume Phase 2 After Prompt Approval (Checkpoint 2)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase2Task,
    name='phase2.resume_after_prompt_approval',
    max_retries=3,
    default_retry_delay=60,
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,
    time_limit=10800,
    soft_time_limit=10500,
)
def resume_phase2_after_prompt_approval_task(
    self,
    job_id: str,
    approval_decision: bool,
    feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Resume Phase 2 workflow after human approves/rejects corrected prompts (Agent 13)

    This task is used after:
    1. User approves corrected prompts (/approve-prompts)
    2. User rejects prompts (workflow ends)

    If approved: Continues to Agent 14 (Imagen Generator)
    If rejected: Ends workflow with status "rejected"

    Args:
        self: Celery task instance
        job_id: Pipeline job identifier
        approval_decision: True = approve and continue, False = reject and end
        feedback: Optional human feedback/comments

    Returns:
        Dict with status and completion info
    """

    pipeline_service = PipelineService()

    try:
        logger.info("="*80)
        logger.info("Resuming Phase 2 After Prompt Approval")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Approval Decision: {'APPROVED' if approval_decision else 'REJECTED'}")
        logger.info(f"Celery Task ID: {self.request.id}")
        logger.info("="*80)

        # Get job to retrieve current state
        job = pipeline_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Verify job is in a valid state for resuming
        # Accept: waiting_for_prompt_approval (expected) or running (already resumed)
        valid_pipeline_statuses = ["waiting_for_prompt_approval", "running"]
        if job.pipeline_status not in valid_pipeline_statuses:
            raise ValueError(
                f"Job cannot be resumed from prompt approval checkpoint. "
                f"Current pipeline_status: {job.pipeline_status}. "
                f"Expected one of: {', '.join(valid_pipeline_statuses)}"
            )
        
        # Check if another task is already processing this job
        if job.pipeline_status == "running":
            logger.warning(f"Job {job_id} is already running. Checking if this is a duplicate task...")
            if job.celery_task_id and job.celery_task_id != self.request.id:
                logger.warning(f"Another task {job.celery_task_id} is already processing this job. Exiting.")
                return {
                    'status': 'duplicate',
                    'job_id': job_id,
                    'message': 'Job is already being processed by another task',
                    'active_task_id': job.celery_task_id,
                    'celery_task_id': self.request.id,
                }
            logger.info("This is a retry of the same task. Continuing...")

        # Get current state from job
        current_state = job.state if hasattr(job, 'state') else {}

        # Ensure critical state fields are preserved/restored
        # MongoDB client is needed for asset library and database operations
        mongodb_client = get_mongodb_client()
        if mongodb_client:
            current_state["mongodb_client"] = mongodb_client

        # Log warnings if critical fields are missing from saved state
        if "movie_id" not in current_state:
            logger.warning(f"movie_id missing from saved state for job {job_id}")
        if "project_id" not in current_state:
            logger.warning(f"project_id missing from saved state for job {job_id}")
        if "show_id" not in current_state:
            logger.warning(f"show_id missing from saved state for job {job_id}")
        if "episode_number" not in current_state:
            logger.warning(f"episode_number missing from saved state for job {job_id}")

        _restore_product_image_url_if_needed(current_state, job_id=job_id)

        # Update state with approval decision
        current_state["prompt_approval_decision"] = approval_decision
        current_state["prompt_approval_feedback"] = feedback

        # If rejected, end workflow
        if not approval_decision:
            logger.info("Prompts rejected by user - ending workflow")

            pipeline_service.update_job_status(
                job_id,
                status="rejected",
                pipeline_status="rejected",
                celery_task_id=self.request.id
            )

            return {
                'status': 'rejected',
                'job_id': job_id,
                'pipeline_status': 'rejected',
                'message': 'Corrected prompts rejected by user',
                'celery_task_id': self.request.id,
            }

        # If approved, continue workflow
        logger.info("Prompts approved - continuing to Agent 14 (Imagen Generator)")

        # Update status to running
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            celery_task_id=self.request.id
        )

        # Report progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_agent': 'agent_14',
                'progress': 60,  # 60% through pipeline
                'message': 'Resuming from Agent 14 (Imagen Generator)...',
                'job_id': job_id,
            }
        )

        # Create workflow and resume from checkpoint
        app = create_phase2_workflow("agent_14")

        # Set current agent to agent_14 to continue
        current_state["current_agent"] = "agent_14"
        current_state["pipeline_status"] = "running"
        # Ensure job_id is preserved
        if "job_id" not in current_state:
            current_state["job_id"] = job_id

        final_state = app.invoke(current_state)

        # Update job with final state
        pipeline_service.update_job_state(job_id, final_state)

        logger.info("="*80)
        logger.info("Phase 2 Workflow Resumed After Prompt Approval")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final Status: {final_state.get('pipeline_status')}")
        logger.info("="*80)

        return {
            'status': 'completed',
            'job_id': job_id,
            'pipeline_status': final_state.get('pipeline_status'),
            'current_agent': final_state.get('current_agent'),
            'requires_human_feedback': final_state.get('requires_human_feedback', False),
            'celery_task_id': self.request.id,
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Resume task {self.request.id} exceeded soft time limit")

        # Check if job reached next checkpoint - don't overwrite if waiting for approval
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status not in ["waiting_for_human_approval", "phase_3_checkpoint", "completed"]:
            pipeline_service.update_job_status(
                job_id,
                status="failed",
                pipeline_status="timeout",
                error_message="Resume task exceeded time limit"
            )

        raise

    except Exception as e:
        logger.error(f"Error resuming Phase 2 after prompt approval: {type(e).__name__}", exc_info=True)

        # Check if workflow reached next checkpoint or completed before failing
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status in ["waiting_for_human_approval", "phase_3_checkpoint", "completed"]:
            logger.warning(f"Job {job_id} reached checkpoint or completed. Not overwriting status due to exception.")
            logger.warning(f"Exception was: {e}")
            # Job successfully reached next checkpoint or completed
            return {
                'status': 'completed',
                'job_id': job_id,
                'pipeline_status': current_job.pipeline_status,
                'message': 'Workflow reached checkpoint or completed',
                'celery_task_id': self.request.id,
            }

        # Otherwise, update job with error
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow resume failed"
        )

        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# Task 4: Resume Phase 2 After Final Approval (Checkpoint 3)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase2Task,
    name='phase2.resume_after_final_approval',
    max_retries=3,
    default_retry_delay=60,
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,
)
def resume_phase2_after_final_approval_task(
    self,
    job_id: str,
    approval_decision: bool,
    feedback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Resume Phase 2 workflow after final human approval of all images

    This task is used after:
    1. User gives final approval (/final-approve)
    2. User rejects (workflow ends)

    If approved: Marks workflow as completed
    If rejected: Ends workflow with status "rejected"

    Args:
        self: Celery task instance
        job_id: Pipeline job identifier
        approval_decision: True = approve and complete, False = reject and end
        feedback: Optional human feedback/comments

    Returns:
        Dict with status and completion info
    """

    pipeline_service = PipelineService()

    try:
        logger.info("="*80)
        logger.info("Processing Final Approval for Phase 2")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Approval Decision: {'APPROVED' if approval_decision else 'REJECTED'}")
        logger.info(f"Celery Task ID: {self.request.id}")
        logger.info("="*80)

        # Get job to retrieve current state
        job = pipeline_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Verify job is in a valid state for final approval
        # Accept: waiting_for_final_approval (expected), completed (already processed), or rejected (already processed)
        valid_pipeline_statuses = ["waiting_for_final_approval", "completed", "rejected"]
        if job.pipeline_status not in valid_pipeline_statuses:
            raise ValueError(
                f"Job cannot process final approval. "
                f"Current pipeline_status: {job.pipeline_status}. "
                f"Expected one of: {', '.join(valid_pipeline_statuses)}"
            )
        
        # If already completed or rejected, return current status (idempotent)
        if job.pipeline_status in ["completed", "rejected"]:
            logger.info(f"Job {job_id} already in final state: {job.pipeline_status}. Returning current status.")
            return {
                'status': job.pipeline_status,
                'job_id': job_id,
                'pipeline_status': job.pipeline_status,
                'message': f'Job already {job.pipeline_status}',
                'celery_task_id': self.request.id,
            }

        # Update job with final decision
        if approval_decision:
            logger.info("Final approval granted - workflow complete")

            pipeline_service.update_job_status(
                job_id,
                status="completed",
                pipeline_status="completed",
                completed_at=datetime.utcnow(),
                celery_task_id=self.request.id
            )

            return {
                'status': 'completed',
                'job_id': job_id,
                'pipeline_status': 'completed',
                'message': 'Phase 2 workflow completed successfully',
                'celery_task_id': self.request.id,
            }
        else:
            logger.info("Final approval rejected by user")

            pipeline_service.update_job_status(
                job_id,
                status="rejected",
                pipeline_status="rejected",
                completed_at=datetime.utcnow(),
                celery_task_id=self.request.id
            )

            return {
                'status': 'rejected',
                'job_id': job_id,
                'pipeline_status': 'rejected',
                'message': 'Final images rejected by user',
                'celery_task_id': self.request.id,
            }

    except Exception as e:
        logger.error(f"Error processing final approval: {type(e).__name__}", exc_info=True)

        # Check if job was already completed before the exception
        current_job = pipeline_service.get_job(job_id)
        if current_job and current_job.status == "completed":
            logger.warning(f"Job {job_id} already completed. Not overwriting status due to exception.")
            logger.warning(f"Exception was: {e}")
            # Job was successfully completed
            return {
                'status': 'completed',
                'job_id': job_id,
                'pipeline_status': 'completed',
                'message': 'Final approval processed successfully',
                'celery_task_id': self.request.id,
            }

        # Otherwise, update job with error
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Final approval processing failed"
        )

        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'run_phase2_workflow_task',
    'resume_phase2_after_strategy_approval_task',
    'resume_phase2_after_prompt_approval_task',
    'resume_phase2_after_final_approval_task',
]
