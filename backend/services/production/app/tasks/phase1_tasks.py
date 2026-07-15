"""
Phase 1 Celery Tasks
===================

This module contains Celery tasks for the Phase 1 workflow (8-agent asset generation pipeline).

Tasks:
------
1. run_phase1_workflow_task: Start a new Phase 1 workflow
2. resume_phase1_workflow_task: Resume from human checkpoint
3. retry_failed_asset_task: Retry a single failed asset generation

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
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List
from typing_extensions import Optional
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
from app.services.movie_service import MovieService
from app.services.phase_1_agents.langgraph_workflow import (
    run_phase1_pipeline,
    create_phase1_workflow
)
from app.services.phase_1_agents.agent_5_image_generator import ImageGeneratorAgent
from app.core.idempotency import (
    get_idempotency_service,
    generate_idempotency_key,
    check_idempotency,
    mark_idempotency_completed,
    mark_idempotency_failed,
)
from app.utils.csv_parser import ShotData
from app.config import get_s3_client, get_bucket_name
import json
from botocore.exceptions import ClientError


# ============================================================================
# Custom Task Base Class (for common functionality)
# ============================================================================

class Phase1Task(Task):
    """
    Custom base task class for Phase 1 tasks

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
        logger.error(f"Task {task_id} failed: {exc}")
        logger.error(f"Exception info: {einfo}")

    def on_success(self, retval: Any, task_id: str, args: tuple, kwargs: dict) -> None:
        """Called when task succeeds"""
        logger.info(f"Task {task_id} completed successfully")

    def on_retry(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """Called when task is retried"""
        logger.warning(f"Task {task_id} is being retried. Reason: {exc}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_movie_shotlist(movie_id: str) -> Optional[List[ShotData]]:
    """
    Load and parse all shotlists from all scenes for a movie.

    Args:
        movie_id: Movie ID

    Returns:
        List of ShotData objects from all scenes, or None if no shotlist available
    """
    try:
        # Get movie document
        movie_service = MovieService()
        movie = movie_service.get_movie_by_id(movie_id)
        if not movie:
            logger.warning(f"Movie {movie_id} not found")
            return None

        scenes = movie.get("scenes", [])
        if not scenes:
            logger.info("Movie has no scenes")
            return None

        # Get S3 client
        s3_client = get_s3_client()
        bucket_name = get_bucket_name()

        if not s3_client or not bucket_name:
            logger.warning("S3 client or bucket not configured")
            return None

        # Load shotlist from each scene
        all_shots = []
        for scene in scenes:
            scene_number = scene.get("scene_number")
            if not scene_number:
                continue

            # Build S3 key for this scene's shotlist
            shotlist_key = f"movies/{movie_id}/scenes/scene_{scene_number:02d}_shotlist.json"

            try:
                # Load shotlist JSON from S3
                obj = s3_client.get_object(Bucket=bucket_name, Key=shotlist_key)
                payload = obj["Body"].read().decode("utf-8")
                shotlist_json = json.loads(payload)

                # Extract shots array
                shots_array = shotlist_json.get("shot_list", {}).get("shots", [])

                # Convert to ShotData objects
                for shot_dict in shots_array:
                    scene_num = shot_dict.get("scene_number")
                    seq_num = shot_dict.get("sequence_number")
                    # Reconstruct shot_number in format "scene.sequence" (e.g., "1.1")
                    shot_number = f"{scene_num}.{seq_num}" if scene_num and seq_num else str(seq_num)

                    shot_data = ShotData(
                        scene_number=scene_num,
                        shot_number=shot_number,
                        shot_type=shot_dict.get("shot_style", ""),
                        camera_movement=shot_dict.get("camera_movement", ""),
                        description=shot_dict.get("description", ""),
                        characters=shot_dict.get("characters", []),
                        locations=shot_dict.get("locations", ""),
                        product_present=shot_dict.get("product_present", False),
                    )
                    all_shots.append(shot_data)

                logger.info(f"Loaded {len(shots_array)} shots from scene {scene_number}")

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                if error_code == "NoSuchKey":
                    logger.info(f"No shotlist found for scene {scene_number} (key: {shotlist_key})")
                else:
                    logger.warning(f"Error loading shotlist for scene {scene_number}: {e}")
            except Exception as e:
                logger.warning(f"Error processing shotlist for scene {scene_number}: {e}")

        if all_shots:
            logger.info(f"✓ Loaded total of {len(all_shots)} shots from {len(scenes)} scenes")
            return all_shots
        else:
            logger.info("No shotlist data found for any scene")
            return None

    except Exception as e:
        logger.error(f"Error loading movie shotlist: {e}")
        return None


# ============================================================================
# Task 1: Run Phase 1 Workflow (Full Pipeline)
# ============================================================================

@celery_app.task(
    bind=True,  # Pass task instance as first argument
    base=Phase1Task,  # Use custom base class
    name='phase1.run_workflow',  # Explicit task name
    max_retries=3,  # Retry up to 3 times on failure
    default_retry_delay=60,  # Wait 60 seconds between retries
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,  # Track when task starts
)
def run_phase1_workflow_task(
    self,
    job_id: str,
    movie_id: Optional[str] = None,
    assets_collection_id: Optional[str] = None,
    project_id: Optional[str] = None,
    script_content: Optional[str] = None,
    visual_style: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    user_id: Optional[str] = None,
    v1_project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 1 workflow (8 agents) as a Celery task

    Supports two modes:
    - Movie mode: movie_id + assets_collection_id (saves to assets_collections)
    - Legacy mode: project_id (saves to production_projects)

    This task:
    1. Updates job status to "running"
    2. Executes all 8 agents in the Phase 1 pipeline
    3. Pauses at human checkpoint (after Agent 6/7)
    4. Updates job with final state
    5. Reports progress throughout execution

    Args:
        self: Celery task instance (injected by bind=True)
        job_id: Pipeline job identifier
        movie_id: Movie identifier (for movie workflow, optional)
        assets_collection_id: Assets collection identifier (for movie workflow, optional)
        project_id: Project identifier (for legacy workflow, optional)
        script_content: Script content to process
        visual_style: Visual style for image generation (realistic, pixar, anime, etc.)

    Returns:
        Dict with status, job_id, and pipeline_status

    Raises:
        SoftTimeLimitExceeded: If task exceeds soft time limit (cleanup time)
        Exception: Any other errors (will trigger retry)

    Why this approach?
    -----------------
    - Celery task_id stored in job record for status queries
    - Progress updates allow UI to show real-time status
    - Errors automatically trigger retries (resilience)
    - Task persists in SQS if worker crashes
    """

    pipeline_service = PipelineService()

    try:
        # ===== STEP 0: Idempotency Check =====
        # Generate idempotency key
        scene_id = project_id or movie_id or assets_collection_id or job_id
        idempotency_key_value = generate_idempotency_key(
            user_id=user_id,
            scene_id=scene_id,
            phase_number=1,
            idempotency_key_header=idempotency_key,
        )
        
        # Build payload for idempotency check
        # Note: job_id is NOT included because it's generated fresh for each request
        # and doesn't represent the semantic intent (which is based on project/movie/script)
        payload = {
            "project_id": project_id,
            "movie_id": movie_id,
            "assets_collection_id": assets_collection_id,
            "script_length": len(script_content) if script_content else 0,
            "visual_style": visual_style,
        }
        
        # Check idempotency
        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase1.run_workflow",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )
        
        if is_duplicate:
            if cached_response:
                # Return cached response
                logger.info(f"Returning cached response for idempotency key: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if this is a true duplicate or first attempt
                logger.warning(f"Duplicate request detected for idempotency key: {idempotency_key_value}")
                
                # Get the idempotency record to check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase1.run_workflow",
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
                            'project_id': project_id,
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
        
        # Attach task reference to idempotency record
        try:
            idempotency_service.attach_task_reference(
                endpoint="phase1.run_workflow",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                metadata={
                    "project_id": project_id,
                    "movie_id": movie_id,
                    "assets_collection_id": assets_collection_id,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to attach task reference to idempotency record: {e}")

        # ===== STEP 1: Initialize Task =====
        logger.info("="*80)
        logger.info("Starting Phase 1 Workflow")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Idempotency Key: {idempotency_key_value}")
        if movie_id:
            logger.info(f"Movie ID: {movie_id}")
            logger.info(f"Assets Collection ID: {assets_collection_id}")
            logger.info("Mode: Movie Workflow")
        else:
            logger.info(f"Project ID: {project_id}")
            logger.info("Mode: Legacy Workflow")
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
        progress_meta = {
            'current_agent': 'initializing',
            'progress': 0,
            'message': 'Initializing Phase 1 pipeline...',
            'job_id': job_id,
        }
        if movie_id:
            progress_meta['movie_id'] = movie_id
            progress_meta['assets_collection_id'] = assets_collection_id
        if project_id:
            progress_meta['project_id'] = project_id

        self.update_state(state='PROGRESS', meta=progress_meta)

        # ===== STEP 3: Load Shotlist + resolve v1_project_id (if movie_id provided) =====
        shotlist_shots = None
        if movie_id:
            logger.info(f"Loading shotlist for movie {movie_id}")
            shotlist_shots = _load_movie_shotlist(movie_id)
            if shotlist_shots:
                logger.info(f"✓ Loaded shotlist with {len(shotlist_shots)} shots")
            else:
                logger.info("No shotlist available - Agent 1 will extract from script only")

            # Resolve v1_project_id from movie document if not explicitly provided
            if not v1_project_id:
                try:
                    movie_service_local = MovieService()
                    movie_doc = movie_service_local.get_movie_by_id(movie_id)
                    if movie_doc:
                        v1_project_id = movie_doc.get("v1_project_id")
                        if v1_project_id:
                            logger.info(f"✓ Resolved v1_project_id from movie: {v1_project_id}")
                except Exception as exc:
                    logger.warning(f"Could not resolve v1_project_id from movie: {exc}")

        # ===== STEP 4: Run the Phase 1 Pipeline =====
        # This runs all 8 agents using LangGraph workflow
        logger.info(f"Processing script ({len(script_content)} characters)")
        if visual_style:
            logger.info(f"Visual style: {visual_style}")

        final_state = run_phase1_pipeline(
            script_content=script_content,
            project_id=project_id,
            movie_id=movie_id,
            assets_collection_id=assets_collection_id,
            job_id=job_id,
            visual_style=visual_style,
            shotlist_shots=shotlist_shots,
            v1_project_id=v1_project_id,
        )

        # ===== STEP 5: Update Job with Final State =====
        pipeline_service.update_job_state(job_id, final_state)

        # ===== STEP 6: Report Completion =====
        final_status = final_state.get('pipeline_status', 'completed')
        logger.info("="*80)
        logger.info("Phase 1 Workflow Completed")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final Status: {final_status}")
        logger.info(f"Current Agent: {final_state.get('current_agent', 'N/A')}")
        logger.info("="*80)

        # Return result (stored in MongoDB result backend)
        result = {
            'status': 'completed',
            'job_id': job_id,
            'project_id': project_id,
            'pipeline_status': final_status,
            'current_agent': final_state.get('current_agent'),
            'celery_task_id': self.request.id,
        }
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase1.run_workflow",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                response_payload=result,
            )
        except Exception as e:
            logger.warning(f"Failed to mark idempotency as completed: {e}")
        
        return result

    except SoftTimeLimitExceeded:
        # Soft time limit reached - cleanup time before hard kill
        logger.warning(f"Task {self.request.id} exceeded soft time limit")
        logger.warning("Performing cleanup before termination...")

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="timeout",
            error_message="Task exceeded time limit (2 hours)"
        )

        # Re-raise to mark task as failed
        raise

    except Exception as e:
        # Any other error - update job and retry (use safe error message)
        logger.error(f"Error in Phase 1 workflow: {type(e).__name__}", exc_info=True)

        # Mark idempotency as failed
        try:
            scene_id = project_id or movie_id or assets_collection_id or job_id
            idempotency_key_value = generate_idempotency_key(
                user_id=user_id,
                scene_id=scene_id,
                phase_number=1,
                idempotency_key_header=idempotency_key,
            )
            mark_idempotency_failed(
                endpoint="phase1.run_workflow",
                idempotency_key=idempotency_key_value,
                error_message=f"Workflow execution failed: {str(e)}",
            )
        except Exception as idemp_error:
            logger.warning(f"Failed to mark idempotency as failed: {idemp_error}")

        # Update job with error
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
# Task 2: Resume Phase 1 Workflow (from Checkpoint)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase1Task,
    name='phase1.resume_workflow',
    max_retries=3,
    default_retry_delay=60,
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,
)
def resume_phase1_workflow_task(
    self,
    job_id: str,
    current_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Resume Phase 1 workflow from human checkpoint as a Celery task

    This task is used after:
    1. User approves assets (/checkpoint/finalize)
    2. User edits a prompt (/checkpoint/edit-prompt)

    It resumes the workflow from the saved state and continues processing.

    Args:
        self: Celery task instance
        job_id: Pipeline job identifier
        current_state: Current workflow state (from checkpoint)

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
        logger.info("Resuming Phase 1 Workflow")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"From Agent: {current_state.get('current_agent')}")
        logger.info(f"Celery Task ID: {self.request.id}")
        logger.info("="*80)

        # Update status to running
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status=current_state.get("pipeline_status", "running"),
            celery_task_id=self.request.id
        )

        # Report progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_agent': current_state.get('current_agent'),
                'progress': 50,  # Already halfway through
                'message': f"Resuming from {current_state.get('current_agent')}...",
                'job_id': job_id,
            }
        )

        # Create workflow and resume from checkpoint
        app = create_phase1_workflow()
        final_state = app.invoke(current_state)

        # Update job with final state
        pipeline_service.update_job_state(job_id, final_state)

        logger.info("="*80)
        logger.info("Phase 1 Workflow Resumed and Completed")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"Final Status: {final_state.get('pipeline_status')}")
        logger.info("="*80)

        return {
            'status': 'completed',
            'job_id': job_id,
            'pipeline_status': final_state.get('pipeline_status'),
            'celery_task_id': self.request.id,
        }

    except SoftTimeLimitExceeded:
        logger.warning(f"Resume task {self.request.id} exceeded soft time limit")

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="timeout",
            error_message="Resume task exceeded time limit"
        )

        raise

    except Exception as e:
        logger.error(f"Error resuming Phase 1 workflow: {type(e).__name__}", exc_info=True)

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Workflow resume failed"
        )

        raise self.retry(exc=e, countdown=60, max_retries=3)


# ============================================================================
# Task 3: Retry Single Failed Asset
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase1Task,
    name='phase1.retry_failed_asset',
    max_retries=2,  # Fewer retries for individual assets
    default_retry_delay=30,
    acks_late=False,  # Acknowledge immediately to prevent duplicate processing
    track_started=True,
)
def retry_failed_asset_task(
    self,
    job_id: str,
    asset_id: str,
    asset_name: str,
    asset_type: str
) -> Dict[str, Any]:
    """
    Retry generation of a single failed asset as a Celery task

    This allows users to retry specific failed assets without re-running
    the entire pipeline.

    Args:
        self: Celery task instance
        job_id: Pipeline job identifier
        asset_id: UUID of the failed asset
        asset_name: Name of the asset
        asset_type: Type (character, location, prop)

    Returns:
        Dict with success status and details

    Why as Celery task?
    ------------------
    - Asset generation can take 1-2 minutes
    - Multiple assets can be retried in parallel
    - Each retry tracked independently
    """

    pipeline_service = PipelineService()
    project_service = ProjectService()

    try:
        logger.info("Retrying Asset Generation")
        logger.info(f"Asset: {asset_name} ({asset_type})")
        logger.info(f"Asset ID: {asset_id}")
        logger.info(f"Job ID: {job_id}")

        # Get job and project
        job = pipeline_service.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        project = project_service.get_project(job.project_id)
        if not project:
            raise ValueError(f"Project {job.project_id} not found")

        # Get state from job
        state = job.state if hasattr(job, 'state') else {}

        # Verify Agent 5 has run
        agent5_status = state.get("agent5_status")
        if agent5_status != "completed":
            raise ValueError(f"Agent 5 hasn't completed yet. Status: {agent5_status}")

        # Initialize Agent 5
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not configured")

        agent = ImageGeneratorAgent(api_key=api_key)

        # Restore agent state
        agent.final_prompts = state.get("optimized_prompts", {})
        agent.generated_images = state.get("generated_images", {
            "characters": {}, "locations": {}, "props": {}
        })
        agent.failed_generations = state.get("failed_generations", [])

        logger.info(f"Current failed count: {len(agent.failed_generations)}")

        # Retry the asset
        result = agent.retry_single_asset(
            asset_name=asset_name,
            asset_type=asset_type
        )

        if result['success']:
            # Update job state
            state["generated_images"] = agent.generated_images
            state["failed_generations"] = agent.failed_generations
            pipeline_service.update_job_state(job_id, state)

            # Update project agent output
            try:
                project_service.update_agent_output(
                    project_id=job.project_id,
                    agent_number=5,
                    status="completed",
                    output={
                        "generated_images": agent.generated_images,
                        "failed_generations": agent.failed_generations,
                    }
                )
                logger.info("Updated project in MongoDB")
            except Exception as db_error:
                logger.warning(f"Failed to update MongoDB: {db_error}")

            logger.info(f"Retry successful! Remaining failed: {len(agent.failed_generations)}")
        else:
            logger.error(f"Retry failed: {result.get('error')}")

        return result

    except Exception as e:
        logger.error(f"Error retrying asset: {type(e).__name__}", exc_info=True)

        # Retry the task
        raise self.retry(exc=e, countdown=30, max_retries=2)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'run_phase1_workflow_task',
    'resume_phase1_workflow_task',
    'retry_failed_asset_task',
]
