"""
Phase 3 Celery Tasks
====================

This module provides Celery tasks for the Phase 3 video generation workflow.
It mirrors the Celery + Amazon SQS integration used in Phase 1 and Phase 2,
ensuring consistent task orchestration, retry behaviour, logging, and human
checkpoint handling across all phases.

Tasks:
------
1. run_phase3_workflow_task: Start the full Phase 3 workflow for a shot
2. resume_phase3_workflow_task: Resume workflow after human decision/checkpoint
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.services.pipeline_service import PipelineService
from app.services.phase_3_agents.langgraph_workflow import (
    run_phase3_pipeline,
    create_phase3_workflow,
)
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

def build_phase3_job_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise LangGraph workflow state into the structure expected by PipelineService.

    Args:
        state: Raw workflow state returned by LangGraph

    Returns:
        Dict containing pipeline tracking fields (agent statuses, checkpoint info, etc.)
    """
    job_state = deepcopy(state)
    job_state["pipeline_status"] = state.get("pipeline_status", job_state.get("pipeline_status", "running"))
    job_state["shot_id"] = state.get("shot_id")
    job_state["show_id"] = state.get("show_id")
    job_state["video_generation_attempt"] = state.get("video_generation_attempt", 0)
    job_state["human_regeneration_attempt"] = state.get("human_regeneration_attempt", 0)

    # Current agent/node tracking
    job_state["current_agent"] = state.get("current_node", "initialize")

    # Agent status derivations
    video_prompt_exists = bool(state.get("video_prompt"))
    generation_status = (state.get("video_generation_status") or "").lower()
    review_status = (state.get("ai_review_status") or "").lower()

    # Agent 17 (Prompt generation)
    if state.get("agent17_status"):
        job_state["agent17_status"] = state["agent17_status"]
    elif state.get("pipeline_status") == "failed" and job_state["current_agent"] in {"initialize", "prompt_router"}:
        job_state["agent17_status"] = "failed"
    else:
        job_state["agent17_status"] = "completed" if video_prompt_exists else "running"

    # Agent 18 (Video generation)
    if state.get("agent18_status"):
        job_state["agent18_status"] = state["agent18_status"]
    elif generation_status == "failed":
        job_state["agent18_status"] = "failed"
    elif state.get("generated_video_url"):
        job_state["agent18_status"] = "completed"
    elif generation_status in {"processing", "running"}:
        job_state["agent18_status"] = "running"
    else:
        job_state["agent18_status"] = "pending"

    # Agent 19 (AI review)
    if state.get("agent19_status"):
        job_state["agent19_status"] = state["agent19_status"]
    elif review_status == "failed":
        job_state["agent19_status"] = "failed"
    elif state.get("review_result") or review_status == "completed":
        job_state["agent19_status"] = "completed"
    elif review_status in {"running", "in_progress"}:
        job_state["agent19_status"] = "running"
    else:
        job_state["agent19_status"] = "pending"

    # Phase 3 human checkpoint
    pipeline_status = state.get("pipeline_status", "running")
    if pipeline_status == "waiting_for_human":
        job_state["phase_3_checkpoint_status"] = "waiting"
    elif pipeline_status == "failed":
        job_state["phase_3_checkpoint_status"] = "failed"
    else:
        job_state["phase_3_checkpoint_status"] = state.get("phase_3_checkpoint_status", "completed")

    # Regeneration tracking + output files
    job_state["regeneration_count"] = state.get("human_regeneration_attempt", 0)

    output_files = []
    for version in state.get("video_versions", []):
        if isinstance(version, dict):
            url = version.get("video_url") or version.get("url")
            if url:
                output_files.append(url)
    if state.get("generated_video_url"):
        if state["generated_video_url"] not in output_files:
            output_files.append(state["generated_video_url"])
    job_state["output_files"] = output_files

    return job_state


# ============================================================================
# Custom Task Base Class
# ============================================================================

class Phase3Task(Task):
    """
    Custom base class for Phase 3 tasks providing consistent logging hooks.
    """

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"❌ Phase 3 Task {task_id} failed: {exc}")
        print(f"   Exception info: {einfo}")

    def on_success(self, retval, task_id, args, kwargs):
        print(f"✅ Phase 3 Task {task_id} completed successfully")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        print(f"🔄 Phase 3 Task {task_id} is being retried. Reason: {exc}")


# ============================================================================
# Task 1: Run Phase 3 Workflow (Full Pipeline)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase3Task,
    name="phase3.run_workflow",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    track_started=True,
    time_limit=10800,       # 3 hours hard limit
    soft_time_limit=10500,  # 2h 55m soft limit
)
def run_phase3_workflow_task(
    self,
    job_id: str,
    shot_id: str,
    show_id: str,
    idempotency_key: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the complete Phase 3 workflow (video prompt → video generation → AI review).

    Args:
        job_id: Pipeline job identifier
        shot_id: Shot identifier (Phase 3 input)
        show_id: Show/Project identifier used for MongoDB lookups

    Returns:
        Dict summarising execution outcome and key metadata
    """
    pipeline_service = PipelineService()

    try:
        # ===== STEP 0: Idempotency Check =====
        # Generate idempotency key
        # IMPORTANT: Include shot_id in the scene_id to ensure different shots get different keys
        # Format: show_id:shot_id (consistent with phase3.py /start endpoint)
        scene_id = f"{show_id}:{shot_id}" if show_id and shot_id else (show_id or shot_id or job_id)

        # DO NOT use idempotency_key_header from frontend because it may reuse
        # the same UUID for different shots. Always generate our own key.
        idempotency_key_value = generate_idempotency_key(
            user_id=user_id,
            scene_id=scene_id,
            phase_number=3,
            idempotency_key_header=None,  # Don't use frontend key - generate our own
        )

        # Build payload for idempotency check
        # Note: job_id is NOT included because it's generated fresh for each request
        # and doesn't represent the semantic intent (which is based on shot/show)
        payload = {
            "shot_id": shot_id,
            "show_id": show_id,
        }
        
        # Check idempotency
        idempotency_service = get_idempotency_service()
        is_duplicate, cached_response = check_idempotency(
            endpoint="phase3.run_workflow",
            idempotency_key=idempotency_key_value,
            payload=payload,
            service=idempotency_service,
        )
        
        if is_duplicate:
            if cached_response:
                # Return cached response
                print(f"Returning cached response for idempotency key: {idempotency_key_value}")
                return cached_response
            else:
                # Processing in progress - check if this is a true duplicate or first attempt
                print(f"Duplicate request detected for idempotency key: {idempotency_key_value}")
                
                # Get the idempotency record to check if task_id is set
                record = idempotency_service.get_record(
                    endpoint="phase3.run_workflow",
                    key=idempotency_key_value
                )
                
                # If record has no task_id, this might be the first attempt
                # (record was created but task not yet attached)
                # Only return early if task_id is already set (true duplicate)
                if record and record.task_id:
                    print(f"Task already running with ID: {record.task_id}")
                    job = pipeline_service.get_job(job_id)
                    if job:
                        return {
                            "status": job.pipeline_status or "running",
                            "job_id": job_id,
                            "shot_id": shot_id,
                            "show_id": show_id,
                            "current_agent": getattr(job, "current_agent", None),
                            "celery_task_id": record.task_id,
                            "message": "Workflow already in progress",
                        }
                else:
                    # No task_id set yet - this is the first attempt
                    # Continue with execution (will attach task_id below)
                    print(f"No task_id in record yet - proceeding with workflow execution")
                    # Continue to next step (don't return early)
        
        # Attach task reference to idempotency record
        try:
            idempotency_service.attach_task_reference(
                endpoint="phase3.run_workflow",
                key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                metadata={
                    "show_id": show_id,
                    "shot_id": shot_id,
                },
            )
        except Exception as e:
            print(f"Warning: Failed to attach task reference to idempotency record: {e}")

        print(f"\n{'='*80}")
        print("🚀 Starting Phase 3 Workflow")
        print(f"   Job ID: {job_id}")
        print(f"   Shot ID: {shot_id}")
        print(f"   Show ID: {show_id}")
        print(f"   Idempotency Key: {idempotency_key_value}")
        print(f"   Celery Task ID: {self.request.id}")
        print(f"{'='*80}\n")

        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            current_agent="initialize",
            started_at=datetime.utcnow(),
            celery_task_id=self.request.id,
            state={
                "shot_id": shot_id,
                "show_id": show_id,
                "current_node": "initialize",
                "pipeline_status": "running",
            },
        )

        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "shot_id": shot_id,
                "current_agent": "initialize",
                "message": "Initializing Phase 3 workflow...",
                "progress": 0,
            },
        )

        final_state = run_phase3_pipeline(
            shot_id=shot_id,
            show_id=show_id,
        )

        job_state = build_phase3_job_state(final_state)
        pipeline_service.update_job_state(job_id, job_state)

        print(f"\n{'='*80}")
        print("✅ Phase 3 Workflow Completed")
        print(f"   Job ID: {job_id}")
        print(f"   Pipeline Status: {job_state.get('pipeline_status')}")
        print(f"   Current Agent: {job_state.get('current_agent')}")
        print(f"{'='*80}\n")

        result = {
            "status": job_state.get("pipeline_status", "completed"),
            "job_id": job_id,
            "shot_id": shot_id,
            "show_id": show_id,
            "current_agent": job_state.get("current_agent"),
            "generated_video_url": job_state.get("generated_video_url"),
            "review_score": job_state.get("review_score"),
            "celery_task_id": self.request.id,
        }
        
        # Mark idempotency as completed
        try:
            mark_idempotency_completed(
                endpoint="phase3.run_workflow",
                idempotency_key=idempotency_key_value,
                workflow_id=job_id,
                task_id=self.request.id,
                response_payload=result,
            )
        except Exception as e:
            print(f"Warning: Failed to mark idempotency as completed: {e}")
        
        return result

    except SoftTimeLimitExceeded:
        print(f"⏰ Phase 3 Task {self.request.id} exceeded soft time limit")
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="timeout",
            error_message="Phase 3 task exceeded execution time limit",
            state={"shot_id": shot_id, "show_id": show_id, "pipeline_status": "timeout"},
        )
        raise

    except Exception as exc:
        error_message = str(exc)
        print(f"❌ Error in Phase 3 workflow: {error_message}")

        # Mark idempotency as failed
        try:
            scene_id = show_id or shot_id or job_id
            idempotency_key_value = generate_idempotency_key(
                user_id=user_id,
                scene_id=scene_id,
                phase_number=3,
                idempotency_key_header=idempotency_key,
            )
            mark_idempotency_failed(
                endpoint="phase3.run_workflow",
                idempotency_key=idempotency_key_value,
                error_message=f"Workflow execution failed: {error_message}",
            )
        except Exception as idemp_error:
            print(f"Warning: Failed to mark idempotency as failed: {idemp_error}")

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message=error_message,
            state={
                "shot_id": shot_id,
                "show_id": show_id,
                "pipeline_status": "failed",
                "error_message": error_message,
            },
        )

        raise self.retry(exc=exc, countdown=60, max_retries=3)


# ============================================================================
# Task 2: Resume Phase 3 Workflow (Human Checkpoint)
# ============================================================================

@celery_app.task(
    bind=True,
    base=Phase3Task,
    name="phase3.resume_workflow",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
    track_started=True,
    time_limit=10800,
    soft_time_limit=10500,
)
def resume_phase3_workflow_task(
    self,
    job_id: str,
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resume Phase 3 workflow after a human checkpoint decision.

    Args:
        job_id: Pipeline job identifier
        current_state: Workflow state (including human decision/feedback) to resume from

    Returns:
        Dict summarising execution outcome and key metadata
    """
    pipeline_service = PipelineService()

    print(f"\n{'='*80}")
    print("🔄 Resuming Phase 3 Workflow")
    print(f"   Job ID: {job_id}")
    print(f"   Human Decision: {current_state.get('human_decision')}")
    print(f"   Celery Task ID: {self.request.id}")
    print(f"{'='*80}\n")

    try:
        pipeline_service.update_job_status(
            job_id,
            status="running",
            pipeline_status="running",
            current_agent=current_state.get("current_node", "human_checkpoint"),
            celery_task_id=self.request.id,
            state=current_state,
        )

        self.update_state(
            state="PROGRESS",
            meta={
                "job_id": job_id,
                "shot_id": current_state.get("shot_id"),
                "current_agent": current_state.get("current_node"),
                "message": "Resuming Phase 3 workflow from human checkpoint...",
                "progress": 60,
            },
        )

        app = create_phase3_workflow()
        final_state = app.invoke(current_state)

        job_state = build_phase3_job_state(final_state)
        pipeline_service.update_job_state(job_id, job_state)

        print(f"\n{'='*80}")
        print("✅ Phase 3 Workflow Resumed and Completed")
        print(f"   Job ID: {job_id}")
        print(f"   Pipeline Status: {job_state.get('pipeline_status')}")
        print(f"   Current Agent: {job_state.get('current_agent')}")
        print(f"{'='*80}\n")

        return {
            "status": job_state.get("pipeline_status", "completed"),
            "job_id": job_id,
            "shot_id": job_state.get("shot_id"),
            "current_agent": job_state.get("current_agent"),
            "generated_video_url": job_state.get("generated_video_url"),
            "review_score": job_state.get("review_score"),
            "celery_task_id": self.request.id,
        }

    except SoftTimeLimitExceeded:
        print(f"⏰ Resume task {self.request.id} exceeded soft time limit")
        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="timeout",
            error_message="Phase 3 resume task exceeded time limit",
            state=current_state,
        )
        raise

    except Exception as exc:
        error_message = str(exc)
        print(f"❌ Error resuming Phase 3 workflow: {error_message}")

        pipeline_service.update_job_status(
            job_id,
            status="failed",
            pipeline_status="failed",
            error_message=error_message,
            state=current_state,
        )

        raise self.retry(exc=exc, countdown=60, max_retries=3)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "run_phase3_workflow_task",
    "resume_phase3_workflow_task",
]

