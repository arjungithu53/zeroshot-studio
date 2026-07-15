"""
Phase 4 Celery Task
====================
Single task that receives the master_job_id from the API endpoint,
looks up all required identifiers from the stored job state, and
runs the Phase 4 LangGraph pipeline synchronously inside the worker.

No retries — Phase 4 is expensive (Gemini Pro, TTS, Lyria, FFmpeg mix).
Re-trigger manually from the API if it fails.
"""

from __future__ import annotations

from typing import Any, Dict

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.services.pipeline_service import PipelineService
from app.services.phase_4_agents.langgraph_workflow import run_phase4_pipeline


class Phase4Task(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"Phase4Task {task_id} failed: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        print(f"Phase4Task {task_id} completed successfully")


@celery_app.task(
    bind=True,
    base=Phase4Task,
    name="phase4.dispatch",
    max_retries=0,
    acks_late=False,
    track_started=True,
    time_limit=7200,       # 2 h hard limit
    soft_time_limit=6900,  # 1 h 55 m — lets us mark "failed" cleanly
)
def dispatch_phase4_task(self, master_job_id: str) -> Dict[str, Any]:
    """
    Run the full Phase 4 pipeline for a master job that has completed
    Phase 3 and is waiting in 'waiting_for_video_review' status.

    Args:
        master_job_id: The production_pipelines job_id for this movie run.

    Returns:
        Summary dict with pipeline_status and key S3 output keys.
    """
    pipeline_service = PipelineService()

    # ── 1. Load master job ────────────────────────────────────────────────
    job = pipeline_service.get_job(master_job_id)
    if not job:
        raise ValueError(f"Master job {master_job_id} not found in production_pipelines.")

    saved_state: Dict[str, Any] = job.state or {}

    movie_id: str | None = saved_state.get("movie_id") or job.movie_id or None
    show_id: str = saved_state.get("show_id") or job.project_id or movie_id or ""
    episode_number: int = int(saved_state.get("episode_number") or 1)
    episode_id: str = saved_state.get("episode_id") or ""

    if not show_id:
        raise ValueError(
            f"Cannot start Phase 4: show_id/movie_id is missing from master job {master_job_id}. "
            "Ensure the Phase 1–3 pipeline stored movie_id in the job state."
        )

    # ── 2. Mark running ───────────────────────────────────────────────────
    pipeline_service.update_job_status(
        master_job_id,
        status="running",
        pipeline_status="phase4_running",
        current_agent="phase4_initialize",
    )

    # ── 3. Run pipeline ───────────────────────────────────────────────────
    try:
        final_state = run_phase4_pipeline(
            show_id=show_id,
            episode_number=episode_number,
            episode_id=episode_id,
            movie_id=movie_id,
            project_id=job.project_id,
            job_id=master_job_id,
            # title / script_content / shot_list left None — initialize_node
            # loads them from MongoDB so we don't need to re-serialise them here.
        )

    except SoftTimeLimitExceeded:
        pipeline_service.update_job_status(
            master_job_id,
            status="failed",
            pipeline_status="failed",
            error_message="Phase 4 soft time limit exceeded (1 h 55 m).",
        )
        raise

    except Exception as exc:
        pipeline_service.update_job_status(
            master_job_id,
            status="failed",
            pipeline_status="failed",
            error_message=str(exc)[:500],
        )
        raise

    # ── 4. Mark completed ─────────────────────────────────────────────────
    phase_status = final_state.get("pipeline_status", "completed")
    if phase_status == "failed":
        pipeline_service.update_job_status(
            master_job_id,
            status="failed",
            pipeline_status="failed",
            error_message="; ".join(
                e.get("error", "") for e in final_state.get("errors", [])
            )[:500] or "Phase 4 pipeline reported failure.",
        )
    else:
        pipeline_service.update_job_status(
            master_job_id,
            status="completed",
            pipeline_status="completed",
        )

    return {
        "master_job_id": master_job_id,
        "pipeline_status": phase_status,
        "final_master_s3_key": final_state.get("final_master_s3_key"),
        "captions_s3_key": final_state.get("captions_s3_key"),
        "platform_exports": final_state.get("platform_exports"),
        "errors": final_state.get("errors", []),
    }
