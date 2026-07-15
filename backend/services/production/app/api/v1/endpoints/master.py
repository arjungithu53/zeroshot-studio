"""
Master Pipeline Endpoint
=========================

POST /api/v1/master/run-pipeline
  Start the full end-to-end pipeline (Phase 1 → Phase 2 → Phase 3) for a movie.
  Accepts the same multipart/form-data as POST /api/v1/movies/create.
  All human checkpoints are auto-approved — no manual intervention required.

GET /api/v1/master/status/{master_job_id}
  Poll overall pipeline status and per-scene/shot progress.
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

# Add backend root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception

logger = get_logger(__name__)

from shared.auth.dependencies import validate_admin_from_header, AdminUser

from app.services.pipeline_service import PipelineService
from app.config import get_workflow_queue_name, upload_file_wrapper
from app.core.quota import QuotaManager, get_quota_manager
from app.utils.csv_parser import parse_movie_csv, validate_csv_format, parse_shotlist_csv
from app.tasks.master_tasks import run_master_pipeline_task, poll_phase2_task, dispatch_phase3_task
from app.tasks.phase2_tasks import resume_phase2_after_final_approval_task
from app.services.phase_3_agents.video_generation.video_model import VideoModel

# ============================================================================
# Router + service singletons
# ============================================================================

router = APIRouter(prefix="/master", tags=["Master Pipeline"])
pipeline_service = PipelineService()


# ============================================================================
# Response models
# ============================================================================

class MasterPipelineStartResponse(BaseModel):
    """Response returned when the master pipeline is started."""
    success: bool
    master_job_id: str
    celery_task_id: str
    movie_id: Optional[str] = None   # not yet known at dispatch time
    total_scenes: int
    status: str
    message: str
    created_at: datetime


class MasterPipelineStatusResponse(BaseModel):
    """Status response for polling pipeline progress."""
    master_job_id: str
    status: str
    pipeline_status: str
    current_phase: str
    movie_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    total_scenes: int
    scene_data: List[Dict[str, Any]]
    phase1_complete: bool
    phase2_complete: bool
    phase3_complete: bool
    failed_scenes: List[Any]
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# Endpoints
# ============================================================================

@router.post("/run-pipeline", response_model=MasterPipelineStartResponse, status_code=202)
async def run_master_pipeline(
    request: Request,
    # Required fields
    title: str = Form(..., description="Movie title"),
    script_csv: UploadFile = File(..., description="CSV with columns: scene_number, scene_name, scene_script"),
    shotlist_csv: UploadFile = File(..., description="CSV with columns: scene_number, shot_number, shot_type, camera_movement, description"),
    # Optional movie metadata
    description: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    visual_style: Optional[str] = Form(None, description="One of: realistic, pixar, 2d"),
    aspect_ratio: Optional[str] = Form(None, description="One of: 9:16, 16:9, 2.39:1"),
    video_model: VideoModel = Form(..., description="Video generation model"),
    user_id: Optional[str] = Form(None),
    product_image_file: Optional[UploadFile] = File(None, description="Product image (PNG/JPG/JPEG, optional)"),
    # Auth + quota
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager),
) -> MasterPipelineStartResponse:
    """
    Start the complete end-to-end production pipeline in a single call.

    **What this does (automatically, no human input required):**
    1. Creates movie document + uploads scripts/shotlists to S3
    2. Runs Phase 1 — asset generation (8 agents)
    3. Auto-approves Phase 1 human checkpoint (after Agent 7)
    4. Runs Phase 2 per scene — shot image generation (7+ agents, each in its own Celery worker)
    5. Auto-approves all 3 Phase 2 checkpoints (strategy → prompt correction → final images)
    6. Runs Phase 3 per shot — video generation (3 agents per shot)
    7. Auto-approves Phase 3 video checkpoint

    **Quota:** consumes 1 unit.

    **Polling:** use GET /api/v1/master/status/{master_job_id} to track progress.

    Args:
        title: Movie title
        script_csv: CSV with scene_number, scene_name, scene_script columns
        shotlist_csv: CSV with scene_number, shot_number, shot_type, camera_movement, description columns
        visual_style: Visual style (realistic | pixar | 2d)
        aspect_ratio: Aspect ratio (9:16 | 16:9 | 2.39:1)
        video_model: Video generation model (Veo 3.1 | Omni Flash)

    Returns:
        master_job_id for polling and celery_task_id for low-level task monitoring
    """
    try:
        # ===== Validate inputs =====
        if visual_style and visual_style not in ("realistic", "pixar", "2d"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid visual_style '{visual_style}'. Must be one of: realistic, pixar, 2d",
            )
        if aspect_ratio and aspect_ratio not in ("9:16", "16:9", "2.39:1"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aspect_ratio '{aspect_ratio}'. Must be one of: 9:16, 16:9, 2.39:1",
            )
        # video_model is already validated/coerced to a VideoModel enum member by FastAPI
        # (it's typed as VideoModel above, which also renders as a dropdown in the OpenAPI docs).
        video_model = video_model.value

        # ===== Parse CSVs (must be done here — UploadFile can't be serialised to Celery) =====
        validate_csv_format(script_csv)
        scenes_data = await parse_movie_csv(script_csv)

        validate_csv_format(shotlist_csv)
        shots_data = await parse_shotlist_csv(shotlist_csv)

        if not shots_data:
            raise HTTPException(
                status_code=400,
                detail="shotlist_csv is required and must contain at least one shot for the full pipeline.",
            )

        scenes_dicts = [s.to_dict() for s in scenes_data]
        shots_dicts = [s.to_dict() for s in shots_data]

        # ===== Consume quota =====
        effective_user_id = user_id or admin_user.user_id
        quota_manager.consume(
            user_id=effective_user_id,
            pipeline_name="production_workflow",
        )
        logger.info(f"[Master] Quota consumed for user {effective_user_id}")

        # ===== Create master pipeline job =====
        job_data = {
            "type": "master_pipeline",
            "status": "pending",
            "pipeline_status": "pending",
            "current_agent": "master_orchestrator",
        }
        job_result = pipeline_service.create_job(job_data)
        master_job_id: str = job_result["job_id"]

        # ===== Upload product image to S3 if provided =====
        # UploadFile can't be serialised to Celery, so we upload here and pass the URL
        product_image_s3_url = None
        if product_image_file and product_image_file.filename:
            allowed_exts = {'.png', '.jpg', '.jpeg'}
            ext = Path(product_image_file.filename).suffix.lower()
            if ext not in allowed_exts:
                raise HTTPException(status_code=400, detail="Product image must be a PNG or JPG file")
            image_bytes = await product_image_file.read()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(image_bytes)
                    tmp_path = tmp.name
                s3_key = f"master/{master_job_id}/product_image{ext}"
                product_image_s3_url = upload_file_wrapper(
                    tmp_path,
                    s3_key=s3_key,
                    content_type=product_image_file.content_type or "image/png",
                    use_presigned_url=False,
                )
                logger.info(f"[Master] Product image uploaded: {product_image_s3_url}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # ===== Dispatch master Celery task =====
        queue_name = get_workflow_queue_name()
        task = run_master_pipeline_task.apply_async(
            args=[master_job_id, scenes_dicts, shots_dicts, title],
            kwargs={
                "description": description,
                "genre": genre,
                "visual_style": visual_style,
                "aspect_ratio": aspect_ratio,
                "video_model": video_model,
                "user_id": effective_user_id,
                "product_image_s3_url": product_image_s3_url,
            },
            queue=queue_name,
            routing_key=queue_name,
        )

        # Update job with Celery task ID and running status
        pipeline_service.update_job_celery_task_id(master_job_id, task.id)
        pipeline_service.update_job_status(
            master_job_id,
            status="running",
            pipeline_status="running",
            celery_task_id=task.id,
            started_at=datetime.utcnow(),
            state={
                "master_pipeline": True,
                "current_phase": "initializing",
                "total_scenes": len(scenes_dicts),
                "phase1_complete": False,
                "phase2_complete": False,
                "phase3_complete": False,
                "failed_scenes": [],
                "scene_data": [],
            },
        )

        logger.info(
            f"[Master] Pipeline started — master_job_id={master_job_id} "
            f"celery_task_id={task.id} scenes={len(scenes_dicts)}"
        )

        job = pipeline_service.get_job(master_job_id)
        return MasterPipelineStartResponse(
            success=True,
            master_job_id=master_job_id,
            celery_task_id=task.id,
            total_scenes=len(scenes_dicts),
            status="running",
            message=(
                f"Master pipeline started — {len(scenes_dicts)} scene(s) will be processed "
                "automatically through Phase 1, Phase 2, and Phase 3. "
                "Poll /api/v1/master/status/{master_job_id} to track progress."
            ),
            created_at=job.created_at if job else datetime.utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, "run_master_pipeline")


@router.get("/find-by-p2-job/{p2_job_id}")
async def find_master_by_p2_job(
    p2_job_id: str,
    request: Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
) -> Dict[str, Any]:
    """Find the master job ID that owns a given Phase 2 job ID."""
    try:
        from app.config import get_pipelines_collection
        _, collection = get_pipelines_collection()
        doc = collection.find_one(
            {"state.scene_data": {"$elemMatch": {"phase2_job_id": p2_job_id}}},
            {"job_id": 1, "status": 1, "pipeline_status": 1, "state.current_phase": 1},
        )
        if not doc:
            raise HTTPException(status_code=404, detail=f"No master job found containing phase2_job_id={p2_job_id}")
        return {
            "master_job_id": doc["job_id"],
            "status": doc.get("status"),
            "pipeline_status": doc.get("pipeline_status"),
            "current_phase": (doc.get("state") or {}).get("current_phase"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, "find_master_by_p2_job")


@router.post("/continue-to-phase3/{master_job_id}")
async def continue_to_phase3(
    master_job_id: str,
    request: Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
) -> Dict[str, Any]:
    """
    Resume the master pipeline from the image-review pause into Phase 3.

    Call this after saving all image selections in the Streamlit review app.
    The pipeline must be in 'waiting_for_image_review' status.
    """
    try:
        job = pipeline_service.get_job(master_job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Master job {master_job_id} not found")

        if job.pipeline_status != "waiting_for_image_review":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Pipeline is not waiting for image review "
                    f"(current status: {job.pipeline_status}). "
                    "Has the pipeline completed Phase 2?"
                ),
            )

        queue_name = get_workflow_queue_name()
        task = dispatch_phase3_task.apply_async(
            args=[master_job_id],
            queue=queue_name,
            routing_key=queue_name,
        )
        pipeline_service.update_job_celery_task_id(master_job_id, task.id)
        logger.info(
            f"[Master] continue-to-phase3: dispatching Phase 3 — "
            f"master_job_id={master_job_id} task={task.id}"
        )

        return {
            "master_job_id": master_job_id,
            "celery_task_id": task.id,
            "message": "Phase 3 dispatched. Poll /api/v1/master/status/{master_job_id} for progress.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, "continue_to_phase3")


@router.post("/rescue/{master_job_id}")
async def rescue_master_pipeline(
    master_job_id: str,
    request: Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
) -> Dict[str, Any]:
    """
    Rescue a stuck master pipeline by re-dispatching poll_phase2_task.

    Use this when the pipeline stopped after Phase 2 and Phase 3 never started.
    Typically caused by the poll_phase2_task worker being killed while the pipeline
    was in waiting_for_final_approval.

    Steps performed:
    1. For each Phase 2 job still at waiting_for_final_approval: dispatches
       resume_phase2_after_final_approval_task to complete it.
    2. Clears the idempotency keys for those jobs so poll_phase2_task will
       detect them as completed on its next tick.
    3. Re-dispatches poll_phase2_task (attempt=0) to trigger Phase 3 dispatch.
    """
    try:
        job = pipeline_service.get_job(master_job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Master job {master_job_id} not found")

        state = job.state or {}
        scene_infos = state.get("scene_data", [])
        if not scene_infos:
            raise HTTPException(status_code=400, detail="No scene_data found in master job state")

        queue_name = get_workflow_queue_name()

        dispatched_resumes: set = {
            tuple(item) for item in state.get("dispatched_p2_resumes", [])
        }

        final_approval_dispatched = []
        already_completed = []

        for info in scene_infos:
            p2_job_id = info.get("phase2_job_id")
            if not p2_job_id:
                continue
            p2_job = pipeline_service.get_job(p2_job_id)
            if not p2_job:
                continue

            if p2_job.pipeline_status == "waiting_for_final_approval":
                key = (p2_job_id, "waiting_for_final_approval")
                dispatched_resumes.discard(key)

                task = resume_phase2_after_final_approval_task.apply_async(
                    args=[p2_job_id, True, None],
                    queue=queue_name,
                    routing_key=queue_name,
                )
                pipeline_service.update_job_celery_task_id(p2_job_id, task.id)
                final_approval_dispatched.append({"p2_job_id": p2_job_id, "task_id": task.id})
                logger.info(f"[Rescue] Dispatched final approval for {p2_job_id} → task={task.id}")

            elif p2_job.status == "completed":
                already_completed.append(p2_job_id)

        state["dispatched_p2_resumes"] = [list(r) for r in dispatched_resumes]
        pipeline_service.update_job_status(
            master_job_id,
            status=job.status,
            pipeline_status=job.pipeline_status or "running",
            state=state,
        )

        poll_task = poll_phase2_task.apply_async(
            args=[master_job_id],
            kwargs={"attempt": 0},
            queue=queue_name,
            routing_key=queue_name,
        )
        logger.info(f"[Rescue] Re-dispatched poll_phase2_task → task={poll_task.id}")

        return {
            "master_job_id": master_job_id,
            "poll_phase2_task_id": poll_task.id,
            "final_approval_dispatched": final_approval_dispatched,
            "already_completed_p2_jobs": already_completed,
            "message": (
                f"Rescue complete: {len(final_approval_dispatched)} final approval(s) dispatched, "
                f"poll_phase2_task re-dispatched (task={poll_task.id}). "
                "Phase 3 will start automatically once Phase 2 jobs complete."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, "rescue_master_pipeline")


@router.get("/status/{master_job_id}", response_model=MasterPipelineStatusResponse)
async def get_master_pipeline_status(
    master_job_id: str,
    request: Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
) -> MasterPipelineStatusResponse:
    """
    Poll the status of a master pipeline run.

    **current_phase** advances through:
      initializing → phase1 → phase2 → phase3 → completed

    **scene_data** lists per-scene job IDs so you can also poll individual
    Phase 2 / Phase 3 jobs for granular progress.

    Args:
        master_job_id: UUID returned by POST /master/run-pipeline

    Returns:
        Current pipeline status with per-scene breakdown
    """
    job = pipeline_service.get_job(master_job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Master job {master_job_id} not found")

    state = job.state or {}

    return MasterPipelineStatusResponse(
        master_job_id=master_job_id,
        status=job.status,
        pipeline_status=job.pipeline_status,
        current_phase=state.get("current_phase", "unknown"),
        movie_id=state.get("movie_id"),
        celery_task_id=job.celery_task_id,
        total_scenes=len(state.get("scene_data", [])) or state.get("total_scenes", 0),
        scene_data=state.get("scene_data", []),
        phase1_complete=state.get("phase1_complete", False),
        phase2_complete=state.get("phase2_complete", False),
        phase3_complete=state.get("phase3_complete", False),
        failed_scenes=state.get("failed_scenes", []),
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
