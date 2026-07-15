"""
Phase 4 API endpoints — Video Review Checkpoint (Agent 0 gate).

Implements:
  GET  /api/v1/phase4/video-review/{movie_id}
  POST /api/v1/phase4/video-review/{movie_id}/{shot_id}/select
  POST /api/v1/phase4/master/continue-to-phase4/{master_job_id}
"""

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.models.responses import ApiResponse
from backend.shared.auth.dependencies import validate_admin_from_header, AdminUser
from backend.shared.utils.error_handlers import handle_api_exception
from backend.shared.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/phase4", tags=["Phase 4 Post-Production"])
pipeline_service = None  # lazy-loaded to avoid circular import at startup
limiter = Limiter(key_func=get_remote_address)


def _get_pipeline_service():
    global pipeline_service
    if pipeline_service is None:
        from app.services.pipeline_service import PipelineService
        pipeline_service = PipelineService()
    return pipeline_service


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class VideoVersionSelection(BaseModel):
    version: str = Field(..., description="Filename version, e.g. 'v1'")
    attempt_key: str = Field(..., description="MongoDB attempt key, e.g. 'v0'")
    s3_key: str = Field(..., description="Durable S3 key (no query string)")
    s3_url: str = Field(default="", description="Presigned URL at save time (informational)")


class VideoSelectionRequest(BaseModel):
    selected: List[VideoVersionSelection] = Field(
        default_factory=list,
        description="0, 1, or many versions. Empty list = explicit no-selection for this shot.",
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_s3_key(presigned_url: str, bucket_name: str) -> str:
    """Strip host, bucket prefix, and query string from a presigned URL to get the durable key."""
    parsed = urlparse(presigned_url)
    path = parsed.path.lstrip("/")
    prefix = bucket_name.rstrip("/") + "/"
    if path.startswith(prefix):
        return path[len(prefix):]
    return path


def _parse_filename_version(s3_key: str) -> str:
    """Extract _v{N} version string from a key like phase3/.../scene_1_shot_1_v2.mp4."""
    filename = s3_key.rsplit("/", 1)[-1]
    m = re.search(r"_v(\d+)\.", filename, re.IGNORECASE)
    return f"v{m.group(1)}" if m else "v1"


def _mint_presigned_url(s3_client: Any, bucket_name: str, s3_key: str,
                         expires: int = 86400 * 7) -> str:
    try:
        return s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=expires,
        )
    except Exception as exc:
        logger.warning(f"Could not mint presigned URL for {s3_key}: {exc}")
        return ""


def _selection_mode(count: int) -> str:
    if count == 0:
        return "none"
    if count == 1:
        return "single"
    return "multi"


def _resolve_project_ids(movies_col: Any, movie_id: str) -> List[str]:
    """Resolve movie _id → list of string show_ids (project_ids)."""
    try:
        movie_doc = movies_col.find_one({"_id": ObjectId(movie_id)})
    except Exception:
        movie_doc = None
    if not movie_doc:
        return []
    return [str(pid) for pid in movie_doc.get("project_ids", [])]


# ---------------------------------------------------------------------------
# GET /video-review/{movie_id}
# ---------------------------------------------------------------------------

@router.get("/video-review/{movie_id}", response_model=ApiResponse[dict])
@limiter.limit("60/minute")
async def get_video_review_gallery(
    request: Request,
    movie_id: str,
    scene_number: Optional[int] = None,
    admin_user: AdminUser = Depends(validate_admin_from_header),
):
    """
    Return all shots for a movie with their per-version video clips and saved selections.
    Walks movies → project_ids → shots, groups by scene_number.
    Optionally filter by scene_number.
    """
    try:
        from app.config import get_shots_service, get_mongo_factory, get_s3_client, get_bucket_name
        shots_service = get_shots_service()
        mongo_factory = get_mongo_factory()
        s3_client = get_s3_client()
        bucket_name = get_bucket_name()
    except (ValueError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=f"Service not configured: {exc}")

    try:
        _, movies_col = mongo_factory.get_collection("movies")
        project_ids = _resolve_project_ids(movies_col, movie_id)

        if not project_ids:
            return ApiResponse(
                success=True,
                data={"movie_id": movie_id, "scenes": []},
                error=None,
            )

        scene_docs = list(
            shots_service.shots_collection.find(
                {"show_id": {"$in": project_ids}}
            ).sort("episode_number", 1)
        )

        # Build scenes map: scene_number → ordered list of shot dicts
        scenes_map: Dict[int, List[dict]] = {}

        for doc in scene_docs:
            raw_shots = doc.get("annotated_shots") or doc.get("shots") or []

            for shot in raw_shots:
                sn = shot.get("scene_number") or 0
                if scene_number is not None and sn != scene_number:
                    continue

                video_obj = shot.get("video") or {}
                versions_out = []

                if isinstance(video_obj, dict):
                    for attempt_key, v in sorted(video_obj.items()):
                        if not (attempt_key.startswith("v") and attempt_key[1:].isdigit()):
                            continue
                        if not isinstance(v, dict):
                            continue
                        urls = v.get("generated_videos_s3") or []
                        if not urls:
                            continue
                        raw_url = urls[0]
                        s3_key = _parse_s3_key(raw_url, bucket_name)
                        fresh_url = _mint_presigned_url(s3_client, bucket_name, s3_key)
                        filename_version = _parse_filename_version(s3_key)
                        versions_out.append({
                            "version": filename_version,
                            "attempt_key": attempt_key,
                            "s3_key": s3_key,
                            "s3_url": fresh_url,
                            "approval_status": v.get("approval_status", "pending"),
                            "prompt": v.get("updated_prompt", ""),
                        })

                if not versions_out:
                    continue

                # Echo saved selection back to the client
                sel_doc = shot.get("video_review_selection") or {}
                current_selection = [
                    s.get("version")
                    for s in sel_doc.get("selected", [])
                    if s.get("version")
                ]

                existing_shots = scenes_map.get(sn, [])
                shot_entry = {
                    "shot_id": shot.get("shot_id"),
                    "scene_number": sn,
                    "shot_number": shot.get("sequence_number") or (len(existing_shots) + 1),
                    "description": shot.get("description", ""),
                    "versions": versions_out,
                    "current_selection": current_selection,
                }
                scenes_map.setdefault(sn, []).append(shot_entry)

        scenes_out = [
            {"scene_number": sn, "shots": shots}
            for sn, shots in sorted(scenes_map.items())
        ]

        return ApiResponse(
            success=True,
            data={"movie_id": movie_id, "scenes": scenes_out},
            error=None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, logger, context="get_video_review_gallery")


# ---------------------------------------------------------------------------
# POST /video-review/{movie_id}/{shot_id}/select
# ---------------------------------------------------------------------------

@router.post("/video-review/{movie_id}/{shot_id}/select", response_model=ApiResponse[dict])
@limiter.limit("120/minute")
async def save_video_selection(
    request: Request,
    movie_id: str,
    shot_id: str,
    body: VideoSelectionRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header),
):
    """
    Persist the human's video version selection for a shot (0, 1, or many versions).
    Writes video_review_selection to the matching annotated_shots entry in MongoDB.
    """
    try:
        from app.config import get_shots_service, get_mongo_factory
        shots_service = get_shots_service()
        mongo_factory = get_mongo_factory()
    except (ValueError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=f"Service not configured: {exc}")

    try:
        _, movies_col = mongo_factory.get_collection("movies")
        project_ids = _resolve_project_ids(movies_col, movie_id)
        if not project_ids:
            raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

        # Find the scene doc containing this shot (annotated_shots first, legacy shots second)
        scene_doc = shots_service.shots_collection.find_one({
            "show_id": {"$in": project_ids},
            "annotated_shots.shot_id": shot_id,
        })
        shots_field = "annotated_shots"
        if not scene_doc:
            scene_doc = shots_service.shots_collection.find_one({
                "show_id": {"$in": project_ids},
                "shots.shot_id": shot_id,
            })
            shots_field = "shots"
        if not scene_doc:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id} not found in movie {movie_id}")

        show_id = scene_doc["show_id"]
        selected_list = [s.model_dump() for s in body.selected]
        mode = _selection_mode(len(selected_list))

        selection_doc = {
            "selected": selected_list,
            "mode": mode,
            "selected_by": admin_user.key if admin_user else "human",
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }

        result = shots_service.shots_collection.update_one(
            {"show_id": show_id, f"{shots_field}.shot_id": shot_id},
            {"$set": {
                f"{shots_field}.$.video_review_selection": selection_doc,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Shot {shot_id} update target not found")

        return ApiResponse(
            success=True,
            data={"ok": True, "shot_id": shot_id, "count": len(selected_list), "mode": mode},
            error=None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, logger, context="save_video_selection")


# ---------------------------------------------------------------------------
# POST /master/continue-to-phase4/{master_job_id}
# ---------------------------------------------------------------------------

@router.post("/master/continue-to-phase4/{master_job_id}")
async def continue_to_phase4(
    master_job_id: str,
    request: Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
) -> Dict[str, Any]:
    """
    Resume the master pipeline from the video-review pause into Phase 4.

    Call this after saving all video selections in the Streamlit review app.
    The pipeline must be in 'waiting_for_video_review' status.
    """
    try:
        svc = _get_pipeline_service()
        job = svc.get_job(master_job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Master job {master_job_id} not found")

        if job.pipeline_status != "waiting_for_video_review":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Pipeline is not waiting for video review "
                    f"(current status: {job.pipeline_status}). "
                    "Has the pipeline completed Phase 3?"
                ),
            )

        try:
            from app.tasks.phase4_tasks import dispatch_phase4_task
            from app.config import get_workflow_queue_name
            queue_name = get_workflow_queue_name()
            task = dispatch_phase4_task.apply_async(
                args=[master_job_id],
                queue=queue_name,
                routing_key=queue_name,
            )
            svc.update_job_celery_task_id(master_job_id, task.id)
            logger.info(
                f"[Master] continue-to-phase4: dispatching Phase 4 — "
                f"master_job_id={master_job_id} task={task.id}"
            )
            return {
                "master_job_id": master_job_id,
                "celery_task_id": task.id,
                "message": "Phase 4 dispatched. Poll /api/v1/master/status/{master_job_id} for progress.",
            }
        except ImportError:
            # Phase 4 Celery task not yet implemented — acknowledge and return
            logger.warning(
                f"[Master] dispatch_phase4_task not yet available; "
                f"returning acknowledgement only. master_job_id={master_job_id}"
            )
            return {
                "master_job_id": master_job_id,
                "celery_task_id": None,
                "message": "Phase 4 task module not yet implemented. Selection acknowledged.",
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise handle_api_exception(exc, logger, context="continue_to_phase4")
