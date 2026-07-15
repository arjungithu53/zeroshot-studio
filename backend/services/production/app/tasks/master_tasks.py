"""
Master Pipeline Celery Tasks
============================

Orchestrates the full production pipeline end-to-end:
  Phase 1 (asset generation) → Phase 2 per scene (shot generation) → Phase 3 per shot (video generation)

All human checkpoints are automatically approved so the pipeline runs unattended.

Architecture — four short-lived tasks (no task holds a worker slot while polling):
  run_master_pipeline_task  — STEP 1+2: create movie, dispatch Phase 1, dispatch poll_phase1_task → returns
  poll_phase1_task          — one tick: check Phase 1, auto-approve checkpoint; when done: dispatch Phase 2 → poll_phase2_task
  poll_phase2_task          — one tick: check Phase 2 scenes, auto-approve checkpoints; when done: dispatch Phase 3 → poll_phase3_task
  poll_phase3_task          — one tick: check Phase 3 shots, auto-approve checkpoints, unblock pending last_frame_seed shots; when done: mark complete

Each poll task reschedules itself with countdown=POLL_INTERVAL rather than sleeping, so it releases its
worker slot immediately and is re-queued after the interval.  All inter-tick state is persisted in the
master pipeline job's `state` dict in MongoDB.
"""

import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from celery.exceptions import SoftTimeLimitExceeded

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

logger = get_logger(__name__)

from app.celery_app import celery_app
from app.config import get_workflow_queue_name, get_database
from app.services.pipeline_service import PipelineService
from app.services.movie_service import MovieService
from app.services.project_service import ProjectService
from app.services.assets_collection_service import AssetsCollectionService
from app.models.mongodb.pipelines import PipelineJobCreate
from app.utils.s3_helpers import (
    upload_scene_script_to_s3,
    create_shotlist_json,
    upload_shotlist_json_to_s3,
)
from app.utils.csv_parser import ShotData, generate_shot_id

from app.tasks.phase1_tasks import run_phase1_workflow_task, resume_phase1_workflow_task
from app.tasks.phase2_tasks import (
    run_phase2_workflow_task,
    resume_phase2_after_strategy_approval_task,
    resume_phase2_after_prompt_approval_task,
    resume_phase2_after_final_approval_task,
    check_phase1_completion,
)
from app.tasks.phase3_tasks import run_phase3_workflow_task, resume_phase3_workflow_task


# ============================================================================
# Constants
# ============================================================================

POLL_INTERVAL = 30              # seconds between rescheduled poll ticks
MAX_ATTEMPTS_PHASE1 = 480       # 4 hours at 30 s intervals
MAX_ATTEMPTS_PHASE2 = 480       # 4 hours
MAX_ATTEMPTS_PHASE3 = 360       # 3 hours

PHASE2_TERMINAL = {"completed", "failed", "rejected"}
PHASE3_TERMINAL = {"completed", "failed"}


# ============================================================================
# State helpers
# ============================================================================


def _save_master_state(
    pipeline_service: PipelineService,
    master_job_id: str,
    state: Dict[str, Any],
    *,
    status: Optional[str] = None,
    pipeline_status: Optional[str] = None,
    **extra_fields,
) -> None:
    """Persist *state* (and any extra top-level fields) back to MongoDB."""
    job = pipeline_service.get_job(master_job_id)
    effective_status = status or (job.status if job else "running")
    pipeline_service.update_job_status(
        master_job_id,
        status=effective_status,
        pipeline_status=pipeline_status or effective_status,
        state=state,
        **extra_fields,
    )


def _fail_master(
    pipeline_service: PipelineService,
    master_job_id: str,
    message: str,
) -> None:
    pipeline_service.update_job_status(
        master_job_id,
        status="failed",
        pipeline_status="failed",
        error_message=message[:500],
    )


# ============================================================================
# Movie / scene setup helpers
# ============================================================================

def _build_combined_script(scenes_dicts: List[Dict[str, Any]]) -> str:
    combined = ""
    for s in sorted(scenes_dicts, key=lambda x: x["scene_number"]):
        combined += f"=== SCENE {s['scene_number']}: {s.get('scene_name', '')} ===\n\n"
        combined += s.get("script", "")
        combined += "\n\n"
    return combined.strip()


def _reconstruct_shot_data_objects(shots_dicts: List[Dict[str, Any]]) -> List[ShotData]:
    return [
        ShotData(
            scene_number=s["scene_number"],
            shot_number=s["shot_number"],
            shot_type=s.get("shot_type", ""),
            camera_movement=s.get("camera_movement", ""),
            description=s.get("description", ""),
            characters=s.get("characters") or [],
            locations=s.get("locations", ""),
            product_present=s.get("product_present", False),
        )
        for s in shots_dicts
    ]


def _build_shot_list_request_dict(
    scene_shots_dicts: List[Dict[str, Any]],
    scene_number: int,
    scene_name: str,
    scene_script: str,
) -> Dict[str, Any]:
    from app.api.v1.endpoints.phase2 import ShotItemRequest, ShotListRequest

    shot_items = []
    for s in sorted(scene_shots_dicts, key=lambda x: x["shot_number"]):
        try:
            shot_id = generate_shot_id(scene_number, s["shot_number"])
        except Exception:
            shot_id = f"{scene_number}.{s['shot_number']}"

        shot_items.append(
            ShotItemRequest(
                shot_id=shot_id,
                description=s.get("description", ""),
                scene_number=scene_number,
                shot_style=s.get("shot_type"),
                camera_movement=s.get("camera_movement"),
                source_type="generated",
                characters=s.get("characters"),
                locations=s.get("locations"),
                product_present=s.get("product_present", False),
            )
        )

    return ShotListRequest(
        episode_id=f"E{scene_number:02d}",
        title=scene_name,
        shots=shot_items,
        scene_description=(scene_script[:500] if scene_script else None),
    ).model_dump()


# ============================================================================
# Phase 3 helpers
# ============================================================================

def _reconstruct_phase3_state(
    shot_id: str,
    show_id: str,
    shots_collection: Any,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "shot_id": shot_id,
        "show_id": show_id,
        "video_versions": [],
        "current_version": 0,
        "pipeline_status": "waiting_for_human",
    }

    shot_doc = shots_collection.find_one(
        {"show_id": show_id, "annotated_shots.shot_id": shot_id}
    )
    if not shot_doc:
        shot_doc = shots_collection.find_one({"shot_id": shot_id})

    if shot_doc:
        for shot_item in shot_doc.get("annotated_shots", []):
            if shot_item.get("shot_id") != shot_id:
                continue
            video_data = shot_item.get("video") or {}
            for version_key in sorted(video_data.keys()):
                urls = video_data[version_key].get("generated_videos_s3", [])
                if urls:
                    state["video_versions"].append(
                        {
                            "version": version_key,
                            "generated_videos_s3": urls,
                            "prompt": video_data[version_key].get("updated_prompt", ""),
                        }
                    )
            if state["video_versions"]:
                state["current_version"] = len(state["video_versions"]) - 1
                state["generated_video_url"] = state["video_versions"][-1]["generated_videos_s3"][0]
            break

    return state


def _seed_shot_has_last_frame(seed_shot_id: str, shots_collection: Any) -> bool:
    """
    Return True if the seed shot has a last-frame URL stored in MongoDB.

    Mirrors the two storage structures read by VideoGenerationAPIAgent.fetch_start_image_url
    for the last_frame_seed strategy:
    - New:  annotated_shots[].video.vN.last_frame_s3
    - Old:  annotated_shots[].generated_video_last_frame_s3
    """
    try:
        episode_doc = shots_collection.find_one({"annotated_shots.shot_id": seed_shot_id})
        if episode_doc:
            for shot_item in episode_doc.get("annotated_shots", []):
                if shot_item.get("shot_id") != seed_shot_id:
                    continue
                if shot_item.get("generated_video_last_frame_s3"):
                    return True
                video_data = shot_item.get("video") or {}
                for version_key, version_val in video_data.items():
                    if (
                        version_key.startswith("v")
                        and isinstance(version_val, dict)
                        and version_val.get("last_frame_s3")
                    ):
                        return True
                return False  # shot found but no last frame yet
        # Fallback: standalone document (old schema)
        doc = shots_collection.find_one({"shot_id": seed_shot_id})
        if doc:
            if doc.get("generated_video_last_frame_s3"):
                return True
            for version_key, version_val in (doc.get("video") or {}).items():
                if (
                    version_key.startswith("v")
                    and isinstance(version_val, dict)
                    and version_val.get("last_frame_s3")
                ):
                    return True
    except Exception:
        logger.exception(f"[Master] Error checking last_frame for seed shot {seed_shot_id}")
    return False  # pessimistic — prevents premature dispatch


def _dispatch_p3_shot(
    shot_id: str,
    show_id: str,
    scene_number: int,
    pipeline_service: PipelineService,
    queue_name: str,
    user_id: Optional[str],
    state: Dict[str, Any],
) -> str:
    """
    Create a Phase 3 pipeline job, dispatch the Celery task, and update
    state["all_phase3_jobs"] and the matching scene's phase3_job_ids in-place.
    """
    p3_job_id = pipeline_service.create_job(
        PipelineJobCreate(project_id=show_id, shot_id=shot_id)
    )["job_id"]

    p3_task = run_phase3_workflow_task.apply_async(
        args=[p3_job_id, shot_id, show_id, None, user_id],
        queue=queue_name,
        routing_key=queue_name,
    )
    pipeline_service.update_job_celery_task_id(p3_job_id, p3_task.id)

    state["all_phase3_jobs"].append([p3_job_id, shot_id, show_id])
    for info in state["scene_data"]:
        if info["scene_number"] == scene_number:
            info["phase3_job_ids"].append(p3_job_id)
            break

    logger.info(f"[Master] Phase 3 dispatched shot={shot_id} job={p3_job_id} task={p3_task.id}")
    return p3_job_id


def _dispatch_p3_unblocked(
    completed_shot_id: str,
    shots_collection: Any,
    pipeline_service: PipelineService,
    queue_name: str,
    user_id: Optional[str],
    state: Dict[str, Any],
) -> int:
    """Dispatch Phase 3 shots that were waiting on completed_shot_id (O(1) dict lookup)."""
    pending_by_seed: Dict[str, List] = state.get("pending_by_seed", {})
    waiting = pending_by_seed.get(completed_shot_id)
    if not waiting:
        return 0
    if not _seed_shot_has_last_frame(completed_shot_id, shots_collection):
        return 0

    count = 0
    failed = []
    for entry in waiting:
        try:
            _dispatch_p3_shot(
                entry["shot_id"], entry["show_id"], entry["scene_number"],
                pipeline_service, queue_name, user_id, state,
            )
            count += 1
            logger.info(f"[Master] Unblocked shot {entry['shot_id']} (seed {completed_shot_id} completed)")
        except Exception as exc:
            logger.error(f"[Master] Failed to dispatch unblocked shot {entry['shot_id']}: {exc}", exc_info=True)
            failed.append(entry)

    if failed:
        pending_by_seed[completed_shot_id] = failed
    else:
        del pending_by_seed[completed_shot_id]
    return count


def _try_unblock_all_p3_pending(
    shots_collection: Any,
    pipeline_service: PipelineService,
    queue_name: str,
    user_id: Optional[str],
    state: Dict[str, Any],
) -> int:
    """Opportunistic sweep each tick — one MongoDB query per unique seed, not per pending shot."""
    pending_by_seed: Dict[str, List] = state.get("pending_by_seed", {})
    if not pending_by_seed:
        return 0

    ready_seeds = {
        seed_id
        for seed_id in list(pending_by_seed)
        if _seed_shot_has_last_frame(seed_id, shots_collection)
    }
    if not ready_seeds:
        return 0

    count = 0
    for seed_id in ready_seeds:
        waiting = pending_by_seed.pop(seed_id, [])
        failed = []
        for entry in waiting:
            try:
                _dispatch_p3_shot(
                    entry["shot_id"], entry["show_id"], entry["scene_number"],
                    pipeline_service, queue_name, user_id, state,
                )
                count += 1
                logger.info(f"[Master] Opportunistically dispatched shot {entry['shot_id']} (seed {seed_id} ready)")
            except Exception as exc:
                logger.error(f"[Master] Failed to dispatch pending shot {entry['shot_id']}: {exc}", exc_info=True)
                failed.append(entry)
        if failed:
            pending_by_seed[seed_id] = failed
    return count


def _dispatch_phase2_for_scenes(
    scene_infos: List[Dict[str, Any]],
    pipeline_service: PipelineService,
    queue_name: str,
    movie_id: str,
    assets_collection_id: str,
    user_id: Optional[str],
    v1_project_id: Optional[str],
) -> None:
    """Dispatch a Phase 2 Celery task for each scene that has shots and hasn't been dispatched yet."""
    for info in scene_infos:
        if info.get("phase2_job_id"):
            continue  # already dispatched

        scene_number = info["scene_number"]
        project_id = info["project_id"]
        scene_shots_dicts = info["shots"]

        if not scene_shots_dicts:
            logger.warning(f"[Master] Scene {scene_number} has no shots — skipping Phase 2/3")
            info["phase2_error"] = "no_shots"
            continue

        try:
            check_phase1_completion(project_id=assets_collection_id, show_id=project_id)

            shot_list_dict = _build_shot_list_request_dict(
                scene_shots_dicts=scene_shots_dicts,
                scene_number=scene_number,
                scene_name=info["scene_name"],
                scene_script=info["script"],
            )

            p2_job_id = pipeline_service.create_job(
                PipelineJobCreate(project_id=assets_collection_id)
            )["job_id"]

            p2_task = run_phase2_workflow_task.apply_async(
                args=[
                    p2_job_id,
                    shot_list_dict,
                    project_id,
                    scene_number,
                    assets_collection_id,
                    info["script"][:500] if info["script"] else None,
                    movie_id,
                ],
                kwargs={"v1_project_id": v1_project_id},
                queue=queue_name,
                routing_key=queue_name,
            )
            pipeline_service.update_job_celery_task_id(p2_job_id, p2_task.id)
            info["phase2_job_id"] = p2_job_id
            info["phase2_celery_task_id"] = p2_task.id
            logger.info(
                f"[Master] Phase 2 dispatched scene {scene_number}: job={p2_job_id} task={p2_task.id}"
            )
        except Exception as exc:
            logger.error(
                f"[Master] Failed to dispatch Phase 2 for scene {scene_number}: {exc}",
                exc_info=True,
            )
            info["phase2_error"] = str(exc)


# ============================================================================
# Task 1 — Setup: create movie, dispatch Phase 1, hand off to poll_phase1_task
# ============================================================================

@celery_app.task(
    bind=True,
    name="master.run_pipeline",
    max_retries=0,
    acks_late=False,
    track_started=True,
    time_limit=600,        # 10 min: movie creation + S3 uploads + one Celery dispatch
    soft_time_limit=540,
)
def run_master_pipeline_task(
    self,
    master_job_id: str,
    scenes_dicts: List[Dict[str, Any]],
    shots_dicts: List[Dict[str, Any]],
    title: str,
    description: Optional[str] = None,
    genre: Optional[str] = None,
    visual_style: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
    video_model: Optional[str] = None,
    user_id: Optional[str] = None,
    v1_project_id: Optional[str] = None,
    product_image_s3_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    STEP 1 + 2: Create movie document, upload S3 assets, dispatch Phase 1.
    Immediately hands off to poll_phase1_task and returns — holds worker slot for < 10 min.
    """
    pipeline_service = PipelineService()
    queue_name = get_workflow_queue_name()

    try:
        # ======================================================================
        # STEP 1: Create Movie
        # ======================================================================
        global_settings: Dict[str, Any] = {}
        if visual_style:
            global_settings["visual_style"] = visual_style
        if aspect_ratio:
            global_settings["aspect_ratio"] = aspect_ratio
        if video_model:
            global_settings["video_model"] = video_model

        movie_svc = MovieService()
        proj_svc = ProjectService()
        assets_svc = AssetsCollectionService()

        movie_result = movie_svc.create_movie(
            title=title,
            scenes=scenes_dicts,
            description=description,
            genre=genre,
            user_id=user_id,
            global_settings=global_settings or None,
            v1_project_id=v1_project_id,
        )
        movie_id: str = movie_result["movie_id"]

        assets_result = assets_svc.create_assets_collection(movie_id)
        assets_collection_id: str = assets_result["assets_collection_id"]
        movie_svc.set_assets_collection_id(movie_id, assets_collection_id)

        shot_data_objects = _reconstruct_shot_data_objects(shots_dicts)

        scene_infos: List[Dict[str, Any]] = []
        for scene_dict in sorted(scenes_dicts, key=lambda x: x["scene_number"]):
            scene_number: int = scene_dict["scene_number"]
            scene_name: str = scene_dict.get("scene_name", f"Scene {scene_number}")
            script: str = scene_dict.get("script", "")

            script_s3_url = upload_scene_script_to_s3(
                movie_id=movie_id,
                scene_number=scene_number,
                script_text=script,
            )

            scene_shots_objects = [s for s in shot_data_objects if s.scene_number == scene_number]
            scene_shots_dicts = [s for s in shots_dicts if s["scene_number"] == scene_number]
            shotlist_s3_url: Optional[str] = None
            if scene_shots_objects:
                shotlist_json = create_shotlist_json(
                    movie_id=movie_id,
                    scene_number=scene_number,
                    scene_name=scene_name,
                    scene_script=script,
                    shots_data=scene_shots_objects,
                    project_id="",
                )
                shotlist_s3_url = upload_shotlist_json_to_s3(
                    movie_id=movie_id,
                    scene_number=scene_number,
                    shotlist_json=shotlist_json,
                )

            project_result = proj_svc.create_scene_project(
                movie_id=movie_id,
                assets_collection_id=assets_collection_id,
                scene_number=scene_number,
                scene_name=scene_name,
                script=script,
                shotlist=scene_dict.get("shotlist"),
                user_id=user_id,
            )
            project_id: str = project_result["project_id"]

            proj_svc.update_s3_urls(project_id, script_s3_url, shotlist_s3_url)
            if product_image_s3_url:
                proj_svc.update_product_image_url(project_id, product_image_s3_url)
            movie_svc.add_project_id(movie_id, project_id)
            movie_svc.update_scene_project_id(movie_id, scene_number, project_id)

            scene_infos.append(
                {
                    "scene_number": scene_number,
                    "scene_name": scene_name,
                    "script": script,
                    "project_id": project_id,
                    "shots": scene_shots_dicts,
                    "phase1_job_id": None,
                    "phase2_job_id": None,
                    "phase2_celery_task_id": None,
                    "phase3_job_ids": [],
                    "phase2_error": None,
                }
            )

        logger.info(f"[Master] Movie created: {movie_id} with {len(scene_infos)} scenes")

        # ======================================================================
        # STEP 2: Dispatch Phase 1
        # ======================================================================
        combined_script = _build_combined_script(scenes_dicts)

        phase1_job_id: str = pipeline_service.create_job(
            {
                "movie_id": movie_id,
                "assets_collection_id": assets_collection_id,
                "type": "phase1_movie",
                "status": "pending",
            }
        )["job_id"]

        p1_task = run_phase1_workflow_task.apply_async(
            args=[phase1_job_id, movie_id, assets_collection_id, None, combined_script],
            queue=queue_name,
            routing_key=queue_name,
        )
        pipeline_service.update_job_celery_task_id(phase1_job_id, p1_task.id)
        movie_svc.update_phase1_status(movie_id, "running")

        for info in scene_infos:
            info["phase1_job_id"] = phase1_job_id

        logger.info(f"[Master] Phase 1 dispatched: job={phase1_job_id} task={p1_task.id}")

        # ======================================================================
        # Persist initial state and hand off to poll_phase1_task
        # ======================================================================
        state: Dict[str, Any] = {
            "master_pipeline": True,
            "current_phase": "phase1",
            "movie_id": movie_id,
            "assets_collection_id": assets_collection_id,
            "scene_data": scene_infos,
            "total_scenes": len(scene_infos),
            "phase1_job_id": phase1_job_id,
            "phase1_checkpoint_dispatched": False,
            "phase1_complete": False,
            "phase2_complete": False,
            "phase3_complete": False,
            "failed_scenes": [],
            "user_id": user_id,
            "v1_project_id": v1_project_id,
            # Phase 2 poll state
            "dispatched_p2_resumes": [],
            # Phase 3 poll state
            "all_phase3_jobs": [],
            "pending_by_seed": {},
            "dispatched_p3_resumes": [],
        }
        _save_master_state(pipeline_service, master_job_id, state)

        poll_phase1_task.apply_async(
            args=[master_job_id],
            kwargs={"attempt": 0},
            queue=queue_name,
            routing_key=queue_name,
        )

        logger.info(f"[Master] Setup complete, poll_phase1_task scheduled — master_job_id={master_job_id}")
        return {"status": "running", "master_job_id": master_job_id, "movie_id": movie_id}

    except SoftTimeLimitExceeded:
        logger.warning(f"[Master {master_job_id}] Setup soft time limit exceeded")
        _fail_master(pipeline_service, master_job_id, "Master setup exceeded time limit")
        raise

    except Exception as exc:
        logger.error(f"[Master {master_job_id}] Setup failed: {exc}", exc_info=True)
        _fail_master(pipeline_service, master_job_id, str(exc))
        raise


# ============================================================================
# Task 2 — Poll Phase 1
# ============================================================================

@celery_app.task(
    bind=False,
    name="master.poll_phase1",
    max_retries=0,
    acks_late=False,
    time_limit=14400,
    soft_time_limit=14100,
)
def poll_phase1_task(master_job_id: str, attempt: int = 0) -> None:
    """
    Looping tick for Phase 1.
    - Auto-approves the human checkpoint after Agent 7.
    - When Phase 1 completes: dispatches Phase 2 per scene, then hands off to poll_phase2_task.
    - Uses a while loop with time.sleep() instead of countdown to avoid SQS visibility loops.
    """
    pipeline_service = PipelineService()
    queue_name = get_workflow_queue_name()

    try:
        while True:
            # Guard: drop duplicate SQS deliveries and ticks for already-terminal jobs
            master_job = pipeline_service.get_job(master_job_id)
            if not master_job:
                logger.error(f"[Master] poll_phase1: master job {master_job_id} not found — dropping tick")
                return
            if master_job.status in ("completed", "failed"):
                logger.debug(f"[Master] poll_phase1: job {master_job_id} is {master_job.status} — dropping tick")
                return

            state = master_job.state or {}
            if not state:
                logger.error(f"[Master] poll_phase1: no state found for {master_job_id}")
                return

            phase1_job_id: str = state.get("phase1_job_id", "")
            if not phase1_job_id:
                _fail_master(pipeline_service, master_job_id, "poll_phase1: phase1_job_id missing from state")
                return

            p1_job = pipeline_service.get_job(phase1_job_id)
            if not p1_job:
                _fail_master(pipeline_service, master_job_id, f"Phase 1 job {phase1_job_id} not found in MongoDB")
                return

            if p1_job.status == "failed":
                _fail_master(pipeline_service, master_job_id, f"Phase 1 failed: {p1_job.error_message}")
                return

            if attempt >= MAX_ATTEMPTS_PHASE1:
                _fail_master(pipeline_service, master_job_id, "Phase 1 timed out after 4 hours")
                return

            # Auto-approve Phase 1 human checkpoint (idempotent via flag in state)
            if (
                p1_job.status == "waiting_for_human_approval"
                and not state.get("phase1_checkpoint_dispatched")
            ):
                p1_resume = resume_phase1_workflow_task.apply_async(
                    args=[phase1_job_id, p1_job.state or {}],
                    queue=queue_name,
                    routing_key=queue_name,
                )
                pipeline_service.update_job_celery_task_id(phase1_job_id, p1_resume.id)
                state["phase1_checkpoint_dispatched"] = True
                _save_master_state(pipeline_service, master_job_id, state)
                logger.info(f"[Master] Phase 1 checkpoint auto-approved → resume task={p1_resume.id}")

            if p1_job.status != "completed":
                # Still running — wait and retry
                attempt += 1
                time.sleep(POLL_INTERVAL)
                continue

            # ======================================================================
            # Phase 1 complete — dispatch Phase 2 per scene, hand off to poll_phase2_task
            # ======================================================================
            logger.info("[Master] Phase 1 completed — dispatching Phase 2 per scene")
            state["phase1_complete"] = True
            state["current_phase"] = "phase2"

            _dispatch_phase2_for_scenes(
                scene_infos=state["scene_data"],
                pipeline_service=pipeline_service,
                queue_name=queue_name,
                movie_id=state["movie_id"],
                assets_collection_id=state["assets_collection_id"],
                user_id=state.get("user_id"),
                v1_project_id=state.get("v1_project_id"),
            )
            _save_master_state(pipeline_service, master_job_id, state)

            poll_phase2_task.apply_async(
                args=[master_job_id],
                kwargs={"attempt": 0},
                queue=queue_name,
                routing_key=queue_name,
            )
            logger.info(f"[Master] Phase 2 dispatched, poll_phase2_task scheduled — master_job_id={master_job_id}")
            break

    except SoftTimeLimitExceeded:
        logger.warning(f"[Master] poll_phase1 soft time limit hit — rescheduling at attempt={attempt}")
        poll_phase1_task.apply_async(
            args=[master_job_id],
            kwargs={"attempt": attempt},
            queue=queue_name,
            routing_key=queue_name,
        )


# ============================================================================
# Task 3 — Poll Phase 2
# ============================================================================

@celery_app.task(
    bind=False,
    name="master.poll_phase2",
    max_retries=0,
    acks_late=False,
    time_limit=14400,
    soft_time_limit=14100,
)
def poll_phase2_task(master_job_id: str, attempt: int = 0) -> None:
    """
    Looping tick for Phase 2 (all scenes).
    - Auto-approves the three human checkpoints per scene (strategy, prompts, final images).
    - When all scenes reach a terminal state (or max attempts exceeded): dispatches Phase 3 per shot,
      then hands off to poll_phase3_task.
    - Uses a while loop with time.sleep() instead of countdown to avoid SQS visibility loops.
    """
    pipeline_service = PipelineService()
    queue_name = get_workflow_queue_name()

    try:
        while True:
            master_job = pipeline_service.get_job(master_job_id)
            if not master_job:
                logger.error(f"[Master] poll_phase2: master job {master_job_id} not found — dropping tick")
                return
            if master_job.status in ("completed", "failed"):
                logger.debug(f"[Master] poll_phase2: job {master_job_id} is {master_job.status} — dropping tick")
                return

            state = master_job.state or {}
            if not state:
                logger.error(f"[Master] poll_phase2: no state found for {master_job_id}")
                return

            scene_infos: List[Dict[str, Any]] = state.get("scene_data", [])
            active_p2 = [info for info in scene_infos if info.get("phase2_job_id")]

            # Race condition guard: phase2_job_ids are written by poll_phase1 just before
            # dispatching poll_phase2. If we read stale state, active_p2 is empty and we'd
            # immediately dispatch phase 3 before any phase 2 work runs.
            if not active_p2 and scene_infos:
                logger.warning("[Master] poll_phase2: no phase2_job_ids in state yet — waiting for dispatch")
                attempt += 1
                time.sleep(POLL_INTERVAL)
                continue

            # Deserialise dispatched_p2_resumes from [[job_id, status], ...] → set of tuples
            dispatched_resumes: set = {
                tuple(item) for item in state.get("dispatched_p2_resumes", [])
            }
            new_resume_added = False

            all_done = True

            for info in active_p2:
                p2_job_id: str = info["phase2_job_id"]
                p2_job = pipeline_service.get_job(p2_job_id)
                if not p2_job or p2_job.status in PHASE2_TERMINAL:
                    continue

                all_done = False
                show_id: str = info["project_id"]
                key = (p2_job_id, p2_job.pipeline_status)

                if key in dispatched_resumes:
                    continue

                if p2_job.pipeline_status == "waiting_for_approval":
                    resume = resume_phase2_after_strategy_approval_task.apply_async(
                        args=[p2_job_id, show_id, info["scene_number"], True, None],
                        queue=queue_name,
                        routing_key=queue_name,
                    )
                    pipeline_service.update_job_celery_task_id(p2_job_id, resume.id)
                    dispatched_resumes.add(key)
                    new_resume_added = True
                    logger.info(f"[Master] Scene {info['scene_number']} strategy auto-approved → resume={resume.id}")

                elif p2_job.pipeline_status == "waiting_for_prompt_approval":
                    resume = resume_phase2_after_prompt_approval_task.apply_async(
                        args=[p2_job_id, True, None],
                        queue=queue_name,
                        routing_key=queue_name,
                    )
                    pipeline_service.update_job_celery_task_id(p2_job_id, resume.id)
                    dispatched_resumes.add(key)
                    new_resume_added = True
                    logger.info(f"[Master] Scene {info['scene_number']} prompt auto-approved → resume={resume.id}")

                elif p2_job.pipeline_status == "waiting_for_final_approval":
                    resume = resume_phase2_after_final_approval_task.apply_async(
                        args=[p2_job_id, True, None],
                        queue=queue_name,
                        routing_key=queue_name,
                    )
                    pipeline_service.update_job_celery_task_id(p2_job_id, resume.id)
                    dispatched_resumes.add(key)
                    new_resume_added = True
                    logger.info(f"[Master] Scene {info['scene_number']} final images auto-approved → resume={resume.id}")

            if new_resume_added:
                state["dispatched_p2_resumes"] = [list(r) for r in dispatched_resumes]
                _save_master_state(pipeline_service, master_job_id, state)

            timed_out = attempt >= MAX_ATTEMPTS_PHASE2

            if not all_done and not timed_out:
                # Still waiting — wait and retry
                attempt += 1
                time.sleep(POLL_INTERVAL)
                continue

            # ======================================================================
            # Phase 2 done (or timed out) — pause for human image review
            # ======================================================================
            if timed_out:
                logger.warning("[Master] Phase 2 monitoring timed out — pausing for image review")
            else:
                logger.info("[Master] Phase 2 complete — pausing for human image review")

            state["phase2_complete"] = True
            state["current_phase"] = "waiting_for_image_review"
            _save_master_state(
                pipeline_service, master_job_id, state,
                status="running",
                pipeline_status="waiting_for_image_review",
            )
            logger.info(
                f"[Master] Pipeline paused — review images in the Streamlit app and click "
                f"'Save All Selections', then POST /api/v1/master/continue-to-phase3/{master_job_id}"
            )
            break

    except SoftTimeLimitExceeded:
        logger.warning(f"[Master] poll_phase2 soft time limit hit — rescheduling at attempt={attempt}")
        poll_phase2_task.apply_async(
            args=[master_job_id],
            kwargs={"attempt": attempt},
            queue=queue_name,
            routing_key=queue_name,
        )


# ============================================================================
# Task 3b — Dispatch Phase 3 (called by continue-to-phase3 endpoint)
# ============================================================================

@celery_app.task(
    bind=False,
    name="master.dispatch_phase3",
    max_retries=0,
    acks_late=False,
    time_limit=300,
    soft_time_limit=270,
)
def dispatch_phase3_task(master_job_id: str) -> None:
    """
    Discovers annotated shots from all Phase 2 scenes and dispatches Phase 3
    per shot, then hands off to poll_phase3_task.

    Called by POST /api/v1/master/continue-to-phase3/{master_job_id} after the
    human has reviewed and saved image selections in the Streamlit review app.
    """
    pipeline_service = PipelineService()
    queue_name = get_workflow_queue_name()

    master_job = pipeline_service.get_job(master_job_id)
    if not master_job:
        logger.error(f"[Master] dispatch_phase3: master job {master_job_id} not found")
        return

    state = master_job.state or {}
    scene_infos: List[Dict[str, Any]] = state.get("scene_data", [])
    active_p2 = [info for info in scene_infos if info.get("phase2_job_id")]

    state["current_phase"] = "phase3"
    state["all_phase3_jobs"] = []
    state["pending_by_seed"] = {}
    state["dispatched_p3_resumes"] = []

    _, db = get_database()
    shots_collection = db["shots"]
    user_id: Optional[str] = state.get("user_id")

    dispatched_shot_ids: set = set()

    for info in active_p2:
        p2_job = pipeline_service.get_job(info["phase2_job_id"])
        if p2_job and p2_job.status == "failed":
            logger.warning(f"[Master] dispatch_phase3: Scene {info['scene_number']} Phase 2 failed — skipping")
            continue

        show_id = info["project_id"]
        shot_doc = shots_collection.find_one({"show_id": show_id})
        if not shot_doc:
            logger.warning(f"[Master] dispatch_phase3: No shots document for show_id={show_id}")
            continue

        for shot_item in shot_doc.get("annotated_shots", []):
            shot_id = shot_item.get("shot_id")
            if not shot_id or shot_id in dispatched_shot_ids:
                continue

            generation_strategy = shot_item.get("generation_strategy", "generate_new")
            seed_shot_id = shot_item.get("seed_shot_id")

            if generation_strategy == "last_frame_seed" and seed_shot_id:
                if not _seed_shot_has_last_frame(seed_shot_id, shots_collection):
                    state["pending_by_seed"].setdefault(seed_shot_id, []).append(
                        {"shot_id": shot_id, "show_id": show_id, "scene_number": info["scene_number"]}
                    )
                    dispatched_shot_ids.add(shot_id)
                    logger.info(f"[Master] dispatch_phase3: Shot {shot_id} pending seed {seed_shot_id}")
                    continue

            _dispatch_p3_shot(shot_id, show_id, info["scene_number"], pipeline_service, queue_name, user_id, state)
            dispatched_shot_ids.add(shot_id)

    total_pending = sum(len(v) for v in state["pending_by_seed"].values())
    if not state["all_phase3_jobs"] and total_pending:
        logger.warning(
            f"[Master] dispatch_phase3: All {total_pending} Phase 3 shots are pending (last_frame_seed) "
            f"with no root shots dispatched — pipeline will time out"
        )

    _save_master_state(pipeline_service, master_job_id, state,
                       status="running", pipeline_status="running")

    poll_phase3_task.apply_async(
        args=[master_job_id],
        kwargs={"attempt": 0},
        queue=queue_name,
        routing_key=queue_name,
    )
    logger.info(f"[Master] dispatch_phase3: Phase 3 dispatched — master_job_id={master_job_id}")


# ============================================================================
# Task 4 — Poll Phase 3
# ============================================================================

@celery_app.task(
    bind=False,
    name="master.poll_phase3",
    max_retries=0,
    acks_late=False,
    time_limit=14400,
    soft_time_limit=14100,
)
def poll_phase3_task(master_job_id: str, attempt: int = 0) -> None:
    """
    Looping tick for Phase 3 (all shots).
    - Batch-fetches all Phase 3 job statuses in one MongoDB query.
    - Auto-approves video checkpoints.
    - Dispatches last_frame_seed shots as their seeds complete.
    - When all shots are terminal and nothing is pending: marks master completed.
    - Uses a while loop with time.sleep() instead of countdown to avoid SQS visibility loops.
    """
    pipeline_service = PipelineService()
    queue_name = get_workflow_queue_name()

    try:
        while True:
            master_job = pipeline_service.get_job(master_job_id)
            if not master_job:
                logger.error(f"[Master] poll_phase3: master job {master_job_id} not found — dropping tick")
                return
            if master_job.status in ("completed", "failed"):
                logger.debug(f"[Master] poll_phase3: job {master_job_id} is {master_job.status} — dropping tick")
                return

            state = master_job.state or {}
            if not state:
                logger.error(f"[Master] poll_phase3: no state found for {master_job_id}")
                return

            _, db = get_database()
            shots_collection = db["shots"]
            user_id: Optional[str] = state.get("user_id")

            # Ensure mutable collections are anchored in state (so _dispatch_p3_shot mutates in-place)
            if "all_phase3_jobs" not in state:
                state["all_phase3_jobs"] = []
            if "pending_by_seed" not in state:
                state["pending_by_seed"] = {}

            all_phase3_jobs: List[List[str]] = state["all_phase3_jobs"]
            pending_by_seed: Dict[str, List] = state["pending_by_seed"]
            dispatched_p3_resumes: set = set(state.get("dispatched_p3_resumes", []))

            prev_p3_count = len(all_phase3_jobs)
            prev_pending_count = sum(len(v) for v in pending_by_seed.values())
            prev_resumes_count = len(dispatched_p3_resumes)

            # Batch-fetch all known Phase 3 job statuses in one query
            known_ids = [entry[0] for entry in all_phase3_jobs]
            jobs_map = pipeline_service.get_jobs(known_ids) if known_ids else {}

            all_done = True

            # Index loop: len(all_phase3_jobs) may grow mid-loop as unblocked shots are dispatched.
            # Newly appended entries won't be in jobs_map this tick — they return None and set all_done=False,
            # which is correct; they'll be fetched on the next tick.
            idx = 0
            while idx < len(all_phase3_jobs):
                p3_job_id, shot_id, show_id = all_phase3_jobs[idx]
                idx += 1

                p3_job = jobs_map.get(p3_job_id)
                if not p3_job:
                    # Newly dispatched this tick — not yet in jobs_map
                    all_done = False
                    continue

                if p3_job.status in PHASE3_TERMINAL:
                    if p3_job.status == "completed":
                        if _dispatch_p3_unblocked(shot_id, shots_collection, pipeline_service, queue_name, user_id, state) > 0:
                            all_done = False
                    continue

                all_done = False

                if (
                    p3_job.status in ("waiting_for_human_approval", "phase_3_checkpoint")
                    and p3_job_id not in dispatched_p3_resumes
                ):
                    p3_state = _reconstruct_phase3_state(shot_id, show_id, shots_collection)
                    p3_state["human_decision"] = "approved"
                    p3_state["human_updated_prompt"] = None
                    p3_state["human_feedback"] = "Auto-approved by master pipeline"

                    p3_resume = resume_phase3_workflow_task.apply_async(
                        args=[p3_job_id, p3_state],
                        queue=queue_name,
                        routing_key=queue_name,
                    )
                    pipeline_service.update_job_celery_task_id(p3_job_id, p3_resume.id)
                    dispatched_p3_resumes.add(p3_job_id)
                    logger.info(f"[Master] Shot {shot_id} video auto-approved → resume={p3_resume.id}")

            if pending_by_seed:
                all_done = False
                _try_unblock_all_p3_pending(shots_collection, pipeline_service, queue_name, user_id, state)

            # Persist state when anything changed
            state["dispatched_p3_resumes"] = list(dispatched_p3_resumes)
            state_changed = (
                len(all_phase3_jobs) != prev_p3_count
                or sum(len(v) for v in pending_by_seed.values()) != prev_pending_count
                or len(dispatched_p3_resumes) != prev_resumes_count
            )
            if state_changed:
                state["phase3_dispatched_count"] = len(all_phase3_jobs)
                state["phase3_pending_count"] = sum(len(v) for v in pending_by_seed.values())
                _save_master_state(pipeline_service, master_job_id, state)

            # ======================================================================
            # Phase 3 terminal checks / Phase 4 review gate
            # ======================================================================
            if attempt >= MAX_ATTEMPTS_PHASE3:
                final_pending = sum(len(v) for v in pending_by_seed.values())
                logger.warning(
                    f"[Master] Phase 3 timed out ({final_pending} shots still pending) "
                    "— pausing for video review"
                )
                state["current_phase"] = "waiting_for_video_review"
                state["phase3_complete"] = True
                pipeline_service.update_job_status(
                    master_job_id,
                    status="running",
                    pipeline_status="waiting_for_video_review",
                    state=state,
                )
                return

            if all_done:
                logger.info(
                    f"[Master] Phase 3 complete — pausing for video review. "
                    f"master_job_id={master_job_id}"
                )
                state["current_phase"] = "waiting_for_video_review"
                state["phase3_complete"] = True
                pipeline_service.update_job_status(
                    master_job_id,
                    status="running",
                    pipeline_status="waiting_for_video_review",
                    state=state,
                )
                return

            # Still running — wait and retry
            attempt += 1
            time.sleep(POLL_INTERVAL)

    except SoftTimeLimitExceeded:
        logger.warning(f"[Master] poll_phase3 soft time limit hit — rescheduling at attempt={attempt}")
        poll_phase3_task.apply_async(
            args=[master_job_id],
            kwargs={"attempt": attempt},
            queue=queue_name,
            routing_key=queue_name,
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "run_master_pipeline_task",
    "poll_phase1_task",
    "poll_phase2_task",
    "poll_phase3_task",
]
