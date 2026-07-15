"""
Movie API Endpoints
===================
FastAPI endpoints for managing movies and their scenes.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from shared.auth.dependencies import validate_admin_from_header, AdminUser

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception

# Initialize logger for this module
logger = get_logger(__name__)

from app.models.requests import CreateMovieRequest, StartMoviePhase1Request, MovieResponse
from app.models.mongodb.pipelines import HumanApprovalRequest, AssetPromptEdit
from app.services.movie_service import MovieService
from app.services.assets_collection_service import AssetsCollectionService
from app.services.project_service import ProjectService
from app.services.pipeline_service import PipelineService
from app.core.quota import QuotaManager, get_quota_manager
from app.api.v1.endpoints.phase2 import (
    Phase2StartRequest,
    ShotItemRequest,
    ShotListRequest,
    start_phase2_pipeline_job,
)
from app.config import get_bucket_name, get_s3_client, get_workflow_queue_name, upload_file_wrapper
from app.utils.csv_parser import (
    parse_movie_csv, validate_csv_format, combine_scripts_for_phase1,
    parse_shotlist_csv, validate_scene_sequence, validate_shot_sequence,
    validate_required_fields, validate_scene_alignment
)
from app.utils.s3_helpers import (
    upload_scene_script_to_s3, create_shotlist_json, upload_shotlist_json_to_s3
)

# Import Celery tasks
from app.tasks.phase1_tasks import run_phase1_workflow_task, resume_phase1_workflow_task

# Import rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/movies", tags=["Movies"])
movie_service = MovieService()
assets_collection_service = AssetsCollectionService()
project_service = ProjectService()
pipeline_service = PipelineService()

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)


class Phase2BootstrapRequest(BaseModel):
    """Request payload for triggering Phase 2 across multiple scenes."""

    scene_numbers: Optional[List[int]] = Field(
        default=None,
        description="Specific scene numbers to bootstrap. Defaults to all scenes configured for the movie.",
    )
    dry_run: bool = Field(
        default=False,
        description="If true, validate readiness for each scene without dispatching Celery jobs.",
    )
    max_scenes: Optional[int] = Field(
        default=None,
        description="Optional cap to limit how many scenes are dispatched in one call.",
    )


class Phase2BootstrapResult(BaseModel):
    """Per-scene result returned by the Phase 2 bootstrap endpoint."""

    scene_number: int
    project_id: Optional[str]
    status: str
    job_id: Optional[str] = None
    celery_task_id: Optional[str] = None
    error: Optional[str] = None


def _build_shotlist_s3_key(movie_id: str, scene_number: int) -> str:
    """Derive the canonical S3 key for a scene's shotlist JSON."""
    return f"movies/{movie_id}/scenes/scene_{scene_number:02d}_shotlist.json"


def _extract_s3_key_from_url(url: str, bucket_name: Optional[str]) -> Optional[str]:
    """
    Attempt to reverse an S3 key from a stored URL.

    Supports formats like:
        https://bucket.s3.amazonaws.com/key
        https://s3.amazonaws.com/bucket/key
        https://custom-endpoint/bucket/key
    """
    if not url:
        return None

    parsed = urlparse(url)
    path = parsed.path.lstrip("/")

    if not path:
        return None

    # Handle virtual-hosted-style URLs (bucket in hostname)
    if bucket_name and path.startswith(f"{bucket_name}/"):
        return path[len(bucket_name) + 1 :]

    return path


def _load_shotlist_json(
    movie_id: str,
    scene_number: int,
    shotlist_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch and deserialize the shotlist JSON for a scene from S3."""
    s3_client = get_s3_client()
    bucket_name = get_bucket_name()

    if not s3_client or not bucket_name:
        raise ValueError("S3 client or bucket configuration is missing")

    candidate_keys = [_build_shotlist_s3_key(movie_id, scene_number)]
    inferred_key = _extract_s3_key_from_url(shotlist_url, bucket_name) if shotlist_url else None
    if inferred_key and inferred_key not in candidate_keys:
        candidate_keys.append(inferred_key)

    last_error: Optional[str] = None
    for key in candidate_keys:
        try:
            logger.debug(f"Attempting to download shotlist from s3://{bucket_name}/{key}")
            obj = s3_client.get_object(Bucket=bucket_name, Key=key)
            payload = obj["Body"].read().decode("utf-8")
            return json.loads(payload)
        except ClientError as exc:
            error_detail = exc.response.get("Error", {}).get("Message", str(exc))
            last_error = f"{exc.response.get('Error', {}).get('Code', 'ClientError')}: {error_detail}"
            logger.warning(
                f"Failed to download shotlist for movie={movie_id}, scene={scene_number} from key={key}: {error_detail}"
            )
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                f"Unexpected error while downloading shotlist for movie={movie_id}, scene={scene_number}, key={key}: {exc}"
            )

    raise ValueError(
        f"Unable to load shotlist JSON for scene {scene_number} (movie {movie_id}): {last_error or 'unknown error'}"
    )


def _build_shot_list_request(
    shotlist_payload: Dict[str, Any],
    fallback_scene_number: int,
    fallback_scene_name: str,
    fallback_scene_description: Optional[str],
) -> ShotListRequest:
    """Convert stored shotlist JSON into the Phase 2 ShotListRequest model."""
    shot_list_section = shotlist_payload.get("shot_list")
    if not shot_list_section:
        raise ValueError("Shotlist JSON is missing the 'shot_list' section")

    raw_shots = shot_list_section.get("shots") or []
    if not raw_shots:
        raise ValueError("Shotlist JSON does not include any shots")

    normalized_shots: List[ShotItemRequest] = []
    for shot in raw_shots:
        cleaned_shot = {key: value for key, value in shot.items() if value is not None}

        # Allow model defaults to kick in when optional values are missing/empty
        if not cleaned_shot.get("source_type"):
            cleaned_shot.pop("source_type", None)

        if "scene_number" not in cleaned_shot:
            cleaned_shot["scene_number"] = fallback_scene_number

        normalized_shots.append(ShotItemRequest(**cleaned_shot))

    return ShotListRequest(
        episode_id=shot_list_section.get("episode_id") or f"E{fallback_scene_number:02d}",
        title=shot_list_section.get("title") or fallback_scene_name,
        shots=normalized_shots,
        scene_description=shot_list_section.get("scene_description") or fallback_scene_description,
    )


@router.post("/validate")
async def validate_movie_csvs(
    request: Request,
    script_csv: UploadFile = File(...),
    shotlist_csv: Optional[UploadFile] = File(None),
    visual_style: Optional[str] = Form(None),
    aspect_ratio: Optional[str] = Form(None),
    admin_user: AdminUser = Depends(validate_admin_from_header)
) -> Dict[str, Any]:
    """
    Validate script and shotlist CSV files without creating a movie.

    This endpoint performs comprehensive validation:
    1. Scene numbers are sequential (1, 2, 3...)
    2. Shot numbers are sequential within scenes (1.1, 1.2, 1.1A, 1.1B...)
    3. No missing required fields (descriptions, shot_type, camera_movement)
    4. Scene alignment between script and shotlist
    5. Visual style is valid (realistic, pixar, or 2d)
    6. Aspect ratio is valid (9:16, 16:9, or 2.39:1)

    Args:
        script_csv: CSV file with scenes (scene_number, scene_name, scene_script)
        shotlist_csv: Optional CSV file with shots (scene_number, shot_number, shot_type, camera_movement, description)
        visual_style: Visual style for the movie (realistic, pixar, or 2d)
        aspect_ratio: Aspect ratio for the movie (9:16 for Vertical, 16:9 for Horizontal, 2.39:1 for Cinematic)

    Returns:
        Dictionary with validation results:
        - success: True if valid, False if errors found
        - errors: List of error messages (empty if valid)
        - preview: Preview data including full script (only if valid)
    """
    try:
        logger.info("Validating CSV files")
        errors = []

        # Validate visual_style if provided
        valid_styles = ["realistic", "pixar", "2d"]
        if visual_style and visual_style not in valid_styles:
            errors.append(f"Invalid visual_style: '{visual_style}'. Must be one of: {', '.join(valid_styles)}")

        # Validate aspect_ratio if provided
        valid_ratios = ["9:16", "16:9", "2.39:1"]
        if aspect_ratio and aspect_ratio not in valid_ratios:
            errors.append(f"Invalid aspect_ratio: '{aspect_ratio}'. Must be one of: {', '.join(valid_ratios)} (Vertical: 9:16, Horizontal: 16:9, Cinematic: 2.39:1)")

        # Validate script CSV format
        try:
            validate_csv_format(script_csv)
        except HTTPException as e:
            errors.append(f"Script CSV: {e.detail}")
            return {
                "success": False,
                "errors": errors,
                "preview": None
            }

        # Parse script CSV
        try:
            scenes_data = await parse_movie_csv(script_csv)
            logger.info(f"Parsed {len(scenes_data)} scenes from script CSV")
        except HTTPException as e:
            errors.append(f"Script CSV: {e.detail}")
            return {
                "success": False,
                "errors": errors,
                "preview": None
            }

        # Validate scene sequence
        scene_numbers = [s.scene_number for s in scenes_data]
        scene_error = validate_scene_sequence(scene_numbers)
        if scene_error:
            errors.append(f"Script CSV: {scene_error}")

        # If shotlist provided, validate it
        shots_data = []
        if shotlist_csv:
            # Validate shotlist CSV format
            try:
                validate_csv_format(shotlist_csv)
            except HTTPException as e:
                errors.append(f"Shotlist CSV: {e.detail}")
                return {
                    "success": False,
                    "errors": errors,
                    "preview": None
                }

            # Parse shotlist CSV
            try:
                shots_data = await parse_shotlist_csv(shotlist_csv)
                logger.info(f"Parsed {len(shots_data)} shots from shotlist CSV")
            except HTTPException as e:
                errors.append(f"Shotlist CSV: {e.detail}")
                return {
                    "success": False,
                    "errors": errors,
                    "preview": None
                }

            # Validate shot sequence
            shot_error = validate_shot_sequence(shots_data)
            if shot_error:
                errors.append(f"Shotlist CSV: {shot_error}")

            # Validate required fields
            fields_error = validate_required_fields(shots_data)
            if fields_error:
                errors.append(f"Shotlist CSV: {fields_error}")

            # Validate scene alignment
            alignment_error = validate_scene_alignment(scenes_data, shots_data)
            if alignment_error:
                errors.append(f"CSV Alignment: {alignment_error}")

        # If any errors found, return them
        if errors:
            return {
                "success": False,
                "errors": errors,
                "preview": None
            }

        # Generate preview data
        full_script = combine_scripts_for_phase1(scenes_data)

        scenes_preview = []
        for scene in scenes_data:
            scene_dict = scene.to_dict()

            # Add shots for this scene if shotlist provided
            if shots_data:
                scene_shots = [s.to_dict() for s in shots_data if s.scene_number == scene.scene_number]
                scene_dict["shots"] = scene_shots

            scenes_preview.append(scene_dict)

        logger.info("CSV validation successful")
        return {
            "success": True,
            "errors": [],
            "preview": {
                "scenes": scenes_preview,
                "full_script": full_script,
                "total_scenes": len(scenes_data),
                "total_shots": len(shots_data) if shots_data else 0
            }
        }

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/create", response_model=MovieResponse)
async def create_movie_from_csv(
    request: Request,
    title: str = Form(...),
    script_csv: UploadFile = File(...),
    shotlist_csv: Optional[UploadFile] = File(None),
    description: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    visual_style: Optional[str] = Form(None),
    aspect_ratio: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    start_phase1: bool = Form(False),
    product_image_file: Optional[UploadFile] = File(None, description="Product image (PNG/JPG/JPEG, optional)"),
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager)
) -> MovieResponse:
    """
    Create a new movie by uploading script CSV and optional shotlist CSV.

    **Quota Enforcement:** This endpoint consumes quota when start_phase1=True

    Script CSV Format:
    - scene_number (integer)
    - scene_name (string)
    - scene_script (string)

    Shotlist CSV Format (optional):
    - scene_number (integer)
    - shot_number (string, e.g., "1.1", "1.2", "1.1A")
    - shot_type (string)
    - camera_movement (string)
    - description (string)

    Process:
    1. Parse and validate CSV files
    2. Upload scene scripts to S3
    3. Upload shotlist JSONs to S3 (if shotlist provided)
    4. Create movie document with all scenes
    5. Create empty assets_collection
    6. Create individual production_projects for each scene with S3 URLs
    7. Optionally start Phase 1 workflow for entire movie

    Args:
        title: Movie title
        script_csv: CSV file with scenes
        shotlist_csv: Optional CSV file with shotlist
        description: Optional movie description
        genre: Optional movie genre
        visual_style: Visual style for the movie (realistic, pixar, or 2d)
        aspect_ratio: Aspect ratio for the movie (9:16 for Vertical, 16:9 for Horizontal, 2.39:1 for Cinematic)
        user_id: Optional user ID
        start_phase1: Whether to start Phase 1 automatically (default: False)

    Returns:
        MovieResponse with movie details and job information
    """
    try:
        logger.info(f"Creating movie: {title}")

        # Enforce quota ONLY if starting Phase 1 workflow (expensive operation)
        if start_phase1:
            user_id_for_quota = user_id or admin_user.user_id
            if not user_id_for_quota:
                raise HTTPException(status_code=400, detail="user_id is required for quota enforcement")

            quota_manager.consume(
                user_id=user_id_for_quota,
                pipeline_name="production_workflow"
            )
            logger.info(f"Quota consumed for user {user_id_for_quota} (create movie with Phase 1)")

        # Validate script CSV format
        validate_csv_format(script_csv)

        # Parse script CSV file
        scenes_data = await parse_movie_csv(script_csv)
        logger.info(f"Parsed {len(scenes_data)} scenes from script CSV")

        # Parse shotlist CSV if provided
        shots_data = []
        if shotlist_csv:
            validate_csv_format(shotlist_csv)
            shots_data = await parse_shotlist_csv(shotlist_csv)
            logger.info(f"Parsed {len(shots_data)} shots from shotlist CSV")

        # Convert SceneData objects to dictionaries for MongoDB
        scenes = [scene.to_dict() for scene in scenes_data]

        # Log received visual_style parameter
        logger.info(f"Received visual_style parameter: {visual_style} (type: {type(visual_style)})")
        logger.info(f"Received aspect_ratio parameter: {aspect_ratio} (type: {type(aspect_ratio)})")

        # Validate visual_style if provided
        if visual_style:
            valid_styles = ["realistic", "pixar", "2d"]
            if visual_style not in valid_styles:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid visual_style: '{visual_style}'. Must be one of: {', '.join(valid_styles)}"
                )

        # Validate aspect_ratio if provided
        if aspect_ratio:
            valid_ratios = ["9:16", "16:9", "2.39:1"]
            if aspect_ratio not in valid_ratios:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid aspect_ratio: '{aspect_ratio}'. Must be one of: {', '.join(valid_ratios)} (Vertical: 9:16, Horizontal: 16:9, Cinematic: 2.39:1)"
                )

        # Prepare global_settings
        global_settings = {}
        if visual_style:
            global_settings["visual_style"] = visual_style
            logger.info(f"Setting visual_style to: {visual_style}")
        else:
            logger.warning("visual_style is None or empty - frontend not sending it!")

        if aspect_ratio:
            global_settings["aspect_ratio"] = aspect_ratio
            logger.info(f"Setting aspect_ratio to: {aspect_ratio}")
        else:
            logger.warning("aspect_ratio is None or empty - frontend not sending it!")

        # Only set global_settings if at least one setting is provided
        global_settings = global_settings if global_settings else None

        # Step 1: Create movie document
        movie_result = movie_service.create_movie(
            title=title,
            scenes=scenes,
            description=description,
            genre=genre,
            user_id=user_id,
            global_settings=global_settings,
        )
        movie_id = movie_result["movie_id"]
        logger.info(f"Movie created: {movie_id}")

        # Step 2: Create assets collection for the movie
        assets_result = assets_collection_service.create_assets_collection(movie_id)
        assets_collection_id = assets_result["assets_collection_id"]
        logger.info(f"Assets collection created: {assets_collection_id}")

        # Step 3: Update movie with assets_collection_id
        movie_service.set_assets_collection_id(movie_id, assets_collection_id)

        # Step 4: Create individual projects for each scene with S3 uploads
        project_ids = []
        for scene in scenes_data:
            # Upload scene script to S3
            script_s3_url = upload_scene_script_to_s3(
                movie_id=movie_id,
                scene_number=scene.scene_number,
                script_text=scene.script
            )
            logger.info(f"Uploaded script for scene {scene.scene_number} to S3")

            # Create and upload shotlist JSON if shotlist CSV was provided
            shotlist_s3_url = None
            if shots_data:
                # Get shots for this scene
                scene_shots = [s for s in shots_data if s.scene_number == scene.scene_number]

                if scene_shots:
                    # Create shotlist JSON structure (project_id will be updated later)
                    shotlist_json = create_shotlist_json(
                        movie_id=movie_id,
                        scene_number=scene.scene_number,
                        scene_name=scene.scene_name,
                        scene_script=scene.script,
                        shots_data=scene_shots,
                        project_id=""  # Empty for now, will be updated
                    )

                    # Upload shotlist JSON to S3
                    shotlist_s3_url = upload_shotlist_json_to_s3(
                        movie_id=movie_id,
                        scene_number=scene.scene_number,
                        shotlist_json=shotlist_json
                    )
                    logger.info(f"Uploaded shotlist for scene {scene.scene_number} to S3")

            # Create scene project
            project_result = project_service.create_scene_project(
                movie_id=movie_id,
                assets_collection_id=assets_collection_id,
                scene_number=scene.scene_number,
                scene_name=scene.scene_name,
                script=scene.script,
                shotlist=scene.shotlist if scene.shotlist else None,
                user_id=user_id
            )
            project_id = project_result["project_id"]
            project_ids.append(project_id)

            # Update project with S3 URLs
            project_service.update_s3_urls(
                project_id=project_id,
                script_s3_url=script_s3_url,
                shotlist_s3_url=shotlist_s3_url
            )
            logger.info(f"Updated S3 URLs for project {project_id}")

            # Add project_id to movie's project_ids array
            movie_service.add_project_id(movie_id, project_id)

            # Update scene's project_id in movie document
            movie_service.update_scene_project_id(movie_id, scene.scene_number, project_id)

            logger.info(f"Scene project created: {project_id} for scene {scene.scene_number}")

        logger.info(f"Created {len(project_ids)} scene projects with S3 uploads")

        # Step 4b: Upload product image once and store URL in all scene projects
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
                s3_key = f"movies/{movie_id}/product_image{ext}"
                product_image_url = upload_file_wrapper(
                    tmp_path,
                    s3_key=s3_key,
                    content_type=product_image_file.content_type or "image/png",
                    use_presigned_url=False,
                )
                for pid in project_ids:
                    project_service.update_product_image_url(pid, product_image_url)
                logger.info(f"Product image uploaded for movie {movie_id}, stored in {len(project_ids)} projects")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Step 5: Start Phase 1 workflow if requested
        phase1_job_id = None
        if start_phase1:
            # Combine all scene scripts for Phase 1 processing
            combined_script = combine_scripts_for_phase1(scenes_data)

            # Create pipeline job for Phase 1
            job_data = {
                "movie_id": movie_id,
                "assets_collection_id": assets_collection_id,
                "type": "phase1_movie",
                "status": "pending",
                "created_at": datetime.utcnow()
            }
            job_result = pipeline_service.create_job(job_data)
            phase1_job_id = job_result["job_id"]

            # Dispatch Celery task for Phase 1 workflow
            task = run_phase1_workflow_task.apply_async(
                args=[phase1_job_id, movie_id, assets_collection_id, None, combined_script]
            )

            # Update job with Celery task ID
            pipeline_service.update_job_celery_task_id(phase1_job_id, task.id)

            # Update movie Phase 1 status
            movie_service.update_phase1_status(movie_id, "running")

            logger.info(f"Phase 1 workflow started: job_id={phase1_job_id}, celery_task_id={task.id}")

        # Return response
        return MovieResponse(
            success=True,
            movie_id=movie_id,
            title=title,
            total_scenes=len(scenes),
            assets_collection_id=assets_collection_id,
            project_ids=project_ids,
            phase1_job_id=phase1_job_id,
            phase1_status="running" if start_phase1 else "pending",
            overall_status="created",
            created_at=movie_result["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create movie: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create movie: {str(e)}")


@router.post("/{movie_id}/phase2/bootstrap", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def bootstrap_phase2_for_movie(
    request: Request,
    movie_id: str,
    bootstrap_request: Phase2BootstrapRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager)
) -> Dict[str, Any]:
    """
    Trigger Phase 2 for every scene in a movie (or a filtered subset).

    **Quota Enforcement:** This endpoint consumes 1 quota unit (not per scene, just once for the bootstrap operation)

    This endpoint:
    1. Fetches the movie + associated scene projects
    2. Loads each scene's shotlist JSON from S3
    3. Normalizes the payload into Phase2StartRequest
    4. Dispatches Celery jobs sequentially (or dry-run to validate readiness)

    Returns a per-scene summary so the frontend can surface successes/failures.
    """
    # Enforce quota ONLY if not a dry run (dry run doesn't start actual work)
    if not bootstrap_request.dry_run:
        quota_manager.consume(
            user_id=admin_user.user_id,
            pipeline_name="production_workflow"
        )
        logger.info(f"Quota consumed for user {admin_user.user_id} (Phase 2 bootstrap for movie {movie_id})")

    movie = movie_service.get_movie(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    assets_collection_id = movie.get("assets_collection_id")
    if not assets_collection_id:
        raise HTTPException(
            status_code=400,
            detail="Movie is missing assets_collection_id. Complete Phase 1 before starting Phase 2."
        )

    scenes = movie.get("scenes") or []
    scene_map: Dict[int, Dict[str, Any]] = {}
    for scene in scenes:
        scene_number = scene.get("scene_number")
        project_id = scene.get("project_id")
        if scene_number is None or not project_id:
            continue
        try:
            normalized_number = int(scene_number)
        except (TypeError, ValueError):
            continue
        scene_map[normalized_number] = scene

    if not scene_map:
        raise HTTPException(
            status_code=400,
            detail="Movie does not have any scenes with linked projects."
        )

    results: List[Phase2BootstrapResult] = []

    if bootstrap_request.scene_numbers:
        target_scene_numbers = []
        for number in bootstrap_request.scene_numbers:
            if number in scene_map:
                target_scene_numbers.append(number)
            else:
                results.append(
                    Phase2BootstrapResult(
                        scene_number=number,
                        project_id=None,
                        status="skipped",
                        error="Scene not found or missing project_id"
                    )
                )
    else:
        target_scene_numbers = sorted(scene_map.keys())

    if bootstrap_request.max_scenes is not None and bootstrap_request.max_scenes > 0:
        target_scene_numbers = target_scene_numbers[: bootstrap_request.max_scenes]

    if not target_scene_numbers:
        return {
            "movie_id": movie_id,
            "assets_collection_id": assets_collection_id,
            "dry_run": bootstrap_request.dry_run,
            "results": [result.model_dump() for result in results],
            "message": "No scenes matched the current filter."
        }

    jobs_started = 0
    assets_collection_id_str = str(assets_collection_id)

    for scene_number in target_scene_numbers:
        scene = scene_map.get(scene_number)
        project_id = scene.get("project_id") if scene else None

        if not scene or not project_id:
            results.append(
                Phase2BootstrapResult(
                    scene_number=scene_number,
                    project_id=project_id,
                    status="skipped",
                    error="Scene is missing required metadata"
                )
            )
            continue

        project_doc = project_service.get_project(str(project_id))
        if not project_doc:
            results.append(
                Phase2BootstrapResult(
                    scene_number=scene_number,
                    project_id=str(project_id),
                    status="error",
                    error="Project not found"
                )
            )
            continue

        try:
            shotlist_payload = _load_shotlist_json(
                movie_id=movie_id,
                scene_number=scene_number,
                shotlist_url=project_doc.get("shotlist_json_s3_url")
            )

            shot_list_request = _build_shot_list_request(
                shotlist_payload,
                fallback_scene_number=scene_number,
                fallback_scene_name=scene.get("scene_name", f"Scene {scene_number}"),
                fallback_scene_description=scene.get("script")
            )

            episode_number = shotlist_payload.get("episode_number") or scene_number
            try:
                episode_number = int(episode_number)
            except (TypeError, ValueError):
                episode_number = scene_number

            scene_description = (
                shotlist_payload.get("scene_description")
                or shot_list_request.scene_description
                or scene.get("scene_name")
            )

            phase2_request = Phase2StartRequest(
                shot_list=shot_list_request,
                show_id=str(project_id),
                episode_number=episode_number,
                project_id=assets_collection_id_str,
                scene_description=scene_description,
                movie_id=str(movie_id),
            )

            if bootstrap_request.dry_run:
                results.append(
                    Phase2BootstrapResult(
                        scene_number=scene_number,
                        project_id=str(project_id),
                        status="ready"
                    )
                )
                continue

            job_response = await start_phase2_pipeline_job(phase2_request)
            jobs_started += 1
            results.append(
                Phase2BootstrapResult(
                    scene_number=scene_number,
                    project_id=str(project_id),
                    status="started",
                    job_id=job_response.job_id,
                    celery_task_id=job_response.celery_task_id
                )
            )

        except HTTPException as exc:
            error_detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
            results.append(
                Phase2BootstrapResult(
                    scene_number=scene_number,
                    project_id=str(project_id),
                    status="error",
                    error=error_detail
                )
            )
        except Exception as exc:
            logger.error(
                f"Failed to bootstrap Phase 2 for movie={movie_id}, scene={scene_number}: {exc}",
                exc_info=True
            )
            results.append(
                Phase2BootstrapResult(
                    scene_number=scene_number,
                    project_id=str(project_id),
                    status="error",
                    error=str(exc)
                )
            )

    successful = sum(1 for result in results if result.status in {"started", "ready"} and not result.error)
    failed = sum(1 for result in results if result.status == "error")

    return {
        "movie_id": movie_id,
        "assets_collection_id": assets_collection_id_str,
        "dry_run": bootstrap_request.dry_run,
        "scenes_requested": bootstrap_request.scene_numbers or "all",
        "scenes_attempted": len(target_scene_numbers),
        "scenes_ready_or_started": successful,
        "scenes_failed": failed,
        "jobs_started": jobs_started,
        "results": [result.model_dump() for result in results]
    }


@router.get("/list", response_model=Dict[str, Any])
async def list_movies(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    admin_user: AdminUser = Depends(validate_admin_from_header)
) -> Dict[str, Any]:
    """
    List all movies with optional filtering.

    Args:
        user_id: Optional user ID to filter by
        limit: Maximum number of movies to return (default: 100)
        offset: Number of movies to skip (default: 0)

    Returns:
        Dictionary with movies array and metadata
    """
    try:
        movies = movie_service.list_movies(user_id=user_id, limit=limit, offset=offset)

        return {
            "success": True,
            "movies": movies,
            "count": len(movies),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Failed to list movies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list movies: {str(e)}")


@router.get("/{movie_id}", response_model=Dict[str, Any])
async def get_movie(movie_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)) -> Dict[str, Any]:
    """
    Get a movie by ID with all details.

    Args:
        movie_id: Movie ID

    Returns:
        Movie document with all scenes and references
    """
    try:
        movie = movie_service.get_movie(movie_id)

        if not movie:
            raise HTTPException(status_code=404, detail=f"Movie not found: {movie_id}")

        return movie

    except ValueError as e:
        logger.warning(f"Invalid movie ID format: {movie_id} - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get movie {movie_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get movie: {str(e)}")


@router.get("/{movie_id}/status", response_model=Dict[str, Any])
async def get_movie_status(movie_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)) -> Dict[str, Any]:
    """
    Get movie Phase 1 status and current pipeline progress.

    This endpoint is used to poll the status of a running Phase 1 workflow.

    Args:
        movie_id: Movie ID

    Returns:
        Status information including current agent and progress
    """
    try:
        # Get movie to verify it exists
        movie = movie_service.get_movie(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail=f"Movie not found: {movie_id}")

        # Get the most recent job for this movie
        job_data = pipeline_service.get_job_by_movie_id(movie_id)

        if not job_data:
            # No job exists yet
            return {
                "success": True,
                "movie_id": movie_id,
                "phase1_status": movie.get("phase1_status", "pending"),
                "job_id": None,
                "job_status": None,
                "current_agent": None,
                "progress": {
                    "completed": 0,
                    "total": 8,
                    "percentage": 0.0
                },
                "waiting_for_approval": False,
                "agent_statuses": {}
            }

        job_id = job_data.get("job_id")
        job = pipeline_service.get_job(job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        # Calculate progress based on agent statuses
        agent_statuses = {
            "agent1": job.agent1_status,
            "agent2": job.agent2_status,
            "agent3": job.agent3_status,
            "agent4": job.agent4_status,
            "agent5": job.agent5_status,
            "agent6": job.agent6_status,
            "agent7": job.agent7_status,
            "agent8": job.agent8_status,
        }

        total_agents = 8
        completed_agents = sum(
            1 for status in agent_statuses.values()
            if status == "completed"
        )

        # Find current agent
        current_agent = None
        for i in range(1, 9):
            status = agent_statuses.get(f"agent{i}")
            if status in ["pending", "running"]:
                current_agent = i
                break

        return {
            "success": True,
            "movie_id": movie_id,
            "phase1_status": movie.get("phase1_status", "pending"),
            "job_id": job.job_id,
            "job_status": job.status,
            "pipeline_status": job.pipeline_status,
            "current_agent": current_agent,
            "progress": {
                "completed": completed_agents,
                "total": total_agents,
                "percentage": round((completed_agents / total_agents) * 100, 2)
            },
            "waiting_for_approval": job.status == "waiting_for_human_approval",
            "checkpoint_approved": job.checkpoint_approved,
            "approved_assets_count": len(job.approved_assets_list or []),
            "regeneration_count": job.regeneration_count,
            "max_regenerations": job.max_regenerations,
            "agent_statuses": agent_statuses,
            "celery_task_id": job.celery_task_id
        }

    except ValueError as e:
        logger.warning(f"Invalid movie ID format: {movie_id} - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get movie status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get movie status: {str(e)}")


@router.get("/{movie_id}/assets", response_model=Dict[str, Any])
async def get_movie_assets(
    movie_id: str,
    include_presigned_urls: bool = True,
    url_expiration: int = 3600,
    admin_user: AdminUser = Depends(validate_admin_from_header)
) -> Dict[str, Any]:
    """
    Get the assets collection for a movie (Phase 1 outputs) with job information.

    Args:
        movie_id: Movie ID
        include_presigned_urls: Whether to generate fresh S3 presigned URLs (default: True)
        url_expiration: Presigned URL expiration in seconds (default: 3600 = 1 hour)

    Returns:
        Assets collection document with all Phase 1 agent outputs, fresh presigned S3 URLs,
        and job information (job_id, status, etc.) for checkpoint operations
    """
    try:
        # Get movie to verify it exists
        movie = movie_service.get_movie(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail=f"Movie not found: {movie_id}")

        # Get assets collection
        assets_collection_id = movie.get("assets_collection_id")
        if not assets_collection_id:
            raise HTTPException(status_code=404, detail=f"Movie has no assets collection yet")

        assets = assets_collection_service.get_assets_collection(
            assets_collection_id,
            include_presigned_urls=include_presigned_urls,
            url_expiration=url_expiration
        )
        if not assets:
            raise HTTPException(status_code=404, detail=f"Assets collection not found")

        # Get the most recent job for this movie to include job_id and status
        job_data = pipeline_service.get_job_by_movie_id(movie_id)

        # Add job information to response
        if job_data:
            job_id = job_data.get("job_id")
            job = pipeline_service.get_job(job_id)

            if job:
                assets["job_id"] = job.job_id
                assets["job_status"] = job.status
                assets["pipeline_status"] = job.pipeline_status
                assets["current_agent"] = job.current_agent
                assets["waiting_for_approval"] = (job.status == "waiting_for_human_approval")
                assets["checkpoint_approved"] = job.checkpoint_approved
                assets["approved_assets_list"] = job.approved_assets_list or []
                assets["regeneration_count"] = job.regeneration_count
                assets["max_regenerations"] = job.max_regenerations
            else:
                logger.warning(f"Job {job_id} not found for movie {movie_id}")
        else:
            logger.warning(f"No job found for movie {movie_id}")

        return assets

    except ValueError as e:
        logger.warning(f"Invalid movie ID format: {movie_id} - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get movie assets: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get movie assets: {str(e)}")


@router.post("/start-phase1", response_model=Dict[str, Any])
async def start_movie_phase1(
    request: StartMoviePhase1Request,
    admin_user: AdminUser = Depends(validate_admin_from_header),
    quota_manager: QuotaManager = Depends(get_quota_manager)
) -> Dict[str, Any]:
    """
    Start Phase 1 workflow for an existing movie.

    This endpoint is used when a movie was created without starting Phase 1,
    or to re-run Phase 1 for a movie.

    **Quota Enforcement:** This endpoint consumes 1 quota unit

    Args:
        request: StartMoviePhase1Request with movie_id

    Returns:
        Dictionary with job details and Celery task ID
    """
    try:
        movie_id = request.movie_id

        # Enforce quota before starting expensive Phase 1 workflow
        quota_manager.consume(
            user_id=admin_user.user_id,
            pipeline_name="production_workflow"
        )
        logger.info(f"Quota consumed for user {admin_user.user_id} (start Phase 1)")

        # Get movie
        movie = movie_service.get_movie(movie_id)
        if not movie:
            raise HTTPException(status_code=404, detail=f"Movie not found: {movie_id}")

        # Get assets collection ID
        assets_collection_id = movie.get("assets_collection_id")
        if not assets_collection_id:
            raise HTTPException(status_code=400, detail=f"Movie has no assets collection")

        # Combine all scene scripts
        scenes = movie.get("scenes", [])
        if not scenes:
            raise HTTPException(status_code=400, detail=f"Movie has no scenes")

        # Import SceneData for compatibility with combine_scripts_for_phase1
        from app.utils.csv_parser import SceneData
        scenes_data = [
            SceneData(
                scene_number=s["scene_number"],
                scene_name=s["scene_name"],
                script=s["script"],
                shotlist=s.get("shotlist", "")
            )
            for s in scenes
        ]
        combined_script = combine_scripts_for_phase1(scenes_data)

        # Create pipeline job
        job_data = {
            "movie_id": movie_id,
            "assets_collection_id": assets_collection_id,
            "type": "phase1_movie",
            "status": "pending",
            "created_at": datetime.utcnow()
        }
        job_result = pipeline_service.create_job(job_data)
        job_id = job_result["job_id"]

        # Dispatch Celery task
        task = run_phase1_workflow_task.apply_async(
            args=[job_id, movie_id, assets_collection_id, None, combined_script]
        )

        # Update job with Celery task ID
        pipeline_service.update_job_celery_task_id(job_id, task.id)

        # Update movie Phase 1 status
        movie_service.update_phase1_status(movie_id, "running")

        logger.info(f"Phase 1 workflow started for movie {movie_id}: job_id={job_id}, celery_task_id={task.id}")

        return {
            "success": True,
            "job_id": job_id,
            "movie_id": movie_id,
            "assets_collection_id": assets_collection_id,
            "celery_task_id": task.id,
            "status": "running",
            "message": "Phase 1 workflow started successfully"
        }

    except ValueError as e:
        logger.warning(f"Invalid movie ID format: {movie_id} - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start Phase 1 for movie: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start Phase 1: {str(e)}")


@router.delete("/{movie_id}", response_model=Dict[str, Any])
async def delete_movie(movie_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)) -> Dict[str, Any]:
    """
    Delete a movie by ID.

    WARNING: This will delete the movie but not the associated projects or assets collection.
    Use with caution.

    Args:
        movie_id: Movie ID

    Returns:
        Dictionary with deletion status
    """
    try:
        result = movie_service.delete_movie(movie_id)

        return {
            "success": True,
            "movie_id": movie_id,
            "message": "Movie deleted successfully"
        }

    except ValueError as e:
        logger.warning(f"Invalid movie ID format: {movie_id} - {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete movie: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete movie: {str(e)}")


@router.post("/checkpoint/approve-by-movie/{movie_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def approve_checkpoint_by_movie_id(
    request: Request,
    movie_id: str,
    approval: HumanApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Approve specific assets at human checkpoint for a movie workflow using movie_id (does NOT proceed to Agent 8)

    This endpoint marks assets as approved and stores them in the job state.
    The job remains at the checkpoint so you can:
    - Approve more assets
    - Edit prompts for other assets
    - Use /checkpoint/finalize when ready to proceed to Agent 8

    User provides a list of approved asset IDs with individual feedback.

    NOTE: This endpoint uses movie_id instead of job_id for convenience.
    It will automatically find the most recent job for this movie.
    """
    # Get the most recent job for this movie
    job_data = pipeline_service.get_job_by_movie_id(movie_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"No job found for movie {movie_id}")

    job_id = job_data["job_id"]
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

    logger.info(f"{len(approved_asset_ids)} new assets approved for movie {movie_id}, job {job_id}")
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

    # Also update the assets_collection with approval status
    try:
        assets_collection_id = job.assets_collection_id
        if assets_collection_id:
            assets_collection_service.update_approval_status(
                assets_collection_id=assets_collection_id,
                approved_assets_list=all_approved,
                checkpoint_approved=False,  # Not finalized yet, just marking assets as approved
                human_approval_feedback={
                    "global_feedback": approval.global_feedback,
                    "approved_assets": [a.dict() for a in approval.approved_assets]
                }
            )
            logger.info(f"Approval status updated in assets_collection {assets_collection_id}")
        else:
            logger.warning(f"Job {job_id} has no assets_collection_id, skipping assets_collection update")
    except Exception as e:
        logger.error(f"Failed to update approval status in assets_collection: {e}")
        # Don't fail the entire request if assets_collection update fails

    logger.info("Assets marked as approved. Job remains at checkpoint.")
    logger.info("Use /checkpoint/finalize to proceed to Agent 8 when ready.")

    # Return updated job (still at checkpoint)
    job = pipeline_service.get_job(job_id)
    response = pipeline_service.to_response(job)
    return response.model_dump()


@router.post("/checkpoint/approve/{job_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def approve_checkpoint_for_movie(
    request: Request,
    job_id: str,
    approval: HumanApprovalRequest,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Approve specific assets at human checkpoint for a movie workflow (does NOT proceed to Agent 8)

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

    # Also update the assets_collection with approval status
    try:
        assets_collection_id = job.assets_collection_id
        if assets_collection_id:
            assets_collection_service.update_approval_status(
                assets_collection_id=assets_collection_id,
                approved_assets_list=all_approved,
                checkpoint_approved=False,  # Not finalized yet, just marking assets as approved
                human_approval_feedback={
                    "global_feedback": approval.global_feedback,
                    "approved_assets": [a.dict() for a in approval.approved_assets]
                }
            )
            logger.info(f"Approval status updated in assets_collection {assets_collection_id}")
        else:
            logger.warning(f"Job {job_id} has no assets_collection_id, skipping assets_collection update")
    except Exception as e:
        logger.error(f"Failed to update approval status in assets_collection: {e}")
        # Don't fail the entire request if assets_collection update fails

    logger.info("Assets marked as approved. Job remains at checkpoint.")
    logger.info("Use /checkpoint/finalize to proceed to Agent 8 when ready.")

    # Return updated job (still at checkpoint)
    job = pipeline_service.get_job(job_id)
    response = pipeline_service.to_response(job)
    return response.model_dump()


@router.post("/checkpoint/finalize/{job_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def finalize_checkpoint_for_movie(
    request: Request,
    job_id: str,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Finalize the checkpoint and proceed to Agent 8 (Variation Generator) for movie workflow

    This endpoint should be called after all assets are approved.
    Only the approved assets will be processed by Agent 8 for variation generation.

    Requirements:
    - Job must be at waiting_for_human_approval status
    - At least one asset must be approved (via /checkpoint/approve)
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

    # Get assets collection data to reconstruct full workflow state
    movie_id = job.movie_id
    assets_collection_id = job.assets_collection_id

    if not movie_id or not assets_collection_id:
        raise HTTPException(status_code=400, detail="Job is not associated with a movie")

    assets_collection = assets_collection_service.get_assets_collection(assets_collection_id)
    if not assets_collection:
        raise HTTPException(status_code=404, detail=f"Assets collection {assets_collection_id} not found")

    # Helper function to safely extract agent output from assets collection
    # Assets collection structure: { "agent5_output": { "output": {...}, "status": "...", "executed_at": "..." } }
    def safe_get_agent_output(assets_col, agent_number, output_key, default=None):
        """Safely extract agent output from assets collection, handling None values"""
        agent_key = f"agent{agent_number}_output"
        agent_data = assets_col.get(agent_key)

        if not agent_data or not isinstance(agent_data, dict):
            return default if default is not None else {}

        output = agent_data.get("output")
        if not output or not isinstance(output, dict):
            return default if default is not None else {}

        return output.get(output_key, default if default is not None else {})

    # Create minimal workflow state for Celery task (to avoid SQS 256KB message size limit)
    # Agent 8 will load full data from assets_collection when needed
    current_state = {
        # Job tracking fields
        "job_id": job.job_id,
        "movie_id": movie_id,
        "assets_collection_id": assets_collection_id,  # Critical: Agent 8 uses this to load data
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
        "script_content": assets_collection.get("combined_script", ""),
    }

    # Resume workflow using Celery (Agent 8 variation generation)
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

    # Update assets_collection to mark checkpoint as finalized
    try:
        if assets_collection_id:
            assets_collection_service.update_approval_status(
                assets_collection_id=assets_collection_id,
                approved_assets_list=approved_assets,
                checkpoint_approved=True,  # Now finalizing the checkpoint
                human_approval_feedback=job.human_approval_feedback
            )
            logger.info(f"Checkpoint finalized in assets_collection {assets_collection_id}")
    except Exception as e:
        logger.error(f"Failed to update checkpoint_approved in assets_collection: {e}")
        # Don't fail the entire request if assets_collection update fails

    # Return updated job (includes new celery_task_id for monitoring)
    job = pipeline_service.get_job(job_id)
    response = pipeline_service.to_response(job)
    return response.model_dump()


@router.post("/agent8/retry/{job_id}", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def retry_agent8_for_movie(
    request: Request,
    job_id: str,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Manually retry Agent 8 (Variation Generator) for a failed job

    This endpoint allows manually retrying Agent 8 when it has failed.
    It resets the Agent 8 status and triggers the workflow to run Agent 8 again.

    Requirements:
    - Job must exist
    - At least one asset must be approved
    """
    job = pipeline_service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check if any assets are approved
    approved_assets = job.approved_assets_list or []
    if not approved_assets:
        raise HTTPException(
            status_code=400,
            detail="No assets have been approved yet. Please approve at least one asset first."
        )

    logger.info(f"Manually retrying Agent 8 for job {job_id}")
    logger.info(f"{len(approved_assets)} approved assets will be processed")

    # Get assets collection data to reconstruct full workflow state
    movie_id = job.movie_id
    assets_collection_id = job.assets_collection_id

    if not movie_id or not assets_collection_id:
        raise HTTPException(status_code=400, detail="Job is not associated with a movie")

    assets_collection = assets_collection_service.get_assets_collection(assets_collection_id)
    if not assets_collection:
        raise HTTPException(status_code=404, detail=f"Assets collection {assets_collection_id} not found")

    # Helper function to safely extract agent output from assets collection
    def safe_get_agent_output(assets_col, agent_number, output_key, default=None):
        """Safely extract agent output from assets collection, handling None values"""
        agent_key = f"agent{agent_number}_output"
        agent_data = assets_col.get(agent_key)

        if not agent_data or not isinstance(agent_data, dict):
            return default if default is not None else {}

        output = agent_data.get("output")
        if not output or not isinstance(output, dict):
            return default if default is not None else {}

        return output.get(output_key, default if default is not None else {})

    # Create minimal state for Agent 8 retry
    # DON'T include large data (images, prompts) - workflow will load from MongoDB
    # This prevents SQS message size limit errors (256KB)
    minimal_state = {
        # Job tracking fields
        "job_id": job.job_id,
        "movie_id": movie_id,
        "assets_collection_id": assets_collection_id,
        "current_agent": "agent_8",
        "pipeline_status": "generating_variations",
        "agent8_status": "pending",
        "agent8_retry_count": 0,  # Reset retry counter for manual retry

        # Asset-level approval tracking (use all approved assets)
        "approved_asset_ids": approved_assets,
    }

    # Update job in database with retry state (so workflow can load full data from there)
    pipeline_service.update_job_state(job_id, minimal_state)

    # Resume workflow using Celery (Agent 8 variation generation)
    # Only pass job_id - workflow will load full state from MongoDB
    queue_name = get_workflow_queue_name()
    task = resume_phase1_workflow_task.apply_async(
        args=[job_id, minimal_state],
        queue=queue_name,
        routing_key=queue_name,
    )

    logger.info(f"Manually triggered Agent 8 retry with {len(approved_assets)} approved assets")
    logger.info(f"Celery Task ID: {task.id}")

    # Update job with new Celery task ID
    pipeline_service.update_job_celery_task_id(job_id, task.id)

    # Update job status to running
    pipeline_service.update_job_status(job_id, "running", pipeline_status="generating_variations")

    # Return updated job (includes new celery_task_id for monitoring)
    job = pipeline_service.get_job(job_id)
    response = pipeline_service.to_response(job)
    return response.model_dump()


@router.post("/checkpoint/edit-prompt/{job_id}", response_model=Dict[str, Any])
@limiter.limit("20/minute")
async def edit_asset_prompt_for_movie(
    request: Request,
    job_id: str,
    edit_request: AssetPromptEdit,
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Edit the prompt for a specific asset and re-run Agent 7 (Image Editor) for movie workflow

    This endpoint allows modifying the prompt for a single asset.
    The edited prompt and current image are sent to Agent 7 for re-processing.
    After Agent 7 completes, the workflow returns to the human checkpoint.
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

    # Get assets collection data to reconstruct full workflow state
    movie_id = job.movie_id
    assets_collection_id = job.assets_collection_id

    if not movie_id or not assets_collection_id:
        raise HTTPException(status_code=400, detail="Job is not associated with a movie")

    assets_collection = assets_collection_service.get_assets_collection(assets_collection_id)
    if not assets_collection:
        raise HTTPException(status_code=404, detail=f"Assets collection {assets_collection_id} not found")

    # Helper function to safely extract agent output from assets collection
    # Assets collection structure: { "agent5_output": { "output": {...}, "status": "...", "executed_at": "..." } }
    def safe_get_agent_output(assets_col, agent_number, output_key, default=None):
        """Safely extract agent output from assets collection, handling None values"""
        agent_key = f"agent{agent_number}_output"
        agent_data = assets_col.get(agent_key)

        if not agent_data or not isinstance(agent_data, dict):
            return default if default is not None else {}

        output = agent_data.get("output")
        if not output or not isinstance(output, dict):
            return default if default is not None else {}

        return output.get(output_key, default if default is not None else {})

    # Get current optimized_prompts and update the specific asset
    optimized_prompts = safe_get_agent_output(assets_collection, 4, "optimized_prompts")

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

    # Reconstruct full workflow state from assets collection data
    current_state = {
        # Job tracking fields
        "job_id": job.job_id,
        "movie_id": movie_id,
        "assets_collection_id": assets_collection_id,
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

        # Critical: Add assets collection data (all agent outputs)
        "script_content": assets_collection.get("combined_script", ""),
        "extracted_assets": safe_get_agent_output(assets_collection, 1, "extracted_assets"),
        "enhanced_assets": safe_get_agent_output(assets_collection, 2, "enhanced_assets"),
        "generated_prompts": safe_get_agent_output(assets_collection, 3, "generated_prompts"),
        "optimized_prompts": optimized_prompts,  # Use the updated prompts
        "generated_images": safe_get_agent_output(assets_collection, 5, "generated_images"),
        "image_reviews": safe_get_agent_output(assets_collection, 6, "image_reviews"),
        "edited_images": safe_get_agent_output(assets_collection, 7, "edited_images"),
        "failed_generations": safe_get_agent_output(assets_collection, 5, "failed_generations", default=[]),
    }

    # Resume workflow using Celery (regenerate with agent_5)
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
    response = pipeline_service.to_response(job)
    return response.model_dump()
