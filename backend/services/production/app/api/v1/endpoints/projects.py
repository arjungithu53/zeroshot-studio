"""Projects endpoint handlers for production service."""
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form, Depends
from backend.services.production.app.services.project_service import ProjectService
from backend.services.production.app.config import upload_file_wrapper
from backend.services.production.app.models.requests import CreateProjectRequest, CreateProjectNameRequest
from backend.services.production.app.models.responses import CreateProjectResponse
from shared.auth.dependencies import validate_admin_from_header, AdminUser

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.shared.utils.error_handlers import handle_api_exception
from backend.shared.models.responses import ApiResponse

# Initialize logger for this module
logger = get_logger(__name__)

# Import rate limiter
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(prefix="/projects", tags=["projects"])

project_service = ProjectService()

# Initialize limiter (will use the one from app.state in practice)
limiter = Limiter(key_func=get_remote_address)


@router.get("", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def list_projects(request: Request, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    List all projects.

    Returns:
        ApiResponse containing list of projects with id, name, status, createdBy, and createdOn

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        projects = project_service.list_projects()

        return ApiResponse(
            success=True,
            data={"projects": projects},
            error=None
        )

    except Exception as e:
        raise handle_api_exception(e, "list_projects")


@router.post("/create", response_model=CreateProjectResponse)
@limiter.limit("10/minute")
async def create_project(request: Request, project_request: CreateProjectRequest, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Create a new project and start the Agent 1-7 pipeline.

    This endpoint:
    1. Creates a new project document in MongoDB
    2. Initializes all agent outputs as "pending"
    3. Sets initial status to "extracting"
    4. Returns project_id for tracking

    TODO: Trigger Agent 1-7 pipeline (async/Celery task)

    Args:
        request: CreateProjectRequest containing name, script, etc.

    Returns:
        CreateProjectResponse with project_id and status

    Raises:
        HTTPException: If project creation fails
    """
    try:
        logger.info(f"Creating project: {project_request.name}")

        result = project_service.create_project(
            name=project_request.name,
            script=project_request.script,
            user_id=project_request.user_id,
            shotlist=project_request.shotlist
        )

        # TODO: Trigger async pipeline here
        # from backend.services.production.app.tasks.pipeline_tasks import run_phase1_pipeline_task
        # task = run_phase1_pipeline_task.apply_async(
        #     kwargs={"project_id": result["project_id"]}
        # )

        return CreateProjectResponse(
            success=result["success"],
            project_id=result["project_id"],
            name=result["name"],
            status=result["status"],
            message="Project created successfully. Pipeline will start shortly.",
            created_at=result["created_at"]
        )

    except Exception as e:
        raise handle_api_exception(e, "create_project")


@router.post("/create-name", response_model=CreateProjectResponse)
@limiter.limit("10/minute")
async def create_project_name(request: Request, project_request: CreateProjectNameRequest, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Create a new project with just a name.

    This endpoint:
    1. Creates a new project document in MongoDB with just the name
    2. Sets status to "draft"
    3. Returns project_id for tracking

    User can later upload script and shotlist using the upload endpoint.

    Args:
        request: CreateProjectNameRequest containing name and optional user_id

    Returns:
        CreateProjectResponse with project_id and status

    Raises:
        HTTPException: If project creation fails
    """
    try:
        logger.info(f"Creating project (name only): {project_request.name}")

        result = project_service.create_project_name_only(
            name=project_request.name,
            user_id=project_request.user_id
        )

        return CreateProjectResponse(
            success=result["success"],
            project_id=result["project_id"],
            name=result["name"],
            status=result["status"],
            message="Project created successfully. Upload script and shotlist to begin processing.",
            created_at=result["created_at"]
        )

    except Exception as e:
        raise handle_api_exception(e, "create_project_name")


@router.post("/{project_id}/upload-files", response_model=ApiResponse[dict])
@limiter.limit("10/minute")
async def upload_script_and_shotlist(
    request: Request,
    project_id: str,
    script_file: UploadFile = File(..., description="Script file (.txt)"),
    shotlist_file: UploadFile = File(None, description="Shotlist file (.txt, optional)"),
    product_image_file: Optional[UploadFile] = File(None, description="Product image (PNG/JPG/JPEG, optional)"),
    admin_user: AdminUser = Depends(validate_admin_from_header)
):
    """
    Upload script and shotlist files to an existing project.

    This endpoint:
    1. Reads the uploaded .txt files
    2. Updates the project with script and shotlist content
    3. Initializes agent outputs
    4. Sets status to "extracting"
    5. Pipeline can be triggered after this

    Args:
        project_id: MongoDB ObjectId as string
        script_file: Script .txt file
        shotlist_file: Optional shotlist .txt file

    Returns:
        ApiResponse with success status

    Raises:
        HTTPException: If upload or update fails
    """
    try:
        logger.info(f"Uploading files to project: {project_id}")

        # Validate file types
        if not script_file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Script file must be a .txt file")

        if shotlist_file and not shotlist_file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Shotlist file must be a .txt file")

        # Read script file
        script_content = await script_file.read()
        script_text = script_content.decode('utf-8')

        # Read shotlist file if provided
        shotlist_text = None
        if shotlist_file:
            shotlist_content = await shotlist_file.read()
            shotlist_text = shotlist_content.decode('utf-8')

        # Update project with script and shotlist
        result = project_service.add_script_and_shotlist(
            project_id=project_id,
            script=script_text,
            shotlist=shotlist_text
        )

        # Upload product image to S3 if provided
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
                s3_key = f"projects/{project_id}/product_image{ext}"
                product_image_url = upload_file_wrapper(
                    tmp_path,
                    s3_key=s3_key,
                    content_type=product_image_file.content_type or "image/png",
                    use_presigned_url=False,
                )
                project_service.update_product_image_url(project_id, product_image_url)
                logger.info(f"Product image uploaded for project {project_id}: {product_image_url}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # TODO: Trigger async pipeline here
        # from backend.services.production.app.tasks.pipeline_tasks import run_phase1_pipeline_task
        # task = run_phase1_pipeline_task.apply_async(
        #     kwargs={"project_id": project_id}
        # )

        return ApiResponse(
            success=True,
            data={
                "project_id": result["project_id"],
                "message": result["message"]
            },
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "upload_script_and_shotlist")


@router.get("/{project_id}", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_project(request: Request, project_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get project details by ID.

    Args:
        project_id: MongoDB ObjectId as string

    Returns:
        Project document with all details wrapped in ApiResponse

    Raises:
        HTTPException: If project not found or retrieval fails
    """
    try:
        project = project_service.get_project(project_id)

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        return ApiResponse(
            success=True,
            data=project,
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "get_project")


@router.get("/{project_id}/status", response_model=ApiResponse[dict])
@limiter.limit("100/minute")
async def get_project_status(request: Request, project_id: str, admin_user: AdminUser = Depends(validate_admin_from_header)):
    """
    Get project status and current pipeline progress.

    Args:
        project_id: MongoDB ObjectId as string

    Returns:
        Status information including current agent and progress wrapped in ApiResponse

    Raises:
        HTTPException: If project not found
    """
    try:
        project = project_service.get_project(project_id)

        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Calculate progress based on completed agents
        agent_outputs = project.get("agent_outputs", {})
        total_agents = 7
        completed_agents = sum(
            1 for agent in agent_outputs.values()
            if agent and agent.get("status") == "completed"
        )

        # Find current agent
        current_agent = None
        for i in range(1, 8):
            agent_key = f"agent{i}"
            agent = agent_outputs.get(agent_key, {})
            if agent.get("status") in ["pending", "running"]:
                current_agent = i
                break

        return ApiResponse(
            success=True,
            data={
                "project_id": project_id,
                "status": project.get("status"),
                "current_agent": current_agent,
                "progress": {
                    "completed": completed_agents,
                    "total": total_agents,
                    "percentage": round((completed_agents / total_agents) * 100, 2)
                },
                "agent_outputs": agent_outputs
            },
            error=None
        )

    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, "get_project_status")
