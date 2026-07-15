"""Project service for production - handles project CRUD operations."""
import sys
from datetime import datetime
from bson import ObjectId
from typing import Dict, Any, Optional
from pathlib import Path
from backend.services.production.app.config import get_projects_collection
from backend.shared.utils.mongodb_validators import validate_object_id

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


class ProjectService:
    """Service class for project operations."""

    def __init__(self) -> None:
        """Initialize the project service."""
        pass

    def create_project_name_only(
        self,
        name: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new project with just a name in MongoDB.

        Args:
            name: Project name
            user_id: Optional user ID

        Returns:
            Dict containing project details including project_id

        Raises:
            Exception: If project creation fails
        """
        client, projects_col = get_projects_collection()

        try:
            # Create project document with minimal fields
            project_doc = {
                "name": name,
                "status": "draft",  # Draft status until script is added
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Add optional fields
            if user_id:
                project_doc["user_id"] = user_id

            # Insert into MongoDB
            result = projects_col.insert_one(project_doc)
            project_id = str(result.inserted_id)

            logger.info(f"Project created (name only): {project_id} - {name}")

            return {
                "success": True,
                "project_id": project_id,
                "name": name,
                "status": "draft",
                "created_at": project_doc["created_at"].isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise Exception(f"Failed to create project: {str(e)}")

    def add_script_and_shotlist(
        self,
        project_id: str,
        script: str,
        shotlist: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add script and shotlist to an existing project and initialize agent pipeline.

        Args:
            project_id: MongoDB ObjectId as string
            script: Full script text content
            shotlist: Optional shotlist/scene breakdown

        Returns:
            Dict containing update status

        Raises:
            Exception: If update fails
        """
        client, projects_col = get_projects_collection()

        try:
            try:
                project_obj_id = validate_object_id(project_id)
            except ValueError as e:
                logger.error(f"Invalid project ID format: {e}")
                raise ValueError(f"Invalid project ID: {str(e)}") from e

            # Initialize all agent outputs as null/pending
            # Phase 1-2: Agents 1-8
            agent_outputs = {
                f"agent{i}": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None
                }
                for i in range(1, 9)  # Agents 1-8
            }

            # Phase 3: Agents 17-19 (video generation pipeline)
            agent_outputs.update({
                "agent17": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video prompt generation"
                },
                "agent18": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video generation"
                },
                "agent19": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "AI video review"
                }
            })

            # Update project with script, shotlist, and agent outputs
            update_data = {
                "script": script,
                "agent_outputs": agent_outputs,
                "status": "extracting",  # Change from draft to extracting
                "updated_at": datetime.utcnow()
            }

            if shotlist:
                update_data["shotlist"] = shotlist

            result = projects_col.update_one(
                {"_id": project_obj_id},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                raise Exception("Project not found")

            logger.info(f"Script and shotlist added to project: {project_id}")

            return {
                "success": True,
                "project_id": project_id,
                "message": "Script and shotlist added successfully. Pipeline will start shortly."
            }

        except Exception as e:
            logger.error(f"Failed to add script and shotlist: {e}")
            raise Exception(f"Failed to add script and shotlist: {str(e)}")

    def create_project(
        self,
        name: str,
        script: str,
        user_id: Optional[str] = None,
        shotlist: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new project in MongoDB.

        Args:
            name: Project name
            script: Full script text
            user_id: Optional user ID
            shotlist: Optional shotlist/scene breakdown

        Returns:
            Dict containing project details including project_id

        Raises:
            Exception: If project creation fails
        """
        client, projects_col = get_projects_collection()

        try:
            # Initialize all agent outputs as null/pending
            # Phase 1-2: Agents 1-8
            agent_outputs = {
                f"agent{i}": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None
                }
                for i in range(1, 9)  # Agents 1-8
            }

            # Phase 3: Agents 17-19 (video generation pipeline)
            agent_outputs.update({
                "agent17": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video prompt generation"
                },
                "agent18": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video generation"
                },
                "agent19": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "AI video review"
                }
            })

            # Create project document
            project_doc = {
                "name": name,
                "script": script,
                "status": "extracting",  # Initial status
                "agent_outputs": agent_outputs,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Add optional fields
            if user_id:
                project_doc["user_id"] = user_id
            if shotlist:
                project_doc["shotlist"] = shotlist

            # Insert into MongoDB
            result = projects_col.insert_one(project_doc)
            project_id = str(result.inserted_id)

            logger.info(f"Project created: {project_id} - {name}")

            return {
                "success": True,
                "project_id": project_id,
                "name": name,
                "status": "extracting",
                "created_at": project_doc["created_at"].isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise Exception(f"Failed to create project: {str(e)}")

    def create_scene_project(
        self,
        movie_id: str,
        assets_collection_id: str,
        scene_number: int,
        scene_name: str,
        script: str,
        shotlist: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new project for a scene in a movie (no Phase 1 agents).

        Args:
            movie_id: MongoDB ObjectId as string
            assets_collection_id: MongoDB ObjectId as string
            scene_number: Scene number in the movie
            scene_name: Scene name/title
            script: Scene script text
            shotlist: Optional scene shotlist
            user_id: Optional user ID

        Returns:
            Dict containing project details including project_id

        Raises:
            Exception: If project creation fails
        """
        client, projects_col = get_projects_collection()

        try:
            movie_obj_id = validate_object_id(movie_id)
            assets_obj_id = validate_object_id(assets_collection_id)

            # Initialize only Phase 2 and Phase 3 agent outputs (no Phase 1)
            agent_outputs = {}

            # Phase 2: Agents 12-15 (shot generation)
            for i in range(12, 16):
                agent_outputs[f"agent{i}"] = {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None
                }

            # Phase 3: Agents 17-19 (video generation)
            agent_outputs.update({
                "agent17": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video prompt generation"
                },
                "agent18": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "Video generation"
                },
                "agent19": {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None,
                    "description": "AI video review"
                }
            })

            # Create project document for the scene
            project_doc = {
                "name": scene_name,
                "script": script,
                "scene_number": scene_number,
                "movie_id": movie_obj_id,
                "assets_collection_id": assets_obj_id,
                "status": "pending",  # Pending until Phase 2 starts
                "agent_outputs": agent_outputs,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Add optional fields
            if user_id:
                project_doc["user_id"] = user_id
            if shotlist:
                project_doc["shotlist"] = shotlist

            # Insert into MongoDB
            result = projects_col.insert_one(project_doc)
            project_id = str(result.inserted_id)

            logger.info(f"Scene project created: {project_id} - Scene {scene_number}: {scene_name} (Movie: {movie_id})")

            return {
                "success": True,
                "project_id": project_id,
                "scene_number": scene_number,
                "name": scene_name,
                "movie_id": movie_id,
                "assets_collection_id": assets_collection_id,
                "status": "pending",
                "created_at": project_doc["created_at"].isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Failed to create scene project: {e}")
            raise Exception(f"Failed to create scene project: {str(e)}")

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get project by ID.

        Args:
            project_id: MongoDB ObjectId as string (may have suffix like _1 or _auto_image_agents)

        Returns:
            Project document or None if not found
        """
        client, projects_col = get_projects_collection()

        try:
            # Strip any suffix from project_id (e.g., "68ff1b7f_1" -> "68ff1b7f")
            clean_project_id = project_id.split('_')[0] if '_' in project_id else project_id

            logger.debug(f"Looking up project with ID: {project_id} (clean: {clean_project_id})")

            try:
                project_obj_id = validate_object_id(clean_project_id)
                project = projects_col.find_one({"_id": project_obj_id})
            except ValueError as e:
                logger.error(f"Invalid project ID format: {e}")
                return None

            if project:
                # Convert ObjectId to string for JSON serialization
                project["_id"] = str(project["_id"])

                # Convert other ObjectId fields if they exist
                if project.get("movie_id"):
                    project["movie_id"] = str(project["movie_id"])
                if project.get("assets_collection_id"):
                    project["assets_collection_id"] = str(project["assets_collection_id"])

                logger.debug(f"Found project: {project.get('name', 'Unknown')}")
                return project
            logger.warning(f"Project not found: {project_id}")
            return None

        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            import traceback
            traceback.print_exc()
            return None  # Return None instead of raising, so caller can handle it

    def update_project_status(
        self,
        project_id: str,
        status: str
    ) -> bool:
        """
        Update project status.

        Args:
            project_id: MongoDB ObjectId as string
            status: New status

        Returns:
            True if successful
        """
        client, projects_col = get_projects_collection()

        try:
            try:
                project_obj_id = validate_object_id(project_id)
            except ValueError as e:
                logger.error(f"Invalid project ID format: {e}")
                raise ValueError(f"Invalid project ID: {str(e)}") from e

            result = projects_col.update_one(
                {"_id": project_obj_id},
                {
                    "$set": {
                        "status": status,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Failed to update project status: {e}")
            raise Exception(f"Failed to update project status: {str(e)}")

    def update_s3_urls(
        self,
        project_id: str,
        script_s3_url: Optional[str] = None,
        shotlist_s3_url: Optional[str] = None
    ) -> bool:
        """
        Update project with S3 URLs for scene script and shotlist JSON.

        Args:
            project_id: MongoDB ObjectId as string
            script_s3_url: S3 URL for scene script text file
            shotlist_s3_url: S3 URL for shotlist JSON file

        Returns:
            True if successful

        Raises:
            Exception: If update fails
        """
        client, projects_col = get_projects_collection()

        try:
            try:
                project_obj_id = validate_object_id(project_id)
            except ValueError as e:
                logger.error(f"Invalid project ID format: {e}")
                raise ValueError(f"Invalid project ID: {str(e)}") from e

            update_data = {"updated_at": datetime.utcnow()}

            if script_s3_url:
                update_data["scene_script_s3_url"] = script_s3_url

            if shotlist_s3_url:
                update_data["shotlist_json_s3_url"] = shotlist_s3_url

            result = projects_col.update_one(
                {"_id": project_obj_id},
                {"$set": update_data}
            )

            if result.modified_count > 0:
                logger.info(f"Updated S3 URLs for project {project_id}")
                return True
            else:
                logger.warning(f"No changes made to project {project_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to update S3 URLs for project: {e}")
            raise Exception(f"Failed to update S3 URLs: {str(e)}")

    def update_product_image_url(self, project_id: str, s3_url: str) -> bool:
        """
        Store the uploaded product image S3 URL on the project document.

        Args:
            project_id: MongoDB ObjectId as string
            s3_url: S3 URL of the uploaded product image

        Returns:
            True if successful
        """
        client, projects_col = get_projects_collection()

        try:
            project_obj_id = validate_object_id(project_id)
            result = projects_col.update_one(
                {"_id": project_obj_id},
                {"$set": {"product_image_s3_url": s3_url, "updated_at": datetime.utcnow()}}
            )
            if result.modified_count > 0:
                logger.info(f"Product image URL stored for project {project_id}")
                return True
            else:
                logger.warning(f"No changes made when storing product image for project {project_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to store product image URL for project: {e}")
            raise Exception(f"Failed to store product image URL: {str(e)}")

    def update_agent_output(
        self,
        project_id: str,
        agent_number: int,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        append_output: bool = False
    ) -> bool:
        """
        Update agent output for a project.

        Args:
            project_id: MongoDB ObjectId as string
            agent_number: Agent number (1-8 for Phase 1, 12-13 for Phase 2, 17-19 for Phase 3)
            status: Agent status (pending/running/completed/failed)
            output: Agent output data (optional)
            error: Error message if failed (optional)
            append_output: If True, append output to outputs array (for Phase 3 multi-shot agents)
                          If False, replace output (default for Phase 1/2 single-run agents)

        Returns:
            True if successful
        """
        client, projects_col = get_projects_collection()

        try:
            try:
                project_obj_id = validate_object_id(project_id)
            except ValueError as e:
                logger.error(f"Invalid project ID format: {e}")
                raise ValueError(f"Invalid project ID: {str(e)}") from e

            agent_key = f"agent{agent_number}"

            # For Phase 3 agents (17, 18, 19) that process multiple shots,
            # append to outputs array instead of replacing
            if append_output and output is not None:
                # Use $push to append to outputs array
                result = projects_col.update_one(
                    {"_id": project_obj_id},
                    {
                        "$set": {
                            f"agent_outputs.{agent_key}.status": status,
                            f"agent_outputs.{agent_key}.executed_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        },
                        "$push": {
                            f"agent_outputs.{agent_key}.outputs": output
                        }
                    }
                )
            else:
                # Standard behavior: replace output (Phase 1/2 agents)
                update_data = {
                    f"agent_outputs.{agent_key}.status": status,
                    f"agent_outputs.{agent_key}.executed_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }

                if output is not None:
                    update_data[f"agent_outputs.{agent_key}.output"] = output

                if error is not None:
                    update_data[f"agent_outputs.{agent_key}.error"] = error

                result = projects_col.update_one(
                    {"_id": project_obj_id},
                    {"$set": update_data}
                )

            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Failed to update agent output: {e}")
            raise Exception(f"Failed to update agent output: {str(e)}")

    def list_projects(self) -> list[Dict[str, Any]]:
        """
        List all projects with basic information.

        Returns:
            List of project dictionaries with id, name, status, createdBy, and createdOn
        """
        client, projects_col = get_projects_collection()

        try:
            # Query all projects, sorted by creation date (newest first)
            projects_cursor = projects_col.find(
                {},
                {
                    "_id": 1,
                    "name": 1,
                    "status": 1,
                    "user_id": 1,
                    "created_at": 1
                }
            ).sort("created_at", -1)

            projects_list = []
            for project in projects_cursor:
                projects_list.append({
                    "id": str(project["_id"]),
                    "name": project.get("name", "Untitled"),
                    "status": project.get("status", "unknown"),
                    "createdBy": project.get("user_id", "Unknown"),
                    "createdOn": project.get("created_at").isoformat() + "Z" if project.get("created_at") else None
                })

            logger.info(f"Retrieved {len(projects_list)} projects")
            return projects_list

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise Exception(f"Failed to list projects: {str(e)}")
