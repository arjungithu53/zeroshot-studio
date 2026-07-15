"""Movie service for production - handles movie CRUD operations."""
import sys
from datetime import datetime
from bson import ObjectId
from typing import Dict, Any, Optional, List
from pathlib import Path
from backend.services.production.app.config import get_database
from backend.shared.utils.mongodb_validators import validate_object_id

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


class MovieService:
    """Service class for movie operations."""

    def __init__(self) -> None:
        """Initialize the movie service."""
        pass

    def create_movie(
        self,
        title: str,
        scenes: List[Dict[str, Any]],
        description: Optional[str] = None,
        genre: Optional[str] = None,
        user_id: Optional[str] = None,
        global_settings: Optional[Dict[str, Any]] = None,
        v1_project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new movie in MongoDB.

        Args:
            title: Movie title
            scenes: List of scene dictionaries (scene_number, scene_name, script, shotlist)
            description: Optional movie description
            genre: Optional movie genre
            user_id: Optional user ID
            global_settings: Optional global movie settings

        Returns:
            Dict containing movie details including movie_id

        Raises:
            Exception: If movie creation fails
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            # Create movie document
            movie_doc = {
                "title": title,
                "scenes": scenes,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "phase1_status": "pending",
                "overall_status": "created",
                "aggregated_data": {
                    "total_scenes": len(scenes),
                    "total_characters": 0,
                    "total_locations": 0,
                    "total_props": 0,
                    "completed_scenes": 0
                }
            }

            # Add optional fields
            if description:
                movie_doc["description"] = description
            if genre:
                movie_doc["genre"] = genre
            if user_id:
                movie_doc["user_id"] = user_id
            if global_settings:
                movie_doc["global_settings"] = global_settings
                logger.info(f"Adding global_settings to movie: {global_settings}")
            if v1_project_id:
                movie_doc["v1_project_id"] = v1_project_id
                logger.info(f"Linking movie to v1 project: {v1_project_id}")

            # Insert into MongoDB
            result = movies_col.insert_one(movie_doc)
            movie_id = str(result.inserted_id)

            logger.info(f"Movie created: {movie_id} - {title} with {len(scenes)} scenes")
            if global_settings:
                logger.info(f"  → visual_style: {global_settings.get('visual_style')}")

            return {
                "success": True,
                "movie_id": movie_id,
                "title": title,
                "total_scenes": len(scenes),
                "created_at": movie_doc["created_at"].isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Failed to create movie: {e}")
            raise Exception(f"Failed to create movie: {str(e)}")

    def get_movie(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a movie by ID.

        Args:
            movie_id: MongoDB ObjectId as string

        Returns:
            Movie document or None if not found

        Raises:
            ValueError: If movie_id format is invalid
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            movie_obj_id = validate_object_id(movie_id)
            movie = movies_col.find_one({"_id": movie_obj_id})

            if movie:
                movie["_id"] = str(movie["_id"])
                # Convert ObjectIds to strings
                if movie.get("assets_collection_id"):
                    movie["assets_collection_id"] = str(movie["assets_collection_id"])
                if movie.get("project_ids"):
                    movie["project_ids"] = [str(pid) for pid in movie["project_ids"]]

                # Convert ObjectIds in scenes
                for scene in movie.get("scenes", []):
                    if scene.get("project_id"):
                        scene["project_id"] = str(scene["project_id"])

            return movie

        except ValueError as e:
            logger.error(f"Invalid movie ID format: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get movie {movie_id}: {e}")
            return None

    def get_movie_by_id(self, movie_id: str) -> Optional[Dict[str, Any]]:
        """
        Backward-compatible alias used by downstream workflows.

        LangGraph Phase 1 calls this method name; keep it thin to avoid diverging
        behavior from the primary getter.
        """
        return self.get_movie(movie_id)

    def update_movie(
        self,
        movie_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update movie document.

        Args:
            movie_id: MongoDB ObjectId as string
            update_data: Dictionary of fields to update

        Returns:
            Dict containing update status

        Raises:
            Exception: If update fails
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            movie_obj_id = validate_object_id(movie_id)

            # Add updated_at timestamp
            update_data["updated_at"] = datetime.utcnow()

            result = movies_col.update_one(
                {"_id": movie_obj_id},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                raise Exception("Movie not found")

            logger.info(f"Movie updated: {movie_id}")

            return {
                "success": True,
                "movie_id": movie_id,
                "modified_count": result.modified_count
            }

        except Exception as e:
            logger.error(f"Failed to update movie {movie_id}: {e}")
            raise Exception(f"Failed to update movie: {str(e)}")

    def set_assets_collection_id(
        self,
        movie_id: str,
        assets_collection_id: str
    ) -> Dict[str, Any]:
        """
        Set the assets_collection_id for a movie.

        Args:
            movie_id: MongoDB ObjectId as string
            assets_collection_id: Assets collection ObjectId as string

        Returns:
            Dict containing update status
        """
        assets_obj_id = validate_object_id(assets_collection_id)

        return self.update_movie(
            movie_id,
            {"assets_collection_id": assets_obj_id}
        )

    def add_project_id(
        self,
        movie_id: str,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Add a project_id to the movie's project_ids array.

        Args:
            movie_id: MongoDB ObjectId as string
            project_id: Project ObjectId as string to add

        Returns:
            Dict containing update status
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            movie_obj_id = validate_object_id(movie_id)
            project_obj_id = validate_object_id(project_id)

            result = movies_col.update_one(
                {"_id": movie_obj_id},
                {
                    "$addToSet": {"project_ids": project_obj_id},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )

            if result.matched_count == 0:
                raise Exception("Movie not found")

            logger.info(f"Project {project_id} added to movie {movie_id}")

            return {
                "success": True,
                "movie_id": movie_id,
                "project_id": project_id
            }

        except Exception as e:
            logger.error(f"Failed to add project to movie: {e}")
            raise Exception(f"Failed to add project to movie: {str(e)}")

    def update_scene_project_id(
        self,
        movie_id: str,
        scene_number: int,
        project_id: str
    ) -> Dict[str, Any]:
        """
        Update the project_id for a specific scene in the movie.

        Args:
            movie_id: MongoDB ObjectId as string
            scene_number: Scene number to update
            project_id: Project ObjectId as string

        Returns:
            Dict containing update status
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            movie_obj_id = validate_object_id(movie_id)
            project_obj_id = validate_object_id(project_id)

            result = movies_col.update_one(
                {
                    "_id": movie_obj_id,
                    "scenes.scene_number": scene_number
                },
                {
                    "$set": {
                        "scenes.$.project_id": project_obj_id,
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            if result.matched_count == 0:
                raise Exception("Movie or scene not found")

            logger.info(f"Scene {scene_number} in movie {movie_id} updated with project {project_id}")

            return {
                "success": True,
                "movie_id": movie_id,
                "scene_number": scene_number,
                "project_id": project_id
            }

        except Exception as e:
            logger.error(f"Failed to update scene project_id: {e}")
            raise Exception(f"Failed to update scene project_id: {str(e)}")

    def update_phase1_status(
        self,
        movie_id: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Update the Phase 1 status for a movie.

        Args:
            movie_id: MongoDB ObjectId as string
            status: New status (pending, running, completed, failed)

        Returns:
            Dict containing update status
        """
        return self.update_movie(
            movie_id,
            {"phase1_status": status}
        )

    def update_overall_status(
        self,
        movie_id: str,
        status: str
    ) -> Dict[str, Any]:
        """
        Update the overall status for a movie.

        Args:
            movie_id: MongoDB ObjectId as string
            status: New status (created, assets_generated, in_production, completed, failed)

        Returns:
            Dict containing update status
        """
        return self.update_movie(
            movie_id,
            {"overall_status": status}
        )

    def update_aggregated_data(
        self,
        movie_id: str,
        aggregated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update aggregated data for a movie.

        Args:
            movie_id: MongoDB ObjectId as string
            aggregated_data: Dictionary with aggregated stats

        Returns:
            Dict containing update status
        """
        return self.update_movie(
            movie_id,
            {"aggregated_data": aggregated_data}
        )

    def list_movies(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List movies with optional filtering.

        Args:
            user_id: Optional user ID to filter by
            limit: Maximum number of movies to return
            offset: Number of movies to skip

        Returns:
            List of movie documents
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            query = {}
            if user_id:
                query["user_id"] = user_id

            movies = list(
                movies_col.find(query)
                .sort("created_at", -1)
                .skip(offset)
                .limit(limit)
            )

            # Convert ObjectIds to strings
            for movie in movies:
                movie["_id"] = str(movie["_id"])
                if movie.get("assets_collection_id"):
                    movie["assets_collection_id"] = str(movie["assets_collection_id"])
                if movie.get("project_ids"):
                    movie["project_ids"] = [str(pid) for pid in movie["project_ids"]]

                # Convert ObjectIds in scenes
                for scene in movie.get("scenes", []):
                    if scene.get("project_id"):
                        scene["project_id"] = str(scene["project_id"])

            return movies

        except Exception as e:
            logger.error(f"Failed to list movies: {e}")
            return []

    def delete_movie(self, movie_id: str) -> Dict[str, Any]:
        """
        Delete a movie by ID.

        Args:
            movie_id: MongoDB ObjectId as string

        Returns:
            Dict containing deletion status

        Raises:
            Exception: If deletion fails
        """
        client, db = get_database()
        movies_col = db.movies

        try:
            movie_obj_id = validate_object_id(movie_id)

            result = movies_col.delete_one({"_id": movie_obj_id})

            if result.deleted_count == 0:
                raise Exception("Movie not found")

            logger.info(f"Movie deleted: {movie_id}")

            return {
                "success": True,
                "movie_id": movie_id,
                "deleted_count": result.deleted_count
            }

        except Exception as e:
            logger.error(f"Failed to delete movie {movie_id}: {e}")
            raise Exception(f"Failed to delete movie: {str(e)}")
