"""Assets Collection service for production - handles movie-level Phase 1 asset operations."""
import sys
import os
from datetime import datetime
from bson import ObjectId
from typing import Dict, Any, Optional
from pathlib import Path
from backend.services.production.app.config import get_database
from backend.shared.utils.mongodb_validators import validate_object_id
import boto3
from botocore.exceptions import ClientError

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


class AssetsCollectionService:
    """Service class for assets collection operations."""

    def __init__(self) -> None:
        """Initialize the assets collection service."""
        self.s3_client = None
        self.s3_bucket = None
        self._initialize_s3()

    def _initialize_s3(self) -> None:
        """Initialize S3 client for generating presigned URLs."""
        try:
            access_key = os.getenv("production_AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("production_AWS_SECRET_ACCESS_KEY")
            region = os.getenv("production_AWS_REGION", "eu-north-1")
            bucket = os.getenv("production_S3_BUCKET_NAME")

            if all([access_key, secret_key, bucket]):
                from botocore.config import Config

                boto_config = Config(
                    signature_version='s3v4',
                    region_name=region
                )

                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region,
                    config=boto_config
                )
                self.s3_bucket = bucket
                logger.info(f"S3 client initialized for assets collection service (bucket: {bucket})")
            else:
                logger.warning("S3 credentials not fully configured, presigned URLs will not be generated")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    def _generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for an S3 object.

        Args:
            s3_key: S3 key of the object
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL or None if generation fails
        """
        if not self.s3_client or not self.s3_bucket:
            return None

        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.warning(f"Failed to generate presigned URL for {s3_key}: {e}")
            return None

    def _enrich_with_presigned_urls(self, data: Any, expiration: int = 3600) -> Any:
        """
        Recursively enrich data structure with fresh presigned URLs.

        Args:
            data: Data structure (dict, list, or primitive)
            expiration: URL expiration time in seconds

        Returns:
            Enriched data with presigned URLs
        """
        if isinstance(data, dict):
            enriched = {}
            for key, value in data.items():
                # Generate presigned URL if s3_key is present
                if key == 's3_key' and value and isinstance(value, str):
                    presigned_url = self._generate_presigned_url(value, expiration)
                    if presigned_url:
                        # Update the s3_url in the parent dict
                        enriched[key] = value
                        enriched['presigned_url'] = presigned_url
                    else:
                        enriched[key] = value
                else:
                    enriched[key] = self._enrich_with_presigned_urls(value, expiration)
            return enriched
        elif isinstance(data, list):
            return [self._enrich_with_presigned_urls(item, expiration) for item in data]
        else:
            return data

    def create_assets_collection(
        self,
        movie_id: str
    ) -> Dict[str, Any]:
        """
        Create a new assets collection for a movie.

        Args:
            movie_id: MongoDB ObjectId as string

        Returns:
            Dict containing assets_collection details including assets_collection_id

        Raises:
            Exception: If creation fails
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            movie_obj_id = validate_object_id(movie_id)

            # Create assets collection document
            assets_doc = {
                "movie_id": movie_obj_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Initialize all Phase 1 agent outputs (agents 1-8) as pending
            for i in range(1, 9):
                assets_doc[f"agent{i}_output"] = {
                    "status": "pending",
                    "output": None,
                    "error": None,
                    "executed_at": None
                }

            # Insert into MongoDB
            result = assets_col.insert_one(assets_doc)
            assets_collection_id = str(result.inserted_id)

            logger.info(f"Assets collection created: {assets_collection_id} for movie {movie_id}")

            return {
                "success": True,
                "assets_collection_id": assets_collection_id,
                "movie_id": movie_id,
                "created_at": assets_doc["created_at"].isoformat() + "Z"
            }

        except Exception as e:
            logger.error(f"Failed to create assets collection: {e}")
            raise Exception(f"Failed to create assets collection: {str(e)}")

    def get_assets_collection(
        self,
        assets_collection_id: str,
        include_presigned_urls: bool = True,
        url_expiration: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        Get an assets collection by ID.

        Args:
            assets_collection_id: MongoDB ObjectId as string
            include_presigned_urls: Whether to generate fresh presigned URLs (default: True)
            url_expiration: Presigned URL expiration in seconds (default: 1 hour)

        Returns:
            Assets collection document or None if not found
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            assets_obj_id = validate_object_id(assets_collection_id)
            assets = assets_col.find_one({"_id": assets_obj_id})

            if assets:
                assets["_id"] = str(assets["_id"])
                if assets.get("movie_id"):
                    assets["movie_id"] = str(assets["movie_id"])

                # Enrich with presigned URLs if requested
                if include_presigned_urls:
                    assets = self._enrich_with_presigned_urls(assets, url_expiration)

            return assets

        except Exception as e:
            logger.error(f"Failed to get assets collection {assets_collection_id}: {e}")
            return None

    def get_assets_collection_by_movie_id(
        self,
        movie_id: str,
        include_presigned_urls: bool = True,
        url_expiration: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        Get an assets collection by movie ID.

        Args:
            movie_id: MongoDB ObjectId as string
            include_presigned_urls: Whether to generate fresh presigned URLs (default: True)
            url_expiration: Presigned URL expiration in seconds (default: 1 hour)

        Returns:
            Assets collection document or None if not found
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            movie_obj_id = validate_object_id(movie_id)
            assets = assets_col.find_one({"movie_id": movie_obj_id})

            if assets:
                assets["_id"] = str(assets["_id"])
                if assets.get("movie_id"):
                    assets["movie_id"] = str(assets["movie_id"])

                # Enrich with presigned URLs if requested
                if include_presigned_urls:
                    assets = self._enrich_with_presigned_urls(assets, url_expiration)

            return assets

        except Exception as e:
            logger.error(f"Failed to get assets collection for movie {movie_id}: {e}")
            return None

    def update_agent_output(
        self,
        assets_collection_id: str,
        agent_number: int,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update a specific agent's output in the assets collection.

        Args:
            assets_collection_id: MongoDB ObjectId as string
            agent_number: Agent number (1-8)
            status: Agent status (pending, running, completed, failed)
            output: Optional agent output data
            error: Optional error message

        Returns:
            Dict containing update status

        Raises:
            Exception: If update fails
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            # Validate agent number
            if agent_number < 1 or agent_number > 8:
                raise ValueError(f"Invalid agent number: {agent_number}. Must be between 1 and 8.")

            assets_obj_id = validate_object_id(assets_collection_id)
            agent_key = f"agent{agent_number}_output"

            # Build update data
            update_data = {
                f"{agent_key}.status": status,
                f"{agent_key}.executed_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            if output is not None:
                update_data[f"{agent_key}.output"] = output

            if error is not None:
                update_data[f"{agent_key}.error"] = error
            else:
                # Clear error if status is not failed
                if status != "failed":
                    update_data[f"{agent_key}.error"] = None

            result = assets_col.update_one(
                {"_id": assets_obj_id},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                raise Exception("Assets collection not found")

            logger.info(f"Agent {agent_number} output updated in assets collection {assets_collection_id}: {status}")

            return {
                "success": True,
                "assets_collection_id": assets_collection_id,
                "agent_number": agent_number,
                "status": status
            }

        except Exception as e:
            logger.error(f"Failed to update agent output in assets collection: {e}")
            raise Exception(f"Failed to update agent output: {str(e)}")

    def get_agent_output(
        self,
        assets_collection_id: str,
        agent_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific agent's output from the assets collection.

        Args:
            assets_collection_id: MongoDB ObjectId as string
            agent_number: Agent number (1-8)

        Returns:
            Agent output data or None if not found
        """
        try:
            # Validate agent number
            if agent_number < 1 or agent_number > 8:
                raise ValueError(f"Invalid agent number: {agent_number}. Must be between 1 and 8.")

            assets = self.get_assets_collection(assets_collection_id)

            if not assets:
                return None

            agent_key = f"agent{agent_number}_output"
            return assets.get(agent_key)

        except Exception as e:
            logger.error(f"Failed to get agent output: {e}")
            return None

    def get_all_agent_outputs(
        self,
        assets_collection_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get all agent outputs from the assets collection.

        Args:
            assets_collection_id: MongoDB ObjectId as string

        Returns:
            Dictionary of all agent outputs (agent1_output through agent8_output)
        """
        try:
            assets = self.get_assets_collection(assets_collection_id)

            if not assets:
                return None

            agent_outputs = {}
            for i in range(1, 9):
                agent_key = f"agent{i}_output"
                if agent_key in assets:
                    agent_outputs[agent_key] = assets[agent_key]

            return agent_outputs

        except Exception as e:
            logger.error(f"Failed to get all agent outputs: {e}")
            return None

    def update_assets_collection(
        self,
        assets_collection_id: str,
        update_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update assets collection document.

        Args:
            assets_collection_id: MongoDB ObjectId as string
            update_data: Dictionary of fields to update

        Returns:
            Dict containing update status

        Raises:
            Exception: If update fails
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            assets_obj_id = validate_object_id(assets_collection_id)

            # Add updated_at timestamp
            update_data["updated_at"] = datetime.utcnow()

            result = assets_col.update_one(
                {"_id": assets_obj_id},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                raise Exception("Assets collection not found")

            logger.info(f"Assets collection updated: {assets_collection_id}")

            return {
                "success": True,
                "assets_collection_id": assets_collection_id,
                "modified_count": result.modified_count
            }

        except Exception as e:
            logger.error(f"Failed to update assets collection {assets_collection_id}: {e}")
            raise Exception(f"Failed to update assets collection: {str(e)}")

    def update_approval_status(
        self,
        assets_collection_id: str,
        approved_assets_list: list,
        checkpoint_approved: bool = False,
        human_approval_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update approval status in the assets collection.

        Args:
            assets_collection_id: MongoDB ObjectId as string
            approved_assets_list: List of approved asset IDs
            checkpoint_approved: Whether the checkpoint has been finalized
            human_approval_feedback: Optional feedback data

        Returns:
            Dict containing update status

        Raises:
            Exception: If update fails
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            assets_obj_id = validate_object_id(assets_collection_id)

            # Build update data
            update_data = {
                "approved_assets_list": approved_assets_list,
                "checkpoint_approved": checkpoint_approved,
                "approval_timestamp": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            if human_approval_feedback is not None:
                update_data["human_approval_feedback"] = human_approval_feedback

            result = assets_col.update_one(
                {"_id": assets_obj_id},
                {"$set": update_data}
            )

            if result.matched_count == 0:
                raise Exception("Assets collection not found")

            logger.info(f"Approval status updated in assets collection {assets_collection_id}: {len(approved_assets_list)} assets approved, checkpoint_approved={checkpoint_approved}")

            return {
                "success": True,
                "assets_collection_id": assets_collection_id,
                "approved_count": len(approved_assets_list),
                "checkpoint_approved": checkpoint_approved
            }

        except Exception as e:
            logger.error(f"Failed to update approval status in assets collection {assets_collection_id}: {e}")
            raise Exception(f"Failed to update approval status: {str(e)}")

    def delete_assets_collection(
        self,
        assets_collection_id: str
    ) -> Dict[str, Any]:
        """
        Delete an assets collection by ID.

        Args:
            assets_collection_id: MongoDB ObjectId as string

        Returns:
            Dict containing deletion status

        Raises:
            Exception: If deletion fails
        """
        client, db = get_database()
        assets_col = db.assets_collections

        try:
            assets_obj_id = validate_object_id(assets_collection_id)

            result = assets_col.delete_one({"_id": assets_obj_id})

            if result.deleted_count == 0:
                raise Exception("Assets collection not found")

            logger.info(f"Assets collection deleted: {assets_collection_id}")

            return {
                "success": True,
                "assets_collection_id": assets_collection_id,
                "deleted_count": result.deleted_count
            }

        except Exception as e:
            logger.error(f"Failed to delete assets collection {assets_collection_id}: {e}")
            raise Exception(f"Failed to delete assets collection: {str(e)}")
