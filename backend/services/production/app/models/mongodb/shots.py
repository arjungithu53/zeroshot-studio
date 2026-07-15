"""
Shots document schema and MongoDB utilities for Phase 2 agents (production Service).

Provides MongoDB validation schemas and connection management for shot data operations.
"""

import os
from typing import List, Dict, Any, Optional, Union, Literal
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
import logging
import certifi

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC DATA MODELS
# ============================================================================

class ShotItem(BaseModel):
    """Individual shot item from the episode shot list."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    shot_id: str = Field(..., description="Unique identifier for the shot")
    description: str = Field(..., description="Detailed description of the shot content")
    duration: Optional[float] = Field(None, description="Shot duration in seconds")
    scene_number: Optional[int] = Field(None, description="Scene number this shot belongs to")
    sequence_number: Optional[int] = Field(None, description="Sequence number within the scene")
    shot_style: Optional[str] = Field(None, description="Shot style (e.g., close_up, wide_shot, medium_shot)")
    camera_movement: Optional[str] = Field(None, description="Camera movement (e.g., push_in, pan, zoom)")
    source_type: Literal["generated", "uploaded"] = Field("generated", description="Source type: generated or uploaded")
    uploaded_image_id: Optional[Union[str, ObjectId]] = Field(None, description="ObjectId from storyboard_images if source_type is 'uploaded'")
    generated_image_id: Optional[Union[str, ObjectId]] = Field(None, description="ObjectId of generated image if source_type is 'generated'")
    generated_video_id: Optional[Union[str, ObjectId]] = Field(None, description="ObjectId of generated video")
    optimized_ai_notes: Optional[str] = Field(None, description="Optimized AI notes for image/video generation")
    characters: Optional[List[str]] = Field(None, description="List of character names appearing in this shot (from CSV)")
    locations: Optional[str] = Field(None, description="Location name for this shot (from CSV)")


class ShotList(BaseModel):
    """Complete episode shot list input."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    episode_id: str = Field(..., description="Unique identifier for the episode")
    title: Optional[str] = Field(None, description="Episode title")
    scene_description: Optional[str] = Field(None, description="Scene description for context")
    shots: List[ShotItem] = Field(..., description="List of all shots in the episode")


class AnnotatedShotItem(ShotItem):
    """Shot item with generation strategy annotation."""
    
    generation_strategy: Literal["multi_shot", "last_frame_seed", "generate_new"] = Field(
        ..., 
        description="Recommended generation strategy for this shot"
    )
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for the strategy recommendation (0.0 to 1.0)"
    )
    seed_shot_id: Optional[str] = Field(
        None, 
        description="ID of the shot to use as seed (for last_frame_seed strategy)"
    )
    continuity_notes: Optional[str] = Field(
        None, 
        description="Notes about maintaining continuity with other shots"
    )
    strategy_approval: Optional[bool] = Field(
        None, 
        description="Human approval status for the strategy (None = pending, True = approved, False = rejected)"
    )
    optimized_ai_notes: Optional[str] = Field(
        None, 
        description="Optimized AI notes for image/video generation"
    )
    image: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="Versioned image generation data with keys like 'v0', 'v1', etc."
    )
    video: Optional[Dict[str, Dict[str, Any]]] = Field(
        None, 
        description="Versioned video generation data with keys like 'v0', 'v1', etc."
    )
    # Agent 12: Shot Design Agent output
    shot_design: Optional[Dict[str, Any]] = Field(
        None,
        description="Shot design output from Agent 12 (feasibility_score, selected_assets, model_recommendation, composition_strategy, warnings)"
    )
    # Agent 13: Prompt Modifier Agent output  
    prompt_modifications: Optional[Dict[str, Any]] = Field(
        None,
        description="Prompt modification output from Agent 13 (corrected_prompt, corrected_assets, warnings_resolved, modifications_made)"
    )
    # Final approval fields
    final_approval_status: Optional[str] = Field(
        None,
        description="Final approval status ('approved', 'max_retries_exceeded', 'pending')"
    )
    final_approval_timestamp: Optional[str] = Field(
        None,
        description="Timestamp of final approval decision"
    )


class AnnotatedShotList(BaseModel):
    """Complete episode shot list with generation strategy annotations."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    episode_id: str = Field(..., description="Unique identifier for the episode")
    title: Optional[str] = Field(None, description="Episode title")
    scene_description: Optional[str] = Field(None, description="Scene description for context")
    annotated_shots: List[AnnotatedShotItem] = Field(..., description="List of annotated shots")
    overall_continuity_notes: Optional[str] = Field(None, description="Overall continuity notes")
    strategy_summary: Optional[Dict[str, Any]] = Field(None, description="Strategy summary")
    processing_metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata about the processing (timestamps, agent versions, etc.)"
    )


class ShotCollectionItem(BaseModel):
    """MongoDB document structure for storing shots in the database."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    show_id: str = Field(..., description="Show identifier")
    episode_number: int = Field(..., description="Episode number")
    episode_id: str = Field(..., description="Episode identifier")
    title: Optional[str] = Field(None, description="Episode title")
    scene_description: Optional[str] = Field(None, description="Scene description")
    annotated_shots: List[AnnotatedShotItem] = Field(..., description="List of annotated shots")
    overall_continuity_notes: Optional[str] = Field(None, description="Overall continuity notes")
    strategy_summary: Optional[Dict[str, Any]] = Field(None, description="Strategy summary")
    processing_metadata: Optional[Dict[str, Any]] = Field(
        None, 
        description="Metadata about the processing"
    )
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")

# ============================================================================
# MONGODB VALIDATION SCHEMAS
# ============================================================================

# Shot item schema for MongoDB validation
SHOT_ITEM_SCHEMA = {
    "bsonType": "object",
    "required": ["shot_id", "description", "generation_strategy", "confidence_score"],
    "properties": {
        "shot_id": {"bsonType": "string", "minLength": 1},
        "description": {"bsonType": "string", "minLength": 1},
        "duration": {"bsonType": ["double", "null"]},
        "scene_number": {"bsonType": ["int", "null"]},
        "sequence_number": {"bsonType": ["int", "null"]},
        "shot_style": {"bsonType": ["string", "null"]},
        "camera_movement": {"bsonType": ["string", "null"]},
        "source_type": {"enum": ["generated", "uploaded"]},
        "uploaded_image_id": {"bsonType": ["string", "objectId", "null"]},
        "generated_image_id": {"bsonType": ["string", "objectId", "null"]},
        "generated_video_id": {"bsonType": ["string", "objectId", "null"]},
        "optimized_ai_notes": {"bsonType": ["string", "null"]},
        "characters": {
            "bsonType": ["array", "null"],
            "items": {"bsonType": "string"},
            "description": "List of character names appearing in this shot (from CSV)"
        },
        "locations": {
            "bsonType": ["string", "null"],
            "description": "Location name for this shot (from CSV)"
        },
        "generation_strategy": {"enum": ["multi_shot", "last_frame_seed", "generate_new"]},
        "confidence_score": {"bsonType": "double", "minimum": 0.0, "maximum": 1.0},
        "seed_shot_id": {"bsonType": ["string", "null"]},
        "continuity_notes": {"bsonType": ["string", "null"]},
        "strategy_approval": {"bsonType": ["bool", "null"]},
        "image": {
            "bsonType": ["object", "null"],
            "properties": {
                "patternProperties": {
                    "^v[0-9]+$": {
                        "bsonType": "object",
                        "properties": {
                            "updated_prompt": {"bsonType": "string"},
                            "changes_made": {"bsonType": "string"},
                            "reasoning": {"bsonType": "string"},
                            "generated_images_s3": {"bsonType": "array", "items": {"bsonType": "string"}}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        },
        "video": {
            "bsonType": ["object", "null"],
            "properties": {
                "patternProperties": {
                    "^v[0-9]+$": {
                        "bsonType": "object",
                        "properties": {
                            "updated_prompt": {"bsonType": "string"},
                            "changes_made": {"bsonType": "string"},
                            "reasoning": {"bsonType": "string"},
                            "generated_videos_s3": {"bsonType": "array", "items": {"bsonType": "string"}}
                        },
                        "additionalProperties": False
                    }
                }
            },
            "additionalProperties": False
        },
        "shot_design": {"bsonType": ["object", "null"]},
        "prompt_modifications": {"bsonType": ["object", "null"]},
        "final_approval_status": {"bsonType": ["string", "null"]},
        "final_approval_timestamp": {"bsonType": ["string", "null"]}
    },
    "additionalProperties": False
}

# Shots collection document schema for MongoDB validation
SHOTS_COLLECTION_SCHEMA = {
    "bsonType": "object",
    "required": ["show_id", "episode_number", "episode_id", "annotated_shots"],
    "properties": {
        "show_id": {"bsonType": "string", "minLength": 1},
        "episode_number": {"bsonType": "int", "minimum": 1},
        "episode_id": {"bsonType": "string", "minLength": 1},
        "title": {"bsonType": ["string", "null"]},
        "scene_description": {"bsonType": ["string", "null"]},
        "annotated_shots": {
            "bsonType": "array",
            "items": SHOT_ITEM_SCHEMA,
            "minItems": 1
        },
        "overall_continuity_notes": {"bsonType": ["string", "null"]},
        "strategy_summary": {"bsonType": ["object", "null"]},
        "processing_metadata": {"bsonType": ["object", "null"]},
        "created_at": {"bsonType": ["string", "null"]},
        "updated_at": {"bsonType": ["string", "null"]}
    },
    "additionalProperties": False
}

# ============================================================================
# MONGODB CLIENT CLASS
# ============================================================================

class MongoDBAtlasClient:
    """
    DEPRECATED: This class is deprecated. Use ShotsService from app.services.shots_service instead.

    MongoDB Atlas client for shot strategy operations.

    Handles connection, collection management, and shot data operations.

    This class is kept for backward compatibility but should not be instantiated directly.
    Use get_shots_service() or get_mongodb_atlas_client() from app.config instead to get
    the singleton instance with proper connection pooling.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "production",
        shots_collection: str = "shots"
    ):
        """
        Initialize MongoDB Atlas client.

        DEPRECATED: Use get_shots_service() from app.config instead.

        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the database
            shots_collection: Name of the shots collection
        """
        import warnings
        warnings.warn(
            "MongoDBAtlasClient is deprecated. Use get_shots_service() from app.config instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.connection_string = connection_string or os.getenv("production_MONGODB_URI") or os.getenv("MONGODB_ATLAS_URI")
        self.database_name = database_name
        self.shots_collection_name = shots_collection
        self.allow_local_mongo = os.getenv("production_ALLOW_LOCAL_MONGO", "false").lower() == "true"
        self._is_local_uri = self._detect_local_uri()
        if self._is_local_uri and not self.allow_local_mongo:
            raise ValueError(
                "Local MongoDB URI detected but production_ALLOW_LOCAL_MONGO is not enabled. "
                "Set production_ALLOW_LOCAL_MONGO=true in your environment for local testing."
            )
        
        if not self.connection_string:
            raise ValueError("MongoDB connection string is required. Set production_MONGODB_URI (preferred) or MONGODB_ATLAS_URI environment variable or pass connection_string parameter.")
        
        self.client: Optional[MongoClient] = None
        self.database: Optional[Database] = None
        self.shots_collection: Optional[Collection] = None
        
    def _detect_local_uri(self) -> bool:
        """Detect if URI points to localhost."""
        uri = (self.connection_string or "").lower()
        return uri.startswith("mongodb://localhost") or uri.startswith("mongodb://127.0.0.1")

    def _use_tls(self) -> bool:
        """Determine if TLS should be used."""
        if self._is_local_uri and self.allow_local_mongo:
            return False
        uri = (self.connection_string or "").lower()
        return uri.startswith("mongodb+srv://") or "ssl=true" in uri or "tls=true" in uri
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create client with optimized connection pool settings
            # Conservative pool size to prevent exhaustion
            use_tls = self._use_tls()
            connection_params = {
                'serverSelectionTimeoutMS': 60000,
                'connectTimeoutMS': 60000,
                'socketTimeoutMS': 120000,
                'retryWrites': True,
                'retryReads': True,
                'waitQueueTimeoutMS': 30000,
                'maxPoolSize': 50,  # Reduced from 100 to prevent exhaustion
                'minPoolSize': 1,   # Reduced from 5 to prevent excessive connections at startup
                'maxIdleTimeMS': 45000,  # Close idle connections after 45s
                'heartbeatFrequencyMS': 10000,  # Check server health every 10s
            }

            if use_tls:
                # Configure TLS/SSL settings with certifi
                try:
                    connection_params['tlsCAFile'] = certifi.where()
                    logger.info("Using certifi CA bundle for TLS")
                except ImportError:
                    logger.warning("certifi not available, using system CA bundle")

                connection_params['tls'] = True
                connection_params['tlsAllowInvalidHostnames'] = False  # More secure
            else:
                connection_params['tls'] = False

            self.client = MongoClient(self.connection_string, **connection_params)
            self.database = self.client[self.database_name]
            self.shots_collection = self.database[self.shots_collection_name]

            # Test connection
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB Atlas database: {self.database_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB Atlas: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB Atlas")
    
    def save_annotated_shots_to_atlas(
        self,
        annotated_shots: List[AnnotatedShotItem],
        show_id: str,
        episode_number: int,
        episode_id: str,
        title: Optional[str] = None,
        scene_description: Optional[str] = None,
        overall_continuity_notes: Optional[str] = None,
        strategy_summary: Optional[Dict[str, Any]] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save annotated shots to MongoDB Atlas.

        Args:
            annotated_shots: List of annotated shot items
            show_id: Show identifier
            episode_number: Episode number
            episode_id: Episode identifier
            title: Episode title
            scene_description: Scene description
            overall_continuity_notes: Overall continuity notes
            strategy_summary: Strategy summary
            processing_metadata: Processing metadata

        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False

        try:
            from datetime import datetime

            # Convert AnnotatedShotItem objects to dicts if needed
            serialized_shots = []
            for shot in annotated_shots:
                # Handle Pydantic models (including serialized/deserialized instances)
                if hasattr(shot, 'model_dump') and callable(getattr(shot, 'model_dump')):
                    serialized_shots.append(shot.model_dump())
                elif isinstance(shot, dict):
                    serialized_shots.append(shot)
                # Also check by type name for Celery-deserialized instances
                elif type(shot).__name__ == 'AnnotatedShotItem':
                    try:
                        serialized_shots.append(shot.model_dump())
                    except Exception as e:
                        logger.error(f"Failed to dump shot {type(shot)}: {e}")
                        return False
                else:
                    logger.error(f"Invalid shot type: {type(shot)}")
                    return False

            # Create document
            document = ShotCollectionItem(
                show_id=show_id,
                episode_number=episode_number,
                episode_id=episode_id,
                title=title,
                scene_description=scene_description,
                annotated_shots=serialized_shots,
                overall_continuity_notes=overall_continuity_notes,
                strategy_summary=strategy_summary,
                processing_metadata=processing_metadata,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )

            # Convert to dict for MongoDB
            doc_dict = document.model_dump()

            # Replace the annotated_shots with the serialized version
            doc_dict['annotated_shots'] = serialized_shots

            # Insert document
            result = self.shots_collection.insert_one(doc_dict)
            logger.info(f"Successfully saved {len(annotated_shots)} annotated shots to MongoDB Atlas. Document ID: {result.inserted_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save annotated shots to MongoDB Atlas: {e}")
            return False
    
    def get_shots_from_atlas(
        self, 
        show_id: str, 
        episode_number: int
    ) -> Optional[ShotCollectionItem]:
        """
        Retrieve shots from MongoDB Atlas.
        Backward compatible - normalizes 'shots' field to 'annotated_shots'.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            
        Returns:
            ShotCollectionItem if found, None otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return None
        
        try:
            # Query document - only use show_id (episode_number is not needed for shot lookup)
            query = {"show_id": show_id, "annotated_shots": {"$exists": True}}
            document = self.shots_collection.find_one(query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                query = {"show_id": show_id}
                document = self.shots_collection.find_one(query)
            
            if document:
                # Convert ObjectId to string for JSON serialization
                if "_id" in document:
                    document["_id"] = str(document["_id"])
                
                # Backward compatibility: rename 'shots' to 'annotated_shots' if needed
                if "shots" in document and "annotated_shots" not in document:
                    document["annotated_shots"] = document.pop("shots")
                    logger.debug("Normalized 'shots' field to 'annotated_shots' for backward compatibility")
                
                # Handle missing required fields with defaults
                if "episode_id" not in document:
                    document["episode_id"] = f"E{episode_number:02d}"  # Default episode ID
                    logger.debug(f"Added default episode_id: {document['episode_id']}")
                
                if "annotated_shots" not in document:
                    document["annotated_shots"] = []  # Default empty list
                    logger.debug("Added default empty annotated_shots list")
                
                try:
                    return ShotCollectionItem(**document)
                except Exception as validation_error:
                    logger.error(f"Document validation failed: {validation_error}")
                    logger.error(f"Document keys: {list(document.keys())}")
                    logger.error(f"Document content: {document}")
                    return None
            else:
                logger.info(f"No shots found for show_id: {show_id}, episode_number: {episode_number}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to retrieve shots from MongoDB Atlas: {e}")
            return None
    
    def update_shot_strategy_approval(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        approval_status: bool,
        feedback: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update strategy approval status for a specific shot.
        Backward compatible with both 'shots' and 'annotated_shots' field names.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            shot_id: Shot identifier
            approval_status: Approval status (True/False)
            feedback: Optional feedback data
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False
        
        try:
            from datetime import datetime
            
            # First, determine which field exists - prioritize documents with annotated_shots
            base_query = {
                "show_id": show_id,
                "episode_number": episode_number,
                "annotated_shots": {"$exists": True}
            }
            document = self.shots_collection.find_one(base_query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                base_query = {
                    "show_id": show_id,
                    "episode_number": episode_number
                }
                document = self.shots_collection.find_one(base_query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Determine which field to update (backward compatibility)
            # Check if field exists AND is a non-empty list/array
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Update query - use the determined shots_field consistently
            query = {
                "show_id": show_id,
                "episode_number": episode_number,
                f"{shots_field}.shot_id": shot_id
            }
            
            # Update data
            update_data = {
                "$set": {
                    f"{shots_field}.$.strategy_approval": approval_status,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            # Add feedback if provided
            if feedback:
                update_data["$set"][f"{shots_field}.$.feedback"] = feedback
            
            result = self.shots_collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated strategy approval for shot {shot_id} using field '{shots_field}'")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update strategy approval: {e}")
            return False
    
    def update_all_shots_strategy_approval(
        self,
        show_id: str,
        episode_number: int,
        approval_status: bool,
        feedback: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Update strategy approval status for all shots in an episode.
        Backward compatible with both 'shots' and 'annotated_shots' field names.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            approval_status: Approval status (True/False)
            feedback: Optional feedback data
            
        Returns:
            int: Number of shots updated
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return 0
        
        try:
            from datetime import datetime
            
            # Update query - prioritize documents with annotated_shots field
            query = {
                "show_id": show_id,
                "episode_number": episode_number,
                "annotated_shots": {"$exists": True}
            }
            
            # First, check which field name exists in the document
            document = self.shots_collection.find_one(query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                query = {
                    "show_id": show_id,
                    "episode_number": episode_number
                }
                document = self.shots_collection.find_one(query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}, episode_number: {episode_number}")
                return 0
            
            # Determine which field to update (backward compatibility)
            # Check if field exists AND is a non-empty list/array
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document for show_id: {show_id}, episode_number: {episode_number}")
                return 0
            
            # Update data
            update_data = {
                "$set": {
                    f"{shots_field}.$[].strategy_approval": approval_status,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            # Add feedback if provided
            if feedback:
                update_data["$set"][f"{shots_field}.$[].feedback"] = feedback
            
            # Use the same query logic for the update
            update_query = {
                "show_id": show_id,
                "episode_number": episode_number,
                "annotated_shots": {"$exists": True}
            }
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document.get("annotated_shots") and not document.get("shots"):
                update_query = {
                    "show_id": show_id,
                    "episode_number": episode_number
                }
            
            result = self.shots_collection.update_one(update_query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated strategy approval for all shots using field '{shots_field}'")
            else:
                logger.warning(f"No shots were modified. Check if strategy_approval values are already set.")
            
            return result.modified_count
                
        except Exception as e:
            logger.error(f"Failed to update strategy approval for all shots: {e}")
            return 0
    
    def update_shot_image_version(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        version: str,
        updated_prompt: str,
        changes_made: str,
        reasoning: str,
        generated_images_s3: Optional[List[str]] = None
    ) -> bool:
        """
        Update image generation data for a specific shot version.
        Backward compatible with both 'shots' and 'annotated_shots' field names.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            shot_id: Shot identifier
            version: Version key (e.g., 'v0', 'v1', etc.)
            updated_prompt: Updated image prompt
            changes_made: Description of changes made
            reasoning: Reasoning for the changes
            generated_images_s3: Optional list of S3 URLs for generated images
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False
        
        try:
            from datetime import datetime
            
            # First, determine which field exists - prioritize documents with annotated_shots
            base_query = {
                "show_id": show_id,
                "episode_number": episode_number,
                "annotated_shots": {"$exists": True}
            }
            document = self.shots_collection.find_one(base_query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                base_query = {
                    "show_id": show_id,
                    "episode_number": episode_number
                }
                document = self.shots_collection.find_one(base_query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Determine which field to update (backward compatibility)
            # Check if field exists AND is a non-empty list/array
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Update query - use the determined shots_field consistently
            query = {
                "show_id": show_id,
                "episode_number": episode_number,
                f"{shots_field}.shot_id": shot_id
            }
            
            # Prepare version data
            version_data = {
                "updated_prompt": updated_prompt,
                "changes_made": changes_made,
                "reasoning": reasoning,
                "generated_images_s3": generated_images_s3 or []
            }
            
            # Update data - use $set to initialize image field and add version
            # First, we need to handle the case where image field is null or doesn't exist
            update_data = {
                "$set": {
                    f"{shots_field}.$.image.{version}": version_data,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            # If the image field is null, we need to initialize it first
            # Check if the image field exists and is not null
            shot_doc = self.shots_collection.find_one(query)
            if shot_doc:
                # Find the specific shot in the array
                for shot_item in shot_doc.get(shots_field, []):
                    if shot_item.get("shot_id") == shot_id:
                        if shot_item.get("image") is None:
                            # Initialize image field as empty object first
                            init_update = {
                                "$set": {
                                    f"{shots_field}.$.image": {},
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                            }
                            self.shots_collection.update_one(query, init_update)
                        break
            
            result = self.shots_collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated image version {version} for shot {shot_id} using field '{shots_field}'")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update image version: {e}")
            return False
    
    def update_shot_video_version(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        version: str,
        updated_prompt: str,
        changes_made: str,
        reasoning: str,
        generated_videos_s3: Optional[List[str]] = None
    ) -> bool:
        """
        Update video generation data for a specific shot version.
        Backward compatible with both 'shots' and 'annotated_shots' field names.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            shot_id: Shot identifier
            version: Version key (e.g., 'v0', 'v1', etc.)
            updated_prompt: Updated video prompt
            changes_made: Description of changes made
            reasoning: Reasoning for the changes
            generated_videos_s3: Optional list of S3 URLs for generated videos
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False
        
        try:
            from datetime import datetime
            
            # First, determine which field exists - prioritize documents with annotated_shots
            base_query = {
                "show_id": show_id,
                "episode_number": episode_number,
                "annotated_shots": {"$exists": True}
            }
            document = self.shots_collection.find_one(base_query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                base_query = {
                    "show_id": show_id,
                    "episode_number": episode_number
                }
                document = self.shots_collection.find_one(base_query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Determine which field to update (backward compatibility)
            # Check if field exists AND is a non-empty list/array
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document for show_id: {show_id}, episode_number: {episode_number}")
                return False
            
            # Update query - use the determined shots_field consistently
            query = {
                "show_id": show_id,
                "episode_number": episode_number,
                f"{shots_field}.shot_id": shot_id
            }
            
            # Prepare version data
            version_data = {
                "updated_prompt": updated_prompt,
                "changes_made": changes_made,
                "reasoning": reasoning,
                "generated_videos_s3": generated_videos_s3 or []
            }
            
            # Update data - set the video version
            update_data = {
                "$set": {
                    f"{shots_field}.$.video.{version}": version_data,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            # If the video field is null, we need to initialize it first
            # Check if the video field exists and is not null
            shot_doc = self.shots_collection.find_one(query)
            if shot_doc:
                # Find the specific shot in the array
                for shot_item in shot_doc.get(shots_field, []):
                    if shot_item.get("shot_id") == shot_id:
                        if shot_item.get("video") is None:
                            # Initialize video field as empty object first
                            init_update = {
                                "$set": {
                                    f"{shots_field}.$.video": {},
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                            }
                            self.shots_collection.update_one(query, init_update)
                        break
            
            result = self.shots_collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated video version {version} for shot {shot_id} using field '{shots_field}'")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update video version: {e}")
            return False
    
    def update_shot_edited_image(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        version: str,
        edited_image_s3_url: str,
        edit_instructions: str,
        edit_prompt: str,
        edit_timestamp: str
    ) -> bool:
        """
        Update edited image data for a specific shot with versioning (v0, v1, v2, v3).
        
        Args:
            show_id: Show identifier
            episode_number: Episode number (used as fallback if not found in document)
            shot_id: Shot identifier
            version: Version key (e.g., 'v0', 'v1', 'v2', 'v3')
            edited_image_s3_url: S3 URL of the edited image
            edit_instructions: Edit instructions from Agent 15
            edit_prompt: Edit prompt used for editing
            edit_timestamp: Timestamp of the edit
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False
        
        try:
            from datetime import datetime
            
            # Find document by show_id only (episode_number is not needed for shot lookup)
            base_query = {
                "show_id": show_id,
                "annotated_shots": {"$exists": True}
            }
            document = self.shots_collection.find_one(base_query)
            
            # Fallback: if no document with annotated_shots found, try without that constraint
            if not document:
                base_query = {"show_id": show_id}
                document = self.shots_collection.find_one(base_query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}")
                return False
            
            # Determine which field to update (backward compatibility)
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document for show_id: {show_id}")
                return False
            
            # Update query - only use show_id and shot_id (no episode_number needed)
            query = {
                "show_id": show_id,
                f"{shots_field}.shot_id": shot_id
            }
            
            # Prepare version data
            version_data = {
                "s3_url": edited_image_s3_url,
                "edit_instructions": edit_instructions,
                "edit_prompt": edit_prompt,
                "edit_timestamp": edit_timestamp
            }
            
            # Initialize edited_image_s3 field if it doesn't exist
            shot_doc = self.shots_collection.find_one(query)
            if shot_doc:
                # Find the specific shot in the array
                for shot_item in shot_doc.get(shots_field, []):
                    if shot_item.get("shot_id") == shot_id:
                        if "edited_image_s3" not in shot_item or shot_item.get("edited_image_s3") is None:
                            # Initialize edited_image_s3 field as empty object first
                            init_update = {
                                "$set": {
                                    f"{shots_field}.$.edited_image_s3": {},
                                    "updated_at": datetime.utcnow().isoformat()
                                }
                            }
                            self.shots_collection.update_one(query, init_update)
                        break
            
            # Update data - set the edited image version
            update_data = {
                "$set": {
                    f"{shots_field}.$.edited_image_s3.{version}": version_data,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            result = self.shots_collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated edited image {version} for shot {shot_id} using field '{shots_field}'")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update edited image: {e}")
            return False
    
    def update_shot_approval_status(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        approval_status: str,
        approval_timestamp: str
    ) -> bool:
        """
        Update final approval status for a shot.
        
        Args:
            show_id: Show identifier
            episode_number: Episode number
            shot_id: Shot identifier
            approval_status: Status ('approved', 'max_retries_exceeded', 'pending')
            approval_timestamp: Timestamp of approval decision
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return False
        
        try:
            from datetime import datetime
            
            # Find document by show_id only (episode_number is not needed for shot lookup)
            base_query = {
                "show_id": show_id,
                "annotated_shots": {"$exists": True}
            }
            document = self.shots_collection.find_one(base_query)
            
            if not document:
                base_query = {"show_id": show_id}
                document = self.shots_collection.find_one(base_query)
            
            if not document:
                logger.warning(f"No document found for show_id: {show_id}")
                return False
            
            # Determine which field to update
            has_annotated_shots = "annotated_shots" in document and isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots", [])) > 0
            has_shots = "shots" in document and isinstance(document.get("shots"), list) and len(document.get("shots", [])) > 0
            
            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No valid shots array found in document")
                return False
            
            # Update query - only use show_id and shot_id (no episode_number needed)
            query = {
                "show_id": show_id,
                f"{shots_field}.shot_id": shot_id
            }
            
            # Update data
            update_data = {
                "$set": {
                    f"{shots_field}.$.final_approval_status": approval_status,
                    f"{shots_field}.$.final_approval_timestamp": approval_timestamp,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
            
            result = self.shots_collection.update_one(query, update_data)
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated approval status for shot {shot_id}: {approval_status}")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update approval status: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict containing database statistics
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return {}
        
        try:
            stats = self.database.command("dbStats")
            return {
                "database": stats.get("db", "unknown"),
                "collections": stats.get("collections", 0),
                "objects": stats.get("objects", 0),
                "avgObjSize": stats.get("avgObjSize", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0),
                "indexSize": stats.get("indexSize", 0)
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def get_shots_count(self) -> int:
        """
        Get total number of shot documents in the collection.
        
        Returns:
            int: Number of shot documents
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return 0
        
        try:
            return self.shots_collection.count_documents({})
        except Exception as e:
            logger.error(f"Failed to get shots count: {e}")
            return 0
    
    def list_episodes(self, show_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all episodes in the database.
        Backward compatible with both 'shots' and 'annotated_shots' field names.
        
        Args:
            show_id: Optional show ID to filter by
            
        Returns:
            List of episode information
        """
        if not self.client:
            logger.error("MongoDB client not connected. Call connect() first.")
            return []
        
        try:
            query = {"show_id": show_id} if show_id else {}
            projection = {
                "show_id": 1,
                "episode_number": 1,
                "episode_id": 1,
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "annotated_shots": 1,
                "shots": 1
            }
            
            episodes = list(self.shots_collection.find(query, projection))
            
            # Process episodes and calculate shots count
            result = []
            for episode in episodes:
                # Convert ObjectId to string
                if "_id" in episode:
                    episode["_id"] = str(episode["_id"])
                
                # Calculate shots count (backward compatible)
                shots_array = episode.get("annotated_shots", episode.get("shots", []))
                episode["shots_count"] = len(shots_array)
                
                # Remove the shots arrays from response
                episode.pop("annotated_shots", None)
                episode.pop("shots", None)
                
                result.append(episode)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list episodes: {e}")
            return []