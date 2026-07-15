"""
ShotsService - Service layer for MongoDB shots collection operations.

This service uses MongoClientFactory from config.py to ensure singleton
connection management and prevent connection pool exhaustion.

IMPORTANT: This replaces the deprecated MongoDBAtlasClient from:
- app/models/mongodb/shots.py
- app/services/phase_2_agents/agent_shot_strategy/mongodb_utils.py

All Phase 2 and Phase 3 agents should use this service for shots operations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pymongo.collection import Collection
from pymongo import MongoClient

from app.models.mongodb.shots import (
    AnnotatedShotItem,
    ShotCollectionItem,
)

logger = logging.getLogger(__name__)


class ShotsService:
    """
    Service layer for MongoDB shots collection operations.

    Uses MongoClientFactory singleton to ensure only ONE connection
    is used across the entire application.
    """

    def __init__(self, client: MongoClient, collection: Collection):
        """
        Initialize ShotsService with MongoDB client and collection.

        Args:
            client: MongoClient instance from MongoClientFactory
            collection: Collection instance for shots
        """
        self.client = client
        self.shots_collection = collection
        logger.debug("ShotsService initialized with singleton MongoDB connection")

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
        try:
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
        try:
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

    def update_shot_generation_strategy(
        self,
        show_id: str,
        episode_number: int,
        shot_id: str,
        generation_strategy: str,
        optimized_ai_notes: Optional[str] = None,
        confidence_score: Optional[float] = None,
        seed_shot_id: Optional[str] = None,
        continuity_notes: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update generation strategy data for a specific shot.

        Args:
            show_id: Show identifier (project_id)
            episode_number: Scene number / episode number
            shot_id: Shot identifier within the scene
            generation_strategy: New generation strategy value
            optimized_ai_notes: Optional optimized notes
            confidence_score: Optional confidence score override
            seed_shot_id: Optional seed shot identifier for last_frame_seed strategy
            continuity_notes: Optional continuity notes

        Returns:
            Updated shot document (dict) if successful, None otherwise.
        """
        try:
            document = self.shots_collection.find_one({
                "show_id": show_id,
                "episode_number": episode_number
            })

            if not document:
                logger.warning(f"No document found for show_id={show_id}, episode_number={episode_number}")
                return None

            has_annotated_shots = isinstance(document.get("annotated_shots"), list) and len(document.get("annotated_shots") or []) > 0
            has_shots = isinstance(document.get("shots"), list) and len(document.get("shots") or []) > 0

            if has_annotated_shots:
                shots_field = "annotated_shots"
            elif has_shots:
                shots_field = "shots"
            else:
                logger.error(f"No shots array found in document for show_id={show_id}, episode_number={episode_number}")
                return None

            update_fields = {
                f"{shots_field}.$.generation_strategy": generation_strategy,
                "updated_at": datetime.utcnow().isoformat()
            }

            if optimized_ai_notes is not None:
                update_fields[f"{shots_field}.$.optimized_ai_notes"] = optimized_ai_notes

            if confidence_score is not None:
                update_fields[f"{shots_field}.$.confidence_score"] = confidence_score

            if seed_shot_id is not None:
                update_fields[f"{shots_field}.$.seed_shot_id"] = seed_shot_id

            if continuity_notes is not None:
                update_fields[f"{shots_field}.$.continuity_notes"] = continuity_notes

            query = {
                "show_id": show_id,
                "episode_number": episode_number,
                f"{shots_field}.shot_id": shot_id
            }

            result = self.shots_collection.update_one(query, {"$set": update_fields})

            if result.matched_count == 0:
                logger.warning(f"No shot matched show_id={show_id}, episode_number={episode_number}, shot_id={shot_id}")
                return None

            updated_doc = self.shots_collection.find_one(
                {
                    "show_id": show_id,
                    "episode_number": episode_number
                },
                {
                    "_id": 0,
                    "show_id": 1,
                    "episode_number": 1,
                    shots_field: {
                        "$elemMatch": {
                            "shot_id": shot_id
                        }
                    }
                }
            )

            if not updated_doc or shots_field not in updated_doc or not updated_doc[shots_field]:
                return None

            shot_payload = updated_doc[shots_field][0]
            shot_payload["show_id"] = show_id
            shot_payload["episode_number"] = episode_number
            return shot_payload

        except Exception as exc:
            logger.error(
                f"Failed to update generation strategy for shot_id={shot_id} (show_id={show_id}, episode_number={episode_number}): {exc}"
            )
            return None

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
        try:
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
        try:
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

    def set_shot_image_selection(
        self,
        show_id: str,
        shot_id: str,
        version: str,
        index: int,
        url: str,
        selected_by: str = "human",
        episode_number: Optional[int] = None,
    ) -> bool:
        """
        Persist a human's image selection for a shot.
        Sets image.selected on the matching annotated_shots entry.
        Finds the document by show_id + shot_id (episode_number is ignored — shots are keyed by ID).
        """
        try:
            # Find the document that actually contains this shot_id (show_id is optional)
            base = {"show_id": show_id} if show_id else {}
            query = {**base, "annotated_shots.shot_id": shot_id}
            document = self.shots_collection.find_one(query)
            if not document:
                query = {**base, "shots.shot_id": shot_id}
                document = self.shots_collection.find_one(query)

            if not document:
                logger.warning(f"No document found containing shot_id: {shot_id} for show_id: {show_id}")
                return False

            # Use the actual show_id from the found document — the caller may have passed None
            actual_show_id = document.get("show_id", show_id)

            has_annotated_shots = (
                "annotated_shots" in document
                and isinstance(document.get("annotated_shots"), list)
                and len(document.get("annotated_shots", [])) > 0
            )
            shots_field = "annotated_shots" if has_annotated_shots else "shots"

            query = {
                "show_id": actual_show_id,
                f"{shots_field}.shot_id": shot_id
            }

            # Init image field if null (same guard as update_shot_image_version)
            shot_doc = self.shots_collection.find_one(query)
            if shot_doc:
                for shot_item in shot_doc.get(shots_field, []):
                    if shot_item.get("shot_id") == shot_id:
                        if shot_item.get("image") is None:
                            self.shots_collection.update_one(
                                query,
                                {"$set": {f"{shots_field}.$.image": {}, "updated_at": datetime.utcnow().isoformat()}}
                            )
                        break

            selection = {
                "version": version,
                "index": index,
                "url": url,
                "selected_by": selected_by,
                "selected_at": datetime.utcnow().isoformat()
            }
            result = self.shots_collection.update_one(
                query,
                {"$set": {
                    f"{shots_field}.$.image.selected": selection,
                    "updated_at": datetime.utcnow().isoformat()
                }}
            )

            if result.matched_count > 0:
                logger.info(f"Saved image selection for shot {shot_id}: {version}[{index}]")
                return True
            else:
                logger.warning(f"No shot found with shot_id: {shot_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to set image selection: {e}")
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
        try:
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
        try:
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
        try:
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
        try:
            database = self.client[self.shots_collection.database.name]
            stats = database.command("dbStats")
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

    def get_shot_by_id(
        self,
        shot_id: str,
        show_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific shot by shot_id.
        Searches in annotated_shots array first, then falls back to individual documents.

        Args:
            shot_id: Shot identifier (e.g., 'S01E01_001')
            show_id: Optional show identifier to narrow the search

        Returns:
            Dict containing shot data if found, None otherwise
        """
        try:
            # Priority 1: Search in annotated_shots array within episode documents
            query = {"annotated_shots": {"$exists": True}}
            if show_id:
                query["show_id"] = show_id

            episode_doc = self.shots_collection.find_one(query)

            if episode_doc and "annotated_shots" in episode_doc:
                # Find the specific shot in the annotated_shots array
                for shot in episode_doc["annotated_shots"]:
                    if shot.get("shot_id") == shot_id:
                        # Add episode context to the shot
                        shot_with_context = {
                            **shot,
                            "episode_id": episode_doc.get("_id"),
                            "episode_title": episode_doc.get("title"),
                            "episode_description": episode_doc.get("scene_description"),
                            "show_id": episode_doc.get("show_id"),
                            "episode_number": episode_doc.get("episode_number")
                        }
                        logger.info(f"Found shot {shot_id} in annotated_shots array")
                        return shot_with_context

            # Priority 2: Fallback to individual shot document
            if show_id:
                query = {
                    "shot_id": shot_id,
                    "$or": [
                        {"show_id": show_id},
                        {"project_id": show_id}
                    ]
                }
            else:
                query = {"shot_id": shot_id}

            shot = self.shots_collection.find_one(query)
            if shot:
                logger.info(f"Found shot {shot_id} as individual document")
                return shot

            logger.warning(f"Shot {shot_id} not found")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve shot {shot_id}: {e}")
            return None

    def connect(self) -> bool:
        """
        Compatibility method for legacy code.
        Connection is already established via MongoClientFactory.

        Returns:
            bool: Always True
        """
        logger.debug("connect() called - connection already established via MongoClientFactory")
        return True

    def disconnect(self):
        """
        Compatibility method for legacy code.
        Connection management is handled by MongoClientFactory.
        """
        logger.debug("disconnect() called - connection managed by MongoClientFactory singleton")
        pass
