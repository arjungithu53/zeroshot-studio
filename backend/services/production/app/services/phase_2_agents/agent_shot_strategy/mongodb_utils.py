"""
MongoDB Atlas utilities for shot strategy agent.

Provides connection management and database operations for MongoDB Atlas.
"""

import os
import ssl
import certifi
from typing import List, Dict, Any, Optional, Union
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

from app.models.mongodb.shots import ShotCollectionItem, AnnotatedShotList

logger = logging.getLogger(__name__)


class MongoDBAtlasClient:
    """
    MongoDB Atlas client for shot strategy operations.
    
    Handles connection, collection management, and shot data operations.
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "production",
        shots_collection: str = "shots"
    ):
        """
        Initialize MongoDB Atlas client.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Name of the database
            shots_collection: Name of the shots collection
        """
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
        
        self._connect()
    
    def _detect_local_uri(self) -> bool:
        """Detect if URI points to localhost."""
        uri = (self.connection_string or "").lower()
        return uri.startswith("mongodb://localhost") or uri.startswith("mongodb://127.0.0.1")
    
    def _use_tls(self) -> bool:
        """Determine if TLS should be used for the current URI."""
        if self._is_local_uri and self.allow_local_mongo:
            return False
        uri = (self.connection_string or "").lower()
        # Use TLS for Atlas/SRV or when explicitly requested via query params
        return uri.startswith("mongodb+srv://") or "ssl=true" in uri or "tls=true" in uri
    
    def _connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            # Try multiple connection strategies with optimized settings
            use_tls = self._use_tls()
            connection_attempts = self._build_connection_attempts(use_tls)

            last_error = None
            for attempt_num, conn_params in enumerate(connection_attempts, 1):
                try:
                    logger.info(f"MongoDB connection attempt {attempt_num}/{len(connection_attempts)}")
                    self.client = MongoClient(self.connection_string, **conn_params)

                    # Test connection
                    self.client.admin.command('ping')
                    logger.info(f"✅ MongoDB connected successfully (attempt {attempt_num})")
                    break  # Connection successful

                except Exception as e:
                    last_error = e
                    if attempt_num < len(connection_attempts):
                        logger.warning(f"Connection attempt {attempt_num} failed, trying next method...")
                    continue
            else:
                # All attempts failed
                raise last_error if last_error else ConnectionFailure("All connection attempts failed")
            
            self.database = self.client[self.database_name]
            self.shots_collection = self.database[self.shots_collection_name]
            
            logger.info(f"Successfully connected to MongoDB Atlas database: {self.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {str(e)}")
            raise ConnectionFailure(f"Could not connect to MongoDB Atlas: {str(e)}")

    def _build_connection_attempts(self, use_tls: bool):
        """Build connection attempts depending on TLS usage."""
        base_params = {
            'serverSelectionTimeoutMS': 60000,
            'connectTimeoutMS': 60000,
            'socketTimeoutMS': 120000,
            'retryWrites': True,
            'retryReads': True,
            'waitQueueTimeoutMS': 30000,
            'maxPoolSize': 50,
            'minPoolSize': 1,  # Reduced from 5 to prevent excessive connections at startup
            'maxIdleTimeMS': 45000,
            'heartbeatFrequencyMS': 10000,
        }

        if not use_tls:
            # Local MongoDB - no TLS
            return [
                {
                    **base_params,
                    'tls': False,
                }
            ]

        attempts = []
        try:
            attempts.append({
                **base_params,
                'tlsCAFile': certifi.where(),
                'tls': True,
            })
        except ImportError:
            attempts.append({
                **base_params,
                'tls': True,
            })

        # Allow invalid certificates (development)
        attempts.append({
            **base_params,
            'tlsAllowInvalidCertificates': True,
            'tlsAllowInvalidHostnames': True,
            'tls': True,
        })

        # Minimal TLS settings
        attempts.append({
            **base_params,
            'tls': True,
        })

        return attempts
    
    def is_connected(self) -> bool:
        """Check if connection to MongoDB Atlas is active."""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except Exception:
            pass
        return False
    
    def reconnect(self):
        """Reconnect to MongoDB Atlas."""
        if self.client:
            self.client.close()
        self._connect()
    
    def close(self):
        """Close connection to MongoDB Atlas."""
        if self.client:
            self.client.close()
            logger.info("MongoDB Atlas connection closed")
    
    def insert_shots(self, shots_data: List[Dict[str, Any]]) -> List[str]:
        """
        Insert shot documents into MongoDB Atlas.
        
        Args:
            shots_data: List of shot documents to insert
            
        Returns:
            List of inserted document IDs
        """
        if not self.is_connected():
            self.reconnect()
        
        try:
            result = self.shots_collection.insert_many(shots_data)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} shots into MongoDB Atlas")
            return [str(doc_id) for doc_id in result.inserted_ids]
            
        except Exception as e:
            logger.error(f"Failed to insert shots into MongoDB Atlas: {str(e)}")
            raise
    
    def get_shots_by_episode(self, show_id: str, episode_number: int) -> List[Dict[str, Any]]:
        """
        Retrieve shots for a specific episode from MongoDB Atlas.
        
        Args:
            show_id: Show ID to filter by
            episode_number: Episode number to filter by
            
        Returns:
            List of shot documents
        """
        if not self.is_connected():
            self.reconnect()
        
        try:
            shots = list(self.shots_collection.find({
                "show_id": show_id,
                "episode_number": episode_number
            }).sort("scene_number", 1).sort("shot_number", 1))
            
            logger.info(f"Retrieved {len(shots)} shots for show {show_id}, episode {episode_number}")
            return shots
            
        except Exception as e:
            logger.error(f"Failed to retrieve shots from MongoDB Atlas: {str(e)}")
            raise
    
    def update_shot_strategy(self, shot_id: str, strategy_data: Dict[str, Any]) -> bool:
        """
        Update shot strategy information in MongoDB Atlas.
        
        Args:
            shot_id: ID of the shot to update
            strategy_data: Strategy data to update
            
        Returns:
            True if update was successful
        """
        if not self.is_connected():
            self.reconnect()
        
        try:
            result = self.shots_collection.update_one(
                {"_id": shot_id},
                {"$set": strategy_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Successfully updated shot {shot_id} strategy in MongoDB Atlas")
                return True
            else:
                logger.warning(f"No shot found with ID {shot_id} to update")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update shot strategy in MongoDB Atlas: {str(e)}")
            raise
    
    def get_shot_by_id(self, shot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific shot by ID from MongoDB Atlas.
        Searches in annotated_shots array first, then falls back to standalone documents.

        Args:
            shot_id: ID of the shot to retrieve

        Returns:
            Shot document or None if not found
        """
        if not self.is_connected():
            self.reconnect()

        try:
            # First, try to find in annotated_shots array (new structure)
            episode_doc = self.shots_collection.find_one({
                "annotated_shots.shot_id": shot_id
            })

            if episode_doc and "annotated_shots" in episode_doc:
                # Find the specific shot in the annotated_shots array
                for shot in episode_doc["annotated_shots"]:
                    if shot.get("shot_id") == shot_id:
                        # Add episode context to the shot
                        shot_with_context = {
                            **shot,
                            "episode_id": episode_doc.get("_id"),
                            "show_id": episode_doc.get("show_id"),
                            "episode_number": episode_doc.get("episode_number")
                        }
                        logger.info(f"Retrieved shot {shot_id} from annotated_shots array")
                        return shot_with_context

            # Fallback: try standalone document (old structure)
            shot = self.shots_collection.find_one({"_id": shot_id})
            if shot:
                logger.info(f"Retrieved shot {shot_id} as standalone document")
                return shot

            logger.warning(f"Shot {shot_id} not found in MongoDB Atlas")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve shot from MongoDB Atlas: {str(e)}")
            raise
    
    def create_indexes(self):
        """
        Create useful indexes for the shots collection.
        
        This improves query performance for common operations.
        """
        if not self.is_connected():
            self.reconnect()
        
        try:
            # Create compound index for show_id and episode_number
            self.shots_collection.create_index([
                ("show_id", 1),
                ("episode_number", 1),
                ("scene_number", 1),
                ("shot_number", 1)
            ])
            
            # Create index for generation_strategy
            self.shots_collection.create_index("generation_strategy")
            
            # Create index for source_type
            self.shots_collection.create_index("source_type")
            
            # Create index for seed_shot_id
            self.shots_collection.create_index("seed_shot_id")
            
            logger.info("Successfully created indexes for shots collection")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the shots collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.is_connected():
            self.reconnect()
        
        try:
            stats = self.database.command("collStats", self.shots_collection_name)
            
            # Get additional statistics
            total_shots = self.shots_collection.count_documents({})
            
            # Count by generation strategy
            strategy_counts = {}
            for strategy in ["generate_new", "last_frame_seed", "multi_shot"]:
                count = self.shots_collection.count_documents({"generation_strategy": strategy})
                strategy_counts[strategy] = count
            
            # Count by source type
            source_counts = {}
            for source_type in ["generated", "uploaded"]:
                count = self.shots_collection.count_documents({"source_type": source_type})
                source_counts[source_type] = count
            
            return {
                "total_shots": total_shots,
                "strategy_distribution": strategy_counts,
                "source_distribution": source_counts,
                "collection_size": stats.get("size", 0),
                "document_count": stats.get("count", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection statistics: {str(e)}")
            raise


def create_mongodb_atlas_client(
    connection_string: Optional[str] = None,
    database_name: str = "production",
    shots_collection: str = "shots"
) -> MongoDBAtlasClient:
    """
    Factory function to create MongoDB Atlas client.
    
    Args:
        connection_string: MongoDB Atlas connection string
        database_name: Name of the database
        shots_collection: Name of the shots collection
        
    Returns:
        MongoDBAtlasClient instance
    """
    return MongoDBAtlasClient(
        connection_string=connection_string,
        database_name=database_name,
        shots_collection=shots_collection
    )


def save_annotated_shots_to_atlas(
    annotated_list: AnnotatedShotList,
    show_id: str,
    episode_number: int,
    mongodb_client: MongoDBAtlasClient
) -> List[str]:
    """
    Save annotated shot list to MongoDB Atlas using upsert.
    
    This function uses upsert to update existing documents or insert new ones,
    preventing duplicate shot_id entries.
    
    Args:
        annotated_list: Annotated shot list to save
        show_id: Show ID for the shots
        episode_number: Episode number
        mongodb_client: MongoDB Atlas client instance
        
    Returns:
        List of upserted document IDs
    """
    from .shot_strategy_agent import ShotStrategyAgent
    
    # Create a temporary agent instance to use the conversion method
    agent = ShotStrategyAgent(llm=None)  # We only need the conversion method
    
    # Convert to MongoDB format
    mongodb_docs = agent.to_mongodb_collection(annotated_list, show_id, episode_number)
    
    # Use upsert to prevent duplicates
    upserted_ids = []
    
    for doc in mongodb_docs:
        # Use shot_id, show_id, and episode_number as unique filter
        filter_query = {
            "shot_id": doc["shot_id"],
            "show_id": doc["show_id"],
            "episode_number": doc["episode_number"]
        }
        
        # Upsert: update if exists, insert if not
        result = mongodb_client.shots_collection.update_one(
            filter_query,
            {"$set": doc},
            upsert=True
        )
        
        # Get the document ID (either matched or newly inserted)
        if result.upserted_id:
            upserted_ids.append(str(result.upserted_id))
            logger.info(f"✅ Inserted new shot: {doc['shot_id']}")
        else:
            # Document was updated, get its _id
            existing_doc = mongodb_client.shots_collection.find_one(filter_query, {"_id": 1})
            if existing_doc:
                upserted_ids.append(str(existing_doc["_id"]))
                logger.info(f"🔄 Updated existing shot: {doc['shot_id']}")
    
    logger.info(f"Upserted {len(upserted_ids)} shots to MongoDB Atlas")
    return upserted_ids


def update_shot_prompts_in_atlas(
    annotated_list: AnnotatedShotList,
    show_id: str,
    episode_number: int,
    mongodb_client: MongoDBAtlasClient
) -> int:
    """
    Update image prompts for shots in MongoDB Atlas.
    
    Args:
        annotated_list: Annotated shot list with image prompts
        show_id: Show ID for the shots
        episode_number: Episode number
        mongodb_client: MongoDB Atlas client instance
./start.sh        
    Returns:
        Number of successfully updated documents
    """
    updated_count = 0
    
    for shot in annotated_list.annotated_shots:
        if shot.prompt_image_draft:
            try:
                # Find the document by shot_id, show_id, and episode_number
                filter_query = {
                    "shot_id": shot.shot_id,
                    "show_id": show_id,
                    "episode_number": episode_number
                }
                
                # Update the prompt_image_draft field
                update_data = {
                    "prompt_image_draft": shot.prompt_image_draft
                }
                
                result = mongodb_client.shots_collection.update_one(
                    filter_query,
                    {"$set": update_data}
                )
                
                if result.matched_count > 0:
                    if result.modified_count > 0:
                        updated_count += 1
                        logger.debug(f"Updated prompt for shot {shot.shot_id}")
                    else:
                        logger.debug(f"Prompt for shot {shot.shot_id} already up-to-date")
                else:
                    logger.warning(f"Shot {shot.shot_id} not found in MongoDB for update (filter: {filter_query})")
                    
            except Exception as e:
                logger.error(f"Error updating prompt for shot {shot.shot_id}: {str(e)}")
                continue
    
    logger.info(f"Updated {updated_count}/{len(annotated_list.annotated_shots)} shot prompts in MongoDB")
    return updated_count