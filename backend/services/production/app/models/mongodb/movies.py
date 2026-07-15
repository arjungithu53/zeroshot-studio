"""Movies document schema for MongoDB validation (production Service)."""

from pymongo.errors import OperationFailure
from typing import Any

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Scene schema structure (embedded in movie document)
SCENE_SCHEMA = {
    "bsonType": "object",
    "required": ["scene_number", "scene_name", "script"],
    "properties": {
        "scene_number": {
            "bsonType": "int",
            "description": "Scene number/order in the movie"
        },
        "scene_name": {
            "bsonType": "string",
            "minLength": 1,
            "description": "Scene name/title"
        },
        "script": {
            "bsonType": "string",
            "minLength": 1,
            "description": "Scene script content"
        },
        "shotlist": {
            "bsonType": ["string", "null"],
            "description": "Scene shotlist (optional)"
        },
        "project_id": {
            "bsonType": ["objectId", "null"],
            "description": "Reference to production_projects document for this scene"
        }
    },
    "additionalProperties": False
}


# Aggregated data schema
AGGREGATED_DATA_SCHEMA = {
    "bsonType": ["object", "null"],
    "properties": {
        "total_scenes": {
            "bsonType": ["int", "null"],
            "description": "Total number of scenes in the movie"
        },
        "total_characters": {
            "bsonType": ["int", "null"],
            "description": "Total number of unique characters"
        },
        "total_locations": {
            "bsonType": ["int", "null"],
            "description": "Total number of unique locations"
        },
        "total_props": {
            "bsonType": ["int", "null"],
            "description": "Total number of unique props"
        },
        "completed_scenes": {
            "bsonType": ["int", "null"],
            "description": "Number of scenes that have completed production"
        }
    },
    "additionalProperties": False
}


# Global settings schema
GLOBAL_SETTINGS_SCHEMA = {
    "bsonType": ["object", "null"],
    "properties": {
        "aspect_ratio": {
            "bsonType": ["string", "null"],
            "enum": ["9:16", "16:9", "2.39:1", None],
            "description": "Default aspect ratio for the movie (Vertical: 9:16, Horizontal: 16:9, Cinematic: 2.39:1)"
        },
        "visual_style": {
            "bsonType": ["string", "null"],
            "enum": ["realistic", "pixar", "2d", None],
            "description": "Overall visual style/aesthetic (realistic, pixar, or 2d)"
        },
        "color_palette": {
            "bsonType": ["string", "null"],
            "description": "Color palette description"
        },
        "video_model": {
            "bsonType": ["string", "null"],
            "enum": ["Veo 3.1", "Omni Flash", None],
            "description": "Video generation model used for Phase 3 (Veo 3.1 or Omni Flash)"
        }
    },
    "additionalProperties": True
}


MOVIES_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["title", "scenes", "created_at", "updated_at"],
        "properties": {
            "_id": {
                "bsonType": "objectId",
                "description": "Auto-generated MongoDB ObjectId"
            },
            "title": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Movie title"
            },
            "description": {
                "bsonType": ["string", "null"],
                "description": "Movie description/synopsis"
            },
            "genre": {
                "bsonType": ["string", "null"],
                "description": "Movie genre"
            },
            "user_id": {
                "bsonType": ["string", "null"],
                "description": "User who created the movie"
            },

            # All scenes defined at creation (immutable)
            "scenes": {
                "bsonType": "array",
                "minItems": 1,
                "items": SCENE_SCHEMA,
                "description": "All scenes in the movie (defined at creation)"
            },

            # Reference to shared assets
            "assets_collection_id": {
                "bsonType": ["objectId", "null"],
                "description": "Reference to assets_collections document"
            },

            # Project references (populated after scene projects are created)
            "project_ids": {
                "bsonType": ["array", "null"],
                "items": {
                    "bsonType": "objectId"
                },
                "description": "References to production_projects documents (one per scene)"
            },

            # Aggregated data from all scenes
            "aggregated_data": AGGREGATED_DATA_SCHEMA,

            # Global movie settings
            "global_settings": GLOBAL_SETTINGS_SCHEMA,

            # Status tracking
            "phase1_status": {
                "bsonType": ["string", "null"],
                "enum": ["pending", "running", "completed", "failed", "skipped", None],
                "description": "Phase 1 (asset generation) status for the entire movie"
            },
            "overall_status": {
                "bsonType": ["string", "null"],
                "enum": ["created", "assets_generated", "in_production", "completed", "failed", None],
                "description": "Overall movie production status"
            },

            "created_at": {
                "bsonType": "date",
                "description": "Timestamp when movie was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Timestamp when movie was last updated"
            }
        },
        "additionalProperties": True
    }
}


def apply_schema(db: Any) -> None:
    """
    Apply schema validation to movies collection.

    Args:
        db: MongoDB database instance
    """
    collection_name = "movies"
    try:
        db.command("collMod", collection_name, validator=MOVIES_SCHEMA, validationLevel="moderate")
        logger.info(f"Schema applied to '{collection_name}'")
    except OperationFailure as e:
        if "NamespaceNotFound" in str(e) or "ns not found" in str(e):
            db.create_collection(collection_name, validator=MOVIES_SCHEMA, validationLevel="moderate")
            logger.info(f"Collection created with schema: '{collection_name}'")

            # Create indexes
            db[collection_name].create_index("user_id")
            db[collection_name].create_index("created_at")
            db[collection_name].create_index("overall_status")
            logger.info(f"Indexes created for '{collection_name}'")
        else:
            logger.error(f"Failed to apply schema to '{collection_name}': {e}")
