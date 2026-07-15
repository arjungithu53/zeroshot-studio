"""Assets document schema for MongoDB validation (production Service - Phase 1)."""

from pymongo.errors import OperationFailure
from typing import Any

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Image version schema (reusable for all angle types)
IMAGE_VERSION_SCHEMA = {
    "bsonType": "object",
    "required": ["version", "prompt", "image_url"],
    "properties": {
        "version": {
            "bsonType": "int",
            "minimum": 1,
            "description": "Version number (1, 2, 3, ...)"
        },
        "prompt": {
            "bsonType": "string",
            "minLength": 1,
            "description": "Prompt used to generate this image"
        },
        "image_url": {
            "bsonType": "string",
            "minLength": 1,
            "description": "S3 URL or path to the generated image"
        },
        "change_needed": {
            "bsonType": ["string", "null"],
            "description": "Feedback/changes needed for next iteration"
        }
    },
    "additionalProperties": False
}

ASSETS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["project_id", "name", "type", "data", "created_at", "updated_at"],
        "properties": {
            "_id": {
                "bsonType": "objectId",
                "description": "Auto-generated MongoDB ObjectId (this is the asset_id)"
            },
            "project_id": {
                "bsonType": "objectId",
                "description": "Reference to parent project in production_projects collection"
            },

            # Asset metadata
            "name": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Asset name (e.g., 'BLACK_LAB_PUPPY', 'COFFEE_SHOP')"
            },
            "type": {
                "bsonType": "string",
                "enum": ["character", "location", "prop"],
                "description": "Asset type"
            },

            # Original extracted data from Agent 1
            "data": {
                "bsonType": "object",
                "required": ["description", "scenes", "importance"],
                "properties": {
                    "description": {
                        "bsonType": "string",
                        "minLength": 1,
                        "description": "Visual description of the asset"
                    },

                    # Character-specific fields (optional for locations/props)
                    "age_range": {
                        "bsonType": ["string", "null"],
                        "description": "Age range (e.g., '20-30') - for characters"
                    },
                    "gender": {
                        "bsonType": ["string", "null"],
                        "description": "Gender: male/female/non-binary/unspecified - for characters"
                    },
                    "key_features": {
                        "bsonType": ["array", "null"],
                        "items": {"bsonType": "string"},
                        "description": "List of distinctive visual features"
                    },
                    "clothing_style": {
                        "bsonType": ["string", "null"],
                        "description": "Description of typical outfit - for characters"
                    },
                    "role": {
                        "bsonType": ["string", "null"],
                        "description": "Character role: protagonist/antagonist/supporting - for characters"
                    },

                    # Common fields
                    "scenes": {
                        "bsonType": "array",
                        "items": {"bsonType": "string"},
                        "description": "List of scenes where asset appears"
                    },
                    "importance": {
                        "bsonType": "string",
                        "enum": ["critical", "important", "background"],
                        "description": "Importance level"
                    },

                    # Additional metadata (flexible)
                    "additional_metadata": {
                        "bsonType": ["object", "null"],
                        "description": "Any additional metadata from extraction"
                    }
                },
                "additionalProperties": True
            },

            # ============================================
            # IMAGE VERSIONS BY ANGLE (Arrays)
            # ============================================
            "master": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Master/main angle versions"
            },
            "close_up": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Close-up angle versions"
            },
            "profile_left": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Left profile angle versions"
            },
            "profile_right": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Right profile angle versions"
            },
            "back_shot": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Back shot angle versions"
            },
            "wide_shot": {
                "bsonType": "array",
                "items": IMAGE_VERSION_SCHEMA,
                "description": "Wide shot angle versions"
            },

            # Human approval
            "human_approved": {
                "bsonType": "bool",
                "description": "Whether approved by human reviewer"
            },
            "human_feedback": {
                "bsonType": ["string", "null"],
                "description": "Feedback from human reviewer"
            },
            "human_approved_at": {
                "bsonType": ["date", "null"],
                "description": "Human approval timestamp"
            },

            # Timestamps
            "created_at": {
                "bsonType": "date",
                "description": "Timestamp when asset was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Timestamp when asset was last updated"
            }
        },
        "additionalProperties": True
    }
}


def apply_schema(db: Any) -> None:
    """
    Apply schema validation to production_assets collection.

    Args:
        db: MongoDB database instance
    """
    collection_name = "production_assets"
    try:
        db.command("collMod", collection_name, validator=ASSETS_SCHEMA, validationLevel="moderate")
        logger.info(f"Schema applied to '{collection_name}'")
    except OperationFailure as e:
        if "NamespaceNotFound" in str(e) or "ns not found" in str(e):
            db.create_collection(collection_name, validator=ASSETS_SCHEMA, validationLevel="moderate")
            logger.info(f"Collection created with schema: '{collection_name}'")
        else:
            logger.error(f"Failed to apply schema to '{collection_name}': {e}")
