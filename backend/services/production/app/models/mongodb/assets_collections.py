"""Assets Collections document schema for MongoDB validation (production Service - Phase 1 Movie-Level Assets)."""

from pymongo.errors import OperationFailure
from typing import Any

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Agent output schema structure (reusable for Phase 1 agents 1-8)
AGENT_OUTPUT_SCHEMA = {
    "bsonType": ["object", "null"],
    "properties": {
        "status": {
            "bsonType": ["string", "null"],
            "enum": ["pending", "running", "completed", "failed", "skipped", None],
            "description": "Agent execution status"
        },
        "output": {
            "bsonType": ["object", "null"],
            "description": "Agent output data (flexible structure - contains extracted_assets, enhanced_assets, generated_prompts, etc.)"
        },
        "error": {
            "bsonType": ["string", "null"],
            "description": "Error message if agent failed"
        },
        "executed_at": {
            "bsonType": ["date", "null"],
            "description": "Timestamp when agent was executed"
        }
    },
    "additionalProperties": False
}


ASSETS_COLLECTIONS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["movie_id", "created_at", "updated_at"],
        "properties": {
            "_id": {
                "bsonType": "objectId",
                "description": "Auto-generated MongoDB ObjectId"
            },
            "movie_id": {
                "bsonType": "objectId",
                "description": "Reference to movies collection document"
            },

            # Phase 1 Agent Outputs (1-8) - Movie-level shared assets
            "agent1_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Asset Generator - Extracts characters, locations, props from entire movie script"
            },
            "agent2_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Asset Reviewer - Reviews and enhances extracted assets"
            },
            "agent3_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Prompt Generator - Creates initial image generation prompts for all assets"
            },
            "agent4_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Prompt Optimizer - Refines and optimizes prompts for all assets"
            },
            "agent5_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Image Generator - Generates master images for all assets"
            },
            "agent6_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Image Reviewer - Reviews generated images for quality"
            },
            "agent7_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Image Editor - Edits/refines images based on feedback"
            },
            "agent8_output": {
                **AGENT_OUTPUT_SCHEMA,
                "description": "Variation Generator - Generates camera angle variations for all assets"
            },

            # Human approval tracking
            "approved_assets_list": {
                "bsonType": ["array", "null"],
                "description": "List of approved asset IDs during human checkpoint",
                "items": {
                    "bsonType": "string"
                }
            },
            "checkpoint_approved": {
                "bsonType": ["bool", "null"],
                "description": "Whether the checkpoint has been finalized and approved"
            },
            "human_approval_feedback": {
                "bsonType": ["object", "null"],
                "description": "Human approval feedback data including global and asset-specific feedback"
            },
            "approval_timestamp": {
                "bsonType": ["date", "null"],
                "description": "Timestamp when assets were approved"
            },

            "created_at": {
                "bsonType": "date",
                "description": "Timestamp when assets collection was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Timestamp when assets collection was last updated"
            }
        },
        "additionalProperties": True
    }
}


def apply_schema(db: Any) -> None:
    """
    Apply schema validation to assets_collections collection.

    Args:
        db: MongoDB database instance
    """
    collection_name = "assets_collections"
    try:
        db.command("collMod", collection_name, validator=ASSETS_COLLECTIONS_SCHEMA, validationLevel="moderate")
        logger.info(f"Schema applied to '{collection_name}'")
    except OperationFailure as e:
        if "NamespaceNotFound" in str(e) or "ns not found" in str(e):
            db.create_collection(collection_name, validator=ASSETS_COLLECTIONS_SCHEMA, validationLevel="moderate")
            logger.info(f"Collection created with schema: '{collection_name}'")

            # Create indexes
            db[collection_name].create_index("movie_id")
            db[collection_name].create_index("created_at")
            logger.info(f"Indexes created for '{collection_name}'")
        else:
            logger.error(f"Failed to apply schema to '{collection_name}': {e}")
