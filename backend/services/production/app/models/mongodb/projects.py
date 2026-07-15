"""Projects document schema for MongoDB validation (production Service - Phase 1)."""

from pymongo.errors import OperationFailure
from typing import Any

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


# Agent output schema structure (reusable for all 7 agents)
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
            "description": "Agent output data (flexible structure)"
        },
        "outputs": {
            "bsonType": ["array", "null"],
            "description": "Array of outputs for multi-shot agents (Phase 3)"
        },
        "error": {
            "bsonType": ["string", "null"],
            "description": "Error message if agent failed"
        },
        "executed_at": {
            "bsonType": ["date", "null"],
            "description": "Timestamp when agent was executed"
        },
        "description": {
            "bsonType": ["string", "null"],
            "description": "Human-readable description of agent purpose"
        }
    },
    "additionalProperties": False
}

PROJECTS_SCHEMA = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "status", "created_at", "updated_at"],
        "properties": {
            "_id": {
                "bsonType": "objectId",
                "description": "Auto-generated MongoDB ObjectId"
            },
            "name": {
                "bsonType": "string",
                "minLength": 1,
                "description": "Project name"
            },
            "script": {
                "bsonType": ["string", "null"],
                "minLength": 1,
                "description": "Full script text content"
            },
            "status": {
                "bsonType": "string",
                "enum": ["draft", "pending", "extracting", "prompting", "generating", "reviewing", "completed", "failed"],
                "description": "Project pipeline status"
            },

            # NEW FIELDS FOR MOVIE WORKFLOW
            "movie_id": {
                "bsonType": ["objectId", "null"],
                "description": "Reference to movies collection (for movie-based projects)"
            },
            "assets_collection_id": {
                "bsonType": ["objectId", "null"],
                "description": "Reference to assets_collections document (shared Phase 1 outputs for movie)"
            },
            "scene_number": {
                "bsonType": ["int", "null"],
                "description": "Scene number within the movie (for movie-based projects)"
            },
            "scene_script_s3_url": {
                "bsonType": ["string", "null"],
                "description": "S3 URL for the scene script text file"
            },
            "shotlist_json_s3_url": {
                "bsonType": ["string", "null"],
                "description": "S3 URL for the shotlist JSON file"
            },
            "product_image_s3_url": {
                "bsonType": ["string", "null"],
                "description": "S3 URL of the uploaded product image for product shots"
            },

            # Agent outputs - nested objects for Phase 1 (agents 1-8) and Phase 2 (agents 12-13)
            "agent_outputs": {
                "bsonType": ["object", "null"],
                "properties": {
                    "agent1": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Asset Generator (extracts characters, locations, props from script)"
                    },
                    "agent2": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Asset Reviewer (reviews and enhances extracted assets)"
                    },
                    "agent3": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Prompt Generator (creates initial image generation prompts)"
                    },
                    "agent4": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Prompt Optimizer (refines and optimizes prompts)"
                    },
                    "agent5": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image Generator (generates images from prompts)"
                    },
                    "agent6": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image Reviewer (reviews generated images for quality)"
                    },
                    "agent7": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image Editor (edits/refines images based on feedback)"
                    },
                    "agent8": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Variation Generator (generates camera angle variations)"
                    },
                    "agent9": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image Prompt Generator (Phase 2 - generates v0 prompts for shots)"
                    },
                    "agent10": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Prompt Review Agent (Phase 2 - reviews and refines prompts to v1)"
                    },
                    "agent12": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Shot Design Agent (analyzes shots and selects assets for composition)"
                    },
                    "agent13": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Prompt Modifier Agent (analyzes warnings and corrects prompts)"
                    },
                    "agent14": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Imagen Generator Agent (generates images using Vertex AI Imagen 4.0)"
                    },
                    "agent15": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image Reviewer Agent (reviews generated images using Gemini vision API)"
                    },
                    "agent14_regenerate": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Image regeneration for failed shots"
                    },
                    "agent15A": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Prompt regeneration agent for review feedback"
                    },
                    "agent17": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Video prompt generation"
                    },
                    "agent18": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "Video generation"
                    },
                    "agent19": {
                        **AGENT_OUTPUT_SCHEMA,
                        "description": "AI video review"
                    }
                },
                "additionalProperties": False,
                "description": "Outputs from Phase 1 (agents 1-8), Phase 2 (agents 12-15), and Phase 3 (agents 17-19) agents"
            },

            "created_at": {
                "bsonType": "date",
                "description": "Timestamp when project was created"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "Timestamp when project was last updated"
            }
        },
        "additionalProperties": True
    }
}


def apply_schema(db: Any) -> None:
    """
    Apply schema validation to production_projects collection.

    Args:
        db: MongoDB database instance
    """
    collection_name = "production_projects"
    try:
        db.command("collMod", collection_name, validator=PROJECTS_SCHEMA, validationLevel="moderate")
        logger.info(f"Schema applied to '{collection_name}'")
    except OperationFailure as e:
        if "NamespaceNotFound" in str(e) or "ns not found" in str(e):
            db.create_collection(collection_name, validator=PROJECTS_SCHEMA, validationLevel="moderate")
            logger.info(f"Collection created with schema: '{collection_name}'")
        else:
            logger.error(f"Failed to apply schema to '{collection_name}': {e}")
