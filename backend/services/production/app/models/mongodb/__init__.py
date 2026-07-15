"""
MongoDB schemas for production service (Phase 1: Image-to-Video Pipeline).

Contains MongoDB validation schemas for collections.
"""

from .projects import PROJECTS_SCHEMA, apply_schema as apply_projects_schema
from .assets import ASSETS_SCHEMA, apply_schema as apply_assets_schema
from .movies import MOVIES_SCHEMA, apply_schema as apply_movies_schema
from .assets_collections import ASSETS_COLLECTIONS_SCHEMA, apply_schema as apply_assets_collections_schema

__all__ = [
    "PROJECTS_SCHEMA",
    "ASSETS_SCHEMA",
    "MOVIES_SCHEMA",
    "ASSETS_COLLECTIONS_SCHEMA",
    "apply_projects_schema",
    "apply_assets_schema",
    "apply_movies_schema",
    "apply_assets_collections_schema",
]
