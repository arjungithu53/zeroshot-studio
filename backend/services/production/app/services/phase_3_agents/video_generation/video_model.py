"""
Video generation model selection.

Defines the set of supported video-generation backends and a helper to
resolve which one a movie was configured with (stored in
movies.global_settings.video_model, set once at pipeline-start time via
POST /api/v1/master/run-pipeline).
"""

import logging
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class VideoModel(str, Enum):
    """Supported video generation backends (friendly display values)."""
    veo_3_1     = "Veo 3.1"
    omni_flash  = "Omni Flash"


# Real model IDs passed to the Google GenAI SDK — kept separate from the
# friendly enum values above so API requests/Mongo documents can stay
# human-readable while the actual SDK calls still use the exact model string
# Google expects.
VIDEO_MODEL_API_IDS: Dict[VideoModel, str] = {
    VideoModel.veo_3_1: "veo-3.1-generate-preview",
    VideoModel.omni_flash: "gemini-omni-flash-preview",
}

_API_ID_TO_VIDEO_MODEL: Dict[str, VideoModel] = {
    api_id: model for model, api_id in VIDEO_MODEL_API_IDS.items()
}


def parse_video_model(value: str) -> VideoModel:
    """
    Resolve a VideoModel from either its friendly value ("Veo 3.1") or the
    raw API model ID ("veo-3.1-generate-preview") — accepting both keeps
    older callers/env vars that reference the raw ID working.

    Raises ValueError if the value matches neither.
    """
    try:
        return VideoModel(value)
    except ValueError:
        pass
    if value in _API_ID_TO_VIDEO_MODEL:
        return _API_ID_TO_VIDEO_MODEL[value]
    raise ValueError(f"'{value}' is not a valid VideoModel")


def resolve_video_model(show_id: str) -> Optional[VideoModel]:
    """
    Resolve the video generation model configured for a movie, given a shot's show_id.

    Mirrors the show_id -> production_projects.movie_id -> movies two-hop lookup
    used elsewhere in Phase 3 (see VideoGenerationAPIAgent._get_movie_folder).

    Returns None if no show_id is given or the setting can't be resolved — callers
    should fall back to their own default in that case.
    """
    if not show_id:
        return None

    try:
        from backend.services.production.app.config import get_mongo_factory
        from backend.shared.utils.mongodb_validators import validate_object_id

        mongo_factory = get_mongo_factory()
        _, projects_col = mongo_factory.get_collection("production_projects")
        project = projects_col.find_one({"_id": validate_object_id(show_id)}, {"movie_id": 1})
        if not project or not project.get("movie_id"):
            return None

        movie_id = project["movie_id"]
        _, movies_col = mongo_factory.get_collection("movies")
        movie = movies_col.find_one({"_id": movie_id}, {"global_settings.video_model": 1})
        if not movie:
            return None

        video_model_value = (movie.get("global_settings") or {}).get("video_model")
        if not video_model_value:
            return None

        return parse_video_model(video_model_value)
    except Exception as e:
        logger.warning(f"Could not resolve video_model for show {show_id}: {e}")
        return None
