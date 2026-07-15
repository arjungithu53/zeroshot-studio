"""
Video Generation API Agent for Phase 3.

This agent handles video generation API calls to Gemini Veo 3.1.
It fetches the appropriate start image based on generation strategy and makes
API calls to generate videos.

Strategies:
- generate_new: Uses the generated image from prompt A (text-to-video)
- multi_shot: Uses the generated image from prompt B (image-to-video with first_frame)
- last_frame_seed: Uses last frame of previous shot (image-to-video with last_frame)

Based on BytePlus ARK SDK: https://ark.ap-
east.bytepluses.com/api/v3
"""

import os
import sys
import json
import base64
import requests
import logging
import time
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import cv2
import numpy as np

# Google GenAI SDK for Veo 3.1 / Omni Flash video generation
from google import genai as google_genai
from google.genai import types as google_genai_types
from io import BytesIO

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, '../..')
sys.path.insert(0, services_dir)

# Add infrastructure directory to path
root_dir = os.path.abspath(os.path.join(current_dir, '../../../../../../../'))
sys.path.insert(0, root_dir)

from app.services.shots_service import ShotsService
from infrastructure.s3.client import S3ClientFactory, S3Config
from infrastructure.s3.upload import upload_file
from backend.shared.utils.mongodb_validators import validate_object_id
from phase_3_agents.video_generation.video_model import VideoModel, VIDEO_MODEL_API_IDS, parse_video_model

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Both supported models are constrained to the same output shape.
TARGET_DURATION_SECONDS = 8
TARGET_ASPECT_RATIO = "9:16"


class VideoGenerationAPIAgent:
    """
    AI agent for making video generation API calls to Gemini Veo 3.1.

    This agent:
    1. Fetches the appropriate start image based on generation strategy
    2. Makes API calls to BytePlus ARK API
    3. Polls for video generation completion
    4. Saves results to MongoDB
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        video_model: Optional[VideoModel] = None,
        base_url: str = None,  # Unused — kept for API compatibility
        enable_saving: bool = True,
        enable_s3: bool = True,
        output_dir: str = "phase_3_agents/video_generation_output",
        generate_audio: bool = True
    ):
        """
        Initialize the Video Generation API Agent.

        Args:
            api_key: Google API key (optional, will use GOOGLE_API_KEY env var if not provided)
            model_id: Model ID string (optional). Must match a VideoModel value.
            video_model: VideoModel enum member (optional, takes priority over model_id).
                Falls back to VIDEO_MODEL_ID/VEO_MODEL_ID env vars, then VideoModel.veo_3_1.
            base_url: Unused — kept for backwards-compatible call signatures
            enable_saving: Whether to save results to files (default: True)
            enable_s3: Whether to upload videos to S3 (default: True)
            output_dir: Directory to save output files
            generate_audio: Unused — Veo generates audio by default
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if video_model is not None:
            self.video_model = video_model
        elif model_id is not None:
            self.video_model = parse_video_model(model_id)
        else:
            env_model_id = os.getenv("VIDEO_MODEL_ID") or os.getenv("VEO_MODEL_ID")
            self.video_model = parse_video_model(env_model_id) if env_model_id else VideoModel.veo_3_1
        # The real model string passed to the Google GenAI SDK (distinct from the
        # friendly VideoModel.value used in the API/Mongo layer).
        self.model_id = VIDEO_MODEL_API_IDS[self.video_model]

        logger.info("=" * 60)
        logger.info("VIDEO GENERATION API AGENT CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"🎯 SELECTED MODEL: {self.model_id}")
        logger.info("=" * 60)

        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize the Google GenAI client for whichever backend is selected.
        # Kept as separate attributes (rather than one shared client) since Veo uses the
        # long-running client.models.generate_videos()/operations API while Omni Flash uses
        # the client.interactions.create() API — different enough call shapes that keeping
        # them distinct avoids ambiguity about which surface is in use.
        self.veo_client = None
        self.omni_client = None
        if self.video_model == VideoModel.veo_3_1:
            self.veo_client = google_genai.Client(api_key=self.api_key)
        else:
            self.omni_client = google_genai.Client(api_key=self.api_key)

        # Pending Veo operation (set during submit, used during poll)
        self._pending_operation = None
        # Omni Flash result, captured synchronously at submit time (see _submit_omni_flash)
        self._pending_omni_result = None

        self.enable_saving = enable_saving
        self.output_dir = output_dir
        self.enable_s3 = enable_s3
        self.s3_client = None
        self.s3_bucket = None
        self.s3_region = None
        self.s3_endpoint_url = None

        # Create output directory if it doesn't exist
        if self.enable_saving:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize S3 if enabled
        if self.enable_s3:
            self._init_s3()

        logger.info(f"Initialized VideoGenerationAPIAgent with {self.video_model.name} (Model: {self.model_id})")

    def _init_s3(self) -> None:
        """Initialize S3 client from environment variables"""
        try:
            access_key = os.getenv("production_AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("production_AWS_SECRET_ACCESS_KEY")
            bucket = os.getenv("production_S3_BUCKET_NAME")
            region = os.getenv("production_AWS_REGION", "us-east-1")
            endpoint_url = os.getenv("production_S3_ENDPOINT_URL")  # For S3-compatible services

            if not all([access_key, secret_key, bucket]):
                logger.warning("⚠️  S3 credentials not found in environment, disabling S3 upload")
                self.enable_s3 = False
                return

            config = S3Config(
                access_key_id=access_key,
                secret_access_key=secret_key,
                bucket_name=bucket,
                region=region,
                endpoint_url=endpoint_url
            )

            factory = S3ClientFactory(config)
            self.s3_client = factory.get_client()
            self.s3_bucket = bucket
            self.s3_region = region
            self.s3_endpoint_url = endpoint_url

            logger.info(f"✅ S3 client initialized (bucket: {bucket})")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize S3 client: {e}")
            self.enable_s3 = False

    def _get_movie_folder(self, show_id: str) -> str:
        """
        Fetch movie title and ID from MongoDB to build the S3 folder segment.
        Returns "{title_slug}_{movie_id}", or "" if lookup fails.
        """
        try:
            import re as _re
            from backend.services.production.app.config import get_mongo_factory
            from backend.shared.utils.mongodb_validators import validate_object_id

            mongo_factory = get_mongo_factory()
            _, projects_col = mongo_factory.get_collection("production_projects")
            project = projects_col.find_one({"_id": validate_object_id(show_id)}, {"movie_id": 1})
            if not project or not project.get("movie_id"):
                return ""

            movie_id = project["movie_id"]
            _, movies_col = mongo_factory.get_collection("movies")
            movie = movies_col.find_one({"_id": movie_id}, {"title": 1})
            if not movie or not movie.get("title"):
                return ""

            title_slug = _re.sub(r'[^A-Za-z0-9]+', '_', movie["title"]).strip('_')
            return f"{title_slug}_{movie_id}"
        except Exception as e:
            logger.warning(f"Could not resolve movie folder for show {show_id}: {e}")
            return ""

    def _download_and_upload_video(self, video_url: str, shot_id: str, scene_number: Optional[int] = None, sequence_number: Optional[int] = None, version: int = 1, show_id: str = "") -> Dict[str, str]:
        """
        Download video from URL, save locally, and optionally upload to S3

        Args:
            video_url: URL of the video from Freepik API
            shot_id: Shot ID for naming (fallback when scene/sequence unavailable)
            scene_number: Scene number for S3 key naming
            sequence_number: Shot sequence number within scene for S3 key naming
            version: Version number (1-indexed) for S3 key naming

        Returns:
            Dict with 'local_path' and 's3_url' (if S3 enabled)
        """
        result = {}

        try:
            if scene_number is not None and sequence_number is not None:
                filename = f"scene_{scene_number}_shot_{sequence_number}_v{version}.mp4"
            else:
                filename = f"{shot_id}_v{version}.mp4"
            local_path = os.path.join(self.output_dir, filename)

            # Download video
            logger.info(f"Downloading video from: {video_url}")
            response = requests.get(video_url, timeout=120, stream=True)
            response.raise_for_status()

            # Save locally
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            result['local_path'] = local_path
            logger.info(f"✅ Video downloaded to: {local_path}")

            # Upload to S3 if enabled
            if self.enable_s3 and self.s3_client:
                try:
                    # Generate S3 key
                    movie_folder = self._get_movie_folder(show_id) if show_id else ""
                    prefix = f"phase3/{movie_folder}/generated_videos" if movie_folder else "phase3/generated_videos"
                    s3_key = f"{prefix}/{filename}"

                    # Upload to S3
                    s3_url = upload_file(
                        file_path=local_path,
                        s3_client=self.s3_client,
                        bucket_name=self.s3_bucket,
                        s3_key=s3_key,
                        content_type="video/mp4",
                        region=self.s3_region,
                        endpoint_url=self.s3_endpoint_url
                    )
                    result['s3_url'] = s3_url
                    logger.info(f"✅ Uploaded to S3: {s3_key}")
                except Exception as s3_error:
                    logger.warning(f"⚠️  S3 upload failed: {s3_error}")
                    # Continue without S3 - local file still available

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download video: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error downloading/uploading video: {str(e)}")
            return {}

    def _extract_last_frame(self, video_path: str, shot_id: str, scene_number: Optional[int] = None, sequence_number: Optional[int] = None, version: int = 1, show_id: str = "") -> Optional[str]:
        """
        Extract the last frame from a video file and upload to S3.

        Args:
            video_path: Local path to the video file
            shot_id: Shot ID for naming (fallback when scene/sequence unavailable)
            scene_number: Scene number for S3 key naming
            sequence_number: Shot sequence number within scene for S3 key naming
            version: Version number (1-indexed) for S3 key naming

        Returns:
            S3 URL of the extracted last frame, or None if failed
        """
        try:
            logger.info(f"Extracting last frame from video: {video_path}")

            # Open the video file
            video = cv2.VideoCapture(video_path)

            if not video.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return None

            # Get total frame count
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Total frames in video: {total_frames}")

            if total_frames == 0:
                logger.error("Video has 0 frames")
                video.release()
                return None

            # Set to the last frame
            video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

            # Read the last frame
            ret, frame = video.read()
            video.release()

            if not ret or frame is None:
                logger.error("Failed to read last frame")
                return None

            # Save frame to temporary file
            if scene_number is not None and sequence_number is not None:
                frame_name = f"scene_{scene_number}_shot_{sequence_number}_v{version}_last_frame"
            else:
                frame_name = f"{shot_id}_v{version}_last_frame"
            temp_frame_path = os.path.join(tempfile.gettempdir(), f"{frame_name}.jpg")

            # Write frame as JPEG
            cv2.imwrite(temp_frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            logger.info(f"✅ Last frame saved to: {temp_frame_path}")

            # Upload to S3 if enabled
            if self.enable_s3 and self.s3_client:
                try:
                    movie_folder = self._get_movie_folder(show_id) if show_id else ""
                    prefix = f"phase3/{movie_folder}/last_frames" if movie_folder else "phase3/last_frames"
                    s3_key = f"{prefix}/{frame_name}.jpg"

                    s3_url = upload_file(
                        file_path=temp_frame_path,
                        s3_client=self.s3_client,
                        bucket_name=self.s3_bucket,
                        s3_key=s3_key,
                        content_type="image/jpeg",
                        region=self.s3_region,
                        endpoint_url=self.s3_endpoint_url
                    )

                    logger.info(f"✅ Last frame uploaded to S3: {s3_url}")

                    # Clean up temp file
                    try:
                        os.remove(temp_frame_path)
                    except:
                        pass

                    return s3_url

                except Exception as s3_error:
                    logger.error(f"Failed to upload last frame to S3: {s3_error}")
                    return None
            else:
                logger.warning("S3 not enabled, cannot upload last frame")
                return None

        except Exception as e:
            logger.error(f"Error extracting last frame: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _fetch_image_from_shots(self, shot_id: str, show_id: str = None, episode_number: int = None, image_version: str = None) -> Optional[str]:
        """
        Fetch image URL from shots collection.

        Searches for image URLs in this priority order:
        1. generated_images_s3[0] - Images from phase 1/2
        2. image_s3_url - Direct image URL field
        3. Inside annotated_shots array (new structure)

        Args:
            shot_id: Shot ID to search for
            show_id: Optional Show ID to filter
            episode_number: Optional episode number to filter
            image_version: Optional version string (e.g., "v0", "v1", "v2"). If not provided, uses "v0" (latest is determined by highest version number if multiple exist)

        Returns:
            Image S3 URL if found, None otherwise
        """
        try:
            from backend.services.production.app.config import get_mongo_factory

            mongo_factory = get_mongo_factory()
            client, shots_collection = mongo_factory.get_collection("shots")

            try:
                # Use requested version; None means resolve to latest available per document
                version_to_use = image_version
                if version_to_use:
                    logger.info(f"Using requested image version: {version_to_use}")

                # Try to find in annotated_shots array first (new structure)
                query = {"annotated_shots.shot_id": shot_id}
                if show_id:
                    query["show_id"] = show_id
                if episode_number is not None:
                    query["episode_number"] = episode_number

                logger.info(f"Querying shots collection for shot {shot_id}")
                episode_doc = shots_collection.find_one(query)

                if episode_doc and "annotated_shots" in episode_doc:
                    # Find the specific shot in the annotated_shots array
                    for shot in episode_doc["annotated_shots"]:
                        if shot.get("shot_id") == shot_id:
                            # Priority 0: Human selection (most authoritative)
                            image_obj = shot.get("image", {})
                            if isinstance(image_obj, dict):
                                selected = image_obj.get("selected", {})
                                if isinstance(selected, dict) and selected.get("url"):
                                    url = selected["url"]
                                    logger.info(f"✅ Using human-selected image for shot {shot_id}: {url}")
                                    return url

                            # Priority 1: Check nested image.{version}.generated_images_s3 structure (new format)
                            image_obj = shot.get("image", {})
                            if isinstance(image_obj, dict):
                                # Resolve to latest version key when not explicitly requested
                                if version_to_use:
                                    v = version_to_use
                                else:
                                    v_keys = [k for k in image_obj.keys() if k.startswith('v') and k[1:].isdigit()]
                                    v = sorted(v_keys, key=lambda x: int(x[1:]))[-1] if v_keys else "v0"
                                    logger.info(f"Using image version: {v} (latest available)")
                                version_image = image_obj.get(v, {})
                                if isinstance(version_image, dict):
                                    generated_images = version_image.get("generated_images_s3", [])
                                    if generated_images and len(generated_images) > 0:
                                        image_url = generated_images[-1]
                                        logger.info(f"✅ Found image URL in shots collection (annotated_shots.image.{v}.generated_images_s3): {image_url}")
                                        return image_url
                                    else:
                                        logger.warning(f"❌ Version {v} exists but has no images in generated_images_s3")
                                else:
                                    logger.warning(f"❌ Version {v} not found in image object, available versions: {list(image_obj.keys())}")

                            # Priority 2: generated_images_s3 array (legacy format)
                            generated_images = shot.get("generated_images_s3", [])
                            if generated_images and len(generated_images) > 0:
                                image_url = generated_images[-1]
                                logger.info(f"✅ Found image URL in shots collection (annotated_shots.generated_images_s3): {image_url}")
                                return image_url

                            # Priority 3: Direct image_s3_url field
                            image_url = shot.get("image_s3_url")
                            if image_url:
                                logger.info(f"✅ Found image URL in shots collection (annotated_shots.image_s3_url): {image_url}")
                                return image_url

                # Fallback: Try standalone document (old structure)
                query = {"shot_id": shot_id}
                if show_id:
                    query["show_id"] = show_id
                if episode_number is not None:
                    query["episode_number"] = episode_number

                shot_doc = shots_collection.find_one(query)

                if shot_doc:
                    # Priority 0: Human selection (most authoritative)
                    image_obj = shot_doc.get("image", {})
                    if isinstance(image_obj, dict):
                        selected = image_obj.get("selected", {})
                        if isinstance(selected, dict) and selected.get("url"):
                            url = selected["url"]
                            logger.info(f"✅ Using human-selected image for shot {shot_id}: {url}")
                            return url

                    # Priority 1: Check nested image.{version}.generated_images_s3 structure (new format)
                    image_obj = shot_doc.get("image", {})
                    if isinstance(image_obj, dict):
                        # Resolve to latest version key when not explicitly requested
                        if version_to_use:
                            v = version_to_use
                        else:
                            v_keys = [k for k in image_obj.keys() if k.startswith('v') and k[1:].isdigit()]
                            v = sorted(v_keys, key=lambda x: int(x[1:]))[-1] if v_keys else "v0"
                            logger.info(f"Using image version: {v} (latest available)")
                        version_image = image_obj.get(v, {})
                        if isinstance(version_image, dict):
                            generated_images = version_image.get("generated_images_s3", [])
                            if generated_images and len(generated_images) > 0:
                                image_url = generated_images[-1]
                                logger.info(f"✅ Found image URL in shots collection (image.{v}.generated_images_s3): {image_url}")
                                return image_url
                            else:
                                logger.warning(f"❌ Version {v} exists but has no images in generated_images_s3")
                        else:
                            logger.warning(f"❌ Version {v} not found in image object, available versions: {list(image_obj.keys())}")

                    # Priority 2: generated_images_s3 array (legacy format)
                    generated_images = shot_doc.get("generated_images_s3", [])
                    if generated_images and len(generated_images) > 0:
                        image_url = generated_images[-1]
                        logger.info(f"✅ Found image URL in shots collection (generated_images_s3): {image_url}")
                        return image_url

                    # Priority 3: Direct image_s3_url field
                    image_url = shot_doc.get("image_s3_url")
                    if image_url:
                        logger.info(f"✅ Found image URL in shots collection (image_s3_url): {image_url}")
                        return image_url

                logger.warning(f"❌ No image found in shots collection for shot {shot_id}")
                return None

            finally:
                pass  # Don't close singleton client

        except Exception as e:
            logger.error(f"❌ Error fetching image from shots collection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _fetch_image_from_production_projects(self, shot_id: str, show_id: str = None) -> Optional[str]:
        """
        Fetch image URL from production_projects collection (BACKUP/FALLBACK).

        The image URL is stored in:
        agent_outputs.agent_15.output.reviews[].image_s3_url
        where each review has a shot_id that we match against.

        Args:
            shot_id: Shot ID to search for
            show_id: Optional Show ID to filter the project

        Returns:
            Image S3 URL if found, None otherwise
        """
        try:
            from backend.services.production.app.config import get_mongo_factory

            mongo_factory = get_mongo_factory()
            client, production_projects_collection = mongo_factory.get_collection("production_projects")

            try:
                # If show_id is provided, query by _id directly (more efficient)
                if show_id:
                    try:
                        # Validate and convert show_id to ObjectId
                        show_id_obj = validate_object_id(show_id)
                        project_doc = production_projects_collection.find_one({"_id": show_id_obj})

                        if not project_doc:
                            logger.error(f"❌ No project document found with _id: {show_id}")
                            return None

                        logger.info(f"✅ Found project document with _id: {show_id}")

                        # Navigate to agent15 output reviews (note: no underscore in agent15)
                        agent_outputs = project_doc.get("agent_outputs", {})
                        agent_15 = agent_outputs.get("agent15", {})
                        output = agent_15.get("output", {})
                        reviews = output.get("reviews", [])

                        logger.info(f"Found {len(reviews)} reviews in agent_15 output")

                        # Find the review matching this shot_id
                        for i, review in enumerate(reviews):
                            review_shot_id = review.get("shot_id")
                            if review_shot_id == shot_id:
                                image_url = review.get("image_s3_url")
                                if image_url:
                                    logger.info(f"✅ Found image URL for {shot_id} at review index {i}: {image_url}")
                                    return image_url
                                else:
                                    logger.warning(f"❌ Review found for {shot_id} but no image_s3_url")
                                    return None

                        logger.warning(f"❌ No review found for shot {shot_id} in {len(reviews)} reviews")
                        logger.info(f"Available shot_ids: {[r.get('shot_id') for r in reviews[:5]]}")

                    except ValueError as e:
                        logger.error(f"❌ Invalid show_id format: {e}")
                        return None
                    except Exception as e:
                        logger.error(f"❌ Error querying by _id: {e}")
                        return None
                else:
                    # Fallback: query by shot_id in reviews array (when show_id not provided)
                    logger.info(f"No show_id provided, querying by shot_id only")
                    query = {"agent_outputs.agent15.output.reviews.shot_id": shot_id}
                    logger.info(f"Querying production_projects with: {query}")

                    project_doc = production_projects_collection.find_one(query)

                    if not project_doc:
                        logger.error(f"❌ No project document found for shot {shot_id}")
                        return None

                    logger.info(f"✅ Found project document")

                    # Navigate to agent15 output reviews (note: no underscore in agent15)
                    agent_outputs = project_doc.get("agent_outputs", {})
                    agent_15 = agent_outputs.get("agent15", {})
                    output = agent_15.get("output", {})
                    reviews = output.get("reviews", [])

                    # Find the review matching this shot_id
                    for review in reviews:
                        if review.get("shot_id") == shot_id:
                            image_url = review.get("image_s3_url")
                            if image_url:
                                logger.info(f"✅ Found image URL in production_projects for {shot_id}")
                                return image_url
                            else:
                                logger.warning(f"❌ Review found for {shot_id} but no image_s3_url")
                                return None

                    logger.warning(f"❌ No review found for shot {shot_id} in agent_15 reviews")
                    return None

            finally:
                pass  # Don't close singleton client

        except Exception as e:
            logger.error(f"❌ Error fetching image from production_projects: {e}")
            return None

    def fetch_start_image_url(
        self,
        shot: Dict[str, Any],
        mongodb_client,
        image_version: str = None
    ) -> Optional[str]:
        """
        Fetch the appropriate start image URL based on generation strategy.

        Strategy logic:
        - generate_new: Use image from shots collection (primary), fallback to production_projects
        - multi_shot: Use image from shots collection (primary), fallback to production_projects
        - last_frame_seed: Fetch last frame of previous shot from MongoDB

        Args:
            shot: Shot document from MongoDB
            mongodb_client: MongoDB collection object or MongoDBAtlasClient instance
            image_version: Optional version string (e.g., "v0", "v1", "v2") to fetch specific image version

        Returns:
            S3 URL of the start image, or None if not found
        """
        shot_id = shot.get("shot_id", "Unknown")
        generation_strategy = shot.get("generation_strategy", "")

        logger.info(f"Fetching start image for shot {shot_id} with strategy: {generation_strategy}")
        if image_version:
            logger.info(f"Requested image version: {image_version}")

        try:
            if generation_strategy in ["generate_new", "multi_shot"]:
                # Get show_id and episode_number from shot data if available
                show_id = shot.get("show_id", "")
                episode_number = shot.get("episode_number")

                # PRIMARY: Try to fetch image URL from shots collection first
                logger.info(f"🔍 Trying shots collection first for shot {shot_id}")
                start_image_url = self._fetch_image_from_shots(shot_id, show_id, episode_number, image_version)

                if start_image_url:
                    logger.info(f"✅ Found start image in SHOTS collection for {shot_id}: {start_image_url}")
                    return start_image_url

                # FALLBACK: Try production_projects collection
                logger.info(f"🔄 Falling back to production_projects collection for shot {shot_id}")
                start_image_url = self._fetch_image_from_production_projects(shot_id, show_id)

                if start_image_url:
                    logger.info(f"✅ Found start image in production_PROJECTS (fallback) for {shot_id}: {start_image_url}")
                    return start_image_url
                else:
                    logger.warning(f"❌ No image found in both shots and production_projects for shot {shot_id}")
                    return None

            elif generation_strategy == "last_frame_seed":
                # Fetch last frame of previous shot
                seed_shot_id = shot.get("seed_shot_id")

                if not seed_shot_id:
                    logger.warning(f"❌ No seed_shot_id found for last_frame_seed strategy in shot {shot_id}")
                    return None

                # Fetch the seed shot scoped to this movie's show_id to avoid cross-movie contamination
                current_show_id = shot.get("show_id", "")
                seed_shot = self.fetch_seed_shot(seed_shot_id, mongodb_client, show_id=current_show_id)

                if not seed_shot:
                    logger.warning(f"❌ Seed shot {seed_shot_id} not found in MongoDB")
                    return None

                # Get the last frame from the seed shot's generated video
                # Check old structure first
                last_frame_url = seed_shot.get("generated_video_last_frame_s3")

                if last_frame_url:
                    logger.info(f"✅ Found last frame for seed shot {seed_shot_id}: {last_frame_url}")
                    return last_frame_url

                # Check new structure (video.v0/v1/v2.last_frame_s3)
                video_data = seed_shot.get("video")
                if video_data and isinstance(video_data, dict):
                    # Get the latest version (highest v number)
                    versions = [k for k in video_data.keys() if k.startswith('v')]
                    if versions:
                        # Sort versions by number
                        latest_version = sorted(versions, key=lambda x: int(x[1:]))[-1]
                        version_data = video_data.get(latest_version, {})

                        if isinstance(version_data, dict):
                            last_frame_url = version_data.get("last_frame_s3")
                            if last_frame_url:
                                logger.info(f"✅ Found last frame in {latest_version} for seed shot {seed_shot_id}: {last_frame_url}")
                                return last_frame_url

                # Last frame not available - try fallback strategies
                logger.warning(f"❌ No last frame found for seed shot {seed_shot_id}, trying fallbacks...")

                # Fallback 1: Get the start image that was used for the seed shot from production_projects
                # This is the best fallback since it's the image used to generate the seed shot video
                show_id = seed_shot.get("show_id", "")
                if show_id:
                    logger.info(f"Trying to get start image for seed shot from production_projects...")
                    seed_start_image = self._fetch_image_from_production_projects(seed_shot_id, show_id)
                    if seed_start_image:
                        logger.info(f"✅ Using seed shot's start image from production_projects: {seed_start_image}")
                        return seed_start_image

                # Fallback 2: use generated_images_s3 from seed shot if available
                seed_images = seed_shot.get("generated_images_s3", [])
                if seed_images and len(seed_images) > 0:
                    fallback_url = seed_images[-1]
                    logger.info(f"⚠️ Using fallback image from seed shot's generated_images_s3: {fallback_url}")
                    return fallback_url

                logger.error(f"❌ No suitable image found for last_frame_seed strategy")
                return None

            else:
                logger.warning(f"❌ Unknown generation strategy: {generation_strategy}")
                return None

        except Exception as e:
            logger.error(f"Error fetching start image for shot {shot_id}: {str(e)}")
            return None

    def fetch_seed_shot(
        self,
        seed_shot_id: str,
        mongodb_client,
        show_id: str = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch seed shot from MongoDB by shot_id.

        Searches in annotated_shots array first, then falls back to standalone document.

        Args:
            seed_shot_id: ID of the seed shot
            mongodb_client: MongoDB collection object or MongoDBAtlasClient instance
            show_id: Optional show_id to restrict search to the current movie's scenes.
                     Without this, a shot_id shared across movies returns the oldest document.

        Returns:
            Seed shot document or None if not found
        """
        try:
            # Handle both collection object and MongoDBAtlasClient
            # Check by type name since pymongo Collection also has attributes with collection names
            client_type = type(mongodb_client).__name__

            if client_type == 'MongoDBAtlasClient':
                # It's a MongoDBAtlasClient instance
                collection = mongodb_client.shots_collection
            else:
                # It's already a collection object (pymongo.collection.Collection)
                collection = mongodb_client

            # First, try to find in annotated_shots array (new structure)
            # Always filter by show_id when available so we don't bleed into other movies
            query: dict = {"annotated_shots.shot_id": seed_shot_id}
            if show_id:
                query["show_id"] = show_id
                logger.info(f"🔍 Fetching seed shot {seed_shot_id} with show_id filter: {show_id}")

            episode_doc = collection.find_one(query)

            if episode_doc and "annotated_shots" in episode_doc:
                # Find the specific shot in the annotated_shots array
                for shot in episode_doc["annotated_shots"]:
                    if shot.get("shot_id") == seed_shot_id:
                        # Add episode context to the shot (needed for fetching from production_projects)
                        shot_with_context = {
                            **shot,
                            "episode_id": episode_doc.get("_id"),
                            "show_id": episode_doc.get("show_id"),
                            "episode_number": episode_doc.get("episode_number")
                        }
                        logger.info(f"✅ Found seed shot {seed_shot_id} in annotated_shots array")
                        return shot_with_context

            # Fallback: try standalone document (old structure)
            standalone_query: dict = {"shot_id": seed_shot_id}
            if show_id:
                standalone_query["show_id"] = show_id
            seed_shot = collection.find_one(standalone_query)

            if seed_shot:
                logger.info(f"✅ Found seed shot {seed_shot_id} as standalone document")
                return seed_shot
            else:
                logger.warning(f"❌ Seed shot {seed_shot_id} not found in MongoDB")
                return None

        except Exception as e:
            logger.error(f"Error fetching seed shot {seed_shot_id}: {str(e)}")
            return None

    def get_video_prompt(self, shot: Dict[str, Any]) -> str:
        """
        Get the appropriate video prompt based on generation strategy.

        For generate_new/last_frame_seed: Use video_prompt_reviewed_A.updated_prompt
        For multi_shot: Use video_prompt_reviewed_B.updated_prompt

        Args:
            shot: Shot document from MongoDB

        Returns:
            Video prompt string
        """
        generation_strategy = shot.get("generation_strategy", "")

        if generation_strategy in ["generate_new", "last_frame_seed"]:
            # Use reviewed prompt A
            video_prompt_reviewed_A = shot.get("video_prompt_reviewed_A", {})
            prompt = video_prompt_reviewed_A.get("updated_prompt", "")

            if not prompt:
                # Fallback to draft prompt
                prompt = shot.get("prompt_video_draft", "")
                logger.warning(f"No reviewed prompt A found, using draft prompt")

        elif generation_strategy == "multi_shot":
            # Use reviewed prompt B
            video_prompt_reviewed_B = shot.get("video_prompt_reviewed_B", {})
            prompt = video_prompt_reviewed_B.get("updated_prompt", "")

            if not prompt:
                # Fallback to draft prompt
                prompt = shot.get("prompt_video_draft", "")
                logger.warning(f"No reviewed prompt B found, using draft prompt")

        else:
            # Fallback to draft prompt
            prompt = shot.get("prompt_video_draft", "")

        return prompt

    def create_video_generation_request(
        self,
        start_image_url: str,
        prompt: str,
        duration: int = 5,
        image_role: str = "first_frame"
    ) -> Dict[str, Any]:
        """
        Create the request payload for Gemini Veo 3.1.

        Based on official SDK format:
        - Uses 'content' array with type-based objects
        - Text object: {"type": "text", "text": "..."}
        - Image object: {"type": "image_url", "image_url": {"url": "..."}, "role": "first_frame" or "last_frame"}

        Args:
            start_image_url: URL of the start image
            prompt: Video generation prompt
            duration: Video duration in seconds (default: 5)
            image_role: Role of the image - "first_frame" or "last_frame" (default: "first_frame")

        Returns:
            Request payload dictionary
        """
        content = []

        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt
        })

        # Add image if provided (for image-to-video)
        if start_image_url:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": start_image_url
                },
                "role": image_role  # "first_frame" or "last_frame"
            })

        return {
            "model": self.model_id,
            "content": content
        }

    def submit_video_generation(
        self,
        start_image_url: str,
        prompt: str,
        duration: int = 5,
        image_role: str = "first_frame",
        generate_audio: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a video generation request to the configured backend (Veo 3.1 or Omni Flash).

        Args:
            start_image_url: URL of the start image (S3 or public URL)
            prompt: Video generation prompt
            duration: Video duration in seconds (passed as hint in prompt)
            image_role: Role of the image — "first_frame" or "last_frame" (informational)
            generate_audio: Unused — both backends generate audio automatically

        Returns:
            Dict with operation id and status, or None if failed
        """
        if self.video_model == VideoModel.omni_flash:
            return self._submit_omni_flash(start_image_url, prompt, duration, image_role)
        return self._submit_veo(start_image_url, prompt, duration, image_role)

    def _submit_veo(
        self,
        start_image_url: str,
        prompt: str,
        duration: int = 5,
        image_role: str = "first_frame",
    ) -> Optional[Dict[str, Any]]:
        """Submit a video generation request to Google Veo 3.1."""
        try:
            logger.info("=" * 60)
            logger.info("SUBMITTING VIDEO GENERATION REQUEST (Veo 3.1)")
            logger.info("=" * 60)
            logger.info(f"🎯 MODEL: {self.model_id}")
            logger.info(f"⏱️  Duration hint: {duration}s")
            logger.info(f"📝 Prompt: {prompt}")
            if start_image_url:
                logger.info(f"🖼️  Start image: {start_image_url[:100]}...")
            logger.info("=" * 60)

            # Build Veo request
            veo_kwargs: Dict[str, Any] = {
                "model": self.model_id,
                "prompt": prompt,
                "config": google_genai_types.GenerateVideosConfig(
                    aspect_ratio=TARGET_ASPECT_RATIO,
                    number_of_videos=1,
                ),
            }

            # Attach start image for image-to-video if provided
            if start_image_url:
                img_response = requests.get(start_image_url, timeout=60)
                img_response.raise_for_status()
                veo_kwargs["image"] = google_genai_types.Image(
                    image_bytes=img_response.content,
                    mime_type="image/jpeg"
                )

            operation = self.veo_client.models.generate_videos(**veo_kwargs)
            self._pending_operation = operation

            logger.info("=" * 60)
            logger.info(f"✅ VIDEO GENERATION REQUEST SUBMITTED SUCCESSFULLY")
            logger.info(f"🎯 Model used: {self.model_id}")
            logger.info(f"📋 Operation name: {operation.name}")
            logger.info("=" * 60)

            return {"id": operation.name, "status": "submitted"}

        except Exception as e:
            logger.error(f"Error submitting video generation request: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _submit_omni_flash(
        self,
        start_image_url: str,
        prompt: str,
        duration: int = 5,
        image_role: str = "first_frame",
    ) -> Optional[Dict[str, Any]]:
        """
        Submit a video generation request to Gemini Omni Flash.

        Requires google-genai >= 2.0.0: Google made a breaking change to the
        Interactions API wire schema (the previously-installed 1.66.0 raised
        "The legacy Interactions API schema is no longer supported" — see
        https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026).
        This is written against the upgraded SDK (2.10.0), verified directly
        against its real types (google.genai.interactions):
        - `generation_config.video_config.task` accepts
          "text_to_video"/"image_to_video"/"reference_to_video"/"edit"
          (interactions.Task).
        - `response_format` for video is a `VideoResponseFormatParam`:
          {"type": "video", "aspect_ratio": "9:16"|"16:9", "duration": <str>,
          "delivery": "inline"|"uri"}. `duration` is typed as a plain string;
          Google's APIs conventionally serialize protobuf Duration as "<n>s",
          which is what's used here — not independently confirmed against a
          live call, since the exact accepted string format isn't documented.
        - The response `Interaction` model declares `output_video` (an SDK
          convenience) and `steps` directly (both were undeclared/legacy in
          1.66.0) — handled in `_extract_omni_video_content`.
        The call is left non-streaming/non-background (synchronous).
        """
        try:
            logger.info("=" * 60)
            logger.info("SUBMITTING VIDEO GENERATION REQUEST (Omni Flash)")
            logger.info("=" * 60)
            logger.info(f"🎯 MODEL: {self.model_id}")
            logger.info(f"⏱️  Duration target: {duration}s")
            logger.info(f"📝 Prompt: {prompt}")
            if start_image_url:
                logger.info(f"🖼️  Start image: {start_image_url[:100]}...")
            logger.info("=" * 60)

            content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

            if start_image_url:
                img_response = requests.get(start_image_url, timeout=60)
                img_response.raise_for_status()
                content.append({
                    "type": "image",
                    "data": base64.b64encode(img_response.content).decode("utf-8"),
                    "mime_type": "image/jpeg",
                })

            interaction = self.omni_client.interactions.create(
                model=self.model_id,
                input=content,
                generation_config={
                    "video_config": {
                        "task": "image_to_video" if start_image_url else "text_to_video",
                    },
                },
                response_format={
                    "type": "video",
                    "aspect_ratio": TARGET_ASPECT_RATIO,
                    "duration": f"{TARGET_DURATION_SECONDS}s",
                    "delivery": "uri",
                },
            )
            self._pending_omni_result = interaction

            interaction_id = getattr(interaction, "id", None) or f"omni_{int(time.time())}"
            status = getattr(interaction, "status", None)

            logger.info("=" * 60)
            logger.info(f"✅ VIDEO GENERATION REQUEST SUBMITTED (Omni Flash)")
            logger.info(f"📋 Interaction id: {interaction_id} (status: {status})")
            logger.info("=" * 60)

            if status not in (None, "completed"):
                logger.error(f"❌ Omni Flash interaction did not complete synchronously (status: {status})")
                return {"id": interaction_id, "status": "failed", "error": f"interaction status: {status}"}

            return {"id": interaction_id, "status": "submitted"}

        except Exception as e:
            logger.error(f"Error submitting Omni Flash video generation request: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def check_video_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Check the status of a pending video generation request for the configured backend.

        Args:
            task_id: Operation/interaction id from submit_video_generation (used for logging only)

        Returns:
            Status response dictionary, or None if failed
        """
        if self.video_model == VideoModel.omni_flash:
            return self._check_omni_flash_status(task_id)
        return self._check_veo_status(task_id)

    def _check_veo_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Check the status of a Veo video generation operation."""
        try:
            if self._pending_operation is None:
                logger.error("No pending Veo operation to check")
                return None

            operation = self.veo_client.operations.get(self._pending_operation)
            self._pending_operation = operation  # Keep reference up-to-date

            status = "succeeded" if operation.done else "processing"
            result = {"id": task_id, "status": status}

            if operation.done and hasattr(operation, "error") and operation.error:
                result["status"] = "failed"
                result["error"] = str(operation.error)

            logger.info(f"📊 Veo operation status: {status}")
            return result

        except Exception as e:
            logger.error(f"Error checking Veo operation status: {str(e)}")
            return None

    def _check_omni_flash_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Check the status of an Omni Flash interaction.

        interactions.create() is treated as synchronous (see
        _submit_omni_flash), so the result is already available — this just maps
        the real Interaction.status field (google.genai.interactions.InteractionStatus:
        "in_progress", "requires_action", "completed", "failed", "cancelled",
        "incomplete", "budget_exceeded") onto this agent's status vocabulary.
        """
        if self._pending_omni_result is None:
            logger.error("No pending Omni Flash result to check")
            return None
        status = getattr(self._pending_omni_result, "status", None)
        if status == "completed":
            mapped = "succeeded"
        elif status in ("failed", "cancelled", "budget_exceeded"):
            mapped = "failed"
        else:
            mapped = "processing"
        return {"id": task_id, "status": mapped}

    def poll_for_completion(
        self,
        task_id: str,
        max_wait_time: int = 600,
        poll_interval: int = 10,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
        show_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Poll for video generation completion, download the result, and upload to S3.

        Args:
            task_id: Operation/interaction id (used for logging and file naming)
            max_wait_time: Maximum time to wait in seconds (default: 600 = 10 minutes)
            poll_interval: Time between polls in seconds (default: 10)

        Returns:
            Dict with status, video_url (S3 URL), and local_path, or None on failure/timeout
        """
        if self.video_model == VideoModel.omni_flash:
            return self._poll_omni_flash(
                task_id, scene_number=scene_number, sequence_number=sequence_number,
                version=version, show_id=show_id,
            )
        return self._poll_veo(
            task_id, max_wait_time=max_wait_time, poll_interval=poll_interval,
            scene_number=scene_number, sequence_number=sequence_number,
            version=version, show_id=show_id,
        )

    def _poll_veo(
        self,
        task_id: str,
        max_wait_time: int = 600,
        poll_interval: int = 10,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
        show_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Poll for Veo 3.1 video generation completion, download the result, and upload to S3."""
        if self._pending_operation is None:
            logger.error("No pending Veo operation to poll")
            return None

        start_time = time.time()
        operation = self._pending_operation
        logger.info(f"Polling Veo operation for completion (operation: {task_id})")

        while (time.time() - start_time) < max_wait_time:
            try:
                operation = self.veo_client.operations.get(operation)
                self._pending_operation = operation
            except Exception as e:
                logger.error(f"Error polling Veo operation: {e}")
                return None

            if operation.done:
                # Check for error
                if hasattr(operation, "error") and operation.error:
                    logger.error(f"❌ Veo generation failed: {operation.error}")
                    return {"status": "error", "message": str(operation.error)}

                logger.info("✅ Veo video generation completed!")
                try:
                    if not getattr(operation.response, "generated_videos", None):
                        rai_count = getattr(operation.response, "rai_media_filtered_count", 0) or 0
                        rai_reasons = getattr(operation.response, "rai_media_filtered_reasons", []) or []
                        if rai_count > 0:
                            logger.error(
                                f"❌ Veo RAI audio filter blocked video. "
                                f"Count: {rai_count}, Reasons: {rai_reasons}"
                            )
                            return {
                                "status": "rai_filtered",
                                "message": f"RAI audio filter: {rai_reasons}",
                                "rai_reasons": rai_reasons,
                            }
                        err = f"API completed but returned no video metadata. Response: {operation.response}"
                        logger.error(err)
                        return {"status": "error", "message": err}

                    video = operation.response.generated_videos[0]
                    # Download video bytes from Google Files API
                    self.veo_client.files.download(file=video.video)

                    if scene_number is not None and sequence_number is not None:
                        filename = f"scene_{scene_number}_shot_{sequence_number}_v{version}.mp4"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"veo_{task_id.replace('/', '_')}_{timestamp}.mp4"
                    local_path = os.path.join(self.output_dir, filename)
                    os.makedirs(self.output_dir, exist_ok=True)
                    video.video.save(local_path)
                    logger.info(f"✅ Video saved locally: {local_path}")

                    # Upload to S3
                    s3_url = None
                    if self.enable_s3 and self.s3_client:
                        try:
                            movie_folder = self._get_movie_folder(show_id) if show_id else ""
                            prefix = f"phase3/{movie_folder}/generated_videos" if movie_folder else "phase3/generated_videos"
                            s3_key = f"{prefix}/{filename}"
                            s3_url = upload_file(
                                file_path=local_path,
                                s3_client=self.s3_client,
                                bucket_name=self.s3_bucket,
                                s3_key=s3_key,
                                content_type="video/mp4",
                                region=self.s3_region,
                                endpoint_url=self.s3_endpoint_url
                            )
                            logger.info(f"✅ Video uploaded to S3: {s3_url}")
                        except Exception as s3_err:
                            logger.error(f"❌ S3 upload failed — no public video URL will be available: {s3_err}")

                    if not s3_url:
                        logger.error(
                            "❌ S3 upload did not produce a URL. "
                            "video_url will be None — downstream save will fail intentionally "
                            "rather than storing an unusable local path in MongoDB."
                        )

                    return {
                        "status": "success",
                        "video_url": s3_url,   # None when S3 unavailable — caller must treat as failure
                        "local_path": local_path,
                        "task_id": task_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                except Exception as e:
                    logger.error(f"Error extracting Veo video: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"status": "error", "message": str(e)}

            logger.info(f"⏳ Veo still generating, waiting {poll_interval}s...")
            time.sleep(poll_interval)

        logger.error(f"❌ Timeout waiting for Veo video generation (max: {max_wait_time}s)")
        return None

    def _poll_omni_flash(
        self,
        task_id: str,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
        show_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the result of an Omni Flash interaction, download it, and upload to S3.

        Since _submit_omni_flash treats interactions.create() as synchronous,
        the result is already available in self._pending_omni_result. The video
        block (found via _extract_omni_video_content — checks output_video,
        outputs, and steps) has either a `uri` or inline base64 `data` (see
        google.genai.interactions.VideoContent).

        IMPORTANT: `uri` is a Gemini Files API URI
        (generativelanguage.googleapis.com/v1beta/files/...), not a public URL —
        a plain unauthenticated `requests.get` gets a 403 Forbidden. It must be
        fetched via `self.omni_client.files.download(file=uri)` (the same
        authenticated mechanism Veo's `_poll_veo` uses via
        `self.veo_client.files.download(file=video.video)`), which returns raw
        bytes directly.
        """
        if self._pending_omni_result is None:
            logger.error("No pending Omni Flash result to poll")
            return None

        interaction = self._pending_omni_result

        try:
            video_content = self._extract_omni_video_content(interaction)
            if not video_content:
                err = f"Omni Flash interaction completed but no video output was found. outputs={getattr(interaction, 'outputs', None)}"
                logger.error(err)
                return {"status": "error", "message": err}

            def _field(obj, key):
                return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)

            video_uri = _field(video_content, "uri")
            video_data_b64 = _field(video_content, "data")

            logger.info("✅ Omni Flash video generation completed!")

            if video_uri:
                logger.info(f"Downloading video via Files API: {video_uri}")
                video_bytes = self.omni_client.files.download(file=video_uri)
            elif video_data_b64:
                video_bytes = base64.b64decode(video_data_b64)
            else:
                err = "Omni Flash video output has neither uri nor data"
                logger.error(err)
                return {"status": "error", "message": err}

            if scene_number is not None and sequence_number is not None:
                filename = f"scene_{scene_number}_shot_{sequence_number}_v{version}.mp4"
            else:
                filename = f"omni_{task_id}_v{version}.mp4"

            os.makedirs(self.output_dir, exist_ok=True)
            local_path = os.path.join(self.output_dir, filename)
            with open(local_path, "wb") as f:
                f.write(video_bytes)
            logger.info(f"✅ Video saved locally: {local_path}")

            s3_url = None
            if self.enable_s3 and self.s3_client:
                try:
                    movie_folder = self._get_movie_folder(show_id) if show_id else ""
                    prefix = f"phase3/{movie_folder}/generated_videos" if movie_folder else "phase3/generated_videos"
                    s3_key = f"{prefix}/{filename}"
                    s3_url = upload_file(
                        file_path=local_path,
                        s3_client=self.s3_client,
                        bucket_name=self.s3_bucket,
                        s3_key=s3_key,
                        content_type="video/mp4",
                        region=self.s3_region,
                        endpoint_url=self.s3_endpoint_url,
                    )
                    logger.info(f"✅ Video uploaded to S3: {s3_url}")
                except Exception as s3_err:
                    logger.error(f"❌ S3 upload failed — no public video URL will be available: {s3_err}")

            if not s3_url:
                logger.error(
                    "❌ S3 upload did not produce a URL. "
                    "video_url will be None — downstream save will fail intentionally "
                    "rather than storing an unusable local path in MongoDB."
                )

            return {
                "status": "success",
                "video_url": s3_url,
                "local_path": local_path,
                "task_id": task_id,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error extracting Omni Flash video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    @staticmethod
    def _extract_omni_video_content(interaction: Any) -> Optional[Any]:
        """
        Find the video content block in an Omni Flash interaction result.

        Checks each shape google.genai.interactions.Interaction (>= 2.0.0) can
        surface a video in, in order of convenience:

        1. `interaction.output_video` — SDK-computed convenience field holding
           the last model output's video content directly, when present.
        2. `interaction.outputs` — legacy field name from pre-2.0.0 SDKs; kept
           as a fallback in case an older response shape is ever returned.
        3. `interaction.steps` — the full step list (user_input/thought/
           model_output); scans the last "model_output" step's `content` for
           a video block. This is the field the API populates by default.
        """
        def _item_type(item):
            return item.get("type") if isinstance(item, dict) else getattr(item, "type", None)

        output_video = getattr(interaction, "output_video", None)
        if output_video is not None:
            return output_video

        outputs = getattr(interaction, "outputs", None) or []
        for item in outputs:
            if _item_type(item) == "video":
                return item

        steps = getattr(interaction, "steps", None) or []
        for step in reversed(steps):
            step_type = step.get("type") if isinstance(step, dict) else getattr(step, "type", None)
            if step_type != "model_output":
                continue
            step_content = step.get("content") if isinstance(step, dict) else getattr(step, "content", None)
            for item in (step_content or []):
                if _item_type(item) == "video":
                    return item

        return None

    def save_to_mongo(
        self,
        shot_id: str,
        show_id: str,
        episode_number: int,
        video_result: Dict[str, Any],
        mongodb_client,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
    ) -> bool:
        """
        Save video generation results to MongoDB.

        Updates the shot document with:
        - generated_video_url: URL of the generated video from Freepik
        - generated_video_s3_url: S3 URL of the downloaded video (if S3 enabled)
        - generated_video_local_path: Local path of the downloaded video
        - generated_video_id: Job ID from Freepik
        - generated_video_metadata: Full API response
        - generated_video_last_frame_s3: (Optional) URL of last frame for next shot

        Args:
            shot_id: ID of the shot
            show_id: Show ID
            episode_number: Episode number
            video_result: Video generation result from API
            mongodb_client: MongoDB collection object or MongoDBAtlasClient instance

        Returns:
            True if save was successful
        """
        try:
            # Handle both collection object and MongoDBAtlasClient
            # Check by type name since pymongo Collection also has attributes with collection names
            client_type = type(mongodb_client).__name__

            if client_type == 'MongoDBAtlasClient':
                # It's a MongoDBAtlasClient instance
                collection = mongodb_client.shots_collection
            else:
                # It's already a collection object (pymongo.collection.Collection)
                collection = mongodb_client

            filter_query = {
                "shot_id": shot_id,
                "show_id": show_id,
                "episode_number": episode_number
            }

            # Extract video URL from result
            video_url = video_result.get("data", {}).get("video_url") or video_result.get("video_url")
            job_id = video_result.get("job_id") or video_result.get("id")

            update_data = {
                "generated_video_url": video_url,
                "generated_video_id": job_id,
                "generated_video_metadata": video_result,
                "video_generation_timestamp": datetime.now().isoformat()
            }

            # Download and upload video to S3
            if video_url:
                logger.info(f"Downloading and uploading video for shot {shot_id}")
                download_result = self._download_and_upload_video(
                    video_url, shot_id,
                    scene_number=scene_number,
                    sequence_number=sequence_number,
                    version=version,
                    show_id=show_id,
                )

                if download_result.get('local_path'):
                    update_data['generated_video_local_path'] = download_result['local_path']
                    logger.info(f"✅ Video saved locally: {download_result['local_path']}")

                if download_result.get('s3_url'):
                    update_data['generated_video_s3_url'] = download_result['s3_url']
                    logger.info(f"✅ Video uploaded to S3: {download_result['s3_url']}")

                # Extract last frame from downloaded video
                if download_result.get('local_path'):
                    logger.info(f"Extracting last frame from video...")
                    last_frame_s3_url = self._extract_last_frame(
                        download_result['local_path'], shot_id,
                        scene_number=scene_number,
                        sequence_number=sequence_number,
                        version=version,
                        show_id=show_id,
                    )
                    if last_frame_s3_url:
                        update_data['generated_video_last_frame_s3'] = last_frame_s3_url
                        logger.info(f"✅ Last frame extracted and uploaded: {last_frame_s3_url}")
                    else:
                        logger.warning(f"⚠️  Failed to extract last frame")

            result = collection.update_one(
                filter_query,
                {"$set": update_data}
            )

            if result.matched_count > 0:
                logger.info(f"✅ Saved video generation results for shot {shot_id} to MongoDB")
                return True
            else:
                logger.warning(f"❌ Shot {shot_id} not found in MongoDB for update")
                return False

        except Exception as e:
            logger.error(f"Error saving video results to MongoDB for shot {shot_id}: {str(e)}")
            return False

    def save_to_file(
        self,
        shot_id: str,
        video_result: Dict[str, Any]
    ) -> str:
        """
        Save video generation result to local JSON file.

        Args:
            shot_id: Shot ID
            video_result: Video generation result

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_result_{shot_id}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(video_result, f, indent=2, ensure_ascii=False)

        logger.info(f"Video result saved to: {filepath}")
        return filepath

    def generate_video(
        self,
        shot: Dict[str, Any],
        video_prompt: str,
        start_image_url: str,
        mongodb_client = None,
        duration: int = 5,
        poll_for_result: bool = True,
        generate_audio: Optional[bool] = None,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
        show_id: str = "",
    ) -> Dict[str, Any]:
        """
        Generate video with explicit prompt and image URL.

        This is a simplified version for workflows that already have
        the prompt and image URL (e.g., Phase 3 workflow with AI suggestions).

        Args:
            shot: Shot document
            video_prompt: Video generation prompt
            start_image_url: S3 URL of start image
            mongodb_client: MongoDB client (optional, not used in this method)
            duration: Video duration in seconds (default: 5)
            poll_for_result: Whether to poll for completion (default: True)
            generate_audio: Whether to generate audio (default: use instance setting)

        Returns:
            Result dictionary with video_url and task_id
        """
        shot_id = shot.get("shot_id", "Unknown")
        logger.info(f"Generating video for shot {shot_id}")
        logger.info(f"Prompt length: {len(video_prompt) if video_prompt else 0} chars")
        logger.info(f"Prompt: {video_prompt[:200] if video_prompt else 'EMPTY'}")

        try:
            # Submit video generation request
            submission_result = self.submit_video_generation(
                start_image_url=start_image_url,
                prompt=video_prompt,
                duration=duration,
                generate_audio=generate_audio
            )

            if not submission_result:
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "Failed to submit video generation request"
                }

            # Extract task_id from response (Freepik returns it in data.task_id)
            task_id = (submission_result.get("data", {}).get("task_id") or
                      submission_result.get("job_id") or
                      submission_result.get("id") or
                      submission_result.get("task_id"))

            if not task_id:
                logger.error("No task_id in submission result")
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "No task_id returned from API",
                    "submission_result": submission_result
                }

            logger.info(f"Video generation submitted, task_id: {task_id}")

            # Poll for completion if requested
            if poll_for_result:
                logger.info("Polling for video completion...")
                poll_result = self.poll_for_completion(
                    task_id,
                    scene_number=scene_number,
                    sequence_number=sequence_number,
                    version=version,
                    show_id=show_id,
                )

                if poll_result and poll_result.get("status") == "success":
                    video_url = poll_result.get("video_url")
                    if not video_url:
                        logger.error("Video generation succeeded but no S3 URL is available (S3 upload failed). Treating as error.")
                        return {
                            "status": "error",
                            "shot_id": shot_id,
                            "message": "Video generated but S3 upload failed — no public URL available",
                            "task_id": task_id,
                            "local_path": poll_result.get("local_path"),
                        }
                    last_frame_url = None

                    # poll_for_completion already handles S3 upload for Veo;
                    # extract last frame from the local file it saved.
                    local_path = poll_result.get("local_path")
                    if local_path and os.path.exists(local_path):
                        logger.info("Extracting last frame from Veo video...")
                        last_frame_url = self._extract_last_frame(
                            local_path, shot_id,
                            scene_number=scene_number,
                            sequence_number=sequence_number,
                            version=version,
                            show_id=show_id,
                        )
                        if last_frame_url:
                            logger.info(f"✅ Last frame extracted and uploaded: {last_frame_url}")
                        else:
                            logger.warning("⚠️  Failed to extract last frame")

                    result_data = {
                        "status": "success",
                        "video_url": video_url,
                        "task_id": task_id,
                        "timestamp": poll_result.get("timestamp")
                    }

                    if last_frame_url:
                        result_data["last_frame_url"] = last_frame_url

                    return result_data
                else:
                    return {
                        "status": "error",
                        "shot_id": shot_id,
                        "message": "Video generation failed or timed out",
                        "task_id": task_id
                    }
            else:
                return {
                    "status": "submitted",
                    "shot_id": shot_id,
                    "task_id": task_id,
                    "message": "Video generation submitted, polling disabled"
                }

        except Exception as e:
            logger.error(f"Error generating video for shot {shot_id}: {e}")
            return {
                "status": "error",
                "shot_id": shot_id,
                "message": str(e)
            }

    def generate_video_for_shot(
        self,
        shot: Dict[str, Any],
        mongodb_client: ShotsService,
        duration: int = 5,
        poll_for_result: bool = True,
        generate_audio: Optional[bool] = None,
        scene_number: Optional[int] = None,
        sequence_number: Optional[int] = None,
        version: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate video for a single shot.

        Complete workflow:
        1. Fetch start image based on strategy
        2. Get video prompt
        3. Submit video generation request
        4. Poll for completion (optional)
        5. Save results to MongoDB

        Args:
            shot: Shot document from MongoDB
            mongodb_client: MongoDB client instance
            duration: Video duration in seconds (default: 5)
            poll_for_result: Whether to poll for completion (default: True)
            generate_audio: Whether to generate audio (default: use instance setting)

        Returns:
            Result dictionary with status and data
        """
        shot_id = shot.get("shot_id", "Unknown")
        show_id = shot.get("show_id", "")
        episode_number = shot.get("episode_number", 0)
        generation_strategy = shot.get("generation_strategy", "")

        logger.info(f"Starting video generation for shot {shot_id} (strategy: {generation_strategy})")

        try:
            # Step 1: Fetch start image
            start_image_url = self.fetch_start_image_url(shot, mongodb_client)

            if not start_image_url:
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "Failed to fetch start image URL",
                    "error": "No start image URL found"
                }

            # Step 2: Get video prompt
            video_prompt = self.get_video_prompt(shot)

            if not video_prompt:
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "No video prompt found",
                    "error": "Missing video prompt"
                }

            logger.info(f"Using prompt: {video_prompt[:100]}...")

            # Step 3: Submit video generation request
            submission_result = self.submit_video_generation(
                start_image_url=start_image_url,
                prompt=video_prompt,
                duration=duration,
                generate_audio=generate_audio
            )

            if not submission_result:
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "Failed to submit video generation request",
                    "error": "API submission failed"
                }

            # Extract task_id from response (Freepik returns it in data.task_id)
            job_id = (submission_result.get("data", {}).get("task_id") or
                     submission_result.get("job_id") or
                     submission_result.get("id") or
                     submission_result.get("task_id"))

            if not job_id:
                logger.error("No task_id/job_id in submission result")
                return {
                    "status": "error",
                    "shot_id": shot_id,
                    "message": "No job_id returned from API",
                    "error": "Missing job_id",
                    "submission_result": submission_result
                }

            logger.info(f"Video generation submitted, job_id: {job_id}")

            # Step 4: Poll for completion (optional)
            final_result = submission_result

            if poll_for_result:
                logger.info("Polling for video completion...")
                poll_result = self.poll_for_completion(
                    job_id,
                    scene_number=scene_number,
                    sequence_number=sequence_number,
                    version=version,
                    show_id=show_id,
                )

                if poll_result:
                    final_result = poll_result
                else:
                    logger.warning("Polling failed or timed out, using submission result")

            # Step 5: Save results to MongoDB
            if self.save_to_mongo(shot_id, show_id, episode_number, final_result, mongodb_client,
                                  scene_number=scene_number, sequence_number=sequence_number, version=version):
                logger.info(f"✅ Saved results to MongoDB")

            # Step 6: Save to local file (optional)
            if self.enable_saving:
                local_file = self.save_to_file(shot_id, final_result)
            else:
                local_file = None

            logger.info(f"✅ Video generation completed for shot {shot_id}")

            return {
                "status": "success",
                "shot_id": shot_id,
                "job_id": job_id,
                "video_result": final_result,
                "local_file": local_file,
                "message": f"Video generation {'completed' if poll_for_result else 'submitted'} successfully"
            }

        except Exception as e:
            logger.error(f"Error generating video for shot {shot_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "status": "error",
                "shot_id": shot_id,
                "message": f"Error generating video: {str(e)}",
                "error": str(e)
            }

    def generate_videos_for_episode(
        self,
        show_id: str,
        episode_number: int,
        mongodb_client: ShotsService,
        duration: int = 5,
        poll_for_result: bool = True,
        filter_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate videos for all shots in an episode.

        Args:
            show_id: Show ID
            episode_number: Episode number
            mongodb_client: MongoDB client instance
            duration: Video duration in seconds (default: 5)
            poll_for_result: Whether to poll for completion (default: True)
            filter_strategy: Optional strategy filter ('generate_new', 'multi_shot', 'last_frame_seed')

        Returns:
            Summary dictionary with status and results
        """
        logger.info(f"Starting video generation for episode {show_id} - {episode_number}")

        try:
            # Fetch all shots for the episode
            shots = mongodb_client.get_shots_by_episode(show_id, episode_number)

            if not shots:
                return {
                    "status": "error",
                    "message": f"No shots found for show {show_id}, episode {episode_number}",
                    "videos_generated": 0
                }

            # Filter by strategy if specified
            if filter_strategy:
                shots = [s for s in shots if s.get("generation_strategy") == filter_strategy]
                logger.info(f"Filtered to {len(shots)} shots with strategy: {filter_strategy}")

            # Generate videos for each shot
            results = []
            success_count = 0
            error_count = 0

            for shot in shots:
                result = self.generate_video_for_shot(
                    shot=shot,
                    mongodb_client=mongodb_client,
                    duration=duration,
                    poll_for_result=poll_for_result
                )

                results.append(result)

                if result.get("status") == "success":
                    success_count += 1
                else:
                    error_count += 1

            logger.info(f"✅ Video generation completed: {success_count} success, {error_count} errors")

            return {
                "status": "success",
                "message": f"Processed {len(shots)} shots",
                "total_shots": len(shots),
                "success_count": success_count,
                "error_count": error_count,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error generating videos for episode: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "status": "error",
                "message": f"Error generating videos: {str(e)}",
                "videos_generated": 0
            }


# Convenience functions

def generate_video_for_shot(
    shot: Dict[str, Any],
    mongodb_client: ShotsService,
    api_key: Optional[str] = None,
    duration: int = 5,
    poll_for_result: bool = True,
    generate_audio: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to generate video for a single shot.

    Args:
        shot: Shot document from MongoDB
        mongodb_client: MongoDB client instance
        api_key: Freepik API key (optional)
        duration: Video duration in seconds
        poll_for_result: Whether to poll for completion
        generate_audio: Whether to generate audio (default: True)

    Returns:
        Result dictionary
    """
    agent = VideoGenerationAPIAgent(api_key=api_key, generate_audio=generate_audio)
    return agent.generate_video_for_shot(
        shot=shot,
        mongodb_client=mongodb_client,
        duration=duration,
        poll_for_result=poll_for_result
    )


def generate_videos_for_episode(
    show_id: str,
    episode_number: int,
    mongodb_client: ShotsService,
    api_key: Optional[str] = None,
    duration: int = 5,
    poll_for_result: bool = True,
    filter_strategy: Optional[str] = None,
    generate_audio: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to generate videos for all shots in an episode.

    Args:
        show_id: Show ID
        episode_number: Episode number
        mongodb_client: MongoDB client instance
        api_key: Freepik API key (optional)
        duration: Video duration in seconds
        poll_for_result: Whether to poll for completion
        filter_strategy: Optional strategy filter
        generate_audio: Whether to generate audio (default: True)

    Returns:
        Summary dictionary
    """
    agent = VideoGenerationAPIAgent(api_key=api_key, generate_audio=generate_audio)
    return agent.generate_videos_for_episode(
        show_id=show_id,
        episode_number=episode_number,
        mongodb_client=mongodb_client,
        duration=duration,
        poll_for_result=poll_for_result,
        filter_strategy=filter_strategy
    )
