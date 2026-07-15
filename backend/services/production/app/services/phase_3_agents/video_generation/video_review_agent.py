"""
Video Review Agent for Phase 3.

This agent reviews generated videos using Gemini's multimodal capabilities with structured output.
It analyzes both the video and the prompt to provide structured feedback and
prompt refinement suggestions.

Inputs:
- Generated video URL
- Original video generation prompt
- Shot metadata

Outputs:
- Structured review with scores (using Pydantic models)
- Decision: approved/refine_prompt/regenerate
- Refined prompt suggestions if needed
- Saves to MongoDB
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, '../..')
sys.path.insert(0, services_dir)

from app.services.shots_service import ShotsService

# Type alias for backward compatibility
MongoDBAtlasClient = ShotsService
from phase_3_agents.video_generation.video_review_models import VideoReviewResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_review_to_file(
    review_data: Dict[str, Any],
    show_id: str,
    episode_number: int,
    output_dir: str = "phase_3_agents/video_review_output"
) -> str:
    """
    Save video review results to a JSON file with timestamp.

    Args:
        review_data: Dictionary containing review results
        show_id: Show ID for filename
        episode_number: Episode number for filename
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_review_{show_id}_{episode_number}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Video review saved to: {filepath}")
    return filepath


class VideoReviewAgent:
    """
    AI agent for reviewing generated videos using Gemini's multimodal capabilities.

    This agent:
    1. Downloads/fetches the generated video
    2. Reviews video quality against the prompt
    3. Provides structured feedback with scores (using Pydantic schema)
    4. Suggests prompt refinements if needed
    5. Saves results to MongoDB
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.1-pro-preview",
        enable_saving: bool = True,
        output_dir: str = "phase_3_agents/video_review_output"
    ):
        """
        Initialize the Video Review Agent.

        Args:
            api_key: Google API key (optional, will use environment variable if not provided)
            model_name: Gemini model with multimodal capabilities
            enable_saving: Whether to save review results to files (default: True)
            output_dir: Directory to save review files
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.enable_saving = enable_saving
        self.output_dir = output_dir

        if self.enable_saving:
            os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized VideoReviewAgent with Gemini model: {self.model_name}")

    def download_video(self, video_url: str, save_path: Optional[str] = None) -> Optional[str]:
        """
        Download video from URL for processing.

        Args:
            video_url: URL of the video to download
            save_path: Optional path to save the video

        Returns:
            Path to downloaded video or None if failed
        """
        try:
            logger.info(f"Downloading video from: {video_url}")

            response = requests.get(video_url, timeout=60, stream=True)
            response.raise_for_status()

            if not save_path:
                # Create temp file path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"temp_video_{timestamp}.mp4")

            # Download in chunks
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"✅ Video downloaded to: {save_path}")
            return save_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download video: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return None

    def _create_review_prompt(
        self,
        shot_id: str,
        video_prompt: str,
        shot_description: Optional[str] = None,
        generation_strategy: Optional[str] = None
    ) -> str:
        """
        Create review prompt for Gemini (no JSON format instructions - handled by structured output).

        Args:
            shot_id: ID of the shot
            video_prompt: The prompt used to generate the video
            shot_description: Original shot description
            generation_strategy: Strategy used (generate_new, multi_shot, last_frame_seed)

        Returns:
            Review prompt for Gemini
        """
        review_prompt = f"""You are an expert AI video quality reviewer specializing in AI-generated videos for film/video production.

Your task is to review a generated video against its prompt and provide structured feedback.

**SHOT INFORMATION:**
- Shot ID: {shot_id}
- Generation Strategy: {generation_strategy or "Unknown"}
- Original Description: {shot_description or "N/A"}

**VIDEO GENERATION PROMPT:**
{video_prompt}

**YOUR TASK:**
Review the provided video and assess its quality against the prompt.

**EVALUATION CRITERIA:**

1. **Prompt Accuracy** (40 points):
   - Does the video accurately represent the prompt?
   - Are key visual elements, actions, and movements correct?
   - Missing minor details OK (35-40), missing critical elements NOT OK (0-20)

2. **Visual Quality** (30 points):
   - Video clarity, sharpness, and resolution
   - Proper lighting and color consistency
   - No major visual artifacts or glitches
   - 25+ points = high quality

3. **Motion Quality** (20 points):
   - Smoothness of movement and transitions
   - Natural motion (not jerky or warped)
   - Camera movement matches prompt
   - 15+ points = good motion

4. **Production Readiness** (10 points):
   - Usable in final production
   - Minimal post-processing needed
   - 7+ points = production ready

**DECISION CRITERIA:**

- **approved** (70-100 points): Excellent quality, matches prompt well, ready to use
- **refine_prompt** (50-69 points): Video has issues that could be fixed with better prompt
- **regenerate** (0-49 points): Major quality issues or doesn't match prompt

**PROMPT REFINEMENT GUIDANCE:**
If the video needs improvement, suggest specific changes to the prompt:
- More specific visual descriptions
- Better motion/camera direction
- Clearer subject emphasis
- Timing or pacing adjustments

**ASSESSMENT GUIDELINES:**
- Strengths: What works well in the video
- Issues: Problems you identified
- Missing elements: Key elements from prompt that are absent
- Artifacts: Any AI glitches, warping, unnatural motion

**PRODUCTION NOTES:**
Provide practical notes for the production team about using this video.

Analyze the video carefully and provide your structured review.
"""
        return review_prompt

    def review_video(
        self,
        video_url: str,
        shot_id: str,
        video_prompt: str,
        shot_description: Optional[str] = None,
        generation_strategy: Optional[str] = None,
        local_video_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Review a single video using Gemini's multimodal capabilities with structured output.

        Args:
            video_url: URL of the generated video
            shot_id: ID of the shot
            video_prompt: Prompt used to generate the video
            shot_description: Original shot description
            generation_strategy: Generation strategy used
            local_video_path: Optional local path if video already downloaded

        Returns:
            Review result dictionary or None if failed
        """
        logger.info(f"\n🎥 Reviewing video for shot: {shot_id}")

        try:
            # Download video if not already local
            video_path = local_video_path
            if not video_path:
                video_path = self.download_video(video_url)
                if not video_path:
                    logger.error("Failed to download video")
                    return None

            # Read video file
            with open(video_path, 'rb') as f:
                video_data = f.read()

            # Create review prompt (without JSON format instructions)
            review_prompt = self._create_review_prompt(
                shot_id, video_prompt, shot_description, generation_strategy
            )

            # Prepare content with video and text
            content_parts = [
                types.Part.from_text(text=review_prompt),
                types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4"
                )
            ]

            logger.info("Sending video to Gemini for review with structured output...")

            # Use structured output with Pydantic schema (like phase_1_agents)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VideoReviewResult,
                }
            )

            # Get the parsed object directly
            parsed_review: VideoReviewResult = response.parsed

            # Convert to dictionary
            review_result = parsed_review.model_dump()
            review_result['video_url'] = video_url
            review_result['video_path'] = video_path

            decision = parsed_review.decision
            score = parsed_review.overall_score

            logger.info(f"✅ Review completed: {decision.upper()} (Score: {score}/100)")

            return review_result

        except Exception as e:
            logger.error(f"Error reviewing video: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            # Return fallback
            return {
                "shot_id": shot_id,
                "decision": "error",
                "overall_score": 0,
                "scores": {
                    "prompt_accuracy": 0,
                    "visual_quality": 0,
                    "motion_quality": 0,
                    "production_readiness": 0
                },
                "assessment": {
                    "strengths": [],
                    "issues": [f"Review failed: {str(e)}"],
                    "missing_elements": [],
                    "artifacts": []
                },
                "prompt_suggestions": {
                    "original_prompt": video_prompt,
                    "suggested_prompt": video_prompt,
                    "changes_made": "Error during review",
                    "reasoning": f"Review failed: {str(e)}"
                },
                "production_notes": "Review failed, manual review required",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def save_to_mongo(
        self,
        shot_id: str,
        show_id: str,
        episode_number: int,
        review_result: Dict[str, Any],
        mongodb_client: MongoDBAtlasClient
    ) -> bool:
        """
        Save video review results to MongoDB.

        Args:
            shot_id: ID of the shot
            show_id: Show ID
            episode_number: Episode number
            review_result: Review result dictionary
            mongodb_client: MongoDB client instance

        Returns:
            True if save was successful
        """
        try:
            filter_query = {
                "shot_id": shot_id,
                "show_id": show_id,
                "episode_number": episode_number
            }

            # Structure the review data for MongoDB
            update_data = {
                "video_review": {
                    "decision": review_result.get("decision"),
                    "overall_score": review_result.get("overall_score"),
                    "scores": review_result.get("scores"),
                    "assessment": review_result.get("assessment"),
                    "prompt_suggestions": review_result.get("prompt_suggestions"),
                    "production_notes": review_result.get("production_notes"),
                    "timestamp": review_result.get("timestamp")
                }
            }

            result = mongodb_client.shots_collection.update_one(
                filter_query,
                {"$set": update_data}
            )

            if result.matched_count > 0:
                logger.info(f"✅ Saved video review for shot {shot_id} to MongoDB")
                return True
            else:
                logger.warning(f"❌ Shot {shot_id} not found in MongoDB for update")
                return False

        except Exception as e:
            logger.error(f"Error saving video review to MongoDB for shot {shot_id}: {str(e)}")
            return False

    def review_videos_for_episode(
        self,
        show_id: str,
        episode_number: int,
        mongodb_client: MongoDBAtlasClient,
        filter_strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Review all generated videos for an episode.

        Args:
            show_id: Show ID
            episode_number: Episode number
            mongodb_client: MongoDB client instance
            filter_strategy: Optional strategy filter

        Returns:
            Summary dictionary with status and results
        """
        logger.info(f"Starting video review for episode {show_id} - {episode_number}")

        try:
            # Fetch all shots with generated videos
            shots = mongodb_client.get_shots_by_episode(show_id, episode_number)

            if not shots:
                return {
                    "status": "error",
                    "message": f"No shots found for show {show_id}, episode {episode_number}",
                    "reviews_completed": 0
                }

            # Filter shots with generated videos (prefer S3 URL, fallback to Freepik URL)
            shots_with_videos = [
                s for s in shots
                if s.get("generated_video_s3_url") or s.get("generated_video_url")
            ]

            if filter_strategy:
                shots_with_videos = [
                    s for s in shots_with_videos
                    if s.get("generation_strategy") == filter_strategy
                ]

            logger.info(f"Found {len(shots_with_videos)} shots with generated videos")

            if not shots_with_videos:
                return {
                    "status": "success",
                    "message": "No shots with generated videos found",
                    "reviews_completed": 0
                }

            # Review each video
            results = []
            success_count = 0
            error_count = 0

            for shot in shots_with_videos:
                shot_id = shot.get("shot_id")
                # Prefer S3 URL over Freepik URL for faster/reliable access
                video_url = shot.get("generated_video_s3_url") or shot.get("generated_video_url")

                # Get the reviewed prompt
                prompt_data = shot.get("video_prompt_reviewed_A") or shot.get("video_prompt_reviewed_B") or {}
                video_prompt = prompt_data.get("updated_prompt") or shot.get("prompt_video_draft", "")

                shot_description = shot.get("description")
                generation_strategy = shot.get("generation_strategy")
                local_video_path = shot.get("generated_video_local_path")

                # Review video (use local path if available to skip download)
                review_result = self.review_video(
                    video_url=video_url,
                    shot_id=shot_id,
                    video_prompt=video_prompt,
                    shot_description=shot_description,
                    generation_strategy=generation_strategy,
                    local_video_path=local_video_path
                )

                if review_result and review_result.get("decision") != "error":
                    # Save to MongoDB
                    if self.save_to_mongo(shot_id, show_id, episode_number, review_result, mongodb_client):
                        success_count += 1
                    results.append(review_result)
                else:
                    error_count += 1
                    if review_result:
                        results.append(review_result)

            # Save local output
            if results and self.enable_saving:
                review_data = {
                    "show_id": show_id,
                    "episode_number": episode_number,
                    "reviewed_at": datetime.now().isoformat(),
                    "total_reviews": len(results),
                    "reviews": results
                }
                local_file = save_review_to_file(review_data, show_id, episode_number, self.output_dir)
            else:
                local_file = None

            logger.info(f"✅ Video review completed: {success_count} success, {error_count} errors")

            return {
                "status": "success",
                "message": f"Reviewed {len(results)} videos",
                "total_shots": len(shots_with_videos),
                "success_count": success_count,
                "error_count": error_count,
                "results": results,
                "local_file": local_file
            }

        except Exception as e:
            logger.error(f"Error reviewing videos for episode: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                "status": "error",
                "message": f"Error reviewing videos: {str(e)}",
                "reviews_completed": 0
            }


# Convenience functions

def review_video(
    video_url: str,
    shot_id: str,
    video_prompt: str,
    api_key: Optional[str] = None,
    shot_description: Optional[str] = None,
    generation_strategy: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to review a single video.

    Args:
        video_url: URL of the video
        shot_id: Shot ID
        video_prompt: Video generation prompt
        api_key: Google API key (optional)
        shot_description: Original shot description
        generation_strategy: Generation strategy

    Returns:
        Review result dictionary
    """
    agent = VideoReviewAgent(api_key=api_key)
    return agent.review_video(
        video_url=video_url,
        shot_id=shot_id,
        video_prompt=video_prompt,
        shot_description=shot_description,
        generation_strategy=generation_strategy
    )


def review_videos_for_episode(
    show_id: str,
    episode_number: int,
    mongodb_client: MongoDBAtlasClient,
    api_key: Optional[str] = None,
    filter_strategy: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to review all videos in an episode.

    Args:
        show_id: Show ID
        episode_number: Episode number
        mongodb_client: MongoDB client instance
        api_key: Google API key (optional)
        filter_strategy: Optional strategy filter

    Returns:
        Summary dictionary
    """
    agent = VideoReviewAgent(api_key=api_key)
    return agent.review_videos_for_episode(
        show_id=show_id,
        episode_number=episode_number,
        mongodb_client=mongodb_client,
        filter_strategy=filter_strategy
    )
