#!/usr/bin/env python3
"""
Video Prompt B Pipeline Orchestrator
=====================================
Orchestrates the generation and review of video prompts for multi-shot strategy shots.

This pipeline:
1. Fetches all shots with generation_strategy = "multi_shot" from MongoDB
2. Runs Agent 16 (video_prompt_B.py) to generate draft video prompts
3. Saves draft prompts to MongoDB field: prompt_video_draft
4. Runs video_prompt_review_B_agent.py to review the draft prompts
5. Saves reviewed prompts to MongoDB field: video_prompt_reviewed_B

Only runs if generation_strategy is "multi_shot".
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.shots_service import ShotsService
from phase_3_agents.video_prompt_B.video_prompt_B import MultiShotPrompt, MultiShotVideoGenerator
from phase_3_agents.video_prompt_B.video_prompt_review_B_agent import VideoPromptReviewBAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoPromptBPipeline:
    """
    Orchestrates the complete video prompt B workflow for multi-shot strategy.

    Workflow:
    1. Fetch multi-shot shots from MongoDB
    2. Generate draft prompts using Agent 16
    3. Save draft prompts to MongoDB (prompt_video_draft)
    4. Review prompts using video_prompt_review_B_agent
    5. Save reviewed prompts to MongoDB (video_prompt_reviewed_B)
    """

    def __init__(
        self,
        mongodb_client: ShotsService,
        google_api_key: Optional[str] = None
    ) -> None:
        """
        Initialize the Video Prompt B Pipeline.

        Args:
            mongodb_client: ShotsService instance (singleton)
            google_api_key: Google API key for Gemini (optional, uses env var if not provided)
        """
        self.mongodb_client: ShotsService = mongodb_client
        env_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        if google_api_key is not None:
            self.google_api_key = google_api_key
        else:
            self.google_api_key = env_api_key or ""

        if not self.google_api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass google_api_key parameter."
            )

        # Initialize Agent 16 (Multi-Shot Video Generator)
        self.video_generator = MultiShotVideoGenerator(api_key=self.google_api_key)

        # Initialize Video Prompt Review B Agent
        self.review_agent = VideoPromptReviewBAgent(api_key=self.google_api_key)

        logger.info("Initialized VideoPromptBPipeline")

    def fetch_multi_shot_shots(
        self,
        show_id: str,
        episode_number: int
    ) -> List[Dict[str, Any]]:
        """
        Fetch all shots with generation_strategy = "multi_shot" from MongoDB.

        Args:
            show_id: Show ID
            episode_number: Episode number

        Returns:
            List of multi-shot shot documents
        """
        logger.info(f"Fetching multi-shot shots for show {show_id}, episode {episode_number}")

        try:
            # Get all shots for the episode
            all_shots = self.mongodb_client.get_shots_by_episode(show_id, episode_number)

            if not all_shots:
                logger.warning(f"No shots found for show {show_id}, episode {episode_number}")
                return []

            # Filter for multi-shot strategy
            multi_shot_shots = [
                shot for shot in all_shots
                if shot.get("generation_strategy") == "multi_shot"
            ]

            logger.info(f"Found {len(multi_shot_shots)} multi-shot shots out of {len(all_shots)} total")
            return multi_shot_shots

        except Exception as e:
            logger.error(f"Error fetching multi-shot shots: {str(e)}")
            raise

    def get_reference_shot(
        self,
        reference_shot_id: str,
        all_shots: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Get reference shot by ID from list of shots.

        Args:
            reference_shot_id: ID of the reference shot
            all_shots: List of all shots to search

        Returns:
            Reference shot document or None if not found
        """
        for shot in all_shots:
            if shot.get("shot_id") == reference_shot_id:
                return shot

        logger.warning(f"Reference shot {reference_shot_id} not found")
        return None

    def save_draft_prompt_to_mongo(
        self,
        shot_id: str,
        show_id: str,
        episode_number: int,
        draft_prompt: str
    ) -> bool:
        """
        Save draft video prompt to MongoDB.

        Args:
            shot_id: Shot ID
            show_id: Show ID
            episode_number: Episode number
            draft_prompt: Draft video prompt to save

        Returns:
            True if save was successful
        """
        try:
            filter_query = {
                "shot_id": shot_id,
                "show_id": show_id,
                "episode_number": episode_number
            }

            update_data = {
                "prompt_video_draft": draft_prompt,
                "prompt_video_draft_timestamp": datetime.now().isoformat()
            }

            result = self.mongodb_client.shots_collection.update_one(
                filter_query,
                {"$set": update_data}
            )

            if result.matched_count > 0:
                logger.info(f"✅ Saved draft prompt for shot {shot_id} to MongoDB")
                return True
            else:
                logger.warning(f"❌ Shot {shot_id} not found in MongoDB for update")
                return False

        except Exception as e:
            logger.error(f"Error saving draft prompt to MongoDB for shot {shot_id}: {str(e)}")
            return False

    async def run_pipeline(
        self,
        show_id: str,
        episode_number: int,
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete Video Prompt B pipeline.

        Args:
            show_id: Show ID
            episode_number: Episode number
            scene_description: Optional scene/episode description for context

        Returns:
            Summary dictionary with results
        """
        logger.info(f"Starting Video Prompt B pipeline for show {show_id}, episode {episode_number}")

        try:
            # Step 1: Fetch multi-shot shots
            multi_shot_shots: List[Dict[str, Any]] = self.fetch_multi_shot_shots(show_id, episode_number)

            if not multi_shot_shots:
                return {
                    "status": "success",
                    "message": f"No multi-shot shots found for show {show_id}, episode {episode_number}",
                    "prompts_generated": 0,
                    "prompts_reviewed": 0,
                    "total_shots": 0
                }

            # Get all shots for reference lookup
            all_shots: List[Dict[str, Any]] = self.mongodb_client.get_shots_by_episode(show_id, episode_number)

            # Step 2: Generate draft prompts using Agent 16
            logger.info("=" * 60)
            logger.info("STEP 1: GENERATING DRAFT VIDEO PROMPTS (Agent 16)")
            logger.info("=" * 60)

            generated_count: int = 0
            failed_generation: List[str] = []

            for shot in multi_shot_shots:
                shot_id = shot.get("shot_id", "Unknown")
                description = shot.get("description", "")
                reference_shot_id = shot.get("seed_shot_id")  # reference to generate_new shot

                try:
                    # Get reference shot for context
                    reference_shot: Optional[Dict[str, Any]] = None
                    reference_context: Optional[str] = None
                    reference_image_path: Optional[str] = None

                    if reference_shot_id:
                        reference_shot = self.get_reference_shot(reference_shot_id, all_shots)
                        if reference_shot:
                            reference_context = reference_shot.get("description", "")
                            # Get latest image from reference shot
                            ref_images = reference_shot.get("generated_images_s3", [])
                            if ref_images:
                                reference_image_path = ref_images[-1]

                    # Generate video prompt using Agent 16
                    if not reference_image_path:
                        logger.warning(f"⚠️ No reference image found for shot {shot_id}, using N/A")
                        reference_image_path = "N/A"

                    result: MultiShotPrompt = self.video_generator.generate_multi_shot_prompt(
                        shot_id=shot_id,
                        shot_description=description,
                        reference_shot_id=reference_shot_id or "N/A",
                        reference_image_path=reference_image_path,
                        reference_context=reference_context
                    )

                    draft_prompt = result.video_prompt

                    # Save draft prompt to MongoDB
                    if self.save_draft_prompt_to_mongo(
                        shot_id, show_id, episode_number, draft_prompt
                    ):
                        generated_count += 1

                except Exception as e:
                    logger.error(f"Failed to generate prompt for shot {shot_id}: {str(e)}")
                    failed_generation.append(shot_id)
                    continue

            logger.info(f"✅ Generated {generated_count}/{len(multi_shot_shots)} draft prompts")

            # Step 3: Review prompts using video_prompt_review_B_agent
            logger.info("\n" + "=" * 60)
            logger.info("STEP 2: REVIEWING VIDEO PROMPTS (Review Agent B)")
            logger.info("=" * 60)

            review_result: Dict[str, Any] = await self.review_agent.review_video_prompts_for_episode(
                show_id=show_id,
                episode_number=episode_number,
                mongodb_client=self.mongodb_client,
                scene_description=scene_description
            )

            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)

            return {
                "status": "success",
                "message": f"Successfully completed Video Prompt B pipeline",
                "prompts_generated": generated_count,
                "prompts_reviewed": review_result.get("reviewed_prompts_saved", 0),
                "total_multi_shot_shots": len(multi_shot_shots),
                "failed_generation": failed_generation,
                "review_details": review_result
            }

        except Exception as e:
            logger.error(f"Error in Video Prompt B pipeline: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}",
                "prompts_generated": 0,
                "prompts_reviewed": 0
            }


async def run_video_prompt_B_pipeline(
    show_id: str,
    episode_number: int,
    mongodb_client: Optional[ShotsService] = None,
    scene_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for Video Prompt B pipeline.

    Args:
        show_id: Show ID
        episode_number: Episode number
        mongodb_client: ShotsService instance (optional, will use singleton if not provided)
        scene_description: Optional scene/episode description

    Returns:
        Summary dictionary with results
    """
    # Get ShotsService singleton if not provided
    client: ShotsService
    if mongodb_client is None:
        from app.config import get_shots_service
        client = get_shots_service()
    else:
        client = mongodb_client

    # Initialize pipeline
    pipeline = VideoPromptBPipeline(mongodb_client=client)

    # Run pipeline
    result: Dict[str, Any] = await pipeline.run_pipeline(
        show_id=show_id,
        episode_number=episode_number,
        scene_description=scene_description
    )

    return result


def main() -> None:
    """
    Example usage of Video Prompt B pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run Video Prompt B pipeline for multi-shot strategy")
    parser.add_argument("--show_id", type=str, required=True, help="Show ID")
    parser.add_argument("--episode", type=int, required=True, help="Episode number")
    parser.add_argument("--scene_description", type=str, help="Optional scene description")

    args = parser.parse_args()

    # Run pipeline
    result = asyncio.run(run_video_prompt_B_pipeline(
        show_id=args.show_id,
        episode_number=args.episode,
        scene_description=args.scene_description
    ))

    # Print results
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Message: {result['message']}")
    print(f"Prompts Generated: {result.get('prompts_generated', 0)}")
    print(f"Prompts Reviewed: {result.get('prompts_reviewed', 0)}")
    print(f"Total Multi-Shot Shots: {result.get('total_multi_shot_shots', 0)}")

    if result.get('failed_generation'):
        print(f"\n⚠️ Failed to generate prompts for: {result['failed_generation']}")


if __name__ == "__main__":
    main()
