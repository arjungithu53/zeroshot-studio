"""
Video Generation Agent for Phase 3.

This agent fetches data from MongoDB, reads all previous Phase 2 agent outputs,
and generates structured prompts for video generation for each shot.
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.services.shots_service import ShotsService

# Type alias for backward compatibility
MongoDBAtlasClient = ShotsService
from backend.services.production.app.models.mongodb.shots import (
    AnnotatedShotItem,
    AnnotatedShotList
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_video_prompts_to_file(
    episode_id: str,
    show_id: str,
    episode_number: int,
    video_prompts: List[Dict[str, Any]],
    output_dir: str = "phase_3_agents/output"
) -> str:
    """
    Save generated video prompts to a JSON file with timestamp.
    
    Args:
        episode_id: Episode ID
        show_id: Show ID
        episode_number: Episode number
        video_prompts: List of video prompts with shot data
        output_dir: Directory to save the file (default: "phase_3_agents/output")
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{show_id}_{episode_number}_video_prompts_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dictionary for JSON serialization
    prompts_data = {
        "show_id": show_id,
        "episode_number": episode_number,
        "episode_id": episode_id,
        "generated_at": datetime.now().isoformat(),
        "total_shots": len(video_prompts),
        "video_prompts": video_prompts
    }
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Video prompts saved to: {filepath}")
    return filepath


class VideoPromptOutput(BaseModel):
    video_prompt: str = Field(description="The cinematic video prompt for Veo generation")
    estimated_duration_seconds: int = Field(description="Estimated video duration in seconds")


class VideoGenerationAgent:
    """
    AI agent for generating cinematic video prompts based on all Phase 2 outputs.
    
    This agent takes shot data with generation strategies, image prompts, and
    S3 image URLs to create comprehensive video generation prompts.
    """
    
    def __init__(
        self,
        model_name: str ="gemini-3.1-pro-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        api_key: Optional[str] = None,
        enable_saving: bool = True,
        output_dir: str = "phase_3_agents/output"
    ):
        """
        Initialize the Video Generation Agent.
        
        Args:
            model_name: Gemini model name (default: gemini-3.1-pro-preview)
            temperature: LLM temperature for creative generation (default: 0.7)
            max_tokens: Maximum tokens for LLM output (default: 4096 for detailed prompts)
            api_key: Google API key (optional, will use environment variable if not provided)
            enable_saving: Whether to save generated prompts to files (default: True)
            output_dir: Directory to save prompt files (default: "phase_3_agents/output")
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_saving = enable_saving
        self.output_dir = output_dir

        logger.info(f"Initialized VideoGenerationAgent with Gemini model: {self.model_name}")
    
    def download_image_from_s3(self, s3_url: str) -> Optional[types.Part]:
        """
        Download an image from S3 URL and return as a genai types.Part.
        """
        try:
            logger.info(f"Downloading image from: {s3_url}")
            response = requests.get(s3_url, timeout=30)
            response.raise_for_status()

            content_type = response.headers.get('content-type', 'image/png')
            if not content_type.startswith('image/'):
                if s3_url.lower().endswith(('.jpg', '.jpeg')):
                    content_type = 'image/jpeg'
                elif s3_url.lower().endswith('.webp'):
                    content_type = 'image/webp'
                else:
                    content_type = 'image/png'

            logger.info(f"Successfully downloaded and encoded image from {s3_url}")
            return types.Part.from_bytes(data=response.content, mime_type=content_type)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {s3_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to process image from {s3_url}: {str(e)}")
            return None
    
    def fetch_shot_data(
        self, 
        show_id: str, 
        episode_number: int, 
        mongodb_client: MongoDBAtlasClient
    ) -> List[Dict[str, Any]]:
        """
        Fetch all necessary shot data from MongoDB for video generation.
        
        Args:
            show_id: Show ID to filter by
            episode_number: Episode number to filter by
            mongodb_client: MongoDB client instance
            
        Returns:
            List of shot documents with all necessary fields
        """
        logger.info(f"Fetching shot data for show {show_id}, episode {episode_number}")
        
        try:
            # Fetch shots from MongoDB
            shots = mongodb_client.get_shots_by_episode(show_id, episode_number)
            
            if not shots:
                logger.warning(f"No shots found for show {show_id}, episode {episode_number}")
                return []
            
            logger.info(f"Successfully fetched {len(shots)} shots from MongoDB")
            return shots
            
        except Exception as e:
            logger.error(f"Error fetching shot data from MongoDB: {str(e)}")
            raise
    
    def prepare_video_prompt(
        self, 
        shot: Dict[str, Any],
        scene_description: Optional[str] = None,
        seed_shot_s3_url: Optional[str] = None
    ) -> str:
        """
        Prepare the input for Gemini to generate video prompt for a single shot.
        
        Args:
            shot: Shot document from MongoDB with all fields
            scene_description: Overall scene/episode description for context
            seed_shot_s3_url: S3 URL of seed shot image for last_frame_seed strategy
            
        Returns:
            Structured prompt string for Gemini
        """
        # Extract all relevant fields
        shot_id = shot.get("shot_id", "Unknown")
        description = shot.get("description", "")
        generation_strategy = shot.get("generation_strategy", "generate_new")

        # Extract generated_images_s3 from nested structure if present
        generated_images_s3 = []
        image_obj = shot.get("image", {})
        if isinstance(image_obj, dict):
            v0_image = image_obj.get("v0", {})
            if isinstance(v0_image, dict):
                generated_images_s3 = v0_image.get("generated_images_s3", [])
        if not generated_images_s3:
            generated_images_s3 = shot.get("generated_images_s3", [])

        camera_movement = shot.get("camera_movement", None)
        
        # Build the new concise prompt as a single string
        prompt = f"""You are a video prompt generator specializing in Veo 3.1, Google's state-of-the-art video generation model. Your task is to create visually rich, highly concise video prompts using the official Veo 3.1 structure.

---

STEP 0 — DIALOGUE EXTRACTION (ON-CAMERA ONLY):
Read the Shot Description carefully. Identify any ON-CAMERA spoken dialogue. 
CRITICAL: You MUST IGNORE any dialogue marked as (V.O.) or Voiceover. Voiceovers are added in post-production. Do not include V.O. text in the prompt, as it forces unwanted lip-syncing.
Only include dialogue if the character is explicitly speaking on-screen.

---

VEO 3.1 OFFICIAL STRUCTURE (Maximum 75 words):
Structure every prompt in this exact order, using simple, declarative sentences. Do not use tags like 'VISUAL:' or 'AUDIO:'. 

1. [Camera Instruction]: e.g., Static medium shot, Slow tracking push-in.
2. [Subject & Exact Visual Match]: Describe the subject exactly as they appear in the seed image.
3. [Single Continuous Action]: What are they doing right now? e.g., her fingertips glide downward across her cheek. Do not stack actions.
4. [Environment & Lighting]: Specific and evocative, e.g., gritty fluorescent lighting in a public washroom.
5. [On-Camera Dialogue]: ONLY if identified in Step 0. Format exactly as -> [Subject] says, "[Exact Quote]".

---

MANDATORY RULES (STRICTLY ENFORCED):

1. STRICT SEED IMAGE ALIGNMENT: Your text description MUST perfectly match the image geometry and wardrobe. Do not contradict the visual ground truth.
2. NO NEGATIVE PROMPTING: Never describe what is NOT there.
3. AUDIO SAFETY FILTER (CRITICAL): Veo's audio model will permanently block the video if it generates ASMR-like sounds. NEVER use the words "sighs", "whispers", "breathes", or "gasps". NEVER instruct the audio to sound like "rubbing", "massaging", "wet skin", or "flesh". If a character speaks, only use the verb "says".
4. NO SEQUENTIAL ACTIONS: Do not use the words "then", "next", or "after". Pick ONE primary continuous motion vector.
5. WORD LIMIT: Maximum 75 words. Be extremely concise.

Input:
Shot Description: "{description}"
Strategy: {generation_strategy}"""

        # Add camera_movement if provided
        if camera_movement:
            prompt += f"""

Camera Movement: {camera_movement}"""

        if generation_strategy == "last_frame_seed" and seed_shot_s3_url:
            prompt += f"""

seed_shot_id: {seed_shot_s3_url}"""

        prompt += """

Output:"""

        return prompt
    
    def get_seed_shot_s3_url(self, seed_shot_id: str, mongodb_client, show_id: str = None) -> Optional[str]:
        """
        Get S3 URL from seed shot for last_frame_seed strategy.
        Looks for last frame first, then falls back to generated images.

        Args:
            seed_shot_id: ID of the seed shot
            mongodb_client: MongoDB client instance
            show_id: Optional show ID to filter by (prevents returning wrong shot with same shot_id from different show)

        Returns:
            S3 URL of the seed shot's last frame or image, or None if not found
        """
        try:
            # Query shots collection directly with show_id filter to avoid getting wrong shot
            if hasattr(mongodb_client, 'shots_collection'):
                # Build query with show_id filter
                query = {"annotated_shots.shot_id": seed_shot_id}
                if show_id:
                    query["show_id"] = show_id
                    logger.info(f"🔍 Fetching seed shot {seed_shot_id} with show_id filter: {show_id}")
                else:
                    logger.warning(f"⚠️  Fetching seed shot {seed_shot_id} WITHOUT show_id filter (may return wrong shot!)")

                episode_doc = mongodb_client.shots_collection.find_one(query)

                if episode_doc and "annotated_shots" in episode_doc:
                    # Find the specific shot in the annotated_shots array
                    seed_shot = None
                    for shot in episode_doc["annotated_shots"]:
                        if shot.get("shot_id") == seed_shot_id:
                            # Add episode context to the shot
                            seed_shot = {
                                **shot,
                                "episode_id": episode_doc.get("_id"),
                                "show_id": episode_doc.get("show_id"),
                                "episode_number": episode_doc.get("episode_number")
                            }
                            break

                    if not seed_shot:
                        logger.warning(f"Seed shot {seed_shot_id} not found in annotated_shots")
                        return None
                else:
                    logger.warning(f"Seed shot {seed_shot_id} not found")
                    return None
            else:
                # Fallback to old method if shots_collection not available
                seed_shot = mongodb_client.get_shot_by_id(seed_shot_id)
                if not seed_shot:
                    logger.warning(f"Seed shot {seed_shot_id} not found")
                    return None

            # Priority 1: Look for last frame in new video structure (video.vX.last_frame_s3)
            video_data = seed_shot.get("video")
            if video_data and isinstance(video_data, dict):
                # Get the latest version (highest v number)
                versions = [k for k in video_data.keys() if k.startswith('v')]
                if versions:
                    # Sort versions by number and get the latest
                    latest_version = sorted(versions, key=lambda x: int(x[1:]))[-1]
                    version_data = video_data.get(latest_version, {})

                    if isinstance(version_data, dict):
                        last_frame_url = version_data.get("last_frame_s3")
                        if last_frame_url:
                            logger.info(f"Found last frame in {latest_version} for seed shot {seed_shot_id}")
                            return last_frame_url

            # Priority 2: Check old structure (generated_video_last_frame_s3)
            last_frame_url = seed_shot.get("generated_video_last_frame_s3")
            if last_frame_url:
                logger.info(f"Found last frame (old structure) for seed shot {seed_shot_id}")
                return last_frame_url

            # Priority 3: Fallback to generated_images_s3 (start image)
            # Check nested structure first
            s3_urls = []
            image_obj = seed_shot.get("image", {})
            if isinstance(image_obj, dict):
                v0_image = image_obj.get("v0", {})
                if isinstance(v0_image, dict):
                    s3_urls = v0_image.get("generated_images_s3", [])
            if not s3_urls:
                s3_urls = seed_shot.get("generated_images_s3", [])

            if s3_urls:
                logger.info(f"Using fallback start image for seed shot {seed_shot_id}")
                return s3_urls[0]

            logger.warning(f"No image found for seed shot {seed_shot_id}")
            return None

        except Exception as e:
            logger.error(f"Error fetching seed shot {seed_shot_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_video_prompt(
        self, 
        shot: Dict[str, Any],
        scene_description: Optional[str] = None,
        mongodb_client = None,
        all_shots: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate video prompt for a single shot using Gemini with vision capabilities.
        
        Args:
            shot: Shot document from MongoDB
            scene_description: Overall scene/episode description
            mongodb_client: MongoDB client for fetching seed shot data
            all_shots: List of all shots (to find seed shot S3 URLs from request data)
            
        Returns:
            Dictionary with shot_id, video_prompt, and estimated_duration_seconds
        """
        shot_id = shot.get("shot_id", "Unknown")
        generation_strategy = shot.get("generation_strategy", "generate_new")
        seed_shot_id = shot.get("seed_shot_id")
        show_id = shot.get("show_id", "")

        logger.info(f"Generating video prompt for shot: {shot_id} with strategy: {generation_strategy}")

        try:
            # Handle different strategies
            seed_shot_s3_url = None
            images_to_use = []

            if generation_strategy == "last_frame_seed" and seed_shot_id:
                # First, try to find seed shot in the current batch of shots (from request)
                seed_shot_s3_url = None
                if all_shots:
                    for s in all_shots:
                        if s.get("shot_id") == seed_shot_id:
                            # Check nested structure first
                            seed_s3_urls = []
                            image_obj = s.get("image", {})
                            if isinstance(image_obj, dict):
                                v0_image = image_obj.get("v0", {})
                                if isinstance(v0_image, dict):
                                    seed_s3_urls = v0_image.get("generated_images_s3", [])
                            if not seed_s3_urls:
                                seed_s3_urls = s.get("generated_images_s3", [])

                            if seed_s3_urls:
                                seed_shot_s3_url = seed_s3_urls[0]
                                logger.info(f"Found seed shot {seed_shot_id} in request data")
                                break

                # If not found in request, try MongoDB (with show_id to avoid wrong shot)
                if not seed_shot_s3_url and mongodb_client:
                    seed_shot_s3_url = self.get_seed_shot_s3_url(seed_shot_id, mongodb_client, show_id)
                    if seed_shot_s3_url:
                        logger.info(f"Found seed shot {seed_shot_id} in MongoDB")
                
                if seed_shot_s3_url:
                    images_to_use = [seed_shot_s3_url]
                    logger.info(f"Using seed shot image for {shot_id}: {seed_shot_s3_url}")
                else:
                    logger.warning(f"No S3 URL found for seed shot {seed_shot_id}")
            elif generation_strategy == "generate_new":
                # Use mapped images from current shot
                # Priority 1: Check nested image.v0.generated_images_s3 structure (new format)
                generated_images_s3 = []
                image_obj = shot.get("image", {})
                if isinstance(image_obj, dict):
                    v0_image = image_obj.get("v0", {})
                    if isinstance(v0_image, dict):
                        generated_images_s3 = v0_image.get("generated_images_s3", [])
                        if generated_images_s3:
                            logger.info(f"Found images in image.v0.generated_images_s3 structure")

                # Priority 2: Fallback to root level generated_images_s3 (legacy format)
                if not generated_images_s3:
                    generated_images_s3 = shot.get("generated_images_s3", [])

                if generated_images_s3:
                    images_to_use = generated_images_s3
                    logger.info(f"Using {len(images_to_use)} mapped image(s) for {shot_id}")
                else:
                    logger.warning(f"No mapped images found for shot {shot_id}")
            
            # Build the text prompt for Gemini
            user_prompt = self.prepare_video_prompt(shot, scene_description, seed_shot_s3_url)
            
            # Download images from S3 if available
            downloaded_images = []
            
            if images_to_use:
                logger.info(f"Downloading {len(images_to_use)} reference image(s) for shot {shot_id}")
                for idx, s3_url in enumerate(images_to_use, 1):
                    image = self.download_image_from_s3(s3_url)
                    if image:
                        downloaded_images.append(image)
                        logger.info(f"✅ Downloaded image {idx}/{len(images_to_use)}")
                    else:
                        logger.warning(f"⚠️ Failed to download image {idx}/{len(images_to_use)}")
            
            # Build content parts for genai SDK
            system_text = (
                "You are a professional video prompt generator with visual analysis capabilities."
                if downloaded_images else
                "You are a professional video prompt generator."
            )
            content_parts = [types.Part.from_text(text=f"{system_text}\n\n{user_prompt}")]

            if downloaded_images:
                logger.info(f"Sending {len(downloaded_images)} image(s) to Gemini for visual analysis")
                for idx, part in enumerate(downloaded_images, 1):
                    content_parts.append(part)
                    logger.debug(f"Added image {idx} to content parts")

            # Generate prompt using structured output
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                    "response_schema": VideoPromptOutput,
                }
            )

            parsed: VideoPromptOutput = response.parsed
            if parsed is None:
                raise ValueError(f"Structured output parsing returned None for shot {shot_id}")

            # Create result
            result = {
                "shot_id": shot_id,
                "video_prompt": parsed.video_prompt,
                "estimated_duration_seconds": parsed.estimated_duration_seconds
            }
            
            logger.info(f"Successfully generated video prompt for {shot_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating video prompt for shot {shot_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a fallback prompt
            fallback = {
                "shot_id": shot_id,
                "video_prompt": f"{shot.get('description', '')} - cinematic video, professional quality.",
                "estimated_duration_seconds": int(shot.get("duration") or 3.0)
            }
            logger.warning(f"Using fallback prompt for {shot_id}")
            return fallback
    
    def save_to_mongo(
        self, 
        shot_id: str,
        show_id: str,
        episode_number: int,
        video_prompt: str,
        mongodb_client: MongoDBAtlasClient
    ) -> bool:
        """
        Save generated video prompt to MongoDB under prompt_video_draft field.
        
        Args:
            shot_id: ID of the shot
            show_id: Show ID
            episode_number: Episode number
            video_prompt: Generated video prompt
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
            
            update_data = {
                "prompt_video_draft": video_prompt
            }
            
            result = mongodb_client.shots_collection.update_one(
                filter_query,
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                logger.info(f"✅ Saved video prompt for shot {shot_id} to MongoDB")
                return True
            else:
                logger.warning(f"❌ Shot {shot_id} not found in MongoDB for update")
                return False
                
        except Exception as e:
            logger.error(f"Error saving video prompt to MongoDB for shot {shot_id}: {str(e)}")
            return False
    
    def save_local_output(
        self,
        show_id: str,
        episode_number: int,
        episode_id: str,
        video_prompts: List[Dict[str, Any]]
    ) -> str:
        """
        Save all video prompts to local JSON file.
        
        Args:
            show_id: Show ID
            episode_number: Episode number
            episode_id: Episode ID
            video_prompts: List of video prompts with shot data
            
        Returns:
            Path to saved file
        """
        return save_video_prompts_to_file(
            episode_id=episode_id,
            show_id=show_id,
            episode_number=episode_number,
            video_prompts=video_prompts,
            output_dir=self.output_dir
        )
    
    async def generate_video_prompts_for_episode(
        self,
        show_id: str,
        episode_number: int,
        mongodb_client: MongoDBAtlasClient,
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline: Fetch shots, generate video prompts, save to MongoDB and local file.
        
        Args:
            show_id: Show ID
            episode_number: Episode number
            mongodb_client: MongoDB client instance
            scene_description: Overall scene/episode description
            
        Returns:
            Summary dictionary with status and results
        """
        logger.info(f"Starting video prompt generation pipeline for show {show_id}, episode {episode_number}")
        
        try:
            # Step 1: Fetch shot data from MongoDB
            shots = self.fetch_shot_data(show_id, episode_number, mongodb_client)
            
            if not shots:
                return {
                    "status": "error",
                    "message": f"No shots found for show {show_id}, episode {episode_number}",
                    "video_prompts_saved": 0
                }
            
            # Step 2: Generate video prompts for each shot
            video_prompts = []
            previous_shot_prompt = None
            saved_count = 0
            
            for shot in shots:
                shot_id = shot.get("shot_id", "Unknown")
                
                # Generate video prompt (pass all shots to find seed shot S3 URLs)
                result = self.generate_video_prompt(shot, scene_description, mongodb_client, all_shots=shots)
                
                video_prompt_text = result.get("video_prompt", "")
                estimated_duration = result.get("estimated_duration_seconds", 3)
                
                # Save to MongoDB
                if self.save_to_mongo(shot_id, show_id, episode_number, video_prompt_text, mongodb_client):
                    saved_count += 1
                
                # Extract reference images from nested structure
                reference_images_s3 = []
                image_obj = shot.get("image", {})
                if isinstance(image_obj, dict):
                    v0_image = image_obj.get("v0", {})
                    if isinstance(v0_image, dict):
                        reference_images_s3 = v0_image.get("generated_images_s3", [])
                if not reference_images_s3:
                    reference_images_s3 = shot.get("generated_images_s3", [])

                # Store for local save
                video_prompts.append({
                    "shot_id": shot_id,
                    "description": shot.get("description", ""),
                    "generation_strategy": shot.get("generation_strategy", ""),
                    "video_prompt": video_prompt_text,
                    "estimated_duration_seconds": estimated_duration,
                    "reference_images_s3": reference_images_s3
                })
            
            # Step 3: Save local output
            episode_id = shots[0].get("episode_id", f"E{episode_number:02d}")
            local_file = self.save_local_output(show_id, episode_number, episode_id, video_prompts)
            
            logger.info(f"✅ Video prompt generation completed: {saved_count}/{len(shots)} saved to MongoDB")
            
            return {
                "status": "success",
                "message": f"Successfully generated video prompts for {len(shots)} shots",
                "video_prompts_saved": saved_count,
                "total_shots": len(shots),
                "local_file": local_file,
                "data_preview": video_prompts[:3]  # Preview first 3 prompts
            }
            
        except Exception as e:
            logger.error(f"Error in video prompt generation pipeline: {str(e)}")
            return {
                "status": "error",
                "message": f"Error generating video prompts: {str(e)}",
                "video_prompts_saved": 0
            }


async def generate_video_prompts_pipeline(
    show_id: str,
    episode_number: int,
    mongodb_client: MongoDBAtlasClient,
    scene_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for video prompt generation pipeline.
    
    Args:
        show_id: Show ID
        episode_number: Episode number
        mongodb_client: MongoDB client instance
        scene_description: Overall scene/episode description
        
    Returns:
        Summary dictionary with status and results
    """
    logger.info("Initializing video prompt generation pipeline")
    
    # Initialize agent
    agent = VideoGenerationAgent()
    
    # Run the pipeline
    result = await agent.generate_video_prompts_for_episode(
        show_id=show_id,
        episode_number=episode_number,
        mongodb_client=mongodb_client,
        scene_description=scene_description
    )
    
    return result

