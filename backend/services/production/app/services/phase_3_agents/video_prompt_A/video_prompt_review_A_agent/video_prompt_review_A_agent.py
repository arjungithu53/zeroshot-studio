"""
Video Prompt Review Agent A for Phase 3.

This agent reviews and refines AI-generated video prompts for each shot to ensure
accuracy, visual continuity, and alignment with image inputs and text descriptions.

Uses Gemini 2.5 Pro to check and improve video prompts for:
- Alignment with description and image content
- Continuity correctness for last_frame_seed strategy
- Vague or missing visual cues
- Cinematic consistency (camera angle, tone, flow)
- Prompt length optimization
"""

import os
import json
import logging
import requests
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from app.services.shots_service import ShotsService

# Type alias for backward compatibility
MongoDBAtlasClient = ShotsService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_review_to_file(
    review_data: Dict[str, Any],
    show_id: str,
    episode_number: int,
    output_dir: str = "phase_3_agents/video_prompt_review_A_agent/output"
) -> str:
    """
    Save video prompt review results to a JSON file with timestamp.
    
    Args:
        review_data: Dictionary containing review results for all shots
        show_id: Show ID for filename
        episode_number: Episode number for filename
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"review_{show_id}_{episode_number}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Video prompt review saved to: {filepath}")
    return filepath


class VideoPromptReviewAgent:
    """
    AI agent for reviewing and refining video prompts based on image inputs and text descriptions.
    
    This agent takes video prompts from the Video Generation Agent and reviews them
    for accuracy, continuity, and cinematic quality.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        api_key: Optional[str] = None,
        enable_saving: bool = True,
        output_dir: str = "phase_3_agents/video_prompt_review_A_agent/output"
    ):
        """
        Initialize the Video Prompt Review Agent.
        
        Args:
            model_name: Gemini model name (default: gemini-3.1-pro-preview)
            temperature: LLM temperature for review (default: 0.3 for consistency)
            max_tokens: Maximum tokens for LLM output (default: 4096)
            api_key: Google API key (optional, will use environment variable if not provided)
            enable_saving: Whether to save review results to files (default: True)
            output_dir: Directory to save review files
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_tokens or 4096
        )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_saving = enable_saving
        self.output_dir = output_dir
        
        logger.info(f"Initialized VideoPromptReviewAgent with Gemini model: {self.model_name}")
    
    def download_image_from_s3(self, s3_url: str) -> Optional[Dict[str, Any]]:
        """
        Download an image from S3 URL and return as base64-encoded dict for Gemini.
        
        Args:
            s3_url: S3 URL of the image
            
        Returns:
            Dictionary with image data in format expected by Gemini, or None if download fails
        """
        try:
            logger.info(f"Downloading image from: {s3_url}")
            response = requests.get(s3_url, timeout=30)
            response.raise_for_status()
            
            # Get image data
            image_data = response.content
            
            # Determine MIME type from response headers or URL
            content_type = response.headers.get('content-type', 'image/png')
            if not content_type.startswith('image/'):
                # Try to infer from URL extension
                if s3_url.lower().endswith('.jpg') or s3_url.lower().endswith('.jpeg'):
                    content_type = 'image/jpeg'
                elif s3_url.lower().endswith('.png'):
                    content_type = 'image/png'
                elif s3_url.lower().endswith('.webp'):
                    content_type = 'image/webp'
                else:
                    content_type = 'image/png'  # default
            
            # Encode to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Return in format expected by Gemini
            image_dict = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{content_type};base64,{image_base64}"
                }
            }
            
            logger.info(f"Successfully downloaded and encoded image from {s3_url}")
            return image_dict
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {s3_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to process image from {s3_url}: {str(e)}")
            return None
    
    def fetch_shots_from_mongo(
        self,
        show_id: str,
        episode_number: int,
        mongodb_client: MongoDBAtlasClient
    ) -> List[Dict[str, Any]]:
        """
        Fetch shots from MongoDB that need video prompt review.
        Only fetches shots where:
        - generation_strategy is 'generate_new' or 'last_frame_seed'
        - prompt_video_draft field exists
        
        Args:
            show_id: Show ID to filter by
            episode_number: Episode number to filter by
            mongodb_client: MongoDB client instance
            
        Returns:
            List of shot documents with all necessary fields
        """
        logger.info(f"Fetching shots for video prompt review (show {show_id}, episode {episode_number})")
        
        try:
            # Fetch all shots for the episode
            all_shots = mongodb_client.get_shots_by_episode(show_id, episode_number)
            
            if not all_shots:
                logger.warning(f"No shots found for show {show_id}, episode {episode_number}")
                return []
            
            # Filter shots that need review
            shots_to_review = []
            for shot in all_shots:
                generation_strategy = shot.get("generation_strategy", "")
                has_draft_prompt = "prompt_video_draft" in shot and shot["prompt_video_draft"]
                
                if generation_strategy in ["generate_new", "last_frame_seed"] and has_draft_prompt:
                    shots_to_review.append(shot)
                else:
                    logger.debug(
                        f"Skipping shot {shot.get('shot_id')}: "
                        f"strategy={generation_strategy}, has_draft={has_draft_prompt}"
                    )
            
            logger.info(f"Found {len(shots_to_review)} shots to review out of {len(all_shots)} total")
            return shots_to_review
            
        except Exception as e:
            logger.error(f"Error fetching shots from MongoDB: {str(e)}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the video prompt review agent."""
        return """You are a professional video prompt reviewer with expertise in Veo 3.1, Google's state-of-the-art video generation model.

Your task is to review and refine AI-generated video prompts to ensure they fully follow the official Veo 3.1 structure and bypass safety filters.

OFFICIAL VEO 3.1 STRUCTURE (every prompt must follow this):
[Camera Instruction]. [Subject & Exact Visual Match]. [Single Continuous Action]. [Environment & Lighting]. [On-Camera Dialogue ONLY]

REVIEW PRIORITIES (earlier priorities are absolute and override later ones):

## 1. AUDIO SAFETY & ASMR FILTER (HIGHEST PRIORITY)
Veo 3.1 has an ultra-sensitive audio safety filter that will SILENTLY BLOCK the video if it detects suggestive, ASMR, or fleshy sounds.
- REMOVE any delivery verbs like "sighs", "whispers", "breathes", "gasps", or "moans". Change them to "says".
- REMOVE any audio cues related to skin contact: "rubbing skin", "massaging", "stroking flesh", "wet sounds".
- If the prompt has ANY risk of generating fleshy/skin sound effects, add this exact sentence to the end of the prompt: "SFX: complete silence, gentle room tone only."

## 2. VOICEOVER PURGE
- Check if the prompt contains dialogue. If the original script marks that dialogue as (V.O.) or Voiceover, you MUST DELETE the dialogue from the prompt entirely. V.O. is added in post-production and ruins the video model's lip-sync.

## 3. Brand Name Preservation
- Remove brand names (e.g., Mamaearth) to avoid IP filters, but replace them with a hyper-specific visual description to maintain product consistency (e.g., "a white squeeze tube with green and orange graphics").

## 4. Ban Sequential Actions & Meta-Instructions
- Remove sequential words like "then", "after", or "next". I2V models require ONE primary continuous motion.
- Remove meta-instructions ("maintain consistency", "do not morph"). Describe the physical scene only.

## 5. Length Optimization
- Maximum 75 words. Strip out overly ornate adjectives.

OUTPUT FORMAT:
Return a JSON object in the required schema. Output ONLY valid JSON, no markdown code fences, no preamble."""
    
    async def review_prompt_with_gemini(
        self,
        shot: Dict[str, Any],
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Review a single video prompt using Gemini with multimodal input.
        
        Args:
            shot: Shot document from MongoDB with all fields
            scene_description: Overall scene/episode description for context
            
        Returns:
            Dictionary with review results
        """
        shot_id = shot.get("shot_id", "Unknown")
        description = shot.get("description", "")
        draft_prompt = shot.get("prompt_video_draft", "")
        generation_strategy = shot.get("generation_strategy", "generate_new")
        seed_shot_id = shot.get("seed_shot_id")

        # Read images from nested structure (image.v0.generated_images_s3), with root-level fallback
        generated_images_s3 = []
        image_obj = shot.get("image", {})
        if isinstance(image_obj, dict):
            v0_image = image_obj.get("v0", {})
            if isinstance(v0_image, dict):
                generated_images_s3 = v0_image.get("generated_images_s3", [])
        if not generated_images_s3:
            generated_images_s3 = shot.get("generated_images_s3", [])
        
        logger.info(f"Reviewing video prompt for shot: {shot_id}")
        
        try:
            # Build the review prompt
            prompt_parts = []
            prompt_parts.append("=== SHOT INFORMATION ===")
            prompt_parts.append(f"Shot ID: {shot_id}")
            prompt_parts.append(f"Description: {description}")
            prompt_parts.append(f"Generation Strategy: {generation_strategy}")
            
            if seed_shot_id:
                prompt_parts.append(f"Seed Shot ID: {seed_shot_id}")
                prompt_parts.append("(This shot should continue smoothly from the seed shot)")
            
            if scene_description:
                prompt_parts.append(f"\nScene Context: {scene_description}")
            
            prompt_parts.append(f"\n=== DRAFT VIDEO PROMPT TO REVIEW ===")
            prompt_parts.append(draft_prompt)
            
            prompt_parts.append("\n=== YOUR TASK ===")
            prompt_parts.append(
                "Review the draft video prompt above. Compare it with the description and the image(s) provided. "
                "Check for accuracy, clarity, continuity, and cinematic quality.\n\n"
                "Return a JSON object in this EXACT format:\n"
                "{\n"
                f'  "shot_id": "{shot_id}",\n'
                '  "draft_prompt": "...",\n'
                '  "updated_prompt": "...",\n'
                '  "changes_made": "...",\n'
                '  "reasoning": "...",\n'
                '  "timestamp": "2025-01-01T12:00:00"\n'
                "}\n\n"
                "Output ONLY the JSON object. No markdown, no code fences, no explanation."
            )
            
            user_prompt = "\n".join(prompt_parts)
            
            # Download images from S3 if available
            downloaded_images = []
            if generated_images_s3:
                logger.info(f"Downloading {len(generated_images_s3)} image(s) for shot {shot_id}")
                for idx, s3_url in enumerate(generated_images_s3, 1):
                    image = self.download_image_from_s3(s3_url)
                    if image:
                        downloaded_images.append(image)
                        logger.info(f"✅ Downloaded image {idx}/{len(generated_images_s3)}")
                    else:
                        logger.warning(f"⚠️ Failed to download image {idx}/{len(generated_images_s3)}")
            
            # Prepare messages for Gemini with multimodal support
            if downloaded_images:
                logger.info(f"Sending {len(downloaded_images)} image(s) to Gemini for review")
                
                # Create message content with both text and images
                message_content = [
                    {"type": "text", "text": user_prompt}
                ]
                
                # Add each image to the message content
                for idx, image_dict in enumerate(downloaded_images, 1):
                    message_content.append(image_dict)
                    logger.debug(f"Added image {idx} to message content")
                
                messages = [
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=message_content)
                ]
            else:
                # Text-only message (no images available)
                logger.info("No images available, using text-only review")
                messages = [
                    SystemMessage(content=self._get_system_prompt()),
                    HumanMessage(content=user_prompt)
                ]
            
            # Generate review using Gemini
            response = self.llm.invoke(messages)
            raw_response = response.content if hasattr(response, "content") else str(response)
            
            logger.debug(f"Raw LLM response: {raw_response[:300]}...")
            
            # Parse JSON response
            review_result = self._parse_review_response(raw_response, shot_id, draft_prompt)
            
            logger.info(f"Successfully reviewed video prompt for {shot_id}")
            return review_result
            
        except Exception as e:
            logger.error(f"Error reviewing video prompt for shot {shot_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback review
            fallback = {
                "shot_id": shot_id,
                "draft_prompt": draft_prompt,
                "updated_prompt": draft_prompt,
                "changes_made": "Error during review - using original prompt",
                "reasoning": f"Review failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            logger.warning(f"Using fallback review for {shot_id}")
            return fallback
    
    def _parse_review_response(
        self,
        response: str,
        shot_id: str,
        draft_prompt: str
    ) -> Dict[str, Any]:
        """Parse the JSON review response from Gemini."""
        try:
            # Clean the response
            response = response.strip()
            
            # Remove markdown code fences if present
            if response.startswith("```json"):
                response = response[7:]
            elif response.startswith("```"):
                response = response[3:]
            
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            review_data = json.loads(response)
            
            # Validate format
            if not isinstance(review_data, dict):
                raise ValueError("Review response must be a JSON object")
            
            # Ensure required fields
            required_fields = ['shot_id', 'updated_prompt']
            for field in required_fields:
                if field not in review_data:
                    logger.warning(f"Review missing field '{field}', using fallback")
                    review_data[field] = shot_id if field == 'shot_id' else draft_prompt
            
            # Set defaults for optional fields
            if 'draft_prompt' not in review_data:
                review_data['draft_prompt'] = draft_prompt
            if 'changes_made' not in review_data:
                review_data['changes_made'] = "Not specified"
            if 'reasoning' not in review_data:
                review_data['reasoning'] = "Not specified"
            if 'timestamp' not in review_data:
                review_data['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Successfully parsed review for shot {shot_id}")
            return review_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review response as JSON: {str(e)}")
            logger.error(f"Response preview: {response[:300]}...")
            
            # Return fallback
            return {
                "shot_id": shot_id,
                "draft_prompt": draft_prompt,
                "updated_prompt": draft_prompt,
                "changes_made": "JSON parse error - using original prompt",
                "reasoning": f"Parse error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error parsing review response: {str(e)}")
            
            # Return fallback
            return {
                "shot_id": shot_id,
                "draft_prompt": draft_prompt,
                "updated_prompt": draft_prompt,
                "changes_made": "Parse error - using original prompt",
                "reasoning": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def save_to_mongo_and_json(
        self,
        shot_id: str,
        show_id: str,
        episode_number: int,
        review_result: Dict[str, Any],
        mongodb_client: MongoDBAtlasClient
    ) -> bool:
        """
        Save reviewed video prompt to MongoDB under video_prompt_reviewed_A field.
        
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
            review_data = {
                "draft_prompt": review_result.get("draft_prompt", ""),
                "updated_prompt": review_result.get("updated_prompt", ""),
                "changes_made": review_result.get("changes_made", ""),
                "reasoning": review_result.get("reasoning", ""),
                "timestamp": review_result.get("timestamp", datetime.now().isoformat())
            }
            
            update_data = {
                "video_prompt_reviewed_A": review_data
            }
            
            result = mongodb_client.shots_collection.update_one(
                filter_query,
                {"$set": update_data}
            )
            
            if result.matched_count > 0:
                logger.info(f"✅ Saved reviewed video prompt for shot {shot_id} to MongoDB")
                return True
            else:
                logger.warning(f"❌ Shot {shot_id} not found in MongoDB for update")
                return False
                
        except Exception as e:
            logger.error(f"Error saving reviewed video prompt to MongoDB for shot {shot_id}: {str(e)}")
            return False
    
    async def review_video_prompts_for_episode(
        self,
        show_id: str,
        episode_number: int,
        mongodb_client: MongoDBAtlasClient,
        scene_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main pipeline: Fetch shots, review video prompts, save to MongoDB and local file.
        
        Args:
            show_id: Show ID
            episode_number: Episode number
            mongodb_client: MongoDB client instance
            scene_description: Overall scene/episode description
            
        Returns:
            Summary dictionary with status and results
        """
        logger.info(f"Starting video prompt review pipeline for show {show_id}, episode {episode_number}")
        
        try:
            # Step 1: Fetch shots from MongoDB
            shots = self.fetch_shots_from_mongo(show_id, episode_number, mongodb_client)
            
            if not shots:
                return {
                    "status": "success",
                    "message": f"No shots to review for show {show_id}, episode {episode_number}",
                    "reviewed_prompts_saved": 0,
                    "total_shots": 0
                }
            
            # Step 2: Review video prompts for each shot
            review_results = []
            saved_count = 0
            skipped_count = 0
            
            for shot in shots:
                shot_id = shot.get("shot_id", "Unknown")
                
                try:
                    # Review video prompt
                    review_result = await self.review_prompt_with_gemini(shot, scene_description)
                    
                    # Save to MongoDB
                    if self.save_to_mongo_and_json(
                        shot_id, show_id, episode_number, review_result, mongodb_client
                    ):
                        saved_count += 1
                    
                    # Store for local save
                    review_results.append(review_result)
                    
                except Exception as e:
                    logger.error(f"Failed to review shot {shot_id}: {str(e)}")
                    skipped_count += 1
                    # Continue with next shot instead of crashing
                    continue
            
            # Step 3: Save local output
            if review_results and self.enable_saving:
                review_data = {
                    "show_id": show_id,
                    "episode_number": episode_number,
                    "reviewed_at": datetime.now().isoformat(),
                    "total_shots_reviewed": len(review_results),
                    "reviews": review_results
                }
                
                local_file = save_review_to_file(
                    review_data,
                    show_id,
                    episode_number,
                    self.output_dir
                )
            else:
                local_file = None
            
            logger.info(
                f"✅ Video prompt review completed: {saved_count}/{len(shots)} saved to MongoDB, "
                f"{skipped_count} skipped"
            )
            
            return {
                "status": "success",
                "message": f"Successfully reviewed video prompts for {len(review_results)} shots",
                "reviewed_prompts_saved": saved_count,
                "total_shots": len(shots),
                "skipped_shots": skipped_count,
                "local_file": local_file,
                "data_preview": review_results[:3]  # Preview first 3 reviews
            }
            
        except Exception as e:
            logger.error(f"Error in video prompt review pipeline: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "message": f"Error reviewing video prompts: {str(e)}",
                "reviewed_prompts_saved": 0
            }


async def review_video_prompts_pipeline(
    show_id: str,
    episode_number: int,
    mongodb_client: MongoDBAtlasClient,
    scene_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for video prompt review pipeline.
    
    Args:
        show_id: Show ID
        episode_number: Episode number
        mongodb_client: MongoDB client instance
        scene_description: Overall scene/episode description
        
    Returns:
        Summary dictionary with status and results
    """
    logger.info("Initializing video prompt review pipeline")
    
    # Initialize agent with Gemini 2.5 Pro
    agent = VideoPromptReviewAgent()
    
    # Run the review pipeline
    result = await agent.review_video_prompts_for_episode(
        show_id=show_id,
        episode_number=episode_number,
        mongodb_client=mongodb_client,
        scene_description=scene_description
    )
    
    return result

