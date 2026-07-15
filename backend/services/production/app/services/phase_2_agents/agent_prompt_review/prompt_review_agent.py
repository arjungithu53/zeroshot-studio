"""
Prompt Review Agent for Phase 2.

This agent reviews and refines AI-generated image prompts for each shot to ensure
visual and narrative continuity across the sequence.

Uses Gemini API to check and improve prompts for:
- Visual consistency (lighting, weather, colors)
- Character positions and blocking
- Directional consistency (sunrise/sunset)
- Emotional continuity
- Scene detail consistency
"""


import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

import os
import json
import logging
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from io import BytesIO
import requests
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from backend.services.production.app.models.mongodb.shots import (
    MongoDBAtlasClient,
    AnnotatedShotItem,
    AnnotatedShotList
)
from .data_schema import PromptReviewResponse, PromptReviewItem

# Import AssetLibrary for fetching assets from Agent 5 and Agent 8
from ..helpers.asset_library import AssetLibrary, AssetInfo

# Import name normalization utility
from backend.services.production.app.utils.name_normalization import normalize_asset_name

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)




def save_review_to_file(
    review_data: Dict[str, Any],
    episode_id: str,
    output_dir: str = "phase_2_agents/outputs/agent_prompt_review"
) -> str:
    """
    Save prompt review results to a JSON file with timestamp.
    
    Args:
        review_data: Dictionary containing review results for all shots
        episode_id: Episode ID for filename
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"review_{episode_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Prompt review saved to: {filepath}")
    return filepath


class PromptReviewAgent:
    """
    AI agent for reviewing and refining image prompts to ensure visual and narrative continuity.
    
    This agent takes prompts from the Prompt Generation Agent and reviews them
    for consistency across the entire sequence.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        temperature: float = 0.3,
        max_tokens: Optional[int] = 16384,
        api_key: Optional[str] = None,
        enable_saving: bool = True,
        output_dir: str = "phase_2_agents/outputs/agent_prompt_review",
        max_shots_per_call: int = 4,
        asset_library: Optional[AssetLibrary] = None
    ):
        """
        Initialize the Prompt Review Agent.

        Args:
            model_name: Gemini model name (default: gemini-3.1-pro-preview)
            temperature: LLM temperature for review (default: 0.3 for consistency)
            max_tokens: Maximum tokens for LLM output (default: 16384 for detailed reviews)
            api_key: Google API key (optional, will use environment variable if not provided)
            enable_saving: Whether to save review results to files (default: True)
            output_dir: Directory to save review files
            max_shots_per_call: Maximum shots to send per LLM call to prevent truncation
            asset_library: AssetLibrary instance for fetching assets from Agent 5 and Agent 8 (optional)
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        # Create base LLM with structured output support
        base_llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_tokens or 16384,
            transport="rest"
        )
        # Bind structured output schema for Gemini
        self.llm = base_llm.with_structured_output(PromptReviewResponse)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_saving = enable_saving
        self.output_dir = output_dir
        self.max_shots_per_call = max(1, max_shots_per_call)
        self.asset_library = asset_library  # Store AssetLibrary for accessing Agent 5 and Agent 8 assets

        logger.info(f"Initialized PromptReviewAgent with Gemini model: {self.model_name}")

        # Log asset library status
        if self.asset_library:
            asset_summary = self.asset_library.generate_asset_summary()
            logger.info(f"AssetLibrary loaded with {asset_summary['total_characters']} characters, "
                       f"{asset_summary['total_locations']} locations, and {asset_summary['total_props']} props")
        else:
            logger.warning("No AssetLibrary provided - assets from Agent 5 and Agent 8 will not be available")
    
    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the review agent, optimized for asset-based workflow.

        Note: This prompt focuses on sanitizing prompts for asset-based generation
        where reference images are provided directly to the model.
        """
        return """You are a professional Visual Continuity Supervisor (Script Supervisor) and AI Prompt Editor working on a Nano Banana (Gemini Flash Image) generation pipeline.

YOUR TASK:
Review a sequence of AI image generation prompts to ensure perfect Visual Consistency, Spatial Logic, and Asset Integrity across shots.

CONTEXT:
These prompts are designed for an Asset-Based Workflow (Reference Images are used for characters and locations).
The Problem: Sometimes prompts drift and start describing the character's physical look (e.g., "blonde hair"), which confuses the model when a Reference Image is present.
Your Goal: Sanitize prompts to remove CHARACTER physical descriptions only, while strictly enforcing environmental and spatial continuity between shots.

CRITICAL REVIEW GUIDELINES:

1. ASSET HYGIENE (Sync with Generation Agent) 🧹
You must aggressively edit prompts to respect the "Asset Image" workflow.
REMOVE CHARACTER PHYSICAL DESCRIPTIONS ONLY: If a prompt describes the subject's personal appearance (e.g., "wearing a red coat," "blue eyes," "curly hair," "tall man"), DELETE IT. The prompt must rely on the Reference Image for character appearance.
ENFORCE GENERIC TAGS: Ensure the character is referred to consistently (e.g., "The Protagonist" or "The Subject") rather than "A man" or "A woman."

⚠️ CRITICAL — DO NOT REMOVE CINEMATIC LANGUAGE:
The following are NOT character physical descriptions. They are cinematic and environmental details that MUST be preserved:
- Lighting setups: "three-point softbox", "golden-hour sidelight from camera-left", "overcast diffuse fill", "practical backlight" → KEEP
- Camera/lens specs: "35mm f/2.8", "85mm telephoto with background bokeh", "wide-angle 24mm" → KEEP
- Film stock/color grade: "Kodak Vision3 500T color grade", "ARRI Log-C", "warm natural grain" → KEEP
- Environmental materiality: "rough-hewn limestone", "wet cobblestones", "aged pine floorboards" → KEEP (this is location, not character)
- Physical grounding: "feet planted on the cobblestones", "shoulder pressed against the brick wall" → KEEP (this is spatial, not appearance)
- Lighting interaction: "lit by the warm sidelight from the warehouse window", "casting a shadow on the pavement" → KEEP

2. ATMOSPHERIC & LIGHTING CONTINUITY (The "Glue") 💡
The environment must not change randomly between shots.
Global Light Direction & Quality: If Shot 1 has "soft golden-hour sidelight from camera-left," Shot 2 must maintain the same light direction and warmth — not switch to "hard overhead midday sun."
Color Temperature: Warm/cool balance must match across shots unless a scene change occurs.
Weather: If Shot 1 has "rain dripping from surfaces," Shot 2 must imply wet surfaces or continued rain.
Fog/Haze: Ensure atmospheric density remains consistent.
If a shot is missing a lighting description, ADD one that is consistent with the surrounding shots — do not leave it undefined.

3. SPATIAL BLOCKING & LOGIC 📍
Relative Position: If Shot 1 places the [Subject] to the left of a lamp, Shot 2 must maintain that relationship (unless action occurred).
Gaze/Eyeline: If Shot 1 looks "Up at the tower," Shot 2 (the POV) must be a High Angle looking down, or a Low Angle looking up at the tower.
Prop Permanence: If the subject picked up a prop in Shot 1, they must be holding it in Shot 2 (unless dropped).
Product Permanence (CRITICAL): For any shot marked [PRODUCT SHOT], the PRODUCT must remain clearly visible, in-focus, and prominently placed in every version of the prompt. If a prompt for a [PRODUCT SHOT] does not mention the product or places it out-of-frame / behind something, add explicit language to bring it forward — e.g., "the PRODUCT rests on the surface in the foreground, fully visible and in sharp focus." Never remove or obscure the product.
Physical Grounding: If a prompt is missing grounding details (how the character physically connects to the environment), add a brief grounding phrase — e.g., "feet firmly on the wet cobblestones" or "back against the brick wall" — to prevent the composited/pasted look.

4. CAMERA & STYLE CONSISTENCY 🎥
Frozen Moments: Ensure no motion verbs ("panning," "zooming") exist. Change them to static descriptors ("motion blur," "dynamic framing").
Lens Consistency: If the sequence uses "85mm telephoto," ensure no shot suddenly requests "fisheye" unless story-motivated.
Style Prefix: Ensure each prompt retains its required style prefix (e.g., "realistic-style", "pixar-style") — never remove it.

REVIEW PROCESS:
1. Sanitize: Strip CHARACTER physical descriptions (hair, clothes, body type, eye color). Preserve all cinematic, environmental, and spatial language.
2. Harmonize: Check surrounding shots. Does the lighting direction and quality match? Does the location match?
3. Ground: If physical grounding is missing, add a brief surface-contact phrase.
4. Correct: Edit the reviewed_prompt to fix continuity errors.

OUTPUT FORMAT (JSON ONLY):
You must output a valid JSON object with a single "reviews" key containing an array of review items.

JSON Rules:
- Output ONLY valid JSON. No markdown, no code fences, no intro/outro text.
- Escape all internal quotes properly.
- Required structure:
  {
    "reviews": [
      {
        "shot_id": (String),
        "original_prompt": (String),
        "reviewed_prompt": (String - The sanitized, continuity-checked version),
        "changes_made": (Array of strings - Concise list of what you fixed),
        "shot_modified": (Boolean - true if you changed anything),
        "reason_for_modification": (String - Why you changed it),
        "continuity_observations": (Array of strings - Notes on continuity with surrounding shots),
        "continuity_status": (String - "Pass" if no changes needed, "Fixed" if modified)
      }
    ]
  }

Example Logic:
Input Prompt: "A man with a beard and green shirt runs through the rain. Shot on 35mm f/2.8, overcast diffuse light."
Your Review: "The [Subject] sprinting through heavy rain, feet splashing on the wet pavement. Shot on 35mm f/2.8, overcast diffuse light with flat even fill." (Reason: Removed character physical description; preserved lens spec and lighting; added grounding detail.)

GENERATE THE REVIEW JSON NOW:"""

    def _get_assets_for_shot(self, shot: AnnotatedShotItem) -> Dict[str, List[Dict]]:
        """
        Extract and fetch assets from AssetLibrary for a given shot.

        Uses characters and locations from CSV fields for targeted asset fetching,
        while props are still detected from the description text.

        Args:
            shot: Annotated shot item

        Returns:
            Dictionary with 'characters', 'locations', and 'props' lists containing asset info
        """
        if not self.asset_library:
            logger.warning("No AssetLibrary available - cannot fetch assets")
            return {'characters': [], 'locations': [], 'props': []}

        assets_info = {
            'characters': [],
            'locations': [],
            'props': []
        }

        # Use CSV fields for characters and locations (targeted fetching)
        if shot.characters:
            # Use explicit character list from CSV
            logger.info(f"Fetching assets for characters from CSV: {shot.characters}")
            for char_name in shot.characters:
                # Normalize the name to match asset library naming (UPPERCASE_WITH_UNDERSCORES)
                normalized_name = normalize_asset_name(char_name)

                # Get all assets for this character
                char_assets = self.asset_library.get_all_assets_for_character(normalized_name)
                if char_assets:
                    assets_info['characters'].append({
                        'name': normalized_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'type': 'character'
                            }
                            for asset in char_assets
                        ]
                    })
                else:
                    logger.warning(f"No assets found for character: {normalized_name}")

        if shot.locations:
            # Use explicit location from CSV
            logger.info(f"Fetching assets for location from CSV: {shot.locations}")

            # Check if location has direction suffix (e.g., "JUNGLE_NORTH")
            location_parts = shot.locations.split('_')
            direction_keywords = ['north', 'south', 'east', 'west']

            # Check if the last part is a direction
            if len(location_parts) > 1 and location_parts[-1].lower() in direction_keywords:
                # Location has direction
                direction = location_parts[-1]
                base_location = '_'.join(location_parts[:-1])  # Everything except the last part
                normalized_name = normalize_asset_name(base_location)

                logger.info(f"Location with direction detected: {base_location} in direction {direction}")

                # Get specific direction asset
                loc_asset = self.asset_library.get_location_asset_by_direction(normalized_name, direction)
                if loc_asset:
                    assets_info['locations'].append({
                        'name': normalized_name,
                        'direction': direction,
                        'assets': [
                            {
                                'angle': loc_asset.angle,
                                'local_path': loc_asset.local_path,
                                'url': loc_asset.url,
                                'prompt': loc_asset.prompt,
                                'type': 'location'
                            }
                        ]
                    })
                    logger.info(f"Found asset for location {normalized_name} in direction {direction}: {loc_asset.url}")
                else:
                    logger.warning(f"No asset found for location {normalized_name} with direction {direction}")
            else:
                # No direction specified, use original behavior (get all assets)
                normalized_name = normalize_asset_name(shot.locations)

                # Get all assets for this location
                loc_assets = self.asset_library.get_all_assets_for_character(normalized_name)  # Uses same method
                if loc_assets:
                    assets_info['locations'].append({
                        'name': normalized_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'type': 'location'
                            }
                            for asset in loc_assets
                        ]
                    })
                else:
                    logger.warning(f"No assets found for location: {normalized_name}")

        # Props: Keep original description-based matching
        # (Props are not in CSV, so search through description)
        description = shot.description.lower()
        available_props = self.asset_library.get_available_props()

        for prop_name in available_props:
            prop_lower = prop_name.lower().replace('_', ' ')
            if prop_lower in description:
                # Get all assets for this prop
                prop_assets = self.asset_library.get_all_assets_for_character(prop_name)  # Uses same method
                if prop_assets:
                    assets_info['props'].append({
                        'name': prop_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'type': 'prop'
                            }
                            for asset in prop_assets
                        ]
                    })

        logger.info(f"Found assets for shot {shot.shot_id}: "
                   f"{len(assets_info['characters'])} characters, "
                   f"{len(assets_info['locations'])} locations, "
                   f"{len(assets_info['props'])} props")

        return assets_info

    def _load_pil_image(self, asset: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Load PIL image from asset information (local path or S3 URL).

        Args:
            asset: Asset dictionary containing 'local_path', 'url', etc.

        Returns:
            PIL Image object or None if loading fails
        """
        try:
            # Try local path first
            local_path = asset.get('local_path')
            if local_path and os.path.exists(local_path):
                logger.debug(f"Loading image from local path: {local_path}")
                return Image.open(local_path)

            # Try URL (S3) if local path doesn't exist
            url = asset.get('url')
            if url:
                logger.debug(f"Loading image from URL: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))

            logger.warning(f"No valid local_path or url found in asset: {asset}")
            return None

        except Exception as e:
            logger.error(f"Error loading PIL image from asset: {e}")
            return None

    def _build_review_prompt(
        self,
        annotated_list: AnnotatedShotList,
        scene_description: Optional[str] = None,
        product_shot_ids: Optional[set] = None
    ) -> str:
        """Build the review prompt with all shot information."""
        prompt_parts = []
        
        # Scene description
        if scene_description:
            prompt_parts.append("=== SCENE DESCRIPTION (OVERALL CONTEXT) ===")
            prompt_parts.append(scene_description)
            prompt_parts.append("")
        
        # Episode information
        prompt_parts.append("=== EPISODE INFORMATION ===")
        prompt_parts.append(f"Episode ID: {annotated_list.episode_id}")
        if annotated_list.title:
            prompt_parts.append(f"Title: {annotated_list.title}")
        prompt_parts.append(f"Total Shots: {len(annotated_list.annotated_shots)}")
        prompt_parts.append("")
        
        # Overall continuity notes from strategy agent
        if annotated_list.overall_continuity_notes:
            prompt_parts.append("=== OVERALL CONTINUITY NOTES ===")
            prompt_parts.append(annotated_list.overall_continuity_notes)
            prompt_parts.append("")
        
        # All shots with their prompts
        prompt_parts.append("=== SHOTS TO REVIEW ===")
        prompt_parts.append("Review the following shots for visual and narrative continuity:")
        prompt_parts.append("")
        
        for i, shot in enumerate(annotated_list.annotated_shots):
            is_product = bool(product_shot_ids and shot.shot_id in product_shot_ids)
            product_tag = " [PRODUCT SHOT — PRODUCT MUST BE CLEARLY VISIBLE IN FRAME]" if is_product else ""
            prompt_parts.append(f"--- Shot {i+1}/{len(annotated_list.annotated_shots)}{product_tag} ---")
            prompt_parts.append(f"Shot ID: {shot.shot_id}")
            prompt_parts.append(f"Description: {shot.description}")
            prompt_parts.append(f"Generation Strategy: {shot.generation_strategy}")
            
            if shot.seed_shot_id:
                prompt_parts.append(f"Seed Shot ID: {shot.seed_shot_id}")
            
            # Get reasoning from versioned structure or legacy field
            reasoning = None
            if shot.image and "v0" in shot.image:
                reasoning = shot.image["v0"].get("reasoning")
            elif hasattr(shot, 'reasoning') and shot.reasoning:
                reasoning = shot.reasoning
            
            if reasoning:
                prompt_parts.append(f"Strategy Reasoning: {reasoning}")
            
            if shot.continuity_notes:
                prompt_parts.append(f"Continuity Notes: {shot.continuity_notes}")
            
            if shot.optimized_ai_notes:
                prompt_parts.append(f"AI Notes: {shot.optimized_ai_notes}")
            
            # Check for existing image prompts in versioned structure
            current_prompt = None
            if shot.image and "v0" in shot.image:
                current_prompt = shot.image["v0"].get("updated_prompt")
            elif hasattr(shot, 'prompt_image_draft') and shot.prompt_image_draft:
                current_prompt = shot.prompt_image_draft
            
            if current_prompt:
                prompt_parts.append(f"CURRENT PROMPT: {current_prompt}")
            else:
                prompt_parts.append("CURRENT PROMPT: [NO PROMPT GENERATED]")
            
            prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("=== YOUR TASK ===")
        prompt_parts.append(
            "Review ALL prompts above and ensure visual and narrative continuity. "
            "For each shot, compare with:\n"
            "1. The previous shot (for consecutive continuity)\n"
            "2. The seed shot (if applicable, for strategy-based continuity)\n"
            "3. All other shots (for global consistency)\n\n"
            "Return a JSON array with one object per shot in this EXACT format:\n"
            "[\n"
            "  {\n"
            '    "shot_id": "S01E01_001",\n'
            '    "original_prompt": "...",\n'
            '    "reviewed_prompt": "...",\n'
            '    "changes_made": ["Change 1", "Change 2"],\n'
            '    "shot_modified": false,\n'
            '    "reason_for_modification": "",\n'
            '    "continuity_observations": ["Observation 1", "Observation 2"]\n'
            "  }\n"
            "]\n\n"
            "Output ONLY the JSON array. No markdown, no code fences, no explanation."
        )
        
        return "\n".join(prompt_parts)
    
    def _split_annotated_list(
        self,
        annotated_list: AnnotatedShotList
    ) -> List[AnnotatedShotList]:
        """Split annotated list into batches to avoid oversized prompts."""
        total_shots = len(annotated_list.annotated_shots)
        if total_shots <= self.max_shots_per_call:
            return [annotated_list]
        
        batches: List[AnnotatedShotList] = []
        for i in range(0, total_shots, self.max_shots_per_call):
            batch_shots = annotated_list.annotated_shots[i:i + self.max_shots_per_call]
            # Convert AnnotatedShotItem instances to dicts to avoid Pydantic validation issues
            batch_shots_dicts = [shot.model_dump() for shot in batch_shots]
            batch = AnnotatedShotList(
                episode_id=annotated_list.episode_id,
                title=annotated_list.title,
                scene_description=annotated_list.scene_description,
                annotated_shots=batch_shots_dicts,  # Pass dicts instead of instances
                overall_continuity_notes=annotated_list.overall_continuity_notes,
                strategy_summary=annotated_list.strategy_summary,
                processing_metadata=annotated_list.processing_metadata
            )
            batches.append(batch)
        
        logger.info(
            f"Split {total_shots} shots into {len(batches)} batches "
            f"(max {self.max_shots_per_call} per call)"
        )
        return batches
    
    def _create_fallback_reviews_for_shots(
        self,
        shots: List[AnnotatedShotItem],
        reason: str
    ) -> List[Dict[str, Any]]:
        """Create fallback reviews for a subset of shots."""
        fallback_reviews = []
        for shot in shots:
            original_prompt = ""
            if shot.image and "v0" in shot.image:
                original_prompt = shot.image["v0"].get("updated_prompt", "")
            elif hasattr(shot, 'prompt_image_draft') and shot.prompt_image_draft:
                original_prompt = shot.prompt_image_draft

            fallback_reviews.append({
                "shot_id": shot.shot_id,
                "original_prompt": original_prompt,
                "reviewed_prompt": original_prompt,
                "changes_made": [f"Fallback: {reason}"],
                "shot_modified": False,
                "reason_for_modification": reason,
                "continuity_observations": ["Review failed - using original prompt"],
                "continuity_status": "Pass"
            })

        logger.warning(f"Created {len(fallback_reviews)} fallback reviews ({reason})")
        return fallback_reviews
    
    def _parse_review_response_fallback(self, response: str, annotated_list: AnnotatedShotList) -> List[Dict[str, Any]]:
        """Fallback method to parse JSON response when structured output fails."""
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
            
            # Try to parse as JSON
            try:
                review_data = json.loads(response)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*"reviews".*\}', response, re.DOTALL)
                if json_match:
                    review_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Handle both array format and object with reviews array
            if isinstance(review_data, list):
                reviews = review_data
            elif isinstance(review_data, dict) and "reviews" in review_data:
                reviews = review_data["reviews"]
            else:
                raise ValueError("Invalid response format")
            
            # Validate each review item has required fields
            validated_reviews = []
            for item in reviews:
                if not isinstance(item, dict):
                    continue
                    
                # Ensure all required fields are present
                validated_item = {
                    "shot_id": item.get("shot_id", ""),
                    "original_prompt": item.get("original_prompt", ""),
                    "reviewed_prompt": item.get("reviewed_prompt", ""),
                    "changes_made": item.get("changes_made", []),
                    "shot_modified": item.get("shot_modified", False),
                    "reason_for_modification": item.get("reason_for_modification", ""),
                    "continuity_observations": item.get("continuity_observations", []),
                    "continuity_status": item.get("continuity_status", "Pass")
                }
                validated_reviews.append(validated_item)
            
            logger.info(f"Successfully parsed {len(validated_reviews)} review items using fallback method")
            return validated_reviews
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {str(e)}")
            # Create minimal fallback reviews for each shot
            return self._create_fallback_reviews_for_shots(
                annotated_list.annotated_shots,
                "No changes made due to parsing error"
            )
    
    def _create_fallback_reviews(self, annotated_list: AnnotatedShotList) -> List[Dict[str, Any]]:
        """Create fallback reviews when all parsing methods fail."""
        return self._create_fallback_reviews_for_shots(
            annotated_list.annotated_shots,
            "No changes made due to API error"
        )
    
    def _create_updated_list_with_fallback(self, annotated_list: AnnotatedShotList, review_results: List[Dict[str, Any]]) -> AnnotatedShotList:
        """Create updated annotated list with fallback reviews."""
        updated_shots = []
        for i, shot in enumerate(annotated_list.annotated_shots):
            # Find corresponding review result
            review = next(
                (r for r in review_results if r['shot_id'] == shot.shot_id),
                None
            )
            
            if review:
                reviewed_prompt = review['reviewed_prompt']
            else:
                # Fallback to original if no review found
                if shot.image and "v0" in shot.image:
                    reviewed_prompt = shot.image["v0"].get("updated_prompt", "")
                elif hasattr(shot, 'prompt_image_draft') and shot.prompt_image_draft:
                    reviewed_prompt = shot.prompt_image_draft
                else:
                    reviewed_prompt = "No prompt available"
            
            # Create updated shot with new versioned structure
            updated_shot = AnnotatedShotItem(
                shot_id=shot.shot_id,
                description=shot.description,
                duration=shot.duration,
                scene_number=shot.scene_number,
                sequence_number=shot.sequence_number,
                shot_style=shot.shot_style,
                camera_movement=shot.camera_movement,
                source_type=shot.source_type,
                uploaded_image_id=shot.uploaded_image_id,
                generated_image_id=shot.generated_image_id,
                generated_video_id=shot.generated_video_id,
                optimized_ai_notes=shot.optimized_ai_notes,
                characters=shot.characters,  # Preserve characters from CSV
                locations=shot.locations,  # Preserve locations from CSV
                generation_strategy=shot.generation_strategy,
                continuity_notes=shot.continuity_notes,
                confidence_score=shot.confidence_score,
                seed_shot_id=shot.seed_shot_id,
                # Preserve existing image versions and add new v1
                image=shot.image if shot.image else {},
                video=shot.video if shot.video else {}
            )
            
            # Add the reviewed prompt as v1 in the image structure
            if not updated_shot.image:
                updated_shot.image = {}
            
            # Build changes_made string from the review results
            changes_made = []
            if review:
                if review.get('changes_made'):
                    changes_made.extend(review['changes_made'])
                if review.get('continuity_observations'):
                    changes_made.extend([f"Continuity: {obs}" for obs in review['continuity_observations']])
            
            # Build reasoning from review results
            reasoning_parts = []
            if review:
                if review.get('reason_for_modification'):
                    reasoning_parts.append(f"Modification reason: {review['reason_for_modification']}")
                if review.get('continuity_observations'):
                    reasoning_parts.append(f"Continuity observations: {', '.join(review['continuity_observations'])}")
                if review.get('shot_modified'):
                    reasoning_parts.append("Shot was modified for continuity")
            
            reasoning = '; '.join(reasoning_parts) if reasoning_parts else 'Continuity review completed'
            
            updated_shot.image["v1"] = {
                "updated_prompt": reviewed_prompt,
                "changes_made": changes_made if changes_made else ['Continuity review applied'],
                "reasoning": reasoning,
                "generated_images_s3": []
            }
            
            updated_shots.append(updated_shot)
        
        # Create updated annotated list
        updated_list = AnnotatedShotList(
            episode_id=annotated_list.episode_id,
            title=annotated_list.title,
            annotated_shots=updated_shots,
            overall_continuity_notes=annotated_list.overall_continuity_notes,
            strategy_summary=annotated_list.strategy_summary
        )
        
        return updated_list
    
    def _merge_review_results(
        self,
        existing: List[Dict[str, Any]],
        new_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge review results, preferring latest entries for duplicate shot IDs."""
        merged = {item.get("shot_id"): item for item in existing if item.get("shot_id")}
        for item in new_items:
            shot_id = item.get("shot_id")
            if not shot_id:
                continue
            merged[shot_id] = item
        return list(merged.values())
    
    def _retry_missing_shots(
        self,
        annotated_list: AnnotatedShotList,
        scene_description: Optional[str],
        missing_shot_ids: Set[str]
    ) -> List[Dict[str, Any]]:
        """Retry LLM call for a subset of missing shots."""
        missing_shots = [
            shot for shot in annotated_list.annotated_shots
            if shot.shot_id in missing_shot_ids
        ]
        if not missing_shots:
            return []
        
        sublist = AnnotatedShotList(
            episode_id=annotated_list.episode_id,
            title=annotated_list.title,
            scene_description=annotated_list.scene_description,
            annotated_shots=missing_shots,
            overall_continuity_notes=annotated_list.overall_continuity_notes,
            strategy_summary=annotated_list.strategy_summary,
            processing_metadata=annotated_list.processing_metadata
        )
        
        logger.info(f"Retrying {len(missing_shots)} missing shot reviews")
        user_prompt = self._build_review_prompt(sublist, scene_description)
        system_prompt = self._get_system_prompt()
        response = self._get_review_recommendations(user_prompt, system_prompt, sublist)
        return self._parse_review_response_structured(response, sublist)
    
    def _ensure_complete_batch(
        self,
        batch_list: AnnotatedShotList,
        batch_results: List[Dict[str, Any]],
        scene_description: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Ensure a batch has review results for all shots via retry/fallback."""
        expected_ids = {shot.shot_id for shot in batch_list.annotated_shots}
        present_ids = {item.get("shot_id") for item in batch_results if item.get("shot_id")}
        missing_ids = expected_ids - present_ids
        
        if not missing_ids:
            return batch_results
        
        logger.warning(
            f"Batch incomplete: missing {len(missing_ids)} shot reviews. "
            "Attempting retry for missing shots."
        )
        retry_results = self._retry_missing_shots(batch_list, scene_description, missing_ids)
        batch_results = self._merge_review_results(batch_results, retry_results)
        
        present_ids = {item.get("shot_id") for item in batch_results if item.get("shot_id")}
        still_missing = expected_ids - present_ids
        if still_missing:
            logger.error(
                f"Retry still missing {len(still_missing)} shot reviews. "
                "Falling back to original prompts for remaining shots."
            )
            missing_shots = [
                shot for shot in batch_list.annotated_shots
                if shot.shot_id in still_missing
            ]
            fallback_reviews = self._create_fallback_reviews_for_shots(
                missing_shots,
                "No changes made due to truncated LLM response"
            )
            batch_results = self._merge_review_results(batch_results, fallback_reviews)
        
        return batch_results
    
    def _parse_review_response(self, response: str, annotated_list: AnnotatedShotList) -> List[Dict[str, Any]]:
        """Parse the JSON review response from Gemini with improved error handling."""
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
            
            # Try to find the JSON array in the response
            # Look for the first '[' and last ']' to extract just the JSON part
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response = response[start_idx:end_idx + 1]
            
            # Parse JSON
            review_data = json.loads(response)
            
            # Validate format
            if not isinstance(review_data, list):
                raise ValueError("Review response must be a JSON array")
            
            # Validate each review item has required fields
            validated_reviews = []
            for item in review_data:
                if not isinstance(item, dict):
                    continue
                    
                # Ensure all required fields are present
                validated_item = {
                    "shot_id": item.get("shot_id", ""),
                    "original_prompt": item.get("original_prompt", ""),
                    "reviewed_prompt": item.get("reviewed_prompt", ""),
                    "changes_made": item.get("changes_made", []),
                    "shot_modified": item.get("shot_modified", False),
                    "reason_for_modification": item.get("reason_for_modification", ""),
                    "continuity_observations": item.get("continuity_observations", []),
                    "continuity_status": item.get("continuity_status", "Pass")
                }
                validated_reviews.append(validated_item)
            
            logger.info(f"Successfully parsed {len(validated_reviews)} review items")
            return validated_reviews
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse review response as JSON: {str(e)}")
            logger.error(f"Response preview: {response[:500]}...")
            # Try to extract partial JSON if possible
            try:
                # Look for individual JSON objects and try to parse them
                import re
                json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response)
                if json_objects:
                    logger.info(f"Found {len(json_objects)} potential JSON objects, attempting to parse...")
                    parsed_objects = []
                    for obj_str in json_objects:
                        try:
                            obj = json.loads(obj_str)
                            parsed_objects.append(obj)
                        except:
                            continue
                    if parsed_objects:
                        logger.info(f"Successfully parsed {len(parsed_objects)} objects from partial JSON")
                        return parsed_objects
            except Exception as partial_e:
                logger.error(f"Failed to parse partial JSON: {partial_e}")
            
            # If all parsing fails, create fallback reviews
            logger.warning("All JSON parsing methods failed, creating fallback reviews")
            return self._create_fallback_reviews(annotated_list)
            
        except Exception as e:
            logger.error(f"Error parsing review response: {str(e)}")
            # Create fallback reviews
            return self._create_fallback_reviews(annotated_list)
    
    def _repair_missing_fields(self, raw_data: dict) -> dict:
        """Repair missing required fields in raw LLM output before Pydantic validation."""
        if not isinstance(raw_data, dict):
            return raw_data

        # Repair reviews array if present
        if "reviews" in raw_data and isinstance(raw_data["reviews"], list):
            for item in raw_data["reviews"]:
                if isinstance(item, dict):
                    # Ensure required fields exist with defaults
                    item.setdefault("shot_id", "")
                    item.setdefault("reviewed_prompt", "")
                    item.setdefault("shot_modified", False)
                    # Ensure optional fields have defaults
                    item.setdefault("original_prompt", "")
                    item.setdefault("changes_made", [])
                    item.setdefault("reason_for_modification", "")
                    item.setdefault("continuity_observations", [])
                    item.setdefault("continuity_status", "Pass")

        return raw_data
    
    def _get_review_recommendations(
        self,
        user_prompt: str,
        system_prompt: str,
        annotated_list: AnnotatedShotList,
        product_image_url: Optional[str] = None
    ) -> PromptReviewResponse:
        """Get review recommendations from LLM using structured output with asset images."""
        try:
            # Fetch assets for all shots in the batch
            all_assets = []
            for shot in annotated_list.annotated_shots:
                assets_info = self._get_assets_for_shot(shot)
                if assets_info:
                    all_assets.append((shot.shot_id, assets_info))

            # Prepare multimodal content for Gemini
            content_parts = [{"type": "text", "text": user_prompt}]

            # Add product reference image first so the reviewer knows what the product looks like
            if product_image_url:
                try:
                    from app.services.phase_2_agents.helpers.image_fetch import fetch_image_bytes
                    img_bytes = fetch_image_bytes(product_image_url)
                    if img_bytes:
                        product_pil = Image.open(BytesIO(img_bytes))
                        buf = BytesIO()
                        product_pil.save(buf, format='PNG')
                        b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        })
                        logger.info("Added PRODUCT reference image to review agent context")
                    else:
                        logger.warning("Could not fetch product image for review agent — skipping")
                except Exception as e:
                    logger.warning(f"Failed to load product image for review agent: {e}")

            # Add asset images to content if available
            for shot_id, assets_info in all_assets:
                # Process character assets
                for char_info in assets_info.get('characters', []):
                    char_name = char_info['name']
                    for asset in char_info['assets']:
                        try:
                            pil_image = self._load_pil_image(asset)
                            if pil_image:
                                buf = BytesIO()
                                pil_image.save(buf, format='PNG')
                                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                                })
                                logger.info(f"Added character asset image for {char_name} ({asset.get('angle', 'unknown')} angle) - Shot {shot_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load character asset {char_name} for shot {shot_id}: {e}")

                # Process location assets
                for loc_info in assets_info.get('locations', []):
                    loc_name = loc_info['name']
                    for asset in loc_info['assets']:
                        try:
                            pil_image = self._load_pil_image(asset)
                            if pil_image:
                                buf = BytesIO()
                                pil_image.save(buf, format='PNG')
                                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                                })
                                logger.info(f"Added location asset image for {loc_name} ({asset.get('angle', 'unknown')} angle) - Shot {shot_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load location asset {loc_name} for shot {shot_id}: {e}")

                # Process prop assets
                for prop_info in assets_info.get('props', []):
                    prop_name = prop_info['name']
                    for asset in prop_info['assets']:
                        try:
                            pil_image = self._load_pil_image(asset)
                            if pil_image:
                                buf = BytesIO()
                                pil_image.save(buf, format='PNG')
                                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                                })
                                logger.info(f"Added prop asset image for {prop_name} ({asset.get('angle', 'unknown')} angle) - Shot {shot_id}")
                        except Exception as e:
                            logger.warning(f"Failed to load prop asset {prop_name} for shot {shot_id}: {e}")

            logger.info(f"Prepared {len(content_parts)} content parts (1 text + {len(content_parts)-1} images) for review agent")

            # Create the messages with multimodal content
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts)
            ]

            # Generate response with structured output
            # When using with_structured_output(), the LLM returns the Pydantic model directly
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import AIMessage
            
            if isinstance(self.llm, BaseChatModel) or hasattr(self.llm, 'with_structured_output'):
                # With structured output, invoke returns the Pydantic model directly
                try:
                    response = self.llm.invoke(messages)
                except Exception as invoke_error:
                    logger.error(f"LLM invoke failed with structured output: {invoke_error}")
                    logger.error(f"Error type: {type(invoke_error)}")
                    # Re-raise to let outer exception handler deal with it
                    raise ValueError(f"LLM invoke failed: {str(invoke_error)}") from invoke_error
                
                # Check if response is None - this can happen if structured output parsing fails
                if response is None:
                    logger.warning("LLM returned None with structured output - attempting retry with same structured output")
                    # Retry the structured output call once
                    try:
                        response = self.llm.invoke(messages)
                        if response is None:
                            logger.error("Retry also returned None - structured output is failing")
                            raise ValueError("LLM returned None response after retry. Structured output parsing failed.")
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        raise ValueError(f"Structured output failed after retry: {str(retry_error)}") from retry_error
                
                # If response is already a Pydantic model, return it
                if isinstance(response, PromptReviewResponse):
                    logger.info(f"Received structured response with {len(response.reviews)} reviews")
                    return response
                
                # Check if response is an AIMessage (sometimes structured output wraps it)
                if isinstance(response, AIMessage):
                    # Try to get content from AIMessage
                    if hasattr(response, 'content') and response.content:
                        try:
                            # Try parsing as JSON first, with field repair
                            if isinstance(response.content, str):
                                raw_data = json.loads(response.content)
                                raw_data = self._repair_missing_fields(raw_data)
                                return PromptReviewResponse.model_validate(raw_data)
                            elif isinstance(response.content, dict):
                                raw_data = self._repair_missing_fields(response.content)
                                return PromptReviewResponse.model_validate(raw_data)
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse AIMessage content: {parse_error}")
                            # If parsing fails, check if there's a parsed attribute
                            if hasattr(response, 'parsed') and response.parsed:
                                if isinstance(response.parsed, PromptReviewResponse):
                                    return response.parsed
                                elif isinstance(response.parsed, dict):
                                    raw_data = self._repair_missing_fields(response.parsed)
                                    return PromptReviewResponse.model_validate(raw_data)
                    
                    # If AIMessage has no valid content, raise error
                    logger.error(f"AIMessage has no valid content. Response: {response}")
                    raise ValueError("AIMessage response has no valid content to parse")
                
                # Fallback: if response has content attribute, try to parse it
                elif hasattr(response, "content"):
                    # This shouldn't happen with structured output, but handle gracefully
                    logger.warning("Received response with content attribute instead of structured output")
                    if response.content is None:
                        raise ValueError("Response content is None")
                    if isinstance(response.content, str):
                        raw_data = json.loads(response.content)
                        raw_data = self._repair_missing_fields(raw_data)
                        return PromptReviewResponse.model_validate(raw_data)
                    elif isinstance(response.content, dict):
                        raw_data = self._repair_missing_fields(response.content)
                        return PromptReviewResponse.model_validate(raw_data)
                    else:
                        raise ValueError(f"Unexpected content type: {type(response.content)}")
                else:
                    # Try to parse as JSON string, but check for None first
                    response_str = str(response)
                    if response_str == "None" or response_str.strip() == "":
                        logger.error(f"Invalid response: {response_str} (type: {type(response)})")
                        raise ValueError(f"LLM returned invalid response: {response_str}")
                    raw_data = json.loads(response_str)
                    raw_data = self._repair_missing_fields(raw_data)
                    return PromptReviewResponse.model_validate(raw_data)
            else:
                # Fallback for non-chat models (shouldn't happen with structured output)
                response = self.llm.invoke(messages)
                if response is None:
                    raise ValueError("LLM returned None response")
                if isinstance(response, PromptReviewResponse):
                    return response
                response_str = str(response)
                if response_str == "None" or response_str.strip() == "":
                    raise ValueError(f"LLM returned invalid response: {response_str}")
                raw_data = json.loads(response_str)
                raw_data = self._repair_missing_fields(raw_data)
                return PromptReviewResponse.model_validate(raw_data)
                
        except Exception as e:
            logger.error(f"Error getting review recommendations: {str(e)}")
            if 'response' in locals():
                logger.error(f"Response type: {type(response)}")
                logger.error(f"Response value: {repr(response)[:500] if response is not None else 'None'}")
                if hasattr(response, '__dict__'):
                    logger.error(f"Response attributes: {list(response.__dict__.keys())}")
            raise ValueError(f"LLM analysis failed: {str(e)}")
    
    def _parse_review_response_structured(
        self, 
        response: PromptReviewResponse, 
        annotated_list: AnnotatedShotList
    ) -> List[Dict[str, Any]]:
        """Parse structured Pydantic response and create review results."""
        try:
            # Response is already a validated Pydantic model
            if not isinstance(response, PromptReviewResponse):
                # Fallback: try to parse if it's a dict or string
                if isinstance(response, dict):
                    # Repair missing fields before validation
                    response = self._repair_missing_fields(response)
                    response = PromptReviewResponse.model_validate(response)
                elif isinstance(response, str):
                    # Parse JSON, repair, then validate
                    raw_data = json.loads(response)
                    raw_data = self._repair_missing_fields(raw_data)
                    response = PromptReviewResponse.model_validate(raw_data)
                else:
                    raise ValueError(f"Unexpected response type: {type(response)}")
            
            # Validate completeness: check if we got responses for all shots
            review_items = response.reviews
            if len(review_items) < len(annotated_list.annotated_shots):
                logger.warning(
                    f"INCOMPLETE RESPONSE: Expected {len(annotated_list.annotated_shots)} reviews, "
                    f"but only received {len(review_items)}. Response may be truncated."
                )
            
            # Convert Pydantic models to dictionaries for backward compatibility
            review_results = []
            for review_item in review_items:
                review_dict = {
                    "shot_id": review_item.shot_id,
                    "original_prompt": review_item.original_prompt,
                    "reviewed_prompt": review_item.reviewed_prompt,
                    "changes_made": review_item.changes_made,
                    "shot_modified": review_item.shot_modified,
                    "reason_for_modification": review_item.reason_for_modification,
                    "continuity_observations": review_item.continuity_observations,
                    "continuity_status": review_item.continuity_status
                }
                review_results.append(review_dict)
            
            logger.info(f"Successfully parsed {len(review_results)} review items from structured output")
            return review_results
            
        except Exception as e:
            logger.error(f"Error parsing review response: {e}")
            raise RuntimeError(f"Failed to parse review response: {e}")
    
    async def review_prompts(
        self,
        annotated_list: AnnotatedShotList,
        scene_description: Optional[str] = None,
        product_shot_ids: Optional[set] = None,
        product_image_url: Optional[str] = None
    ) -> tuple[AnnotatedShotList, List[Dict[str, Any]]]:
        """
        Review all image prompts for continuity.

        Args:
            annotated_list: List of annotated shots with draft prompts
            scene_description: Overall scene/episode description

        Returns:
            Tuple of (updated_annotated_list, review_results)
        """
        logger.info(f"Reviewing image prompts for {len(annotated_list.annotated_shots)} shots")

        try:
            system_prompt = self._get_system_prompt()
            batches = self._split_annotated_list(annotated_list)
            all_review_results: List[Dict[str, Any]] = []
            
            for idx, batch in enumerate(batches, start=1):
                logger.info(
                    f"Processing review batch {idx}/{len(batches)} "
                    f"with {len(batch.annotated_shots)} shots"
                )
                user_prompt = self._build_review_prompt(batch, scene_description, product_shot_ids=product_shot_ids)

                review_response = self._get_review_recommendations(user_prompt, system_prompt, batch, product_image_url=product_image_url)
                batch_results = self._parse_review_response_structured(review_response, batch)
                batch_results = self._ensure_complete_batch(batch, batch_results, scene_description)
                all_review_results.extend(batch_results)
            
            # Step 3: Update shots with reviewed prompts using the helper method
            review_results = all_review_results
            updated_list = self._create_updated_list_with_fallback(annotated_list, review_results)

            logger.info(f"Successfully reviewed prompts for all {len(updated_list.annotated_shots)} shots")

            return updated_list, review_results

        except Exception as e:
            logger.error(f"Error reviewing prompts: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to original parsing method if structured output fails
            try:
                logger.warning("Attempting fallback parsing method")
                system_prompt = self._get_system_prompt()
                fallback_results: List[Dict[str, Any]] = []
                
                batches = self._split_annotated_list(annotated_list)
                for batch in batches:
                    user_prompt = self._build_review_prompt(batch, scene_description, product_shot_ids=product_shot_ids)
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    api_key = os.getenv("GOOGLE_API_KEY")
                    fallback_llm = ChatGoogleGenerativeAI(
                        model=self.model_name,
                        temperature=self.temperature,
                        google_api_key=api_key,
                        max_output_tokens=self.max_tokens or 8192,
                        transport="rest"
                    )
                    
                    response = fallback_llm.invoke(messages)
                    raw_response = response.content if hasattr(response, "content") else str(response)
                    batch_results = self._parse_review_response(raw_response, batch)
                    batch_results = self._ensure_complete_batch(batch, batch_results, scene_description)
                    fallback_results.extend(batch_results)
                
                updated_list = self._create_updated_list_with_fallback(annotated_list, fallback_results)
                
                logger.info(f"Fallback parsing succeeded: {len(fallback_results)} reviews")
                return updated_list, fallback_results
            except Exception as fallback_error:
                logger.error(f"Fallback parsing also failed: {str(fallback_error)}")
                raise
    


async def review_image_prompts(
    annotated_list: AnnotatedShotList,
    mongodb_client: Optional[MongoDBAtlasClient] = None,
    scene_description: Optional[str] = None,
    show_id: Optional[str] = None,
    episode_number: Optional[int] = None,
    asset_library: Optional[AssetLibrary] = None
) -> tuple[AnnotatedShotList, Dict[str, Any]]:
    """
    Main function to review image prompts and optionally save to MongoDB.

    Args:
        annotated_list: Annotated shot list with draft prompts from Agent 2
        mongodb_client: Optional MongoDB client to update documents
        scene_description: Overall scene/episode description
        show_id: Show ID for MongoDB updates
        episode_number: Episode number for MongoDB updates
        asset_library: Optional AssetLibrary instance for accessing Agent 5 and Agent 8 assets

    Returns:
        Tuple of (updated_annotated_list, review_summary)
    """
    logger.info("Starting prompt review pipeline")

    # Initialize agent with AssetLibrary
    agent = PromptReviewAgent(asset_library=asset_library)

    # Review prompts for all shots
    updated_list, review_results = await agent.review_prompts(annotated_list, scene_description)
    
    # Create review summary for saving
    review_data = {
        "episode_id": annotated_list.episode_id,
        "title": annotated_list.title,
        "reviewed_at": datetime.now().isoformat(),
        "total_shots": len(updated_list.annotated_shots),
        "shots_modified": len([r for r in review_results if r.get('shot_modified', False)]),
        "strategy_summary": annotated_list.strategy_summary,
        "overall_continuity_notes": annotated_list.overall_continuity_notes,
        "shot_reviews": review_results
    }
    
    # Save review to file if enabled
    if agent.enable_saving:
        try:
            saved_file = save_review_to_file(
                review_data,
                annotated_list.episode_id,
                agent.output_dir
            )
            logger.info(f"Review results saved to: {saved_file}")
        except Exception as e:
            logger.warning(f"Failed to save review to file: {e}")
    
    # Update MongoDB if client is provided
    if mongodb_client and show_id and episode_number:
        logger.info(
            f"Updating MongoDB with reviewed prompts "
            f"(show_id={show_id}, episode_number={episode_number})"
        )
        
        updated_count = 0
        not_found_count = 0
        
        try:
            for shot in updated_list.annotated_shots:
                # Update using new versioned structure
                if shot.image and "v1" in shot.image:
                    v1_data = shot.image["v1"]
                    mongodb_client.update_shot_image_version(
                        show_id=show_id,
                        episode_number=episode_number,
                        shot_id=shot.shot_id,
                        version="v1",
                        updated_prompt=v1_data.get("updated_prompt", ""),
                        changes_made=", ".join(v1_data.get("changes_made", [])),
                        reasoning=v1_data.get("reasoning", ""),
                        generated_images_s3=v1_data.get("generated_images_s3", [])
                    )
                    updated_count += 1
                    logger.info(f"✅ Updated shot {shot.shot_id} with reviewed prompt (v1)")
                else:
                    not_found_count += 1
                    logger.warning(f"❌ Shot {shot.shot_id} not found or no v1 data available")
            
            logger.info(
                f"MongoDB Update Summary: {updated_count} updated, {not_found_count} not found "
                f"out of {len(updated_list.annotated_shots)} total shots"
            )
            
            if updated_count > 0:
                logger.info(f"✅ Successfully updated {updated_count} reviewed prompts in MongoDB")
            else:
                logger.warning("⚠️  No reviewed prompts were updated in MongoDB")
            
        except Exception as e:
            logger.error(f"Error updating MongoDB with reviewed prompts: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    else:
        if not mongodb_client:
            logger.warning("MongoDB client not provided - skipping MongoDB updates")
        elif not show_id:
            logger.warning("show_id not provided - skipping MongoDB updates")
        elif not episode_number:
            logger.warning("episode_number not provided - skipping MongoDB updates")

    # Update production_projects with agent10 output (full per-shot review data)
    if mongodb_client and show_id:
        try:
            from backend.services.production.app.config import get_database
            from bson import ObjectId
            from backend.shared.utils.mongodb_validators import validate_object_id
            from fastapi import HTTPException
            
            try:
                show_id_obj = validate_object_id(show_id)
            except (ValueError, HTTPException) as e:
                logger.error(f"Invalid show_id format for Agent 10 output: {e}")
                raise ValueError(f"Invalid show_id") from e

            # Prepare full per-shot review data
            shots_data = []
            for shot in updated_list.annotated_shots:
                if shot.image and "v1" in shot.image:
                    v1_data = shot.image["v1"]
                    # Find corresponding review result
                    review = next(
                        (r for r in review_results if r.get('shot_id') == shot.shot_id),
                        None
                    )

                    # Capture input assets used for this shot
                    assets_info = agent._get_assets_for_shot(shot)
                    input_assets = {
                        "characters": [
                            {
                                "name": char_info['name'],
                                "s3_urls": [asset['url'] for asset in char_info['assets'] if asset.get('url')],
                                "angles": [asset['angle'] for asset in char_info['assets'] if asset.get('angle')]
                            }
                            for char_info in assets_info.get('characters', [])
                        ],
                        "locations": [
                            {
                                "name": loc_info['name'],
                                "s3_urls": [asset['url'] for asset in loc_info['assets'] if asset.get('url')],
                                "angles": [asset['angle'] for asset in loc_info['assets'] if asset.get('angle')]
                            }
                            for loc_info in assets_info.get('locations', [])
                        ],
                        "props": [
                            {
                                "name": prop_info['name'],
                                "s3_urls": [asset['url'] for asset in prop_info['assets'] if asset.get('url')],
                                "angles": [asset['angle'] for asset in prop_info['assets'] if asset.get('angle')]
                            }
                            for prop_info in assets_info.get('props', [])
                        ]
                    }

                    shots_data.append({
                        "shot_id": shot.shot_id,
                        "input_assets": input_assets,
                        "original_prompt": review.get("original_prompt", "") if review else "",
                        "reviewed_prompt": v1_data.get("updated_prompt", ""),
                        "changes_made": v1_data.get("changes_made", []),
                        "reasoning": v1_data.get("reasoning", ""),
                        "generated_images_s3": v1_data.get("generated_images_s3", []),
                        "shot_modified": review.get("shot_modified", False) if review else False,
                        "continuity_observations": review.get("continuity_observations", []) if review else [],
                    })

            agent10_output = {
                "episode_id": annotated_list.episode_id,
                "title": annotated_list.title,
                "total_shots": len(updated_list.annotated_shots),
                "shots_modified": len([r for r in review_results if r.get('shot_modified', False)]),
                "shots": shots_data,
                "reviewed_at": datetime.now().isoformat()
            }

            # Update production_projects
            client, db = get_database()
            projects_col = db["production_projects"]

            result = projects_col.update_one(
                {"_id": show_id_obj},
                {
                    "$set": {
                        "agent_outputs.agent10.status": "completed",
                        "agent_outputs.agent10.executed_at": datetime.utcnow(),
                        "agent_outputs.agent10.output": agent10_output,
                        "agent_outputs.agent10.description": "Prompt Review Agent (v1 refined prompts)",
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"✅ Agent 10 output saved to production_projects (show_id: {show_id})")
            else:
                logger.warning(f"⚠️  Failed to save Agent 10 output to production_projects")

        except Exception as e:
            logger.error(f"Error saving Agent 10 output to production_projects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without failing the review

    logger.info("Prompt review pipeline completed successfully")

    # Return summary for API response
    summary = {
        "success": True,
        "message": f"Successfully reviewed prompts for {len(updated_list.annotated_shots)} shots",
        "total_shots": len(updated_list.annotated_shots),
        "shots_modified": len([r for r in review_results if r.get('shot_modified', False)]),
        "review_file_saved": agent.enable_saving
    }

    return updated_list, summary


# Example usage and testing
async def test_agent_standalone():
    """Test the agent with example data."""
    from phase_2_agents.agent_shot_strategy import ShotItem, ShotList
    
    # Create example shots with draft prompts
    shot1 = AnnotatedShotItem(
        shot_id="S01E01_001",
        description="Wide establishing shot of a modern office building at sunset",
        duration=4.0,
        scene_number=1,
        sequence_number=1,
        generation_strategy="generate_new",
        reasoning="First shot of the episode",
        confidence_score=0.95,
        prompt_image_draft="A wide cinematic shot of a modern glass office building at golden hour sunset, warm orange light reflecting off the windows, professional architectural photography"
    )
    
    shot2 = AnnotatedShotItem(
        shot_id="S01E01_002",
        description="Close-up of protagonist's face as they look out the window",
        duration=3.0,
        scene_number=1,
        sequence_number=2,
        generation_strategy="last_frame_seed",
        reasoning="Continuous from previous shot",
        confidence_score=0.90,
        seed_shot_id="S01E01_001",
        prompt_image_draft="Close-up of a person's face looking contemplative at sunrise, bright morning light"
    )
    
    annotated_list = AnnotatedShotList(
        episode_id="E01",
        title="Test Episode",
        annotated_shots=[shot1, shot2],
        overall_continuity_notes="Testing continuity review",
        strategy_summary={"generate_new": 1, "last_frame_seed": 1}
    )
    
    # Initialize agent and review
    agent = PromptReviewAgent()
    updated_list, review_results = await agent.review_prompts(annotated_list)
    
    logger.info("Review Results:")
    for result in review_results:
        logger.info(f"\nShot: {result['shot_id']}")
        logger.info(f"Modified: {result.get('shot_modified', False)}")
        if result.get('changes_made'):
            logger.info(f"Changes: {result['changes_made']}")
    
    return updated_list, review_results


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_standalone())

