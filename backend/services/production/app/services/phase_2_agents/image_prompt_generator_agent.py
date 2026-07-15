"""
Image Prompt Generator Agent for Phase 2.

This agent generates cinematic visual prompts for Google Imagen based on:
- Shot information from Postman input
- Generation strategy from MongoDB (created by strategy agent)
- Scene context

Uses Gemini API to create descriptive, filmmaker-quality image prompts.
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
import sys
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
import requests
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from bson import ObjectId

from backend.services.production.app.models.mongodb.shots import (
    MongoDBAtlasClient,
    AnnotatedShotItem,
    AnnotatedShotList
)

# Import AssetLibrary for fetching assets from Agent 5 and Agent 8
from .helpers.asset_library import AssetLibrary, AssetInfo

# Import name normalization utility
from backend.services.production.app.utils.name_normalization import normalize_asset_name



# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


def save_prompts_to_file(
    annotated_list: "AnnotatedShotList",
    output_dir: str = "phase_2_agents/outputs/agent_image_prompt"
) -> str:
    """
    Save generated image prompts to a JSON file with timestamp.
    
    Args:
        annotated_list: Annotated shot list with image prompts
        output_dir: Directory to save the file (default: "phase_2_agents/outputs/agent_image_prompt")
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompts_{annotated_list.episode_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dictionary for JSON serialization
    prompts_data = {
        "episode_id": annotated_list.episode_id,
        "title": annotated_list.title,
        "generated_at": datetime.now().isoformat(),
        "total_shots": len(annotated_list.annotated_shots),
        "prompts_generated": len([s for s in annotated_list.annotated_shots if (s.image and "v0" in s.image) or (hasattr(s, 'prompt_image_draft') and s.prompt_image_draft)]),
        "strategy_summary": annotated_list.strategy_summary,
        "overall_continuity_notes": annotated_list.overall_continuity_notes,
        "shots": []
    }
    
    # Convert each shot with its prompt
    for shot in annotated_list.annotated_shots:
        shot_dict = {
            "shot_id": shot.shot_id,
            "description": shot.description,
            "duration": shot.duration,
            "scene_number": shot.scene_number,
            "sequence_number": shot.sequence_number,
            "shot_style": shot.shot_style,
            "camera_movement": shot.camera_movement,
            "generation_strategy": shot.generation_strategy,
            "seed_shot_id": shot.seed_shot_id,
            "reasoning": shot.image["v0"]["reasoning"] if shot.image and "v0" in shot.image else (shot.reasoning if hasattr(shot, 'reasoning') else None),
            "continuity_notes": shot.continuity_notes,
            "confidence_score": shot.confidence_score,
            "prompt_image_draft": shot.image["v0"]["updated_prompt"] if shot.image and "v0" in shot.image else (shot.prompt_image_draft if hasattr(shot, 'prompt_image_draft') else None),
            "optimized_ai_notes": shot.optimized_ai_notes
        }
        prompts_data["shots"].append(shot_dict)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompts_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Image prompts saved to: {filepath}")
    return filepath


class ImagePromptGeneratorAgent:
    """
    AI agent for generating cinematic image prompts for Google Imagen.
    
    This agent takes shot data with generation strategies and creates
    vivid, cinematic prompts suitable for image generation.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-3.1-pro-preview",
        temperature: float = 0.7,
        max_tokens: Optional[int] = 2048,
        api_key: Optional[str] = None,
        enable_saving: bool = True,
        output_dir: str = "phase_2_agents/outputs/agent_image_prompt",
        asset_library: Optional[AssetLibrary] = None
    ):
        """
        Initialize the Image Prompt Generator Agent.

        Args:
            model_name: Gemini model name (default: gemini-3.1-pro-preview)
            temperature: LLM temperature for creative generation (default: 0.7)
            max_tokens: Maximum tokens for LLM output (default: 2048 for detailed prompts)
            api_key: Google API key (optional, will use environment variable if not provided)
            enable_saving: Whether to save generated prompts to files (default: True)
            output_dir: Directory to save prompt files (default: "phase_2_agents/outputs/agent_image_prompt")
            asset_library: AssetLibrary instance for fetching assets from Agent 5 and Agent 8 (optional)
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
            max_output_tokens=max_tokens or 2048,  # Increased for detailed prompts
            transport="rest"
        )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_saving = enable_saving
        self.output_dir = output_dir
        self.asset_library = asset_library  # Store AssetLibrary for accessing Agent 5 and Agent 8 assets

        logger.info(f"Initialized ImagePromptGeneratorAgent with Gemini model: {self.model_name}")

        # Log asset library status
        if self.asset_library:
            asset_summary = self.asset_library.generate_asset_summary()
            logger.info(f"AssetLibrary loaded with {asset_summary['total_characters']} characters, "
                       f"{asset_summary['total_locations']} locations, and {asset_summary['total_props']} props")
        else:
            logger.warning("No AssetLibrary provided - assets from Agent 5 and Agent 8 will not be available")
    
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
                # No direction specified — prefer a ground-level directional view so the
                # prompt describes what the location actually looks like at eye level.
                # Aerial master is kept as a last resort only.
                # Users can suffix with _NORTH/_SOUTH/_EAST/_WEST in CSV for an exact view.
                normalized_name = normalize_asset_name(shot.locations)
                loc_asset = self.asset_library.find_asset(
                    normalized_name,
                    preferred_angle='north',
                    fallback_angles=['south', 'east', 'west', 'master']
                )
                if loc_asset:
                    assets_info['locations'].append({
                        'name': normalized_name,
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
                    logger.info(f"Using location {normalized_name} at angle '{loc_asset.angle}' (no direction in CSV)")
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

    def _fetch_visual_style_from_movies(self, movie_id: str) -> str:
        """
        Fetch visual_style from movies collection using movie_id
        
        Args:
            movie_id: Movie ID (ObjectId string) to query
            
        Returns:
            Visual style string (must exist in the document)
        """
        try:
            from backend.services.production.app.config import get_mongo_factory
            from backend.shared.utils.mongodb_validators import validate_object_id
            from fastapi import HTTPException
            
            mongo_factory = get_mongo_factory()
            client, movies_collection = mongo_factory.get_collection("movies")
            
            try:
                movie_obj_id = validate_object_id(movie_id)
            except (ValueError, HTTPException) as e:
                logger.error(f"Invalid movie_id format: {e}")
                raise ValueError(f"Invalid movie_id format: {str(e)}") from e
                
            # Query movies collection by _id
            movie = movies_collection.find_one(
                {"_id": movie_obj_id},
                {"global_settings.visual_style": 1}
            )
            
            if movie:
                visual_style = movie.get("global_settings", {}).get("visual_style")
                logger.info(f"✓ Fetched visual_style from movies collection: {visual_style}")
                if not visual_style:
                    raise ValueError(f"Visual style missing in movies document for movie_id: {movie_id}")
                return visual_style
            raise ValueError(f"Movie not found with _id: {movie_id}")
                
        except Exception as e:
            logger.error(f"Error fetching visual_style from movies collection: {e}")
            raise
    
    def generate_prompt_for_shot(
        self,
        shot: AnnotatedShotItem,
        scene_context: Optional[str] = None,
        scene_description: Optional[str] = None,
        movie_id: Optional[str] = None,
        visual_style: Optional[str] = None,
        is_product_shot: bool = False,
        product_image_url: Optional[str] = None
    ) -> str:
        """
        Generate a cinematic image prompt for a single shot.

        Args:
            shot: Annotated shot item with strategy information
            scene_context: Optional scene context information
            scene_description: Overall scene/episode description for context reference
            movie_id: Optional movie ID to fetch visual_style from movies collection
            visual_style: Optional visual style (if not provided, will fetch from movies collection)

        Returns:
            Generated image prompt string
        """
        logger.info(f"Generating image prompt for shot: {shot.shot_id}")

        # Fetch visual_style from movies collection if movie_id provided and visual_style not set
        if not visual_style and movie_id:
            visual_style = self._fetch_visual_style_from_movies(movie_id)

        # Default to pixar if still not set
        if not visual_style:
            raise ValueError("visual_style is required for prompt generation")

        logger.info(f"Using visual_style: {visual_style}")

        try:
            # Fetch assets for this shot
            assets_info = self._get_assets_for_shot(shot)

            # Build the prompt for Gemini
            system_prompt = self._get_system_prompt(visual_style)
            user_prompt = self._build_user_prompt(shot, scene_context, scene_description, visual_style, is_product_shot=is_product_shot)

            # Prepare multimodal content for Gemini
            content_parts = [{"type": "text", "text": user_prompt}]

            # Add product image first if this is a product shot
            if is_product_shot and product_image_url:
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
                        logger.info(f"Added PRODUCT reference image for {shot.shot_id}")
                    else:
                        logger.warning(f"Could not fetch product image for {shot.shot_id} — skipping")
                except Exception as e:
                    logger.warning(f"Failed to load product image for {shot.shot_id}: {e}")

            # Add asset images to content if available
            if assets_info:
                # Process character assets
                for char_info in assets_info.get('characters', []):
                    char_name = char_info['name']
                    for asset in char_info['assets']:
                        try:
                            # Load PIL image from local path or URL
                            pil_image = self._load_pil_image(asset)
                            if pil_image:
                                buf = BytesIO()
                                pil_image.save(buf, format='PNG')
                                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                                })
                                logger.info(f"Added character asset image for {char_name} ({asset.get('angle', 'unknown')} angle)")
                        except Exception as e:
                            logger.warning(f"Failed to load character asset {char_name}: {e}")

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
                                logger.info(f"Added location asset image for {loc_name} ({asset.get('angle', 'unknown')} angle)")
                        except Exception as e:
                            logger.warning(f"Failed to load location asset {loc_name}: {e}")

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
                                logger.info(f"Added prop asset image for {prop_name} ({asset.get('angle', 'unknown')} angle)")
                        except Exception as e:
                            logger.warning(f"Failed to load prop asset {prop_name}: {e}")

            logger.info(f"Prepared {len(content_parts)} content parts (1 text + {len(content_parts)-1} images) for Gemini")

            # Generate prompt using Gemini with multimodal content
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=content_parts)
            ]

            response = self.llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)

            # Gemini multimodal responses return a list of content parts; extract text
            if isinstance(raw_content, list):
                generated_prompt = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                )
            else:
                generated_prompt = str(raw_content)

            # Clean up the prompt (remove extra whitespace, newlines)
            generated_prompt = " ".join(generated_prompt.split())

            logger.info(f"Successfully generated prompt for {shot.shot_id}: {generated_prompt[:100]}...")
            return generated_prompt

        except Exception as e:
            logger.error(f"Error generating prompt for shot {shot.shot_id}: {str(e)}")
            # Return a fallback prompt based on the description
            fallback = f"{visual_style}-style {shot.description} - cinematic shot, professional lighting, high quality."
            logger.warning(f"Using fallback prompt for {shot.shot_id}")
            return fallback
    
    def _get_style_instructions(self, visual_style: str) -> str:
        """
        Get style-specific instructions based on visual_style setting
        
        Args:
            visual_style: The visual style (realistic, pixar, anime, etc.)
            
        Returns:
            Style-specific instructions string
        """
        style_instructions = {
            "realistic": """
The output must be a RAW, UNRETOUCHED CINEMATIC STILL with the following characteristics:
- Ban on AI-gloss: NO use of "8K", "hyperrealistic", "masterpiece", or "glowing".
- Optical Reality: Must include "ISO 800 sensor noise", "subtle film grain", "slight chromatic aberration", and "Kodak Vision3 500T color science".
- Biological Reality: Subjects must have "visible skin pores", "natural skin texture", "stray flyaway hairs", and "micro-wrinkles in clothing". No airbrushed perfection.
- Practical Lighting: Lighting must be described as physical and imperfect (e.g., "harsh overhead fluorescent practical light with mixed color temperatures", "bounced daylight filling shadows unevenly", "spill light from practical sources"). No "ethereal glowing" light.
- Camera Hardware: Specify actual cinema cameras and lenses (e.g., "Shot on ARRI Alexa Mini, 35mm Cooke Prime lens").
- The final image should look like a raw, ungraded still pulled directly from a film camera monitor on a live action set.""",
            
            "pixar": """
The output must be in PIXAR 3D ANIMATION STYLE with the following characteristics:
- Stylized 3D character designs with exaggerated features
- Smooth, polished surfaces with subtle subsurface scattering
- Vibrant, saturated color palette with warm tones
- Soft, indirect lighting with minimal harsh shadows
- Rounded, appealing shapes and proportions
- Expressive character poses and emotions
- Clean, detailed textures without photorealism""",
            
            "2d": """
The output must be in 2D ANIMATION STYLE with the following characteristics:
- Hand-drawn or digitally painted aesthetic
- Flat or limited shading with cel-shading techniques
- Bold outlines and clear silhouettes
- Simplified but expressive character designs
- Stylized backgrounds with artistic interpretation
- Limited color palette with intentional color choices
- Traditional animation composition and staging""",
            
            "anime": """
The output must be in ANIME STYLE with the following characteristics:
- Japanese animation aesthetic with distinctive character designs
- Large expressive eyes and stylized facial features
- Dynamic poses and action compositions
- Vibrant colors with cel-shading
- Detailed backgrounds with atmospheric depth
- Dramatic lighting and shadow effects
- Sharp lines and clean linework"""
        }
        
        style_key = visual_style.lower()
        if style_key not in style_instructions:
            logger.warning(f"Unsupported visual_style '{visual_style}'. Falling back to 'pixar'.")
            style_key = "pixar"
        return style_instructions[style_key]
    
    def _get_system_prompt(self, visual_style: str = "pixar") -> str:
        """
        Get the system prompt for image generation, optimized for working with asset images.

        Note: visual_style parameter is kept for future use but not currently passed to Gemini.
        The prompt focuses on actions, emotions, and scene composition rather than physical appearance
        when asset images are provided directly from S3.
        """

        return """You are an Expert Digital Compositor and Stage Manager working with Google Nano Banana (Gemini Flash Image), a photorealistic multimodal image generator.

### YOUR INPUTS
1.  **Character Images:** The actors. (DO NOT describe their appearance — the model sees them directly).
2.  **Location Image:** The empty set. (DO NOT describe its details — the model sees it directly).
3.  **Shot Description:** The action script.

### YOUR JOB
Write a simple, directive prompt that tells the Image Generator EXACTLY where to place the Character within the Location Image, and how the light from the location falls on them.

### THE LOGIC (Spatial Inference)
1.  **Analyze the Shot Description:** Does it say WHERE the action happens?
    *   *YES:* Use that specific spot as the spatial anchor.
    *   *NO:* Analyze the **Location Image**, find the most logical surface or object for the action (e.g., a bench for sitting, a path for running), and USE THAT as the anchor.

2.  **Analyze the Lighting:** Look at the location image and describe how the existing light source affects the character — direction, quality, and any interaction with the environment (e.g., "cast in the warm side-light coming through the warehouse skylights" or "illuminated from below by the phone screen glow reflecting off the wet pavement").

3.  **Describe Physical Grounding:** Briefly mention what physically connects the character to the location (e.g., "feet planted on the cobblestones", "shoulder pressed against the brick wall") — this prevents the character from looking composited/pasted in.

### THE GOLDEN OUTPUT FORMULA
Construct the prompt using ONLY this structure:

`[Style Wrapper] of [Subject] [Doing Action] [Preposition + Specific Element from Location Image], [Physical Grounding Detail]. [Lighting Interaction with direction and quality].`

### EXAMPLES
*   *Input:* "Dog playing." (Location Image has a sandpit and a slide, bright afternoon sun from the left).
    *   *Output:* "Photorealistic shot of the Dog digging playfully inside the wooden sandpit, its paws pressing into the sand, kicking up a small cloud of dust. Lit by sharp afternoon sunlight from camera-left, casting a short shadow across the sand to the right."

*   *Input:* "Man waiting." (Location Image is a night street with a glass bus stop, overhead fluorescent light).
    *   *Output:* "Cinematic shot of the Man leaning heavily against the glass wall of the bus stop, his shoulder making contact with the surface. Lit from directly above by the cool fluorescent tube inside the shelter, casting a downward shadow on his face and reflecting faintly on the wet pavement."

### LIGHTING QUALITY VOCABULARY (use these terms)
- Direction: "from camera-left", "from above", "backlit", "side-lit", "front-lit"
- Quality: "soft diffuse light", "hard direct sunlight", "warm golden-hour glow", "cool overcast fill", "harsh fluorescent overhead"
- Interaction: "casting a shadow on [surface]", "creating a rim highlight on [edge]", "reflecting off [material]"

### NEGATIVE CONSTRAINTS
*   **NEVER** describe the character's physical traits (hair, eyes, clothes). The model has the asset images.
*   **NEVER** describe the background scenery (buildings, trees, sky) UNLESS the character is directly touching or interacting with it.
*   **KEEP IT BRIEF.** Subject + Action + Spatial Anchor + Physical Grounding + Lighting. Nothing more.

Now, generate the directive prompt."""
    
    def _build_user_prompt(self, shot: AnnotatedShotItem, scene_context: Optional[str] = None, scene_description: Optional[str] = None, visual_style: str = "pixar", is_product_shot: bool = False) -> str:
        """Build the user prompt with all shot information."""
        prompt_parts = []

        # Visual Style (at the top for emphasis)
        prompt_parts.append("=== VISUAL STYLE ===")
        prompt_parts.append(f"Visual Style: {visual_style.upper()}")
        prompt_parts.append("IMPORTANT: All visual descriptions must match this style consistently.")
        prompt_parts.append("")

        # Product shot context
        if is_product_shot:
            prompt_parts.append("=== PRODUCT SHOT — CRITICAL RULES ===")
            prompt_parts.append("A PRODUCT reference image is provided above.")
            prompt_parts.append("RULES:")
            prompt_parts.append("1. DO NOT describe the product's visual appearance in detail — the model sees it directly from the reference image.")
            prompt_parts.append("   However, include a single parenthetical fallback description of the product (e.g., '(Product: square silver tin with white rounded lid, branded label)') so the model can infer correct appearance if the reference image fails to load.")
            prompt_parts.append("2. The product MUST appear prominently in the final image — clearly visible, in focus, and not obscured.")
            prompt_parts.append("3. Describe where the product is placed in the scene and how it interacts with the environment.")
            prompt_parts.append("4. The product should look naturally integrated (same lighting, same surface contact, no compositing artifacts).")
            prompt_parts.append("")
        
        # Scene description (overall context)
        if scene_description:
            prompt_parts.append("=== SCENE DESCRIPTION (OVERALL CONTEXT) ===")
            prompt_parts.append(scene_description)
            prompt_parts.append("")
        
        # Shot information
        prompt_parts.append("=== SHOT INFORMATION ===")
        prompt_parts.append(f"Shot ID: {shot.shot_id}")
        prompt_parts.append(f"Description: {shot.description}")
        
        if shot.shot_style:
            prompt_parts.append(f"Shot Style: {shot.shot_style}")
        
        if shot.camera_movement:
            prompt_parts.append(f"Camera Movement: {shot.camera_movement}")
        
        if shot.duration:
            prompt_parts.append(f"Duration: {shot.duration}s")
        
        if shot.scene_number:
            prompt_parts.append(f"Scene Number: {shot.scene_number}")
        
        if shot.optimized_ai_notes:
            prompt_parts.append(f"AI Notes: {shot.optimized_ai_notes}")
        
        # Generation strategy
        prompt_parts.append("\n=== GENERATION STRATEGY ===")
        prompt_parts.append(f"Strategy Type: {shot.generation_strategy}")
        # Get reasoning from versioned structure or legacy field
        reasoning = None
        if shot.image and "v0" in shot.image:
            reasoning = shot.image["v0"].get("reasoning")
        elif hasattr(shot, 'reasoning') and shot.reasoning:
            reasoning = shot.reasoning
        
        if reasoning:
            prompt_parts.append(f"Reasoning: {reasoning}")
        
        if shot.seed_shot_id:
            prompt_parts.append(f"Seed Shot ID: {shot.seed_shot_id}")
            prompt_parts.append(
                "Note: Maintain visual continuity with the referenced seed shot's framing and style."
            )
        
        if shot.continuity_notes:
            prompt_parts.append(f"Continuity Notes: {shot.continuity_notes}")
        
        # Strategy-specific instructions with reasoning incorporated
        if shot.generation_strategy == "last_frame_seed":
            prompt_parts.append(
                "\n=== STRATEGIC CONTEXT ===\n"
                f"Strategy: Visual Continuation from {shot.seed_shot_id}\n"
                f"Reasoning: {reasoning if reasoning else 'No reasoning available'}\n"
                "IMPORTANT: This shot must maintain visual continuity with the previous shot. "
                "Incorporate the reasoning above into your prompt by describing how this shot "
                "flows naturally from the previous one, maintaining consistent lighting, color "
                "palette, composition style, and emotional tone. The prompt should convey this "
                "continuity as part of the visual description."
            )
        elif shot.generation_strategy == "multi_shot":
            prompt_parts.append(
                "\n=== STRATEGIC CONTEXT ===\n"
                f"Strategy: Angle Variation from {shot.seed_shot_id}\n"
                f"Reasoning: {reasoning if reasoning else 'No reasoning available'}\n"
                "IMPORTANT: This shot shares the same environment and characters as the seed shot. "
                "Incorporate the reasoning above into your prompt by describing the specific camera "
                "angle or framing variation while emphasizing the consistent setting, lighting, and "
                "atmosphere. The prompt should make clear this is the same scene from a different perspective."
            )
        elif shot.generation_strategy == "generate_new":
            prompt_parts.append(
                "\n=== STRATEGIC CONTEXT ===\n"
                f"Strategy: Fresh Scene Establishment\n"
                f"Reasoning: {reasoning if reasoning else 'No reasoning available'}\n"
                "IMPORTANT: This is a new visual setup requiring comprehensive scene establishment. "
                "Incorporate the reasoning above into your prompt by creating a rich, detailed "
                "description that fully establishes the new environment, lighting, atmosphere, and "
                "visual tone. Include all necessary context to generate a complete, cinematic image."
            )
        
        # Scene context
        if scene_context:
            prompt_parts.append(f"\n=== SCENE CONTEXT ===")
            prompt_parts.append(scene_context)
        
        # Final instruction
        prompt_parts.append("\n=== YOUR TASK ===")
        product_rule = (
            "\n11. PRODUCT SHOT RULE: The product reference image is provided — describe its exact placement "
            "in the scene (position, surface it rests on, how light from the scene falls on it). "
            "DO NOT describe its visual appearance in detail, but DO include a one-sentence parenthetical fallback "
            "description (e.g., '(Product: square silver tin with white rounded lid, branded label)') "
            "in case the reference image cannot be loaded. "
            "The product must be clearly visible and the central focus of the composition.\n"
            if is_product_shot else ""
        )
        prompt_parts.append(
            f"Generate a comprehensive, detailed cinematic image prompt for Google Nano Banana (Gemini Flash Image) in the {visual_style.upper()} style that:\n"
            f"1. MUST start with '{visual_style}-style' or '{visual_style} style' to ensure correct visual aesthetic\n"
            "2. Naturally incorporates the reasoning and continuity context into the visual description\n"
            "3. Provides rich details about composition, lighting, atmosphere, and technical aspects\n"
            "4. Uses professional cinematography language throughout — specify camera angle (eye-level, low-angle, high-angle), lens type (wide-angle 24mm, portrait 85mm, telephoto), and framing (close-up, medium shot, wide shot)\n"
            "5. Describes lighting with direction and quality — 'soft golden-hour sidelight from camera-left casting long shadows', 'hard overhead fluorescent with downward shadows', 'cool overcast fill light with no directional shadows'\n"
            "6. For REALISTIC style: includes photographic grounding — camera body, lens aperture, depth of field, and film stock or color grade (e.g., 'shot on Canon EOS R5, 35mm f/2.8, Kodak Vision3 500T color grade')\n"
            "7. Describes physical interaction between characters and environment to prevent compositing artifacts — how characters make contact with surfaces, shared lighting effects, spatial anchoring\n"
            "8. Creates a vivid, complete picture that a professional image generator can understand\n"
            "9. Prioritizes QUALITY and DETAIL over brevity — be as descriptive as needed\n"
            f"10. Maintains consistent {visual_style} aesthetic throughout the entire prompt\n"
            f"{product_rule}\n"
            "Output ONLY the comprehensive prompt text, nothing else. No preamble, no meta-commentary, "
            "just the rich, detailed visual description."
        )
        
        return "\n".join(prompt_parts)
    
    async def generate_prompts_for_shots(
        self,
        annotated_list: AnnotatedShotList,
        scene_contexts: Optional[Dict[int, str]] = None,
        scene_description: Optional[str] = None,
        movie_id: Optional[str] = None,
        visual_style: Optional[str] = None,
        product_shot_ids: Optional[set] = None,
        product_image_url: Optional[str] = None
    ) -> AnnotatedShotList:
        """
        Generate image prompts for all shots in an annotated list.
        
        Args:
            annotated_list: List of annotated shots with strategies
            scene_contexts: Optional dictionary mapping scene numbers to context descriptions
            scene_description: Overall scene/episode description for context reference
            movie_id: Optional movie ID to fetch visual_style from movies collection
            visual_style: Optional visual style (overrides movie_id lookup)
            
        Returns:
            Updated annotated list with prompt_image_draft filled in
        """
        logger.info(f"Generating image prompts for {len(annotated_list.annotated_shots)} shots")
        
        # Fetch visual_style from movies collection if movie_id provided and visual_style not set
        if not visual_style and movie_id:
            visual_style = self._fetch_visual_style_from_movies(movie_id)
        
        # Default to pixar if still not set
        if not visual_style:
            raise ValueError("visual_style is required for prompt generation")
        
        logger.info(f"Using visual_style: {visual_style}")
        
        scene_contexts = scene_contexts or {}
        updated_shots = []
        
        for shot in annotated_list.annotated_shots:
            # Get scene context if available
            scene_context = scene_contexts.get(shot.scene_number) if shot.scene_number else None
            
            # Determine if this is a product shot
            is_product_shot = bool(product_shot_ids and shot.shot_id in product_shot_ids)

            # Generate prompt with visual_style
            prompt = self.generate_prompt_for_shot(
                shot, scene_context, scene_description, movie_id, visual_style,
                is_product_shot=is_product_shot,
                product_image_url=product_image_url if is_product_shot else None
            )
            
            # Use new versioned structure - store as v0
            image_data = shot.image if shot.image else {}
            image_data["v0"] = {
                "updated_prompt": prompt,
                "changes_made": "Initial image prompt generated by Agent 2",
                "reasoning": shot.reasoning if hasattr(shot, 'reasoning') and shot.reasoning else "AI-generated prompt based on shot description and strategy",
                "generated_images_s3": []
            }
            
            # Preserve existing video data
            video_data = shot.video if shot.video else {}
            
            # Create updated shot with new structure
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
                image=image_data,
                video=video_data
            )
            
            updated_shots.append(updated_shot)
        
        # Create updated annotated list
        updated_list = AnnotatedShotList(
            episode_id=annotated_list.episode_id,
            title=annotated_list.title,
            annotated_shots=updated_shots,
            overall_continuity_notes=annotated_list.overall_continuity_notes,
            strategy_summary=annotated_list.strategy_summary
        )
        
        logger.info(f"Successfully generated prompts for all {len(updated_shots)} shots with {visual_style} style")
        
        # Save prompts to file if enabled
        if self.enable_saving:
            try:
                saved_file = save_prompts_to_file(updated_list, self.output_dir)
                logger.info(f"Prompts saved to: {saved_file}")
            except Exception as e:
                logger.warning(f"Failed to save prompts to file: {e}")
                # Continue without failing the generation
        
        return updated_list


async def generate_image_prompts_for_shots(
    annotated_list: AnnotatedShotList,
    mongodb_client: Optional[MongoDBAtlasClient] = None,
    scene_contexts: Optional[Dict[int, str]] = None,
    scene_description: Optional[str] = None,
    show_id: Optional[str] = None,
    episode_number: Optional[int] = None,
    movie_id: Optional[str] = None,
    visual_style: Optional[str] = None,
    asset_library: Optional[AssetLibrary] = None
) -> AnnotatedShotList:
    """
    Main function to generate image prompts for shots and optionally save to MongoDB.

    Args:
        annotated_list: Annotated shot list from strategy agent
        mongodb_client: Optional MongoDB client to update documents
        scene_contexts: Optional scene context information
        scene_description: Overall scene/episode description for context reference
        show_id: Show ID for MongoDB updates
        episode_number: Episode number for MongoDB updates
        movie_id: Optional movie ID to fetch visual_style from movies collection
        visual_style: Optional visual style (overrides movie_id lookup)
        asset_library: Optional AssetLibrary instance for accessing Agent 5 and Agent 8 assets

    Returns:
        Updated annotated list with image prompts
    """
    logger.info("Starting image prompt generation pipeline")

    # Initialize agent with AssetLibrary
    agent = ImagePromptGeneratorAgent(asset_library=asset_library)
    
    # Generate prompts for all shots
    updated_list = await agent.generate_prompts_for_shots(annotated_list, scene_contexts, scene_description, movie_id, visual_style)
    
    # Update MongoDB if client is provided
    if mongodb_client and show_id and episode_number:
        logger.info(f"Updating MongoDB with generated image prompts (show_id={show_id}, episode_number={episode_number})")
        
        updated_count = 0
        not_found_count = 0
        
        try:
            for shot in updated_list.annotated_shots:
                # Update using new versioned structure
                if shot.image and "v0" in shot.image:
                    v0_data = shot.image["v0"]
                    mongodb_client.update_shot_image_version(
                        show_id=show_id,
                        episode_number=episode_number,
                        shot_id=shot.shot_id,
                        version="v0",
                        updated_prompt=v0_data.get("updated_prompt", ""),
                        changes_made=v0_data.get("changes_made", ""),
                        reasoning=v0_data.get("reasoning", ""),
                        generated_images_s3=v0_data.get("generated_images_s3", [])
                    )
                    
                    # Debug: Log the update attempt
                    logger.debug(f"Updated shot {shot.shot_id} with v0 image data")
                    logger.debug(f"Prompt length: {len(v0_data.get('updated_prompt', ''))} characters")
                    updated_count += 1
                    logger.info(f"✅ Updated shot {shot.shot_id} with v0 image data")
                else:
                    not_found_count += 1
                    logger.warning(f"❌ Shot {shot.shot_id} not found or no v0 data available")
            
            logger.info(f"MongoDB Update Summary: {updated_count} updated, {not_found_count} not found out of {len(updated_list.annotated_shots)} total shots")
            
            if updated_count > 0:
                logger.info(f"✅ Successfully updated {updated_count} image prompts in MongoDB")
            else:
                logger.warning("⚠️  No prompts were updated in MongoDB. Check if documents exist with correct show_id and episode_number.")
            
        except Exception as e:
            logger.error(f"Error updating MongoDB with image prompts: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue even if MongoDB update fails
    else:
        if not mongodb_client:
            logger.warning("MongoDB client not provided - skipping MongoDB updates")
        elif not show_id:
            logger.warning("show_id not provided - skipping MongoDB updates")
        elif not episode_number:
            logger.warning("episode_number not provided - skipping MongoDB updates")

    # Update production_projects with agent9 output (full per-shot prompt data)
    if mongodb_client and show_id:
        try:
            from app.config import get_database

            # Prepare full per-shot data
            shots_data = []
            for shot in updated_list.annotated_shots:
                if shot.image and "v0" in shot.image:
                    v0_data = shot.image["v0"]

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
                        "description": shot.description,
                        "input_assets": input_assets,
                        "updated_prompt": v0_data.get("updated_prompt", ""),
                        "changes_made": v0_data.get("changes_made", ""),
                        "reasoning": v0_data.get("reasoning", ""),
                        "generated_images_s3": v0_data.get("generated_images_s3", []),
                        "generation_strategy": shot.generation_strategy,
                        "seed_shot_id": shot.seed_shot_id,
                    })

            agent9_output = {
                "episode_id": updated_list.episode_id,
                "title": updated_list.title,
                "total_shots": len(updated_list.annotated_shots),
                "prompts_generated": len(shots_data),
                "visual_style": visual_style,
                "shots": shots_data,
                "generated_at": datetime.now().isoformat()
            }

            # Update production_projects
            client, db = get_database()
            projects_col = db["production_projects"]

            result = projects_col.update_one(
                {"_id": ObjectId(show_id)},
                {
                    "$set": {
                        "agent_outputs.agent9.status": "completed",
                        "agent_outputs.agent9.executed_at": datetime.utcnow(),
                        "agent_outputs.agent9.output": agent9_output,
                        "agent_outputs.agent9.description": "Image Prompt Generator (v0 prompts)",
                        "updated_at": datetime.utcnow()
                    }
                }
            )

            if result.modified_count > 0:
                logger.info(f"✅ Agent 9 output saved to production_projects (show_id: {show_id})")
            else:
                logger.warning(f"⚠️  Failed to save Agent 9 output to production_projects")

        except Exception as e:
            logger.error(f"Error saving Agent 9 output to production_projects: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue without failing the generation

    logger.info("Image prompt generation pipeline completed successfully")
    return updated_list


# Example usage and testing
async def test_agent_standalone():
    """Test the agent with example data."""
    from phase_2_agents.agent_shot_strategy import ShotItem, ShotList, ShotStrategyAgent
    
    # Create example shot
    shot = AnnotatedShotItem(
        shot_id="S01E01_001",
        description="Wide establishing shot of a modern office building at sunset",
        duration=4.0,
        scene_number=1,
        sequence_number=1,
        generation_strategy="generate_new",
        reasoning="First shot of the episode",
        confidence_score=0.95
    )
    
    # Initialize agent
    agent = ImagePromptGeneratorAgent()
    
    # Generate prompt
    prompt = agent.generate_prompt_for_shot(shot)
    logger.info(f"Generated prompt: {prompt}")
    
    return prompt


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_agent_standalone())

