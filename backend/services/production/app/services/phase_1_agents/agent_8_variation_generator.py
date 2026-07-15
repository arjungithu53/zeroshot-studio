#!/usr/bin/env python3
"""
Agent 8: Variation Generator
=============================
Generates multiple camera angle variations from approved master images.
Creates different shots (close-up, wide, profile, etc.) for I2V scene flexibility.

Flow:
1. Load approved images from Agent 6 (or edited from Agent 7)
2. For each approved master image, generate variations:
   - Close-up shot
   - Wide shot
   - Profile view (left/right)
   - Three-quarter view
   - Action pose (if character)
3. Use Gemini Nano Banana (Flash Image Preview) with reference image + angle prompt
4. Save all variations organized by asset and angle
"""

from PIL import Image
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from io import BytesIO
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# S3 imports
from infrastructure.s3.client import S3ClientFactory, S3Config
from infrastructure.s3.upload import upload_file

# Gemini imports
from google import genai
from google.genai import types


class VariationGeneratorAgent:
    """
    Agent 8: Generates multiple angle variations from master images

    This agent creates different camera angles and shots from approved
    master images to provide flexibility for I2V scene composition.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Variation Generator Agent using Nano Banana (Gemini Flash Image).

        Args:
            api_key: Unused — kept for backwards-compatible call signatures. Uses GEMINI_API_KEY.
        """
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=gemini_api_key)
        self.gemini_model = "gemini-3.1-flash-image-preview"
        logger.info("✓ Initialized with Nano Banana (Gemini Flash Image) for image generation")

        self.approved_images = {}
        self.generated_variations = {}
        self.failed_variations = []

        # S3 configuration
        self.enable_s3 = True
        self.s3_client = None
        self.s3_bucket = None
        self.s3_region = None
        self.s3_endpoint_url = None
        self._init_s3()

        # Define angle variations for different asset types
        self.character_angles = [
            {
                "name": "close_up",
                "prompt": "Using the character in the reference image, create a tight head-and-shoulders close-up portrait. Maintain identical facial features, skin tone, hair texture and style, and costume/clothing details visible at this framing. Crop to chest level. Apply shallow depth of field (f/1.8), soft studio lighting matching the reference, same neutral background and color grading as the reference image.",
                "description": "Head and shoulders close-up"
            },
            {
                "name": "wide_shot",
                "prompt": "Using the character in the reference image, create a full-body wide shot showing the complete figure from head to toe with a small amount of breathing room around the edges. Maintain identical facial features, skin tone, hair, complete costume, proportions, and overall visual style. Use the same neutral background, lighting direction, and color grading as the reference image.",
                "description": "Full body wide shot"
            },
            {
                "name": "profile_left",
                "prompt": "Using the character in the reference image, rotate the character to show a clean left-side profile view. Preserve all costume details, hair silhouette, body proportions, and any accessories as they would appear from the left side. Maintain the same neutral background, lighting quality, and color grading as the reference image.",
                "description": "Left side profile view"
            },
            {
                "name": "profile_right",
                "prompt": "Using the character in the reference image, rotate the character to show a clean right-side profile view. Preserve all costume details, hair silhouette, body proportions, and any accessories as they would appear from the right side. Maintain the same neutral background, lighting quality, and color grading as the reference image.",
                "description": "Right side profile view"
            },
            {
                "name": "back_shot",
                "prompt": "Using the character in the reference image, show the character from behind — a clean full-body back view revealing the rear details of the costume, hairstyle from behind, and complete silhouette. Maintain the same neutral background, lighting direction, and color grading as the reference image. Preserve all costume texture and material details visible from the back.",
                "description": "Back view shot"
            }
        ]

        # Location variations - single 2x2 grid prompt for all directional views
        # Grid layout: [North][East]
        #              [West][South]
        self.location_grid_prompt = """Using this aerial reference image of the location, generate a 2x2 grid of 4 photorealistic ground-level photographs of the same location, each from a different cardinal direction. Lower the virtual camera to eye level (approximately 1.6 meters height) at the center point of the location. Each quadrant must show a unique directional perspective of the environment: top-left=North view (camera looking north), top-right=East view (camera looking east), bottom-left=West view (camera looking west), bottom-right=South view (camera looking south). Infer any unseen angles and environmental details from context in the aerial image. Maintain consistent time of day, natural lighting conditions, atmosphere, weather, and environmental details across all 4 views. Photorealistic, cinematic wide-angle (24mm equivalent), natural available light."""

        # Direction names and descriptions for the 4 cropped images
        self.location_directions = [
            {"name": "north", "description": "Northern view of the location", "grid_position": "top-left"},
            {"name": "east", "description": "Eastern view of the location", "grid_position": "top-right"},
            {"name": "west", "description": "Western view of the location", "grid_position": "bottom-left"},
            {"name": "south", "description": "Southern view of the location", "grid_position": "bottom-right"}
        ]

        # Props use master image only - no variations needed
        self.prop_angles = []

    def _init_s3(self) -> None:
        """Initialize S3 client from environment variables"""
        try:
            access_key = os.getenv("production_AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("production_AWS_SECRET_ACCESS_KEY")
            bucket = os.getenv("production_S3_BUCKET_NAME")
            region = os.getenv("production_AWS_REGION", "us-east-1")
            endpoint_url = os.getenv("production_S3_ENDPOINT_URL")

            logger.info(f"Agent 8: Checking S3 credentials...")
            logger.info(f"   - Access Key: {'✓ Set' if access_key else '✗ Missing (production_AWS_ACCESS_KEY_ID)'}")
            logger.info(f"   - Secret Key: {'✓ Set' if secret_key else '✗ Missing (production_AWS_SECRET_ACCESS_KEY)'}")
            logger.info(f"   - Bucket: {bucket if bucket else '✗ Missing (production_S3_BUCKET_NAME)'}")
            logger.info(f"   - Region: {region}")
            if endpoint_url:
                logger.info(f"   - Endpoint URL: {endpoint_url}")

            if not all([access_key, secret_key, bucket]):
                logger.warning("S3 credentials incomplete, disabling S3 upload for variations")
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

            logger.info(f"Agent 8: S3 client initialized successfully")
            logger.info(f"   - Will use presigned URLs (7-day expiration)")
        except Exception as e:
            logger.warning(f"Agent 8: Failed to initialize S3 client: {e}")
            import traceback
            traceback.print_exc()
            self.enable_s3 = False

    def load_agent6_output(self, agent6_output_path: str) -> None:
        """
        Load approved images from Agent 6 review results
        If using updated Agent 6 file, edited images are already marked as approved

        Args:
            agent6_output_path: Path to Agent 6 review results JSON (original or updated)
        """
        with open(agent6_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is an updated file from Agent 7
        is_updated = data.get('updated_by_agent7', False)
        if is_updated:
            logger.info(f"✓ Using updated Agent 6 results with edited images")

        # Get all approved assets
        approved_assets = data.get('assets_approved', {})

        # Extract image paths from approved assets (now a list structure)
        for asset_type in ['characters', 'locations', 'props']:
            for asset_data in approved_assets.get(asset_type, []):
                asset_name = asset_data.get('name', 'Unknown')
                asset_id = asset_data.get('id')
                reviews = asset_data.get('reviews', [])

                if reviews:
                    # Use the first approved review (master image)
                    master_review = reviews[0]
                    image_path = master_review.get('image_path')

                    # Show if this was edited by Agent 7
                    if master_review.get('edited_by_agent7'):
                        logger.info(f"   ✓ {asset_name}: Using edited image")

                    is_url = image_path and (image_path.startswith("http://") or image_path.startswith("https://"))
                    if image_path and (is_url or os.path.exists(image_path)):
                        if asset_type not in self.approved_images:
                            self.approved_images[asset_type] = []

                        self.approved_images[asset_type].append({
                            'id': asset_id,
                            'name': asset_name,
                            'master_image': image_path,
                            'review_data': master_review
                        })

        total_approved = sum(len(assets) for assets in self.approved_images.values())

        logger.info(f"\n✓ Loaded Agent 6 review results from: {agent6_output_path}")
        logger.info(f"   Total master images: {total_approved}")
        logger.info(f"   Characters: {len(self.approved_images.get('characters', []))}")
        logger.info(f"   Locations: {len(self.approved_images.get('locations', []))}")
        logger.info(f"   Props: {len(self.approved_images.get('props', []))}")

    def _get_aspect_ratio_from_string(self, aspect_ratio_str: str) -> str:
        """
        Convert aspect ratio string to BytePlus format

        Args:
            aspect_ratio_str: Aspect ratio like "square_1_1", "widescreen_16_9", etc.

        Returns:
            BytePlus aspect ratio format (e.g., "1K", "2K")
        """
        # For variations, we'll use 2K as default for better quality
        return "2K"

    def _generate_variation(
        self,
        reference_image_path: str,
        variation_prompt: str,
        aspect_ratio: str = "1:1",
        num_images: int = 1
    ) -> Optional[List[Image.Image]]:
        """
        Generate image variation using configured provider (Gemini Flash Image Preview)

        Args:
            reference_image_path: Path to master/reference image
            variation_prompt: Prompt describing the angle/variation
            aspect_ratio: Aspect ratio (e.g., "1:1", "16:9")
            num_images: Number of variations to generate

        Returns:
            List of PIL Image objects or None if failed
        """
        return self._generate_variation_gemini(reference_image_path, variation_prompt, num_images)

    def _generate_variation_gemini(
        self,
        reference_image_path: str,
        variation_prompt: str,
        num_images: int = 1
    ) -> Optional[List[Image.Image]]:
        """
        Generate image variation using Gemini Flash Image Preview (Nano Banana)

        Args:
            reference_image_path: Path to master/reference image or S3 URL
            variation_prompt: Prompt describing the angle/variation
            num_images: Number of variations to generate

        Returns:
            List of PIL Image objects or None if failed
        """
        try:
            # Check if reference_image_path is an S3 URL or local file path
            if reference_image_path.startswith('http://') or reference_image_path.startswith('https://'):
                # Download image from URL
                import requests
                logger.info(f"   Downloading reference image from S3 URL...")
                img_response = requests.get(reference_image_path, timeout=60)
                if img_response.status_code == 200:
                    reference_image = Image.open(BytesIO(img_response.content))
                    logger.info(f"   ✓ Downloaded reference image successfully")
                else:
                    logger.error(f"   Failed to download image from URL: {img_response.status_code}")
                    return None
            else:
                # Load from local file path
                reference_image = Image.open(reference_image_path)

            generated_images = []

            # Gemini generates one image at a time, so loop for multiple images
            for i in range(num_images):
                try:
                    # Generate image using Gemini Flash Image Preview with reference image
                    response = self.gemini_client.models.generate_content(
                        model=self.gemini_model,
                        contents=[variation_prompt, reference_image],
                        config=types.GenerateContentConfig(
                            response_modalities=['IMAGE'],
                            image_config=types.ImageConfig(
                                aspect_ratio="1:1",  # Default aspect ratio for characters
                                # image_size="2K"  # Optionally can specify resolution
                            ),
                        )
                    )

                    # Extract generated image from response
                    for part in response.parts:
                        # Check if part has inline image data
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            # Convert inline_data to PIL Image
                            image_bytes = part.inline_data.data
                            pil_image = Image.open(BytesIO(image_bytes))
                            generated_images.append(pil_image)
                            break  # Only take first image from response
                        elif hasattr(part, 'as_image'):
                            # Try as_image() method and convert to PIL
                            try:
                                img = part.as_image()
                                if isinstance(img, Image.Image):
                                    # Already a PIL Image
                                    generated_images.append(img)
                                else:
                                    # Convert to PIL Image if needed
                                    generated_images.append(img)
                                break
                            except:
                                continue

                except Exception as img_error:
                    logger.warning(f"   Failed to generate image {i+1}/{num_images}: {img_error}")
                    continue

            return generated_images if generated_images else None

        except Exception as e:
            logger.error(f"   Gemini image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_image(self, image: Image.Image, save_path: str, s3_key: str = None) -> Dict[str, str]:
        """
        Save PIL Image locally and optionally upload to S3

        Args:
            image: PIL Image object
            save_path: Local path to save the image
            s3_key: S3 key for the uploaded file (optional)

        Returns:
            Dict with 'local_path' and 's3_url' (if S3 enabled), or empty dict on failure
        """
        result = {}

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save image locally
            image.save(save_path)
            result['local_path'] = save_path

            # Upload to S3 if enabled
            if self.enable_s3 and self.s3_client:
                try:
                    # Generate S3 key if not provided
                    if not s3_key:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.basename(save_path)
                        s3_key = f"phase1/variations/{timestamp}_{filename}"

                    # Upload to S3 with presigned URL
                    s3_url = upload_file(
                        file_path=save_path,
                        s3_client=self.s3_client,
                        bucket_name=self.s3_bucket,
                        s3_key=s3_key,
                        content_type="image/png",
                        region=self.s3_region,
                        endpoint_url=self.s3_endpoint_url,
                        use_presigned_url=True,
                        presigned_expiration=86400 * 7  # 7 days
                    )
                    result['s3_url'] = s3_url
                    result['s3_key'] = s3_key
                    logger.info(f"   ✓ Uploaded to S3: {s3_key}")
                except Exception as s3_error:
                    logger.warning(f"   S3 upload failed: {s3_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue without S3 - local file still available
            else:
                if not self.enable_s3:
                    logger.warning(f"   S3 upload disabled (enable_s3=False)")
                elif not self.s3_client:
                    logger.warning(f"   S3 client not initialized")

            return result

        except Exception as e:
            logger.error(f"   Save error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _crop_grid_to_directions(self, grid_image: Image.Image) -> Dict[str, Image.Image]:
        """
        Crop a 2x2 grid image into 4 separate directional images.

        Grid layout:
        [North] [East]
        [West]  [South]

        Args:
            grid_image: PIL Image of the 2x2 grid

        Returns:
            Dictionary with direction names as keys and cropped PIL Images as values
            {"north": Image, "east": Image, "west": Image, "south": Image}
        """
        try:
            width, height = grid_image.size
            half_width = width // 2
            half_height = height // 2

            logger.info(f"   📏 Original grid size: {width}x{height} pixels")
            logger.info(f"   📏 Each cropped quadrant will be: {half_width}x{half_height} pixels")
            logger.info(f"   ✂️  Crop coordinates:")

            # Crop the 4 quadrants with detailed logging
            crop_specs = {
                "north": {
                    "coords": (0, 0, half_width, half_height),
                    "position": "top-left",
                    "description": "Northern view"
                },
                "east": {
                    "coords": (half_width, 0, width, half_height),
                    "position": "top-right",
                    "description": "Eastern view"
                },
                "west": {
                    "coords": (0, half_height, half_width, height),
                    "position": "bottom-left",
                    "description": "Western view"
                },
                "south": {
                    "coords": (half_width, half_height, width, height),
                    "position": "bottom-right",
                    "description": "Southern view"
                }
            }

            cropped_images = {}
            for direction, spec in crop_specs.items():
                coords = spec["coords"]
                logger.info(f"      • {direction.upper()}: {spec['position']} - box{coords}")
                cropped_images[direction] = grid_image.crop(coords)

                # Verify cropped image size
                cropped_size = cropped_images[direction].size
                logger.info(f"         Actual cropped size: {cropped_size[0]}x{cropped_size[1]} pixels")

            logger.info(f"   ✓ Successfully cropped grid into {len(cropped_images)} directional images")
            return cropped_images

        except Exception as e:
            logger.error(f"   ❌ FAILED to crop grid image: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def generate_variations(
        self,
        output_dir: str = "output/variations",
        num_variations_per_angle: int = 1
    ) -> Dict[str, Any]:
        """
        Generate angle variations for all approved images

        Args:
            output_dir: Directory to save variations
            num_variations_per_angle: How many variations to generate per angle

        Returns:
            Dictionary with all generated variations
        """
        logger.info("\n" + "="*60)
        logger.info("AGENT 8: VARIATION GENERATION STARTING")
        logger.info("="*60)

        if not self.approved_images:
            logger.warning("\nNo approved images to generate variations from")
            return {"status": "no_approved_images"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(output_dir, timestamp)

        self.generated_variations = {
            "characters": [],
            "locations": [],
            "props": []
        }

        # Generate character variations
        if self.approved_images.get('characters'):
            logger.info("\n🎭 Generating character variations...")

            for char_data in self.approved_images['characters']:
                char_name = char_data.get('name', 'Unknown')
                char_id = char_data.get('id')
                logger.info(f"\n{'='*60}")
                logger.info(f"Character: {char_name}")
                logger.info(f"{'='*60}")

                master_image = char_data['master_image']
                char_variations = {}

                for angle_config in self.character_angles:
                    angle_name = angle_config['name']
                    angle_prompt = angle_config['prompt']
                    angle_desc = angle_config['description']

                    logger.info(f"\n   📸 Generating: {angle_desc}")

                    # Generate variations using BytePlus Seedream
                    generated_images = self._generate_variation(
                        reference_image_path=master_image,
                        variation_prompt=angle_prompt,
                        aspect_ratio="1:1",
                        num_images=num_variations_per_angle
                    )

                    if not generated_images:
                        self.failed_variations.append({
                            'asset_id': char_id,
                            'asset_name': char_name,
                            'asset_type': 'character',
                            'angle': angle_name,
                            'reason': 'Image generation failed'
                        })
                        continue

                    logger.info(f"   ✓ Generated {len(generated_images)} variation(s)")

                    # Save generated images
                    logger.info(f"   💾 Saving variations...")

                    angle_images = []
                    for idx, img in enumerate(generated_images):
                        safe_name = char_name.replace(' ', '_').replace('/', '_')
                        filename = f"{safe_name}_{angle_name}_{idx+1}.png"
                        save_path = os.path.join(base_output_dir, "characters", safe_name, filename)

                        result = self._save_image(img, save_path)
                        if result and 'local_path' in result:
                            logger.info(f"   ✓ Saved: {angle_name} variation {idx+1}")
                            image_data = {
                                'index': idx + 1,
                                'url': result.get('s3_url', save_path),  # Use S3 URL if available, otherwise local path
                                'local_path': save_path,
                                'filename': filename
                            }
                            if 's3_url' in result:
                                image_data['s3_url'] = result['s3_url']
                            angle_images.append(image_data)
                        else:
                            logger.error(f"   Failed to save variation {idx+1}")

                    if angle_images:
                        char_variations[angle_name] = {
                            'angle_description': angle_desc,
                            'prompt': angle_prompt,
                            'images': angle_images
                        }

                if char_variations:
                    self.generated_variations['characters'].append({
                        'id': char_id,
                        'name': char_name,
                        'master_image': master_image,
                        'variations': char_variations,
                        'generation_timestamp': datetime.now().isoformat()
                    })

        # Generate location variations
        if self.approved_images.get('locations'):
            logger.info("\n🗺 Generating location variations...")
            logger.info(f"📋 Total locations to process: {len(self.approved_images['locations'])}")

            for loc_idx, loc_data in enumerate(self.approved_images['locations'], 1):
                loc_name = loc_data.get('name', 'Unknown')
                loc_id = loc_data.get('id')
                logger.info(f"\n{'='*60}")
                logger.info(f"Location {loc_idx}/{len(self.approved_images['locations'])}: {loc_name}")
                logger.info(f"{'='*60}")

                master_image = loc_data['master_image']
                logger.info(f"   📂 Master image path: {master_image}")
                logger.info(f"   🔑 Location ID: {loc_id}")
                loc_variations = {}

                logger.info(f"\n   📸 Generating 2x2 grid with all directional views (North, East, West, South)...")
                logger.info(f"   📝 Prompt being sent to API:")
                logger.info(f"      '{self.location_grid_prompt[:100]}...'")
                logger.info(f"   🔢 Number of grids requested: {num_variations_per_angle}")

                # Generate single 2x2 grid image with all directional views
                generated_grids = self._generate_variation(
                    reference_image_path=master_image,
                    variation_prompt=self.location_grid_prompt,
                    aspect_ratio="16:9",
                    num_images=num_variations_per_angle
                )

                if not generated_grids:
                    logger.error(f"   ❌ FAILED: Grid image generation returned None")
                    self.failed_variations.append({
                        'asset_id': loc_id,
                        'asset_name': loc_name,
                        'asset_type': 'location',
                        'angle': 'all_directions',
                        'reason': 'Grid image generation failed'
                    })
                    continue

                logger.info(f"   ✓ Generated {len(generated_grids)} 2x2 grid image(s)")

                # Process each generated grid
                for grid_idx, grid_image in enumerate(generated_grids):
                    logger.info(f"\n   {'─'*50}")
                    logger.info(f"   📐 Processing grid {grid_idx+1}/{len(generated_grids)}")
                    logger.info(f"   {'─'*50}")

                    # Log grid image details
                    grid_width, grid_height = grid_image.size
                    logger.info(f"   📏 Grid image dimensions: {grid_width}x{grid_height} pixels")
                    logger.info(f"   📏 Grid aspect ratio: {grid_width/grid_height:.2f}")

                    # Crop the grid into 4 directional images
                    logger.info(f"   ✂️  Cropping grid into 4 directional images...")
                    cropped_images = self._crop_grid_to_directions(grid_image)

                    if not cropped_images:
                        logger.error(f"   ❌ FAILED: Cropping returned empty dictionary")
                        continue

                    logger.info(f"   ✓ Successfully cropped into {len(cropped_images)} images")
                    logger.info(f"   📋 Cropped directions: {list(cropped_images.keys())}")

                    # Save each directional image
                    logger.info(f"\n   💾 Saving directional variations...")

                    for direction_config in self.location_directions:
                        direction_name = direction_config['name']
                        direction_desc = direction_config['description']
                        grid_position = direction_config['grid_position']

                        logger.info(f"\n      🧭 Direction: {direction_name.upper()} ({grid_position})")

                        if direction_name not in cropped_images:
                            logger.warning(f"      ⚠️  Missing direction '{direction_name}' in cropped images!")
                            continue

                        cropped_img = cropped_images[direction_name]
                        crop_width, crop_height = cropped_img.size
                        logger.info(f"      📏 Cropped image size: {crop_width}x{crop_height} pixels")

                        safe_name = loc_name.replace(' ', '_').replace('/', '_')

                        # Filename includes grid index if multiple grids
                        if len(generated_grids) > 1:
                            filename = f"{safe_name}_{direction_name}_{grid_idx+1}.png"
                        else:
                            filename = f"{safe_name}_{direction_name}.png"

                        save_path = os.path.join(base_output_dir, "locations", safe_name, filename)
                        logger.info(f"      💾 Saving to: {save_path}")

                        result = self._save_image(cropped_img, save_path)
                        if result and 'local_path' in result:
                            logger.info(f"      ✓ Successfully saved {direction_name} view")
                            if 's3_url' in result:
                                logger.info(f"      ☁️  S3 URL available: Yes")
                            else:
                                logger.info(f"      ☁️  S3 URL available: No (local only)")

                            # Initialize direction in variations if not exists
                            if direction_name not in loc_variations:
                                loc_variations[direction_name] = {
                                    'angle_description': direction_desc,
                                    'prompt': self.location_grid_prompt,
                                    'images': []
                                }

                            image_data = {
                                'index': grid_idx + 1,
                                'url': result.get('s3_url', save_path),  # Use S3 URL if available, otherwise local path
                                'local_path': save_path,
                                'filename': filename,
                                'dimensions': f"{crop_width}x{crop_height}"
                            }
                            if 's3_url' in result:
                                image_data['s3_url'] = result['s3_url']
                            loc_variations[direction_name]['images'].append(image_data)
                        else:
                            logger.error(f"      ❌ FAILED to save {direction_name} variation")

                    logger.info(f"\n   ✓ Completed processing grid {grid_idx+1}")
                    logger.info(f"   📊 Saved {len([d for d in loc_variations])} directional variations")

                if loc_variations:
                    logger.info(f"\n   ✨ Location '{loc_name}' completed successfully")
                    logger.info(f"   📊 Total directions saved: {len(loc_variations)}")
                    for dir_name, dir_data in loc_variations.items():
                        logger.info(f"      • {dir_name}: {len(dir_data['images'])} image(s)")

                    self.generated_variations['locations'].append({
                        'id': loc_id,
                        'name': loc_name,
                        'master_image': master_image,
                        'variations': loc_variations,
                        'generation_timestamp': datetime.now().isoformat()
                    })
                else:
                    logger.warning(f"\n   ⚠️  No variations generated for location '{loc_name}'")

        # Generate prop variations
        if self.approved_images.get('props'):
            logger.info("\nGenerating prop variations...")

            for prop_data in self.approved_images['props']:
                prop_name = prop_data.get('name', 'Unknown')
                prop_id = prop_data.get('id')
                logger.info(f"\n{'='*60}")
                logger.info(f"Prop: {prop_name}")
                logger.info(f"{'='*60}")

                master_image = prop_data['master_image']
                prop_variations = {}

                for angle_config in self.prop_angles:
                    angle_name = angle_config['name']
                    angle_prompt = angle_config['prompt']
                    angle_desc = angle_config['description']

                    logger.info(f"\n   📸 Generating: {angle_desc}")

                    # Generate variations using BytePlus Seedream
                    generated_images = self._generate_variation(
                        reference_image_path=master_image,
                        variation_prompt=angle_prompt,
                        aspect_ratio="1:1",
                        num_images=num_variations_per_angle
                    )

                    if not generated_images:
                        self.failed_variations.append({
                            'asset_id': prop_id,
                            'asset_name': prop_name,
                            'asset_type': 'prop',
                            'angle': angle_name,
                            'reason': 'Image generation failed'
                        })
                        continue

                    logger.info(f"   ✓ Generated {len(generated_images)} variation(s)")

                    # Save generated images
                    logger.info(f"   💾 Saving variations...")

                    angle_images = []
                    for idx, img in enumerate(generated_images):
                        safe_name = prop_name.replace(' ', '_').replace('/', '_')
                        filename = f"{safe_name}_{angle_name}_{idx+1}.png"
                        save_path = os.path.join(base_output_dir, "props", safe_name, filename)

                        result = self._save_image(img, save_path)
                        if result and 'local_path' in result:
                            logger.info(f"   ✓ Saved: {angle_name} variation {idx+1}")
                            image_data = {
                                'index': idx + 1,
                                'url': result.get('s3_url', save_path),  # Use S3 URL if available, otherwise local path
                                'local_path': save_path,
                                'filename': filename
                            }
                            if 's3_url' in result:
                                image_data['s3_url'] = result['s3_url']
                            angle_images.append(image_data)
                        else:
                            logger.error(f"   Failed to save variation {idx+1}")

                    if angle_images:
                        prop_variations[angle_name] = {
                            'angle_description': angle_desc,
                            'prompt': angle_prompt,
                            'images': angle_images
                        }

                if prop_variations:
                    self.generated_variations['props'].append({
                        'id': prop_id,
                        'name': prop_name,
                        'master_image': master_image,
                        'variations': prop_variations,
                        'generation_timestamp': datetime.now().isoformat()
                    })

        logger.info("\n✓ Variation generation completed!")
        self._print_variation_summary()

        return self.generated_variations

    def _print_variation_summary(self) -> None:
        """Print summary of generated variations"""
        logger.info("\n" + "─"*60)
        logger.info("VARIATION SUMMARY")
        logger.info("─"*60)

        total_variations = 0

        for asset_type in ['characters', 'locations', 'props']:
            assets = self.generated_variations.get(asset_type, [])
            if assets:
                logger.info(f"\n{asset_type.upper()}:")
                for asset_data in assets:
                    asset_name = asset_data.get('name', 'Unknown')
                    variations = asset_data.get('variations', {})
                    num_angles = len(variations)
                    num_images = sum(len(v.get('images', [])) for v in variations.values())
                    total_variations += num_images
                    logger.info(f"  • {asset_name}: {num_angles} angles, {num_images} variations")

        logger.info(f"\n✨ Total variations generated: {total_variations}")

        if self.failed_variations:
            logger.error(f"\nFailed variations: {len(self.failed_variations)}")

    def run_full_pipeline(self, agent6_output_path: str) -> Dict[str, Any]:
        """
        Run the complete Agent 8 pipeline

        Args:
            agent6_output_path: Path to Agent 6 review results JSON (original or updated)

        Returns:
            Dictionary with variation generation results
        """
        # Step 1: Load approved images
        self.load_agent6_output(agent6_output_path)

        if not self.approved_images:
            return {
                "status": "no_approved_images",
                "message": "No approved images found to generate variations from"
            }

        # Step 2: Generate variations
        results = self.generate_variations()

        return {
            "status": "completed",
            "generated_variations": results,
            "failed_variations": self.failed_variations
        }


def main():
    """Example usage of Agent 8"""
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)

    # Initialize agent
    agent = VariationGeneratorAgent()

    logger.info("Agent 8: Variation Generator initialized with Nano Banana")
    logger.info("Use run_full_pipeline() with Agent 6 output path")


if __name__ == "__main__":
    main()
