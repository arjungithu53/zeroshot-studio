#!/usr/bin/env python3
"""
Agent 5: Asset Image Generator
===============================
Generates master asset images using Gemini 3.1 Flash Image (Nano Banana) via the
multimodal generate_content endpoint, based on optimized prompts from Agent 4.

Flow:
1. Load final prompts from Agent 4
2. Generate images using Gemini 3.1 Flash Image
3. Upload images to S3
4. Track generation metadata
"""

from google import genai
from google.genai import types
from PIL import Image
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from io import BytesIO
from infrastructure.s3.client import S3ClientFactory, S3Config
from infrastructure.s3.upload import upload_file
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)


class ImageGeneratorAgent:
    """
    Agent 5: Generates images using Google Imagen 4.0 API

    This agent takes optimized prompts from Agent 4 and generates
    actual images, saving them locally with metadata tracking.
    """

    def __init__(self, api_key: str = None, enable_s3: bool = True):
        """
        Initialize Image Generator Agent

        Args:
            api_key: Google API key (optional, will use GOOGLE_API_KEY env var if not provided)
            enable_s3: Whether to upload images to S3 (default: True)
        """
        # Initialize Google Imagen client
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        self.client = genai.Client()
        self.model = 'gemini-3.1-flash-image-preview'
        self.final_prompts = {}
        self.generation_tasks = {}
        self.generated_images = {}
        self.failed_generations = []
        self.enable_s3 = enable_s3
        self.s3_client = None
        self.s3_bucket = None
        self.s3_region = None

        # Initialize S3 if enabled
        if self.enable_s3:
            self._init_s3()

    def load_agent4_output(self, agent4_output_path: str) -> None:
        """
        Load final optimized prompts from Agent 4 output file

        Args:
            agent4_output_path: Path to Agent 4 JSON output
        """
        with open(agent4_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.final_prompts = data.get('optimized_prompts', {})

        logger.info(f"✓ Loaded Agent 4 optimized prompts from: {agent4_output_path}")
        logger.info(f"   Characters: {len(self.final_prompts.get('characters', []))}")
        logger.info(f"   Locations: {len(self.final_prompts.get('locations', []))}")
        logger.info(f"   Props: {len(self.final_prompts.get('props', []))}")

    def _init_s3(self) -> None:
        """Initialize S3 client from environment variables"""
        try:
            access_key = os.getenv("production_AWS_ACCESS_KEY_ID")
            secret_key = os.getenv("production_AWS_SECRET_ACCESS_KEY")
            bucket = os.getenv("production_S3_BUCKET_NAME")
            region = os.getenv("production_AWS_REGION", "us-east-1")
            endpoint_url = os.getenv("production_S3_ENDPOINT_URL")  # For S3-compatible services

            logger.info(f"Checking S3 credentials...")
            logger.info(f"   - Access Key: {'✓ Set' if access_key else '✗ Missing (production_AWS_ACCESS_KEY_ID)'}")
            logger.info(f"   - Secret Key: {'✓ Set' if secret_key else '✗ Missing (production_AWS_SECRET_ACCESS_KEY)'}")
            logger.info(f"   - Bucket: {bucket if bucket else '✗ Missing (production_S3_BUCKET_NAME)'}")
            logger.info(f"   - Region: {region}")
            if endpoint_url:
                logger.info(f"   - Endpoint URL: {endpoint_url}")

            if not all([access_key, secret_key, bucket]):
                logger.warning("S3 credentials incomplete, disabling S3 upload")
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

            logger.info(f"S3 client initialized successfully")
            logger.info(f"   - Bucket: {bucket}")
            logger.info(f"   - Will use presigned URLs (7-day expiration)")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
            import traceback
            traceback.print_exc()
            self.enable_s3 = False

    def _get_aspect_ratio_value(self, aspect_ratio_str: str) -> str:
        """
        Convert aspect ratio string to Google Imagen API format

        Args:
            aspect_ratio_str: Aspect ratio like "16:9", "3:4", "1:1"

        Returns:
            Google Imagen API aspect ratio value
        """
        aspect_ratio_map = {
            "1:1": "1:1",
            "16:9": "16:9",
            "9:16": "9:16",
            "3:4": "3:4",
            "4:3": "4:3",
        }

        return aspect_ratio_map.get(aspect_ratio_str, "1:1")

    def _generate_images(
        self,
        prompt: str,
        negative_prompt: str = "",
        aspect_ratio: str = "1:1",
        num_images: int = 1
    ) -> Optional[List[Image.Image]]:
        """
        Generate images using Gemini 3.1 Flash Image (Nano Banana) multimodal endpoint.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Things to avoid (appended to the prompt — no separate param)
            aspect_ratio: Image aspect ratio
            num_images: Number of images to generate (1-4)

        Returns:
            List of PIL Image objects or None if failed
        """
        try:
            full_prompt = prompt
            if negative_prompt:
                full_prompt += f"\n\nDO NOT INCLUDE: {negative_prompt}"

            generated_images = []

            # Nano Banana generates 1 image per call via generate_content
            for _ in range(min(num_images, 4)):
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                        )
                    )
                )

                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                from PIL import Image as PILImage
                                from io import BytesIO
                                img = PILImage.open(BytesIO(part.inline_data.data))
                                generated_images.append(img)
                                break

            return generated_images if generated_images else None

        except Exception as e:
            logger.error(f"   Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def retry_single_asset(
        self,
        asset_name: str,
        asset_type: str,
        output_dir: str = "output/generated_images"
    ) -> Dict[str, Any]:
        """
        Retry generation for a single failed asset

        Args:
            asset_name: Name of the asset to retry (e.g., "BLACK LAB PUPPY")
            asset_type: Type of asset ('character', 'location', or 'prop')
            output_dir: Directory to save generated images

        Returns:
            Dictionary with retry result:
            {
                'success': bool,
                'asset_name': str,
                'asset_type': str,
                'images': [...] if successful,
                'error': str if failed
            }
        """
        logger.info(f"\nRetrying generation for {asset_type}: {asset_name}")

        # Find the failed generation
        failed_gen = None
        failed_index = None
        for idx, gen in enumerate(self.failed_generations):
            if gen['asset_name'] == asset_name and gen['asset_type'] == asset_type:
                failed_gen = gen
                failed_index = idx
                break

        if not failed_gen:
            return {
                'success': False,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'error': 'Failed generation not found'
            }

        # Extract parameters
        prompt = failed_gen.get('prompt', '')
        negative_prompt = failed_gen.get('negative_prompt', '')
        aspect_ratio = failed_gen.get('aspect_ratio', 'square_1_1')
        num_images = failed_gen.get('num_images', 1)
        tech_specs = failed_gen.get('technical_specs', {})

        # Submit generation request
        logger.info(f"   📤 Generating images...")
        generated_images = self._generate_images(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            num_images=num_images
        )

        if not generated_images:
            return {
                'success': False,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'error': 'Image generation failed'
            }

        logger.info(f"   ✓ Generated {len(generated_images)} image(s)")

        # Upload images to S3 (no local storage)
        logger.info(f"   💾 Uploading {len(generated_images)} image(s) to S3...")

        images = []
        for idx, img in enumerate(generated_images):
            safe_name = asset_name.replace(' ', '_').replace('/', '_')
            filename = f"{safe_name}_{idx+1}.png"
            # Generate a virtual path for S3 key generation (not used for local storage)
            virtual_path = f"{asset_type}s/{safe_name}/{filename}"

            result = self._save_image(img, virtual_path)
            if result and 's3_url' in result:
                logger.info(f"   ✓ Uploaded: {filename}")
                image_data = {
                    'index': idx + 1,
                    'url': result['s3_url'],
                    's3_url': result['s3_url'],
                    's3_key': result.get('s3_key', ''),
                    'filename': filename
                }
                images.append(image_data)
            else:
                logger.error(f"   Failed to upload image {idx+1}")

        if not images:
            return {
                'success': False,
                'asset_name': asset_name,
                'asset_type': asset_type,
                'error': 'Failed to download any images'
            }

        # Store the generated images
        asset_data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'aspect_ratio': aspect_ratio,
            'technical_specs': tech_specs,
            'images': images,
            'generation_timestamp': datetime.now().isoformat()
        }

        # Update generated_images
        asset_type_plural = asset_type + 's' if asset_type == 'character' or asset_type == 'location' else 's'
        if asset_type == 'character':
            self.generated_images['characters'][asset_name] = asset_data
        elif asset_type == 'location':
            self.generated_images['locations'][asset_name] = asset_data
        else:  # prop
            self.generated_images['props'][asset_name] = asset_data

        # Remove from failed_generations
        if failed_index is not None:
            self.failed_generations.pop(failed_index)
            logger.info(f"   Removed from failed generations list")

        logger.info(f"   Retry successful!")

        return {
            'success': True,
            'asset_name': asset_name,
            'asset_type': asset_type,
            'images': images
        }

    def _save_and_upload_image(self, image: Image.Image, save_path: str, s3_key: str = None) -> Dict[str, str]:
        """
        Upload PIL Image to S3 only (no local storage)

        Args:
            image: PIL Image object or Google GenAI Image object
            save_path: Original save path (used only for filename extraction)
            s3_key: S3 key for the uploaded file (optional)

        Returns:
            Dict with 's3_url' and 's3_key'
        """
        result = {}

        try:
            # Upload to S3 (MongoDB-only storage pattern)
            if self.enable_s3 and self.s3_client:
                try:
                    # Generate S3 key if not provided
                    if not s3_key:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = os.path.basename(save_path)
                        s3_key = f"phase1/generated_images/{timestamp}_{filename}"

                    # Convert Google GenAI Image to PIL Image if needed
                    from PIL import Image as PILImage
                    if not isinstance(image, PILImage.Image):
                        # If it's a Google GenAI Image object, access the underlying PIL image
                        if hasattr(image, '_pil_image'):
                            image = image._pil_image
                        else:
                            raise ValueError(f"Cannot convert image of type {type(image)} to PIL Image")

                    # Save image to BytesIO buffer for direct S3 upload
                    from io import BytesIO
                    buffer = BytesIO()
                    image.save(buffer, format='PNG')
                    buffer.seek(0)

                    # Upload buffer to S3
                    self.s3_client.put_object(
                        Bucket=self.s3_bucket,
                        Key=s3_key,
                        Body=buffer.getvalue(),
                        ContentType='image/png'
                    )

                    # Generate presigned URL for access
                    s3_url = self.s3_client.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                        ExpiresIn=86400 * 7  # 7 days
                    )

                    result['s3_url'] = s3_url
                    result['s3_key'] = s3_key
                    logger.info(f"   ✓ Uploaded to S3: {s3_key}")
                except Exception as s3_error:
                    logger.error(f"   S3 upload failed: {s3_error}")
                    import traceback
                    traceback.print_exc()
                    return {}
            else:
                error_msg = "S3 upload disabled (enable_s3=False)" if not self.enable_s3 else "S3 client not initialized"
                logger.error(f"   {error_msg}")
                return {}

            return result

        except Exception as e:
            logger.error(f"   Upload error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _save_image(self, image: Image.Image, save_path: str) -> Dict[str, str]:
        """
        Save PIL Image locally and optionally upload to S3

        Args:
            image: PIL Image object
            save_path: Local path to save the image

        Returns:
            Dict with 'local_path' and optionally 's3_url'
        """
        return self._save_and_upload_image(image, save_path)

    def generate_images(
        self,
        output_dir: str = "output/generated_images",
        num_images_per_asset: int = 1,
        assets_to_regenerate: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate images for all assets from Agent 4 prompts

        Args:
            output_dir: Directory to save generated images
            num_images_per_asset: Number of images to generate per asset
            assets_to_regenerate: Optional list of specific asset IDs to regenerate (format: "asset_type:asset_id")
                                  If provided, only these assets will be regenerated

        Returns:
            Dictionary containing generation results and metadata
        """
        logger.info("\n" + "="*60)
        logger.info("AGENT 5: IMAGE GENERATION STARTING")
        logger.info("="*60)

        # If assets_to_regenerate is provided, log it
        if assets_to_regenerate:
            logger.info(f"\nREGENERATION MODE: Only regenerating {len(assets_to_regenerate)} specific asset(s):")
            for asset_key in assets_to_regenerate:
                logger.info(f"   • {asset_key}")

        if not self.final_prompts:
            raise ValueError("No prompts loaded. Call load_agent4_output() first.")

        self.generated_images = {
            "characters": [],
            "locations": [],
            "props": []
        }

        # Generate character images
        logger.info("\nGenerating character images...")

        for char_data in self.final_prompts.get('characters', []):
            char_name = char_data.get('name') or char_data.get('asset_name', 'Unknown')
            char_id = char_data.get('id')

            # Skip if we're in regeneration mode and this asset is not in the list
            if assets_to_regenerate:
                asset_key = f"characters:{char_id}"
                if asset_key not in assets_to_regenerate:
                    logger.info(f"\n   ⏭Skipping {char_name} (not in regeneration list)")
                    continue

            logger.info(f"\n   Processing: {char_name}")

            final_prompt_data = char_data.get('final_prompt', {})
            prompt = final_prompt_data.get('prompt', '')
            negative_prompt = final_prompt_data.get('negative_prompt', '')
            tech_specs = final_prompt_data.get('technical_specs', {})

            if not prompt:
                logger.warning(f"   No prompt found, skipping")
                continue

            # Get aspect ratio
            aspect_ratio_str = tech_specs.get('aspect_ratio', '1:1')
            aspect_ratio = self._get_aspect_ratio_value(aspect_ratio_str)

            # Generate images
            logger.info(f"   📤 Generating images...")
            generated_images = self._generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                num_images=num_images_per_asset
            )

            if not generated_images:
                self.failed_generations.append({
                    'asset_id': char_id,
                    'asset_name': char_name,
                    'asset_type': 'character',
                    'reason': 'Image generation failed',
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'aspect_ratio': aspect_ratio,
                    'num_images': num_images_per_asset,
                    'technical_specs': tech_specs
                })
                continue

            logger.info(f"   ✓ Generated {len(generated_images)} image(s)")

            # Upload generated images to S3 (no local storage)
            logger.info(f"   💾 Uploading {len(generated_images)} image(s) to S3...")

            char_images = []
            for idx, img in enumerate(generated_images):
                # Create safe filename
                safe_name = char_name.replace(' ', '_').replace('/', '_')
                filename = f"{safe_name}_{idx+1}.png"
                # Generate a virtual path for S3 key generation (not used for local storage)
                virtual_path = f"characters/{safe_name}/{filename}"

                result = self._save_image(img, virtual_path)
                if result and 's3_url' in result:
                    logger.info(f"   ✓ Uploaded: {filename}")
                    image_data = {
                        'index': idx + 1,
                        'url': result['s3_url'],
                        's3_url': result['s3_url'],
                        's3_key': result.get('s3_key', ''),
                        'filename': filename
                    }
                    char_images.append(image_data)
                else:
                    logger.error(f"   Failed to upload image {idx+1}")

            self.generated_images['characters'].append({
                'id': char_id,
                'name': char_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'aspect_ratio': aspect_ratio,
                'technical_specs': tech_specs,
                'images': char_images,
                'generation_timestamp': datetime.now().isoformat()
            })

        # Generate location images
        logger.info("\nGenerating location images...")

        for loc_data in self.final_prompts.get('locations', []):
            loc_name = loc_data.get('name') or loc_data.get('asset_name', 'Unknown')
            loc_id = loc_data.get('id')

            # Skip if we're in regeneration mode and this asset is not in the list
            if assets_to_regenerate:
                asset_key = f"locations:{loc_id}"
                if asset_key not in assets_to_regenerate:
                    logger.info(f"\n   ⏭Skipping {loc_name} (not in regeneration list)")
                    continue

            logger.info(f"\n   Processing: {loc_name}")

            final_prompt_data = loc_data.get('final_prompt', {})
            prompt = final_prompt_data.get('prompt', '')
            negative_prompt = final_prompt_data.get('negative_prompt', '')
            tech_specs = final_prompt_data.get('technical_specs', {})

            if not prompt:
                logger.warning(f"   No prompt found, skipping")
                continue

            aspect_ratio_str = tech_specs.get('aspect_ratio', '9:16')
            aspect_ratio = self._get_aspect_ratio_value(aspect_ratio_str)

            # Generate images
            logger.info(f"   📤 Generating images...")
            generated_images = self._generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                num_images=num_images_per_asset
            )

            if not generated_images:
                self.failed_generations.append({
                    'asset_id': loc_id,
                    'asset_name': loc_name,
                    'asset_type': 'location',
                    'reason': 'Image generation failed',
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'aspect_ratio': aspect_ratio,
                    'num_images': num_images_per_asset,
                    'technical_specs': tech_specs
                })
                continue

            logger.info(f"   ✓ Generated {len(generated_images)} image(s)")

            # Upload generated images to S3 (no local storage)
            logger.info(f"   💾 Uploading {len(generated_images)} image(s) to S3...")

            loc_images = []
            for idx, img in enumerate(generated_images):
                safe_name = loc_name.replace(' ', '_').replace('/', '_')
                filename = f"{safe_name}_{idx+1}.png"
                # Generate a virtual path for S3 key generation (not used for local storage)
                virtual_path = f"locations/{safe_name}/{filename}"

                result = self._save_image(img, virtual_path)
                if result and 's3_url' in result:
                    logger.info(f"   ✓ Uploaded: {filename}")
                    image_data = {
                        'index': idx + 1,
                        'url': result['s3_url'],
                        's3_url': result['s3_url'],
                        's3_key': result.get('s3_key', ''),
                        'filename': filename
                    }
                    loc_images.append(image_data)
                else:
                    logger.error(f"   Failed to upload image {idx+1}")

            self.generated_images['locations'].append({
                'id': loc_id,
                'name': loc_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'aspect_ratio': aspect_ratio,
                'technical_specs': tech_specs,
                'images': loc_images,
                'generation_timestamp': datetime.now().isoformat()
            })

        # Generate prop images
        logger.info("\nGenerating prop images...")

        for prop_data in self.final_prompts.get('props', []):
            prop_name = prop_data.get('name') or prop_data.get('asset_name', 'Unknown')
            prop_id = prop_data.get('id')

            # Skip if we're in regeneration mode and this asset is not in the list
            if assets_to_regenerate:
                asset_key = f"props:{prop_id}"
                if asset_key not in assets_to_regenerate:
                    logger.info(f"\n   ⏭Skipping {prop_name} (not in regeneration list)")
                    continue

            logger.info(f"\n   Processing: {prop_name}")

            final_prompt_data = prop_data.get('final_prompt', {})
            prompt = final_prompt_data.get('prompt', '')
            negative_prompt = final_prompt_data.get('negative_prompt', '')
            tech_specs = final_prompt_data.get('technical_specs', {})

            if not prompt:
                logger.warning(f"   No prompt found, skipping")
                continue

            aspect_ratio_str = tech_specs.get('aspect_ratio', '1:1')
            aspect_ratio = self._get_aspect_ratio_value(aspect_ratio_str)

            # Generate images
            logger.info(f"   📤 Generating images...")
            generated_images = self._generate_images(
                prompt=prompt,
                negative_prompt=negative_prompt,
                aspect_ratio=aspect_ratio,
                num_images=num_images_per_asset
            )

            if not generated_images:
                self.failed_generations.append({
                    'asset_id': prop_id,
                    'asset_name': prop_name,
                    'asset_type': 'prop',
                    'reason': 'Image generation failed',
                    'prompt': prompt,
                    'negative_prompt': negative_prompt,
                    'aspect_ratio': aspect_ratio,
                    'num_images': num_images_per_asset,
                    'technical_specs': tech_specs
                })
                continue

            logger.info(f"   ✓ Generated {len(generated_images)} image(s)")

            # Upload generated images to S3 (no local storage)
            logger.info(f"   💾 Uploading {len(generated_images)} image(s) to S3...")

            prop_images = []
            for idx, img in enumerate(generated_images):
                safe_name = prop_name.replace(' ', '_').replace('/', '_')
                filename = f"{safe_name}_{idx+1}.png"
                # Generate a virtual path for S3 key generation (not used for local storage)
                virtual_path = f"props/{safe_name}/{filename}"

                result = self._save_image(img, virtual_path)
                if result and 's3_url' in result:
                    logger.info(f"   ✓ Uploaded: {filename}")
                    image_data = {
                        'index': idx + 1,
                        'url': result['s3_url'],
                        's3_url': result['s3_url'],
                        's3_key': result.get('s3_key', ''),
                        'filename': filename
                    }
                    prop_images.append(image_data)
                else:
                    logger.error(f"   Failed to upload image {idx+1}")

            self.generated_images['props'].append({
                'id': prop_id,
                'name': prop_name,
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'aspect_ratio': aspect_ratio,
                'technical_specs': tech_specs,
                'images': prop_images,
                'generation_timestamp': datetime.now().isoformat()
            })

        logger.info("\n✓ Image generation completed!")
        self._print_generation_summary()

        return self.generated_images

    def _print_generation_summary(self) -> None:
        """Print summary of generated images"""

        logger.info("\n" + "─"*60)
        logger.info("GENERATION SUMMARY")
        logger.info("─"*60)

        total_images = 0

        for char_data in self.generated_images.get('characters', []):
            char_name = char_data.get('name', 'Unknown')
            num_images = len(char_data.get('images', []))
            logger.info(f"🎭 {char_name}: {num_images} image(s)")
            total_images += num_images

        for loc_data in self.generated_images.get('locations', []):
            loc_name = loc_data.get('name', 'Unknown')
            num_images = len(loc_data.get('images', []))
            logger.info(f"🗺{loc_name}: {num_images} image(s)")
            total_images += num_images

        for prop_data in self.generated_images.get('props', []):
            prop_name = prop_data.get('name', 'Unknown')
            num_images = len(prop_data.get('images', []))
            logger.info(f"{prop_name}: {num_images} image(s)")
            total_images += num_images

        logger.info(f"\n✨ Total images generated: {total_images}")

        if self.failed_generations:
            logger.error(f"\nFailed generations: {len(self.failed_generations)}")
            for failure in self.failed_generations:
                logger.info(f"   • {failure['asset_type']}: {failure['asset_name']} - {failure['reason']}")

    def run_full_pipeline(
        self,
        agent4_output_path: str,
        output_dir: str = "output/generated_images",
        num_images_per_asset: int = 1
    ) -> Dict[str, Any]:
        """
        Run the complete Agent 5 pipeline

        Args:
            agent4_output_path: Path to Agent 4 output JSON
            output_dir: Directory to save generated images
            num_images_per_asset: Number of images per asset

        Returns:
            Dictionary with generation results
        """
        # Step 1: Load Agent 4 output
        self.load_agent4_output(agent4_output_path)

        # Step 2: Generate images
        results = self.generate_images(
            output_dir=output_dir,
            num_images_per_asset=num_images_per_asset
        )

        return {
            "status": "completed",
            "generated_images": results,
            "failed_generations": self.failed_generations
        }


def main():
    """Example usage of Agent 5"""
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)

    # Initialize agent
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        logger.error("ERROR: GOOGLE_API_KEY not found in environment")
        return

    agent = ImageGeneratorAgent(api_key=api_key)

    logger.info("Agent 5: Image Generator initialized with Gemini 3.1 Flash Image (Nano Banana)")
    logger.info("Use run_full_pipeline() with Agent 4 output path")


if __name__ == "__main__":
    main()
