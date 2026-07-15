"""
Agent 14: Imagen Generator Agent
=================================
Generates images using Gemini 2.5 Flash Image based on corrected prompts from Agent 13.

Flow:
1. Load corrected prompts from Agent 13 output in MongoDB
2. Fetch input images from S3 URLs (from corrected_assets)
3. Generate images using Gemini 2.5 Flash Image (image-to-image)
4. Upload images to S3 (productionvideos/phase2/)
5. Generate non-expiring public S3 URLs
6. Save URLs to MongoDB and locally
"""

import os
import logging
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from io import BytesIO
from PIL import Image
from bson import ObjectId

from google import genai
from google.genai import types
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


# --- Configuration ---
# Gemini model configuration (default, can be overridden by VIDEO_MODEL_PROVIDER)
GEMINI_MODEL_FLASH = "gemini-3.1-flash-image-preview"
GEMINI_MODEL_PRO = "gemini-3-pro-image-preview"


@dataclass
class GeneratedImage:
    """Metadata for a generated image"""
    shot_id: str
    prompt: str
    local_path: str
    s3_url: str
    generation_timestamp: str
    assets_used: List[Dict]
    metadata: Dict


class ImagenGeneratorAgent:
    """
    Agent 14: Generates images using Gemini 2.5 Flash Image

    Takes corrected prompts from Agent 13 (stored in MongoDB) and generates final images
    for each shot using Gemini 2.5 Flash Image with image-to-image generation.
    Uploads images to S3 and stores URLs in MongoDB.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        s3_client=None,
        s3_bucket: str = None
    ):
        """
        Initialize Imagen Generator Agent with Gemini client and S3

        Args:
            api_key: Google API key (optional, can use Application Default Credentials)
            s3_client: Optional boto3 S3 client (if None, will create from env vars)
            s3_bucket: S3 bucket name (if None, will use env var)
        """
        # Check which video model provider to use
        self.video_model_provider = os.getenv("VIDEO_MODEL_PROVIDER", "flash").lower()

        # Initialize Gemini client (HttpOptions no longer supports client_args/transport in newer versions)
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()  # Uses ADC

        # Set model based on provider
        if self.video_model_provider == "pro":
            self.model_name = GEMINI_MODEL_PRO
            logger.info("✓ Initialized with Gemini 3 Pro Image Preview for image generation")
        else:
            self.model_name = GEMINI_MODEL_FLASH
            logger.info("✓ Initialized with Gemini 2.5 Flash Image for image generation")

        self.generated_images = []
        self.failed_generations = []

        # S3 setup
        if s3_client is None:
            self.s3_client = self._create_s3_client()
        else:
            self.s3_client = s3_client

        self.s3_bucket = (
            s3_bucket
            or os.getenv("AWS_S3_BUCKET_NAME")
            or os.getenv("production_S3_BUCKET_NAME")
            or "productionvideos"
        )
        self.s3_folder = "phase2"

        logger.info(f"Initialized Imagen Generator Agent with Gemini {self.model_name}")
        logger.info(f"S3 bucket: {self.s3_bucket}/{self.s3_folder}")

    def _create_s3_client(self):
        """Create S3 client from environment variables"""
        access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("production_AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("production_AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN") or os.getenv("production_AWS_SESSION_TOKEN")
        region = os.getenv("AWS_REGION") or os.getenv("production_AWS_REGION") or "us-east-1"
        endpoint_url = os.getenv("AWS_ENDPOINT_URL") or os.getenv("production_AWS_ENDPOINT_URL")

        if not access_key or not secret_key:
            logger.error(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or "
                "production_AWS_ACCESS_KEY_ID/production_AWS_SECRET_ACCESS_KEY."
            )
            return None

        try:
            client_kwargs = {
                'aws_access_key_id': access_key,
                'aws_secret_access_key': secret_key,
                'region_name': region,
            }
            if session_token:
                client_kwargs['aws_session_token'] = session_token
            if endpoint_url:
                client_kwargs['endpoint_url'] = endpoint_url

            s3_client = boto3.client('s3', **client_kwargs)
            logger.info("✓ S3 client created successfully")
            return s3_client
        except Exception as e:
            logger.error(f"Failed to create S3 client: {e}")
            return None

    def _fetch_aspect_ratio_from_movies(self, movie_id: str) -> str:
        """
        Fetch aspect ratio from movies collection using movie_id
        
        Args:
            movie_id: Movie ID (ObjectId string) to query
            
        Returns:
            Aspect ratio string (defaults to "2.38:1" if not found)
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
                return "2.38:1"
                
            # Query movies collection by _id
            movie = movies_collection.find_one(
                {"_id": movie_obj_id},
                {"global_settings.aspect_ratio": 1}
            )
            
            if movie:
                aspect_ratio = movie.get("global_settings", {}).get("aspect_ratio", "2.38:1")
                logger.info(f"✓ Fetched aspect ratio from movies collection: {aspect_ratio}")
                return aspect_ratio
            else:
                logger.warning(f"Movie not found with _id: {movie_id}, using default aspect ratio 2.38:1")
                return "2.38:1"
                
        except Exception as e:
            logger.error(f"Error fetching aspect ratio from movies collection: {e}")
            logger.warning("Falling back to default aspect ratio 2.38:1")
            return "2.38:1"

    def _fetch_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """
        Fetch an image from a URL. Uses boto3 for private S3 URLs (avoids 403).
        """
        from app.services.phase_2_agents.helpers.image_fetch import fetch_image_bytes

        logger.info(f"🔍 Attempting to fetch image from URL: {image_url[:100]}...")
        img_bytes = fetch_image_bytes(image_url)
        if img_bytes:
            try:
                image = Image.open(BytesIO(img_bytes))
                logger.info(f"✓ Successfully fetched image - Size: {image.size}, Mode: {image.mode}, Format: {image.format}")
                return image
            except Exception as e:
                logger.error(f"❌ Error opening image bytes from {image_url[:100]}: {e}")
                return None
        logger.error(f"❌ Failed to fetch image from URL: {image_url[:100]}")
        return None

    def generate_image(self, prompt: str, shot_id: str, input_images: List[Image.Image] = None, movie_id: Optional[str] = None, assets_metadata: List[Dict] = None, negative_prompt: str = "") -> Optional[bytes]:
        """
        Generate a single image using configured Gemini model (Flash or Pro)

        Args:
            prompt: Text prompt for image generation
            shot_id: Shot identifier for logging
            input_images: Optional list of PIL Image objects to use as input (for image-to-image)
            movie_id: Optional movie ID to fetch aspect ratio from movies collection
            assets_metadata: Optional list of asset dictionaries with 'name', 'type', 'character' etc.
            negative_prompt: Things to exclude (Nano Banana has no separate field; appended to text)

        Returns:
            Image data as bytes or None if failed
        """
        # --- Fetch dynamic aspect ratio from movies collection ---
        if movie_id:
            aspect_ratio = self._fetch_aspect_ratio_from_movies(movie_id)
        else:
            aspect_ratio = "9:16"  # Default fallback
            logger.warning(f"No movie_id provided for {shot_id}, using default aspect ratio 9:16")

        # Check if storyboard is present in assets_metadata
        has_storyboard = False
        if assets_metadata:
            has_storyboard = any(asset.get('type') == 'storyboard' for asset in assets_metadata)

        # Add storyboard instruction if present
        storyboard_instruction = ""
        if has_storyboard:
            storyboard_instruction = (
                "\n\nMUST FOLLOW INSTRUCTION: "
                "Ensure that the positioning, location & posture of the subjects EXACTLY matches the storyboard and is exactly the same. "
                "You should follow the storyboard provided properly for staging and composition of the image also. "
                "\n\nIMPORTANT: A storyboard reference image is provided. Use it as a compositional and framing reference "
                "while generating the image. Match the camera angle, subject positioning, and overall scene composition from the storyboard."
            )

        fixed_ratio_instruction = (
            f"\n\nIMPORTANT: ALWAYS GENERATE THE IMAGE IN EXACT ASPECT RATIO {aspect_ratio}. "
            "IGNORE ANY OTHER ASPECT RATIO IN THE PROMPT. THIS OVERRIDES ALL OTHER INSTRUCTIONS."
        )
        prompt = prompt + storyboard_instruction + fixed_ratio_instruction

        if self.video_model_provider == "pro":
            return self._generate_image_gemini_pro(prompt, shot_id, input_images, assets_metadata, aspect_ratio=aspect_ratio, negative_prompt=negative_prompt)
        else:
            return self._generate_image_gemini_flash(prompt, shot_id, input_images, assets_metadata, aspect_ratio=aspect_ratio, negative_prompt=negative_prompt)

    def _generate_image_gemini_flash(self, prompt: str, shot_id: str, input_images: List[Image.Image] = None, assets_metadata: List[Dict] = None, aspect_ratio: str = "9:16", negative_prompt: str = "") -> Optional[bytes]:
        """
        Generate a single image using Gemini 2.5 Flash Image
        Uses the new SDK format with labeled reference images for better image-to-image composition.

        Args:
            prompt: Text prompt for image generation
            shot_id: Shot identifier for logging
            input_images: Optional list of PIL Image objects to use as input (for image-to-image)
            assets_metadata: Optional list of asset dictionaries with 'name', 'type', 'character' etc.

        Returns:
            Image data as bytes or None if failed
        """
        try:
            logger.info(f"Generating image for {shot_id} (Gemini 2.5 Flash)")
            logger.debug(f"Prompt: {prompt[:100]}...")
            if input_images:
                logger.info(f"Using {len(input_images)} input image(s) for composition with labeled reference format")

            # Build contents for Gemini Flash using new labeled reference format
            contents = []

            # NEW SDK FORMAT: Add labeled reference images BEFORE the final instruction
            if input_images:
                # Label each reference image with its actual name from metadata
                has_storyboard_in_assets = False
                for idx, img in enumerate(input_images):
                    # Get asset metadata for this image
                    asset_name = "Unknown"
                    asset_type = "Asset"

                    if assets_metadata and idx < len(assets_metadata):
                        asset = assets_metadata[idx]
                        # Try to get the most descriptive name
                        asset_name = asset.get('character') or asset.get('name', 'Unknown')
                        asset_type = asset.get('type', 'Asset').capitalize()

                        # Check if this is a storyboard
                        if asset.get('type') == 'storyboard':
                            has_storyboard_in_assets = True

                    # Create descriptive label for this reference image
                    if asset_type.lower() == 'storyboard':
                        label = (
                            f"STORYBOARD REFERENCE IMAGE - THIS IS THE COMPOSITION TEMPLATE YOU MUST FOLLOW EXACTLY:\n"
                            f"Match the exact positioning, location, posture, camera angle, and staging shown in this storyboard:"
                        )
                    elif asset_type.lower() == 'product':
                        label = (
                            f"PRODUCT REFERENCE — REPRODUCE THIS EXACTLY:\n"
                            f"The product in the generated image MUST have the identical shape, dimensions, label text, "
                            f"logo, colors, lid design, and branding as shown in this reference image. "
                            f"Do NOT substitute, simplify, or reinterpret the product's appearance:"
                        )
                    else:
                        label = f"Reference image for the {asset_type} ({asset_name}):"

                    contents.append(label)
                    contents.append(img)
                    logger.info(f"  Added labeled reference: {asset_type} - {asset_name}")

                # Add final instruction that references the labeled images
                # Emphasize storyboard matching if present
                if has_storyboard_in_assets:
                    final_instruction = (
                        f"CRITICAL INSTRUCTION: The storyboard image above shows the EXACT composition you must replicate. "
                        f"Ensure that the positioning, location & posture of ALL subjects EXACTLY matches the storyboard. "
                        f"Follow the storyboard's staging, camera angle, and composition PRECISELY. "
                        f"\n\nGenerate a realistic cinematic image combining these reference images above. "
                        f"{prompt}"
                    )
                else:
                    final_instruction = (
                        f"Generate a realistic cinematic image combining these reference images above. "
                        f"{prompt}"
                    )
                has_product_asset = assets_metadata and any(
                    a.get('type') == 'product' for a in assets_metadata
                )
                if has_product_asset:
                    final_instruction += (
                        "\n\nCRITICAL PRODUCT FIDELITY RULE: One of the reference images above is an actual product. "
                        "In the generated image, this product must appear with EXACTLY the same: "
                        "shape, size, label text, logo, colors, lid/cap design, and overall form. "
                        "Do NOT simplify, stylize, or substitute the product's appearance. "
                        "Reproduce it faithfully as if photographed."
                    )
                # Nano Banana has no separate negative_prompt kwarg; concatenate into text.
                if negative_prompt:
                    final_instruction += (
                        f"\n\nCRITICAL CONSTRAINTS - DO NOT INCLUDE UNDER ANY CIRCUMSTANCES: {negative_prompt}"
                    )
                contents.append(final_instruction)
                logger.info("  Added final generation instruction referencing labeled images")
            else:
                # No input images - use text-only generation (original behavior)
                text_only_prompt = prompt
                if negative_prompt:
                    text_only_prompt += (
                        f"\n\nCRITICAL CONSTRAINTS - DO NOT INCLUDE UNDER ANY CIRCUMSTANCES: {negative_prompt}"
                    )
                contents.append(text_only_prompt)

            # 🔍 DEBUG: Log what we're sending to API
            logger.info("="*70)
            logger.info(f"📤 SENDING TO GEMINI API ({self.model_name})")
            logger.info("="*70)
            logger.info(f"Contents array length: {len(contents)}")
            for i, item in enumerate(contents):
                if isinstance(item, Image.Image):
                    logger.info(f"  [{i}] PIL Image - Size: {item.size}, Mode: {item.mode}")
                elif isinstance(item, str):
                    logger.info(f"  [{i}] String (prompt) - Length: {len(item)} chars")
                    logger.info(f"      Preview: {item[:100]}...")
                else:
                    logger.info(f"  [{i}] {type(item)}")
            logger.info("="*70)

            # Generate image using Gemini Flash
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

            # Extract image data from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.inline_data:
                            image_data = part.inline_data.data
                            logger.info(f"✓ Image generated successfully for {shot_id}")
                            return image_data

            logger.error(f"No image data in response for {shot_id}")
            return None

        except Exception as e:
            logger.error(f"Gemini Flash generation failed for {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_image_gemini_pro(self, prompt: str, shot_id: str, input_images: List[Image.Image] = None, assets_metadata: List[Dict] = None, aspect_ratio: str = "9:16", negative_prompt: str = "") -> Optional[bytes]:
        """
        Generate a single image using Gemini 3 Pro Image Preview
        Uses the new SDK format with labeled reference images for better image-to-image composition.

        Args:
            prompt: Text prompt for image generation
            shot_id: Shot identifier for logging
            input_images: Optional list of PIL Image objects to use as input (for image-to-image)
            assets_metadata: Optional list of asset dictionaries with 'name', 'type', 'character' etc.
            aspect_ratio: Aspect ratio for the generated image. Defaults to "9:16".

        Returns:
            Image data as bytes or None if failed
        """
        try:
            logger.info(f"Generating image for {shot_id} (Gemini 3 Pro)")
            logger.debug(f"Prompt: {prompt[:100]}...")
            if input_images:
                logger.info(f"Using {len(input_images)} input image(s) for composition with labeled reference format")

            # Build contents for Gemini Pro using new labeled reference format
            contents = []

            # NEW SDK FORMAT: Add labeled reference images BEFORE the final instruction
            if input_images:
                # Label each reference image with its actual name from metadata
                has_storyboard_in_assets = False
                for idx, img in enumerate(input_images):
                    # Get asset metadata for this image
                    asset_name = "Unknown"
                    asset_type = "Asset"

                    if assets_metadata and idx < len(assets_metadata):
                        asset = assets_metadata[idx]
                        # Try to get the most descriptive name
                        asset_name = asset.get('character') or asset.get('name', 'Unknown')
                        asset_type = asset.get('type', 'Asset').capitalize()

                        # Check if this is a storyboard
                        if asset.get('type') == 'storyboard':
                            has_storyboard_in_assets = True

                    # Create descriptive label for this reference image
                    if asset_type.lower() == 'storyboard':
                        label = (
                            f"STORYBOARD REFERENCE IMAGE - THIS IS THE COMPOSITION TEMPLATE YOU MUST FOLLOW EXACTLY:\n"
                            f"Match the exact positioning, location, posture, camera angle, and staging shown in this storyboard:"
                        )
                    elif asset_type.lower() == 'product':
                        label = (
                            f"PRODUCT REFERENCE — REPRODUCE THIS EXACTLY:\n"
                            f"The product in the generated image MUST have the identical shape, dimensions, label text, "
                            f"logo, colors, lid design, and branding as shown in this reference image. "
                            f"Do NOT substitute, simplify, or reinterpret the product's appearance:"
                        )
                    else:
                        label = f"Reference image for the {asset_type} ({asset_name}):"

                    contents.append(label)
                    contents.append(img)
                    logger.info(f"  Added labeled reference: {asset_type} - {asset_name}")

                # Add final instruction that references the labeled images
                # Emphasize storyboard matching if present
                if has_storyboard_in_assets:
                    final_instruction = (
                        f"CRITICAL INSTRUCTION: The storyboard image above shows the EXACT composition you must replicate. "
                        f"Ensure that the positioning, location & posture of ALL subjects EXACTLY matches the storyboard. "
                        f"Follow the storyboard's staging, camera angle, and composition PRECISELY. "
                        f"\n\nGenerate a realistic cinematic image combining these reference images above. "
                        f"{prompt}"
                    )
                else:
                    final_instruction = (
                        f"Generate a realistic cinematic image combining these reference images above. "
                        f"{prompt}"
                    )
                has_product_asset = assets_metadata and any(
                    a.get('type') == 'product' for a in assets_metadata
                )
                if has_product_asset:
                    final_instruction += (
                        "\n\nCRITICAL PRODUCT FIDELITY RULE: One of the reference images above is an actual product. "
                        "In the generated image, this product must appear with EXACTLY the same: "
                        "shape, size, label text, logo, colors, lid/cap design, and overall form. "
                        "Do NOT simplify, stylize, or substitute the product's appearance. "
                        "Reproduce it faithfully as if photographed."
                    )
                # Nano Banana has no separate negative_prompt kwarg; concatenate into text.
                if negative_prompt:
                    final_instruction += (
                        f"\n\nCRITICAL CONSTRAINTS - DO NOT INCLUDE UNDER ANY CIRCUMSTANCES: {negative_prompt}"
                    )
                contents.append(final_instruction)
                logger.info("  Added final generation instruction referencing labeled images")
            else:
                # No input images - use text-only generation (original behavior)
                text_only_prompt = prompt
                if negative_prompt:
                    text_only_prompt += (
                        f"\n\nCRITICAL CONSTRAINTS - DO NOT INCLUDE UNDER ANY CIRCUMSTANCES: {negative_prompt}"
                    )
                contents.append(text_only_prompt)

            # 🔍 DEBUG: Log what we're sending to API
            logger.info("="*70)
            logger.info(f"📤 SENDING TO GEMINI API ({self.model_name})")
            logger.info("="*70)
            logger.info(f"Contents array length: {len(contents)}")
            for i, item in enumerate(contents):
                if isinstance(item, Image.Image):
                    logger.info(f"  [{i}] PIL Image - Size: {item.size}, Mode: {item.mode}")
                elif isinstance(item, str):
                    logger.info(f"  [{i}] String (prompt) - Length: {len(item)} chars")
                    logger.info(f"      Preview: {item[:100]}...")
                else:
                    logger.info(f"  [{i}] {type(item)}")
            logger.info("="*70)

            # Generate image using Gemini 3 Pro
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

            # STRATEGY 1 (Primary): Try new Gemini response format
            # response.candidates → candidate.content.parts → part.inline_data.data
            try:
                logger.info(f"Using Strategy 1 (candidates.parts) for {shot_id}...")
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                                for part in candidate.content.parts:
                                    if hasattr(part, 'inline_data') and part.inline_data is not None:
                                        if hasattr(part.inline_data, 'data'):
                                            image_data = part.inline_data.data
                                            logger.info(f"✓ Image generated successfully for {shot_id} (Strategy 1)")
                                            return image_data
                logger.debug(f"Strategy 1 found no image data for {shot_id}")
            except Exception as e:
                logger.warning(f"Strategy 1 extraction failed for {shot_id}: {e}")
                import traceback
                traceback.print_exc()

            # STRATEGY 2 (Fallback): Try old Gemini response format
            # response.parts → part.inline_data or part.as_image()
            try:
                logger.info(f"Using Strategy 2 fallback (response.parts) for {shot_id}...")
                if hasattr(response, 'parts') and response.parts:
                    for part in response.parts:
                        # Check if part has inline image data
                        if hasattr(part, 'inline_data') and part.inline_data is not None:
                            if hasattr(part.inline_data, 'data'):
                                image_data = part.inline_data.data
                                logger.info(f"✓ Image generated successfully for {shot_id} (Strategy 2 - inline_data)")
                                return image_data
                        # Try as_image() method
                        elif hasattr(part, 'as_image'):
                            try:
                                image = part.as_image()
                                # Convert PIL Image to bytes
                                img_byte_arr = BytesIO()
                                image.save(img_byte_arr, format='PNG')
                                image_data = img_byte_arr.getvalue()
                                logger.info(f"✓ Image generated successfully for {shot_id} (Strategy 2 - as_image)")
                                return image_data
                            except Exception as conv_error:
                                logger.warning(f"Failed to convert image using as_image() for {shot_id}: {conv_error}")
                                continue
                logger.debug(f"Strategy 2 found no image data for {shot_id}")
            except Exception as e:
                logger.warning(f"Strategy 2 extraction failed for {shot_id}: {e}")
                import traceback
                traceback.print_exc()

            # Both strategies failed
            logger.error(f"No image data in response for {shot_id} (both strategies failed)")
            return None

        except Exception as e:
            logger.error(f"Gemini Pro generation failed for {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def upload_to_s3(self, image_data: bytes, shot_id: str) -> Optional[str]:
        """
        Upload image to S3 and return public URL
        
        Args:
            image_data: Raw image bytes
            shot_id: Shot identifier for filename
            
        Returns:
            Public S3 URL or None if failed
        """
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
            
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_shot_id = shot_id.replace('.', '_').replace('/', '_')
            filename = f"{safe_shot_id}_{timestamp}.png"
            s3_key = f"{self.s3_folder}/{filename}"
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=image_data,
                ContentType='image/png'
            )
            
            # Generate presigned URL (7 days) so private-bucket objects are accessible
            s3_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': s3_key},
                ExpiresIn=604800  # 7 days
            )
            
            logger.info(f"✓ Uploaded to S3: {s3_url}")
            return s3_url
            
        except ClientError as e:
            logger.error(f"S3 upload failed for {shot_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading to S3 for {shot_id}: {e}")
            return None

    def save_image_locally(self, image_data: bytes, save_path: str) -> bool:
        """
        Save image data to local file
        
        Args:
            image_data: Raw image bytes
            save_path: Local path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"✓ Saved locally: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Local save failed: {e}")
            return False

    def _process_single_shot(self, shot_data: Dict, i: int, total: int, base_output_dir: str, movie_id: Optional[str]) -> Dict:
        """
        Process a single shot: fetch assets, generate image, upload to S3, save locally.
        Returns a result dict — never mutates shared instance state directly.
        """
        shot_id = shot_data.get('shot_id', f'shot_{i}')
        corrected_prompt = shot_data.get('corrected_prompt', '')
        corrected_assets = shot_data.get('corrected_assets', [])
        storyboard_s3_link = shot_data.get('storyboard_s3_link')
        negative_prompt = shot_data.get('negative_prompt', '') or ''

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Shot {shot_id} ({i}/{total})")
        logger.info(f"{'='*50}")

        if not corrected_prompt:
            logger.warning(f"No corrected prompt found for {shot_id}, skipping")
            return {'status': 'failed', 'image': None, 'failure': {'shot_id': shot_id, 'reason': 'No corrected prompt available'}}

        # Fetch input images from S3 URLs in corrected_assets
        input_images = []
        assets_metadata = []
        logger.info("="*70)
        logger.info(f"🖼️  ASSET IMAGE FETCHING FOR {shot_id}")
        logger.info("="*70)
        logger.info(f"corrected_assets type: {type(corrected_assets)}")
        logger.info(f"corrected_assets length: {len(corrected_assets) if corrected_assets else 0}")

        if corrected_assets:
            logger.info(f"📦 Attempting to fetch {len(corrected_assets)} asset image(s) from S3...")
            for idx, asset in enumerate(corrected_assets):
                logger.info(f"\n  Asset [{idx}]: {asset}")
                asset_url = asset.get('url', '')
                asset_name = asset.get('character') or asset.get('name', 'unknown')
                asset_type = asset.get('type', 'unknown')

                if asset_url:
                    logger.info(f"  ➤ Fetching {asset_type}: {asset_name}")
                    logger.info(f"    URL: {asset_url[:150]}...")
                    img = self._fetch_image_from_url(asset_url)
                    if img:
                        input_images.append(img)
                        assets_metadata.append(asset)
                        logger.info(f"    ✅ Fetched successfully - Size: {img.size}, Mode: {img.mode}")
                    else:
                        logger.error(f"    ❌ Failed to fetch {asset_name} from {asset_url[:100]}")
                else:
                    logger.warning(f"  ⚠️  No URL found for {asset_type}: {asset_name}")
        else:
            logger.warning("⚠️  No corrected_assets provided")

        logger.info("="*70)
        logger.info(f"📊 ASSET FETCHING SUMMARY for {shot_id}")
        logger.info("="*70)
        logger.info(f"Total assets in corrected_assets: {len(corrected_assets) if corrected_assets else 0}")
        logger.info(f"Successfully fetched images: {len(input_images)}")
        logger.info(f"Failed/missing: {(len(corrected_assets) if corrected_assets else 0) - len(input_images)}")
        logger.info("="*70)

        expected_product = any(a.get('type') == 'product' for a in corrected_assets) if corrected_assets else False
        fetched_product = any(a.get('type') == 'product' for a in assets_metadata)
        if expected_product and not fetched_product:
            logger.error(
                f"❌ PRODUCT IMAGE FETCH FAILED for {shot_id} — "
                f"product reference image could not be loaded. "
                f"The generated image will NOT have the correct product appearance. "
                f"Check that product_image_s3_url is not expired."
            )

        # Fetch storyboard image if provided
        if storyboard_s3_link:
            logger.info("="*70)
            logger.info(f"📋 STORYBOARD IMAGE FETCHING FOR {shot_id}")
            logger.info("="*70)
            logger.info(f"Storyboard S3 link: {storyboard_s3_link[:150]}...")
            storyboard_image = self._fetch_image_from_url(storyboard_s3_link)
            if storyboard_image:
                logger.info(f"✅ Storyboard fetched successfully - Size: {storyboard_image.size}, Mode: {storyboard_image.mode}")
                input_images.append(storyboard_image)
                assets_metadata.append({
                    'name': 'Storyboard',
                    'type': 'storyboard',
                    'character': 'Storyboard'
                })
                logger.info(f"✓ Storyboard added to input images for {shot_id}")
            else:
                logger.error(f"❌ Failed to fetch storyboard from {storyboard_s3_link[:100]}")
            logger.info("="*70)

        if not input_images:
            logger.warning(f"⚠️  No input images available for {shot_id}, will generate from text only")
        else:
            logger.info(f"✓ Will use {len(input_images)} input image(s) for generation")

        image_data = self.generate_image(
            corrected_prompt,
            shot_id,
            input_images,
            movie_id=movie_id,
            assets_metadata=assets_metadata if assets_metadata else None,
            negative_prompt=negative_prompt
        )

        if not image_data:
            return {'status': 'failed', 'image': None, 'failure': {'shot_id': shot_id, 'reason': 'Image generation failed'}}

        s3_url = self.upload_to_s3(image_data, shot_id)
        if not s3_url:
            logger.warning(f"S3 upload failed for {shot_id}, saving locally only")
            s3_url = ""

        safe_shot_id = shot_id.replace('.', '_').replace('/', '_')
        local_path = os.path.join(base_output_dir, f"{safe_shot_id}.png")

        if self.save_image_locally(image_data, local_path):
            logger.info(f"✓ Shot {shot_id} completed successfully")
            return {
                'status': 'success',
                'image': GeneratedImage(
                    shot_id=shot_id,
                    prompt=corrected_prompt,
                    local_path=local_path,
                    s3_url=s3_url,
                    generation_timestamp=datetime.now().isoformat(),
                    assets_used=corrected_assets,
                    metadata=shot_data.get('metadata', {})
                ),
                'failure': None
            }
        else:
            return {'status': 'failed', 'image': None, 'failure': {'shot_id': shot_id, 'reason': 'Image save failed'}}

    def generate_images_for_shots(
        self,
        modified_shots: List[Dict],
        output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_14_generated_images",
        movie_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate images for all shots from Agent 13's modified prompts.
        Shots are processed concurrently (max 3 at a time) using a thread pool.

        Args:
            modified_shots: List of modified shot dictionaries from Agent 13
            output_dir: Directory to save generated images locally
            movie_id: Optional movie ID to fetch aspect ratio from movies collection

        Returns:
            Dictionary containing generation results and metadata
        """
        logger.info("="*60)
        logger.info("AGENT 14: IMAGE GENERATION STARTING (Gemini 2.5 Flash Image)")
        logger.info("="*60)

        if not modified_shots:
            raise ValueError("No modified shots provided")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(output_dir, timestamp)

        logger.info(f"Output directory: {base_output_dir}")
        logger.info(f"Total shots to generate: {len(modified_shots)}")

        results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._process_single_shot, shot_data, i + 1, len(modified_shots), base_output_dir, movie_id): shot_data
                for i, shot_data in enumerate(modified_shots)
            }
            for future in as_completed(futures):
                results.append(future.result())

        for result in results:
            if result['status'] == 'success' and result['image']:
                self.generated_images.append(result['image'])
            elif result['failure']:
                self.failed_generations.append(result['failure'])

        logger.info("\n" + "="*60)
        logger.info("✓ IMAGE GENERATION COMPLETED")
        logger.info("="*60)
        self._print_generation_summary()

        return {
            'generated_images': [asdict(img) for img in self.generated_images],
            'failed_generations': self.failed_generations
        }

    def _print_generation_summary(self) -> None:
        """Print summary of generated images"""
        logger.info("─"*60)
        logger.info("📊 GENERATION SUMMARY")
        logger.info("─"*60)
        logger.info(f"✨ Total images generated: {len(self.generated_images)}")
        
        if self.failed_generations:
            logger.warning(f"❌ Failed generations: {len(self.failed_generations)}")
            for failure in self.failed_generations:
                logger.warning(f"   • {failure['shot_id']}: {failure['reason']}")
        else:
            logger.info("✓ No failures!")

    def save_metadata(self, output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_14_generated_images") -> str:
        """
        Save generation metadata and results
        
        Args:
            output_dir: Directory to save metadata file
            
        Returns:
            Path to saved metadata file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent14_imagen_output_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        metadata = {
            "agent": "Agent 14: Imagen Generator",
            "timestamp": datetime.now().isoformat(),
            "generated_images": [asdict(img) for img in self.generated_images],
            "failed_generations": self.failed_generations,
            "statistics": {
                "total_shots_processed": len(self.generated_images) + len(self.failed_generations),
                "successful_generations": len(self.generated_images),
                "failed_generations": len(self.failed_generations),
                "success_rate": f"{(len(self.generated_images) / (len(self.generated_images) + len(self.failed_generations)) * 100):.1f}%" if (len(self.generated_images) + len(self.failed_generations)) > 0 else "0%"
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Generation metadata saved to: {filepath}")
        return filepath


def save_results(
    generated_images: List[GeneratedImage],
    failed_generations: List[Dict],
    output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_14_generated_images"
) -> str:
    """
    Save agent 14 results to JSON file
    
    Args:
        generated_images: List of GeneratedImage objects
        failed_generations: List of failed generation dictionaries
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agent14_imagen_output_{timestamp}.json"
    
    output_data = {
        "agent": "Agent 14: Imagen Generator",
        "timestamp": datetime.now().isoformat(),
        "generated_images": [asdict(img) for img in generated_images],
        "failed_generations": failed_generations,
        "statistics": {
            "total_shots_processed": len(generated_images) + len(failed_generations),
            "successful_generations": len(generated_images),
            "failed_generations": len(failed_generations),
            "success_rate": f"{(len(generated_images) / (len(generated_images) + len(failed_generations)) * 100):.1f}%" if (len(generated_images) + len(failed_generations)) > 0 else "0%"
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"✓ Results saved to: {filename}")
    return filename


def main():
    """Example usage of Agent 14"""
    logger.info("Agent 14: Imagen Generator")
    logger.info("Usage: Initialize agent and call generate_images_for_shots(modified_shots)")


if __name__ == "__main__":
    main()
