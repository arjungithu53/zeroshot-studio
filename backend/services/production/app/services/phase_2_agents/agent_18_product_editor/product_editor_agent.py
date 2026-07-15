#!/usr/bin/env python3
"""
Agent 18: Product Image Editor using Nano Banana (Phase 2)
==========================================================
Edits generated images to replace incorrectly rendered products with
the correct reference product image using Nano Banana (gemini-3.1-flash-image-preview).

Flow:
1. For each shot in shots_needing_product_fix:
   - Download latest generated/edited image
   - Download product reference image
   - Apply Nano Banana edit using prompt from Agent 17
   - Upload corrected image to S3
   - Store new URL in product_corrected_images
2. Return corrected image URLs for Agent 16 re-review
"""

import os
import logging
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any

from io import BytesIO
from PIL import Image as PILImage

from google import genai
from google.genai import types

from infrastructure.s3.upload import S3ImageUploader
from app.services.phase_2_agents.helpers.image_fetch import fetch_image_bytes

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
NANO_BANANA_MODEL = "gemini-3.1-flash-image-preview"
S3_BUCKET = os.getenv("production_S3_BUCKET_NAME", "productionvideos")


class ProductEditorAgent:
    """
    Agent 18: Replaces incorrectly rendered products using Nano Banana.

    Takes the generated image, the reference product image, and the
    replacement prompt from Agent 17, and produces a corrected image.
    """

    def __init__(self, api_key: str = API_KEY):
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        self.client = genai.Client(api_key=api_key)
        self.nano_banana_model = NANO_BANANA_MODEL
        self.s3_uploader = S3ImageUploader(bucket_name=S3_BUCKET)

        logger.info("=" * 60)
        logger.info("AGENT 18: PRODUCT EDITOR (NANO BANANA) INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Image Edit API: Nano Banana ({NANO_BANANA_MODEL})")
        logger.info(f"S3 Bucket: {S3_BUCKET}")
        logger.info("=" * 60)

    def _fetch_image_bytes(self, url_or_path: str) -> Optional[bytes]:
        """Fetch raw image bytes from a URL or local path."""
        return fetch_image_bytes(url_or_path)

    def _detect_aspect_ratio(self, image_bytes: bytes) -> str:
        """Snap the input image's dimensions to the nearest Gemini-supported aspect ratio."""
        supported = [
            ("21:9", 21 / 9),
            ("16:9", 16 / 9),
            ("4:3", 4 / 3),
            ("1:1", 1.0),
            ("3:4", 3 / 4),
            ("9:16", 9 / 16),
        ]
        try:
            with PILImage.open(BytesIO(image_bytes)) as img:
                ratio = img.width / img.height
        except Exception as e:
            logger.warning(f"Could not detect aspect ratio, defaulting to 9:16: {e}")
            return "9:16"
        return min(supported, key=lambda r: abs(r[1] - ratio))[0]

    def edit_shot(
        self,
        shot_id: str,
        generated_image_url: str,
        product_image_url: str,
        edit_prompt: str,
        show_id: str,
        iteration: int,
    ) -> Optional[str]:
        """
        Edit a single shot image using Nano Banana.

        Args:
            shot_id: Shot identifier
            generated_image_url: URL of the current generated/edited image
            product_image_url: URL of the reference product image
            edit_prompt: Nano Banana editing prompt from Agent 17
            show_id: Show identifier for S3 path
            iteration: Current product fix iteration (for versioning)

        Returns:
            S3 URL of the corrected image, or None if editing failed
        """
        logger.info(f"Editing product for shot: {shot_id} (iteration {iteration})")
        logger.info(f"Edit prompt: {edit_prompt[:120]}...")

        generated_bytes = self._fetch_image_bytes(generated_image_url)
        product_bytes = self._fetch_image_bytes(product_image_url)

        if not generated_bytes or not product_bytes:
            logger.error(f"Failed to fetch images for {shot_id}")
            return None

        try:
            safe_edit_prompt = edit_prompt + "\n\nEnsure film grain, noise, and lighting match seamlessly across the composite boundary. Do not leave a visible bounding box."

            aspect_ratio = self._detect_aspect_ratio(generated_bytes)

            # Nano Banana call:
            # contents[0] = text prompt
            # contents[1] = generated image to edit (the image being modified)
            # contents[2] = product reference image (guidance for what the product should look like)
            response = self.client.models.generate_content(
                model=self.nano_banana_model,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=safe_edit_prompt),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=generated_bytes,
                                )
                            ),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=product_bytes,
                                )
                            ),
                        ]
                    )
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                    ),
                ),
            )

            edited_bytes = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    edited_bytes = part.inline_data.data
                    break

            if not edited_bytes:
                logger.error(f"No image in Nano Banana response for {shot_id}")
                return None

            # Save to temp file for S3 upload
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            edited_pil = PILImage.open(BytesIO(edited_bytes))
            edited_pil.save(tmp.name)
            tmp.close()

            try:
                safe_shot_id = shot_id.replace(".", "_").replace("/", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_key = f"phase2/product_fix/{safe_shot_id}_fix{iteration}_{timestamp}.png"
                s3_url = self.s3_uploader.upload_image(tmp.name, s3_key)

                if not s3_url:
                    logger.error(f"S3 upload failed for {shot_id}")
                    return None

                logger.info(f"Edited image uploaded for {shot_id}: {s3_url[:80]}...")
                return s3_url

            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Nano Banana edit failed for {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def edit_shots_batch(
        self,
        shots_needing_fix: List[str],
        generated_images: Dict[str, str],
        product_image_url: str,
        product_fix_prompts: Dict[str, str],
        product_review_iterations: Dict[str, int],
        show_id: str,
    ) -> Dict[str, Any]:
        """
        Edit a batch of shots with incorrect product rendering.

        Args:
            shots_needing_fix: Shot IDs that need product correction
            generated_images: Map of shot_id -> latest image URL (before correction)
            product_image_url: Reference product image URL
            product_fix_prompts: Map of shot_id -> Nano Banana prompt (from Agent 17)
            product_review_iterations: Current iteration count per shot
            show_id: Show identifier for S3 paths

        Returns:
            Dict with corrected_images (shot_id -> new S3 URL) and failed_edits list
        """
        logger.info("=" * 60)
        logger.info("AGENT 18: PRODUCT EDITING (NANO BANANA) STARTING")
        logger.info("=" * 60)
        logger.info(f"Shots to edit: {shots_needing_fix}")

        corrected_images = {}
        failed_edits = []

        for shot_id in shots_needing_fix:
            image_url = generated_images.get(shot_id)
            edit_prompt = product_fix_prompts.get(shot_id, "")
            iteration = product_review_iterations.get(shot_id, 0)

            if not image_url:
                logger.warning(f"No image URL for {shot_id}, skipping")
                failed_edits.append({"shot_id": shot_id, "reason": "No image URL"})
                continue

            if not edit_prompt:
                logger.warning(f"No edit prompt for {shot_id}, skipping")
                failed_edits.append({"shot_id": shot_id, "reason": "No edit prompt"})
                continue

            corrected_url = self.edit_shot(
                shot_id=shot_id,
                generated_image_url=image_url,
                product_image_url=product_image_url,
                edit_prompt=edit_prompt,
                show_id=show_id,
                iteration=iteration + 1,
            )

            if corrected_url:
                corrected_images[shot_id] = corrected_url
                logger.info(f"  Corrected: {shot_id}")
            else:
                failed_edits.append({"shot_id": shot_id, "reason": "Nano Banana edit failed"})
                logger.warning(f"  Failed: {shot_id}")

        logger.info("=" * 60)
        logger.info("AGENT 18: PRODUCT EDITING COMPLETE")
        logger.info(f"  Corrected: {len(corrected_images)}")
        logger.info(f"  Failed:    {len(failed_edits)}")
        logger.info("=" * 60)

        return {
            "corrected_images": corrected_images,
            "failed_edits": failed_edits,
        }
