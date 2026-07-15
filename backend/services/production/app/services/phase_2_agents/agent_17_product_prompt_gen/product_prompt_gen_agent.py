#!/usr/bin/env python3
"""
Agent 17: Product Position & Replacement Prompt Generator (Phase 2)
====================================================================
Analyzes failed product shots and generates precise Nano Banana editing
prompts that describe where the incorrect product appears and how to
replace it with the correct reference product.

Flow:
1. For each shot that failed Agent 16 review:
   - Download the latest generated/edited image
   - Download the product reference image
   - Retrieve Agent 16's specific issues for the shot
   - Call Gemini vision to produce a Nano Banana replacement prompt
2. Store the prompt per shot for Agent 18 to use
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from google import genai
from google.genai import types
from app.services.phase_2_agents.helpers.image_fetch import fetch_image_bytes

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"


class ProductPromptGenAgent:
    """
    Agent 17: Generates Nano Banana editing prompts for product replacement.

    Uses Gemini vision to identify incorrect product positions in the generated
    image and produce a targeted prompt for Agent 18 (Nano Banana editor).
    """

    def __init__(self, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        logger.info("=" * 60)
        logger.info("AGENT 17: PRODUCT PROMPT GENERATOR INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info("=" * 60)

    def _fetch_image_bytes(self, url_or_path: str) -> Optional[bytes]:
        """Fetch raw image bytes from a URL or local path."""
        return fetch_image_bytes(url_or_path)

    def _build_prompt(self, shot_id: str, issues: List[Dict[str, str]]) -> str:
        issues_text = ""
        if issues:
            issue_lines = "\n".join(
                f"  - {i.get('aspect', 'unknown').upper()}: {i.get('description', '')}"
                for i in issues
            )
            issues_text = f"""
ISSUES IDENTIFIED BY PRODUCT REVIEWER (use these to make your prompt more precise):
{issue_lines}
"""

        return f"""You are an expert AI image editing prompt engineer for Nano Banana (gemini-3.1-flash-image-preview).

You have been provided with TWO images:
1. REFERENCE PRODUCT IMAGE — the correct product that MUST appear in the final image
2. GENERATED SCENE IMAGE — the current image containing an incorrect version of the product
{issues_text}
YOUR TASK:
Generate a single, precise Nano Banana editing prompt that will replace the incorrect product(s)
in the GENERATED SCENE IMAGE with the correct product from the REFERENCE PRODUCT IMAGE.

STEP 1 — IDENTIFY ALL PRODUCT INSTANCES:
Look at the generated scene image carefully. The product may appear once or multiple times.
For each instance, note its exact position (e.g., "center-left", "lower-right quadrant",
"behind the character on the left", etc.).

STEP 2 — WRITE THE EDITING PROMPT:

Your prompt must:

A. DESCRIBE THE PROBLEM (briefly):
   - For each product instance: state its current position and what is wrong
     (shape distorted, text wrong, proportions off, logo incorrect, etc.)
   - If product appears more than once, address each instance separately

B. INSTRUCT THE REPLACEMENT:
   - Replace [description of position] of the incorrectly rendered product with the
     reference product image — matching its exact shape, proportions, and all text/logos
   - If multiple instances: "Replace the product in [position 1] and [position 2]..."

C. ENFORCE CRITICAL CONSTRAINTS — these MUST appear in every prompt:
   - "The replacement product MUST match the reference image EXACTLY in shape and proportions"
   - "All text, numbers, and logos on the product MUST be reproduced character-for-character"
   - "Do NOT alter the product's size relative to the scene — keep the same scale"

D. PRESERVE CLAUSE — MANDATORY:
   - State explicitly what must NOT change: background, other scene elements,
     characters, lighting, shadows, and anything not part of the product itself
   - Formula: "Preserve [everything not being replaced] completely unchanged"

PROMPT STYLE GUIDELINES:
- Write as a direct command (starts with action verb: Replace, Correct, Fix)
- Be specific about positions (use visual landmarks: "the bottle in the lower-left",
  "behind the character's left hand", "on the table in the foreground")
- Positive framing: describe what you WANT, not what you don't want
- Length: 60-120 words — concise but complete
- Do NOT use bullet points — write as flowing prose

OUTPUT:
Output ONLY the Nano Banana editing prompt as plain text. No JSON, no labels, no explanations.
Just the prompt itself.
"""

    def generate_prompt_for_shot(
        self,
        shot_id: str,
        generated_image_url: str,
        product_image_url: str,
        issues: List[Dict[str, str]],
    ) -> str:
        """
        Generate a Nano Banana replacement prompt for a single shot.

        Args:
            shot_id: Shot identifier
            generated_image_url: URL of the current generated/edited image
            product_image_url: URL of the reference product image
            issues: List of ProductIssue dicts from Agent 16

        Returns:
            Nano Banana editing prompt string
        """
        logger.info(f"Generating product replacement prompt for shot: {shot_id}")

        _fallback = (
            "Replace the incorrectly rendered product in this image with the reference product image. "
            "The replacement product MUST match the reference image EXACTLY in shape and proportions. "
            "All text, numbers, and logos on the product MUST be reproduced character-for-character. "
            "Do NOT alter the product's size relative to the scene. "
            "Preserve all other scene elements, background, characters, lighting, and shadows completely unchanged."
        )

        product_bytes = self._fetch_image_bytes(product_image_url)
        generated_bytes = self._fetch_image_bytes(generated_image_url)

        if not product_bytes or not generated_bytes:
            logger.warning(f"Download failed for {shot_id}, using fallback prompt")
            return _fallback

        try:
            prompt_text = self._build_prompt(shot_id, issues)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part(text=prompt_text),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=product_bytes,
                                )
                            ),
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/png",
                                    data=generated_bytes,
                                )
                            ),
                        ]
                    )
                ],
            )

            edit_prompt = response.text.strip()
            logger.info(
                f"Generated prompt for {shot_id} ({len(edit_prompt)} chars): {edit_prompt[:100]}..."
            )
            return edit_prompt

        except Exception as e:
            logger.error(f"Prompt generation failed for {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            return _fallback

    def generate_prompts_batch(
        self,
        shots_needing_fix: List[str],
        generated_images: Dict[str, str],
        product_image_url: str,
        product_review_results: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Generate replacement prompts for a batch of failing shots.

        Args:
            shots_needing_fix: Shot IDs that need product correction
            generated_images: Map of shot_id -> latest image URL
            product_image_url: Reference product image URL
            product_review_results: Agent 16 review results per shot

        Returns:
            Dict of shot_id -> Nano Banana prompt string
        """
        logger.info("=" * 60)
        logger.info("AGENT 17: PRODUCT PROMPT GENERATION STARTING")
        logger.info("=" * 60)
        logger.info(f"Shots to process: {shots_needing_fix}")

        prompts = {}

        for shot_id in shots_needing_fix:
            image_url = generated_images.get(shot_id)
            if not image_url:
                logger.warning(f"No image URL for {shot_id}, using generic fallback prompt")
                prompts[shot_id] = (
                    "Replace the incorrectly rendered product with the reference product image, "
                    "matching shape, proportions, and all text/logos exactly. "
                    "Preserve all other scene elements unchanged."
                )
                continue

            review = product_review_results.get(shot_id, {})
            issues = review.get("issues", [])

            prompts[shot_id] = self.generate_prompt_for_shot(
                shot_id=shot_id,
                generated_image_url=image_url,
                product_image_url=product_image_url,
                issues=issues,
            )

        logger.info("=" * 60)
        logger.info(f"AGENT 17: Generated prompts for {len(prompts)} shots")
        logger.info("=" * 60)

        return prompts
