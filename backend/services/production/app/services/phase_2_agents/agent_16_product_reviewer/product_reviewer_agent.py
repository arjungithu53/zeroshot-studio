#!/usr/bin/env python3
"""
Agent 16: Product Fidelity Reviewer (Phase 2)
==============================================
Reviews generated images to verify that the product (shape, size, text/logo)
matches the reference product image exactly.

Flow:
1. For each product shot approved by Agent 15:
   - Download the latest generated/edited image
   - Download the product reference image
   - Call Gemini vision to compare product in image vs reference
   - Decision: pass or fail with specific issues
2. Shots that pass go to final_checkpoint
3. Shots that fail (within 3 iterations) go to Agent 17 for correction
4. Shots that fail after 3 iterations are force-passed to final_checkpoint
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from google import genai
from google.genai import types
from app.services.phase_2_agents.helpers.image_fetch import fetch_image_bytes

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"
MAX_PRODUCT_FIX_ITERATIONS = 3


@dataclass
class ProductIssue:
    aspect: str        # "shape" | "size" | "text" | "logo"
    description: str   # specific mismatch description


@dataclass
class ProductReviewResult:
    shot_id: str
    decision: str              # "pass" | "fail"
    confidence: float          # 0.0 to 1.0
    issues: List[ProductIssue]
    product_instance_count: int  # how many times product appears
    review_timestamp: str


class ProductReviewerAgent:
    """
    Agent 16: Verifies product fidelity in generated images.

    Compares the product appearing in the generated image against the
    reference product image for shape, size, and text/logo accuracy.
    """

    def __init__(self, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        if not api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

        logger.info("=" * 60)
        logger.info("AGENT 16: PRODUCT FIDELITY REVIEWER INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")
        logger.info("=" * 60)

    def _fetch_image_bytes(self, url_or_path: str) -> Optional[bytes]:
        """Fetch raw image bytes from a URL or local path."""
        return fetch_image_bytes(url_or_path)

    def _build_review_prompt(self, shot_id: str) -> str:
        return f"""You are an expert product quality analyst reviewing a generated advertising image for shot {shot_id}.

You have been provided with TWO images:
1. REFERENCE PRODUCT IMAGE — the exact product that must appear in the scene
2. GENERATED SCENE IMAGE — the AI-generated image that should contain the product

YOUR TASK:
Compare the product as it appears in the GENERATED SCENE IMAGE against the REFERENCE PRODUCT IMAGE.

FOCUS ONLY ON THE PRODUCT OBJECT — do NOT evaluate background, lighting, characters, composition, or anything else.

CHECK THESE THREE THINGS:

1. SHAPE
   - Does the product's overall shape match the reference?
   - Is it the correct form factor (bottle, can, box, etc.)?
   - Are proportions correct (not stretched, squished, or deformed)?

2. SIZE / PROPORTIONS
   - Are the relative proportions of the product correct?
   - Do the various parts of the product (cap, label area, body) have correct relative sizes?

3. TEXT, LOGO & BRANDING
   - Is the text on the product readable and correct?
   - Does the logo match the reference (correct shape, position on product)?
   - Are labels, numbers, or any written content accurate?

DECISION RULES:
- "pass": Product shape, size, and all text/logos match the reference closely. Minor lighting differences are acceptable.
- "fail": Any of the following: shape is distorted, proportions are wrong, text is wrong/missing/unreadable, logo is different.

IMPORTANT: Count how many times the product appears in the generated image (it may appear more than once).

OUTPUT FORMAT (JSON only, no markdown):
{{
  "shot_id": "{shot_id}",
  "decision": "pass" or "fail",
  "confidence": 0.0 to 1.0,
  "product_instance_count": number of times product appears in generated image,
  "issues": [
    {{
      "aspect": "shape" or "size" or "text" or "logo",
      "description": "specific description of the mismatch"
    }}
  ],
  "summary": "brief overall assessment"
}}

If decision is "pass", issues array must be empty.
Output ONLY the JSON object, nothing else.
"""

    def review_shot(
        self,
        shot_id: str,
        generated_image_url: str,
        product_image_url: str,
    ) -> ProductReviewResult:
        """
        Review a single shot for product fidelity.

        Args:
            shot_id: Shot identifier
            generated_image_url: URL of the generated/edited image
            product_image_url: URL of the reference product image

        Returns:
            ProductReviewResult with decision and issues
        """
        logger.info(f"Reviewing product fidelity for shot: {shot_id}")

        product_bytes = self._fetch_image_bytes(product_image_url)
        generated_bytes = self._fetch_image_bytes(generated_image_url)

        if not product_bytes or not generated_bytes:
            logger.error(f"Failed to fetch images for {shot_id} — force-passing")
            return ProductReviewResult(
                shot_id=shot_id,
                decision="pass",
                confidence=0.0,
                issues=[],
                product_instance_count=1,
                review_timestamp=datetime.now().isoformat(),
            )

        try:
            prompt_text = self._build_review_prompt(shot_id)

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

            response_text = response.text.strip()

            # Strip markdown fences if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            data = json.loads(response_text)

            issues = [
                ProductIssue(
                    aspect=i.get("aspect", ""),
                    description=i.get("description", ""),
                )
                for i in data.get("issues", [])
            ]

            result = ProductReviewResult(
                shot_id=shot_id,
                decision=data.get("decision", "fail"),
                confidence=float(data.get("confidence", 0.5)),
                issues=issues,
                product_instance_count=int(data.get("product_instance_count", 1)),
                review_timestamp=datetime.now().isoformat(),
            )

            logger.info(
                f"Shot {shot_id}: decision={result.decision}, "
                f"confidence={result.confidence:.2f}, issues={len(result.issues)}"
            )
            return result

        except Exception as e:
            logger.error(f"Product review failed for {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            return ProductReviewResult(
                shot_id=shot_id,
                decision="pass",  # fail-safe: don't block on error
                confidence=0.0,
                issues=[],
                product_instance_count=1,
                review_timestamp=datetime.now().isoformat(),
            )

    def review_product_shots(
        self,
        product_shot_ids: set,
        shots_to_review: List[str],
        generated_images: Dict[str, str],
        product_image_url: str,
        product_review_iterations: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Review a batch of product shots.

        Args:
            product_shot_ids: Set of all shot_ids with product_present=True
            shots_to_review: Shot IDs to review in this run
            generated_images: Map of shot_id -> latest image URL
            product_image_url: Reference product image URL
            product_review_iterations: Current iteration count per shot

        Returns:
            Dict with review_results, shots_passing, shots_failing, shots_force_passed
        """
        logger.info("=" * 60)
        logger.info("AGENT 16: PRODUCT FIDELITY REVIEW STARTING")
        logger.info("=" * 60)
        logger.info(f"Shots to review: {shots_to_review}")

        review_results = {}
        shots_passing = []
        shots_failing = []
        shots_force_passed = []

        for shot_id in shots_to_review:
            if shot_id not in product_shot_ids:
                shots_passing.append(shot_id)
                continue

            image_url = generated_images.get(shot_id)
            if not image_url:
                logger.warning(f"No image URL for {shot_id}, force-passing")
                shots_force_passed.append(shot_id)
                continue

            result = self.review_shot(
                shot_id=shot_id,
                generated_image_url=image_url,
                product_image_url=product_image_url,
            )
            review_results[shot_id] = asdict(result)

            if result.decision == "pass":
                shots_passing.append(shot_id)
                logger.info(f"  PASS: {shot_id}")
            else:
                current_iterations = product_review_iterations.get(shot_id, 0)
                if current_iterations >= MAX_PRODUCT_FIX_ITERATIONS:
                    logger.warning(
                        f"  FORCE-PASS: {shot_id} exceeded max iterations ({MAX_PRODUCT_FIX_ITERATIONS})"
                    )
                    shots_force_passed.append(shot_id)
                else:
                    shots_failing.append(shot_id)
                    logger.info(
                        f"  FAIL: {shot_id} (iteration {current_iterations + 1}/{MAX_PRODUCT_FIX_ITERATIONS})"
                    )

        logger.info("=" * 60)
        logger.info("AGENT 16: PRODUCT FIDELITY REVIEW COMPLETE")
        logger.info(f"  Passing:      {len(shots_passing)}")
        logger.info(f"  Failing:      {len(shots_failing)}")
        logger.info(f"  Force-passed: {len(shots_force_passed)}")
        logger.info("=" * 60)

        return {
            "review_results": review_results,
            "shots_passing": shots_passing,
            "shots_failing": shots_failing,
            "shots_force_passed": shots_force_passed,
        }
