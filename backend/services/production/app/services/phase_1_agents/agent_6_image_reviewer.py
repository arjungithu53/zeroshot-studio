#!/usr/bin/env python3
"""
Agent 6: Image Reviewer
========================
AI critic that assesses generated image quality by comparing images against prompts.
Uses Gemini 2.5 Pro's vision capabilities to review and provide decisions.

Flow:
1. Load generated images from Agent 5
2. Load corresponding prompts from Agent 4
3. Review each image against its prompt using Gemini Vision
4. Output decision: approved, needs_edit, or regenerate
5. Provide detailed feedback for improvements

Decisions:
- approved: Image matches prompt well, ready for variation generation
- needs_edit: Minor issues, can be fixed with image editing
- regenerate: Major issues, needs complete regeneration with modified prompt
"""

from google import genai
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import base64
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# Import Pydantic models from models.py
from backend.services.production.app.services.phase_1_agents.models import ImageReviewResult


class ImageReviewerAgent:
    """
    Agent 6: Reviews generated images against prompts using Gemini Vision

    This agent acts as an AI critic, ensuring generated images meet
    quality standards and match their prompts before proceeding.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-pro-preview"):
        """
        Initialize Image Reviewer Agent

        Args:
            api_key: Google AI API key
            model_name: Gemini model with vision capabilities
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.generated_images = {}
        self.final_prompts = {}
        self.review_results = {}
        self.statistics = {
            "approved": 0,
            "needs_edit": 0,
            "regenerate": 0,
            "total_reviewed": 0
        }

    def load_agent5_output(self, agent5_metadata_path: str) -> None:
        """
        Load generated images metadata from Agent 5

        Args:
            agent5_metadata_path: Path to Agent 5 metadata JSON
        """
        with open(agent5_metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.generated_images = data.get('generated_images', {})

        logger.info(f"✓ Loaded Agent 5 metadata from: {agent5_metadata_path}")
        logger.info(f"   Characters: {len(self.generated_images.get('characters', []))}")
        logger.info(f"   Locations: {len(self.generated_images.get('locations', []))}")
        logger.info(f"   Props: {len(self.generated_images.get('props', []))}")

    def load_agent4_output(self, agent4_output_path: str) -> None:
        """
        Load final prompts from Agent 4

        Args:
            agent4_output_path: Path to Agent 4 JSON output
        """
        with open(agent4_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.final_prompts = data.get('optimized_prompts', {})

        logger.info(f"✓ Loaded Agent 4 prompts from: {agent4_output_path}")

    def _encode_image(self, image_path: str) -> Optional[bytes]:
        """
        Read and encode image for Gemini Vision API
        Supports both local file paths and S3 URLs

        Args:
            image_path: Path to image file or S3 URL

        Returns:
            Image bytes or None if failed
        """
        try:
            # Check if it's an S3 URL
            if image_path.startswith('http://') or image_path.startswith('https://'):
                import requests
                response = requests.get(image_path)
                response.raise_for_status()
                return response.content
            else:
                # Local file path
                with open(image_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"   Error reading image: {e}")
            return None

    def _create_review_prompt(
        self,
        asset_name: str,
        asset_type: str,
        prompt_data: Dict,
        image_index: int = 1
    ) -> str:
        """
        Create review prompt for Gemini Vision

        Args:
            asset_name: Name of the asset
            asset_type: Type (character, location, prop)
            prompt_data: The final prompt data used for generation
            image_index: Which image variation is being reviewed

        Returns:
            Review prompt for Gemini
        """
        final_prompt = prompt_data.get('final_prompt', {})
        prompt_text = final_prompt.get('prompt', '')
        negative_prompt = final_prompt.get('negative_prompt', '')
        tech_specs = final_prompt.get('technical_specs', {})

        review_prompt = f"""
You are an expert AI image quality critic specializing in reviewing AI-generated images for film/video production.
Your job is to assess whether a generated image matches its prompt and meets production quality standards.

**ASSET INFORMATION:**
- Asset Name: {asset_name}
- Asset Type: {asset_type}
- Image Iteration: #{image_index}

**ORIGINAL GENERATION PROMPT:**
{prompt_text}

**NEGATIVE PROMPT (Things to Avoid):**
{negative_prompt}

**TECHNICAL SPECIFICATIONS:**
{json.dumps(tech_specs, indent=2)}

**YOUR TASK:**
Review the provided image and assess its quality against the prompt.

**EVALUATION CRITERIA:**

1. **Prompt Accuracy** (40 points):
   - Does the image accurately represent key elements in the prompt?
   - Are visual details, colors, textures reasonably correct?
   - Missing minor details is OK (35-40 pts), missing critical elements is not (0-20 pts)

2. **Background Compliance** (30 points):
   - FOR CHARACTERS & PROPS:
     - 25–30 pts: Clean neutral/plain background (solid color, simple gradient)
     - 15–24 pts: Subtle texture or soft gradient variation — still acceptable, route to edit not regenerate
     - 5–14 pts: Mild environmental hints (soft blur of an environment faintly visible) — needs_edit
     - 0–4 pts: Clear recognizable environmental scene (park, landscape, room) — regenerate
   - FOR LOCATIONS: Is the environment properly detailed? Full environment expected and desired.

3. **Technical Quality** (20 points):
   - Image sharpness and clarity (minor softness OK, major blur is not)
   - Proper lighting and composition
   - No major AI artifacts (small imperfections acceptable, distorted anatomy is not)
   - 15+ points = production ready

4. **Production Readiness** (10 points):
   - Suitable for I2V compositing and video production
   - Can this be used as-is or with minimal post-processing?
   - 7+ points = ready to proceed

**DECISION CRITERIA:**

- **approved** (70-100 points): Good to excellent quality, matches prompt well, ready for variation generation
- **needs_edit** (50-69 points): Acceptable but has fixable issues that would significantly improve quality
- **regenerate** (0-49 points): Major issues, doesn't match prompt, or has unfixable quality problems

**IMPORTANT NOTES:**
- Be lenient with good images - don't over-criticize minor imperfections
- Score 70+ should be approved unless there are clear quality issues
- Only suggest edits for issues that will meaningfully improve the final result
- Consider production value: if the image works for I2V compositing, it's likely good enough

**OUTPUT FORMAT (JSON):**
{{
    "asset_name": "{asset_name}",
    "asset_type": "{asset_type}",
    "image_index": {image_index},
    "decision": "approved|needs_edit|regenerate",
    "overall_score": 85,
    "scores": {{
        "prompt_accuracy": 35,
        "background_compliance": 28,
        "technical_quality": 15,
        "production_readiness": 7
    }},
    "assessment": {{
        "strengths": [
            "What works well in this image"
        ],
        "issues": [
            "Problems or concerns identified"
        ],
        "missing_elements": [
            "Elements from prompt that are missing or incorrect"
        ],
        "ai_artifacts": [
            "Any AI generation artifacts detected"
        ]
    }},
    "feedback": {{
        "for_edit": "If needs_edit: Specific edits required (color, crop, etc.)",
        "for_regeneration": "If regenerate: How to modify the prompt for better results",
        "general_notes": "Additional observations"
    }},
    "production_notes": {{
        "compositing_ready": true/false,
        "concerns": ["Any production workflow concerns"],
        "recommendations": ["Suggestions for improvement"]
    }}
}}

**CRITICAL REMINDERS:**
- FOR CHARACTER ASSETS: Prefer neutral backgrounds. Score using the graduated scale above. Only score 0–4 (regenerate territory) if a CLEAR recognizable environmental scene is present. Subtle textures, gradients, or soft out-of-focus backgrounds should score 15–24 (edit, not regenerate).
- FOR PROP ASSETS: No characters, animals, or hands. Prefer neutral background. Apply the same graduated scale — only regenerate if a clear environmental scene is present, not for subtle textures.
- FOR LOCATION ASSETS: Environment and atmosphere are expected and desired
- Be strict but fair - production quality is essential
- Consider the asset's role in video compositing

Please analyze the image and provide your review in JSON format.
"""
        return review_prompt

    def review_image(
        self,
        image_path: str,
        asset_name: str,
        asset_type: str,
        prompt_data: Dict,
        image_index: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Review a single image using Gemini Vision

        Args:
            image_path: Path to the image file
            asset_name: Name of the asset
            asset_type: Type of asset
            prompt_data: Prompt used for generation
            image_index: Image variation number

        Returns:
            Review result dictionary or None if failed
        """
        logger.info(f"\n   Reviewing: {asset_name} - Image #{image_index}")

        # Load image
        image_data = self._encode_image(image_path)
        if not image_data:
            return None

        # Create review prompt
        review_prompt = self._create_review_prompt(
            asset_name, asset_type, prompt_data, image_index
        )

        try:
            # Prepare content with text and image inline part
            from google.genai import types

            content_parts = [
                types.Part.from_text(text=review_prompt),
                types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/png"
                )
            ]

            # Use structured output with Pydantic schema
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ImageReviewResult,
                }
            )

            # Get the parsed object directly
            parsed_review: ImageReviewResult = response.parsed

            # Convert to dictionary and add metadata
            review_result = parsed_review.model_dump()
            review_result['image_path'] = image_path
            review_result['review_timestamp'] = datetime.now().isoformat()

            decision = parsed_review.decision
            score = parsed_review.overall_score

            logger.info(f"   ✓ Decision: {decision.upper()} (Score: {score}/100)")

            return review_result

        except Exception as e:
            logger.error(f"   Review failed: {e}")
            return None

    def review_all_images(self, assets_to_review: Optional[List[str]] = None, previous_reviews: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Review generated images from Agent 5 with structured output

        Args:
            assets_to_review: Optional list of asset IDs to review. If None, reviews all assets.
                             If provided, only reviews assets with IDs in this list.
                             Format: List of asset UUIDs
            previous_reviews: Optional previous review results to preserve for assets not being re-reviewed.
                            If provided with assets_to_review, previous results will be kept for skipped assets.

        Returns:
            Dictionary containing all review results
        """
        logger.info("\n" + "="*60)
        logger.info("AGENT 6: IMAGE REVIEW STARTING (Structured Output)")
        logger.info("="*60)

        if not self.generated_images:
            raise ValueError("No images loaded. Call load_agent5_output() first.")

        if not self.final_prompts:
            raise ValueError("No prompts loaded. Call load_agent4_output() first.")

        # If assets_to_review is provided, only review those specific assets
        if assets_to_review:
            logger.info(f"\nSelective Review Mode: Only reviewing {len(assets_to_review)} specific asset(s)")
            logger.info(f"   Asset IDs: {assets_to_review}")
        else:
            logger.info("\n📋 Full Review Mode: Reviewing all assets")

        # Start with previous results if doing selective review, otherwise start fresh
        if assets_to_review and previous_reviews:
            import copy
            self.review_results = copy.deepcopy(previous_reviews)
            logger.info(f"   Preserving previous review results for non-reviewed assets")
        else:
            self.review_results = {
                "characters": [],
                "locations": [],
                "props": []
            }

        # Review character images
        logger.info("\n🎭 Reviewing character images...")
        for char_data in self.generated_images.get('characters', []):
            char_name = char_data.get('name', 'Unknown')
            char_id = char_data.get('id')

            # Skip if selective review and this asset is not in the list
            if assets_to_review and char_id not in assets_to_review:
                logger.info(f"\n⏭Skipping {char_name} (not in review list)")
                continue

            logger.info(f"\n📸 Character: {char_name}")

            images = char_data.get('images', [])
            # Find prompt data by UUID or name
            prompt_data = None
            for p_data in self.final_prompts.get('characters', []):
                if p_data.get('id') == char_id or p_data.get('name') == char_name:
                    prompt_data = p_data
                    break

            if not prompt_data:
                prompt_data = {}

            if not images:
                logger.warning(f"   No images found")
                continue

            char_reviews = []
            for img_data in images:
                # Support both S3 URLs and local paths
                img_path = img_data.get('s3_url') or img_data.get('url') or img_data.get('local_path')
                img_index = img_data.get('index', 1)

                if not img_path:
                    logger.warning(f"   Image path/URL not found in data")
                    continue

                # Skip file existence check for URLs
                if not (img_path.startswith('http://') or img_path.startswith('https://')) and not os.path.exists(img_path):
                    logger.warning(f"   Image not found: {img_path}")
                    continue

                review = self.review_image(
                    img_path, char_name, 'character', prompt_data, img_index
                )

                if review:
                    char_reviews.append(review)
                    decision = review.get('decision')
                    self.statistics[decision] = self.statistics.get(decision, 0) + 1
                    self.statistics['total_reviewed'] += 1

            # Update existing entry if present (selective review), otherwise append new
            existing_entry = None
            for i, entry in enumerate(self.review_results['characters']):
                if entry.get('id') == char_id:
                    existing_entry = i
                    break

            new_entry = {
                'id': char_id,
                'name': char_name,
                'reviews': char_reviews
            }

            if existing_entry is not None:
                self.review_results['characters'][existing_entry] = new_entry
            else:
                self.review_results['characters'].append(new_entry)

        # Review location images
        logger.info("\n🗺Reviewing location images...")
        for loc_data in self.generated_images.get('locations', []):
            loc_name = loc_data.get('name', 'Unknown')
            loc_id = loc_data.get('id')

            # Skip if selective review and this asset is not in the list
            if assets_to_review and loc_id not in assets_to_review:
                logger.info(f"\n⏭Skipping {loc_name} (not in review list)")
                continue

            logger.info(f"\n📸 Location: {loc_name}")

            images = loc_data.get('images', [])
            # Find prompt data by UUID or name
            prompt_data = None
            for p_data in self.final_prompts.get('locations', []):
                if p_data.get('id') == loc_id or p_data.get('name') == loc_name:
                    prompt_data = p_data
                    break

            if not prompt_data:
                prompt_data = {}

            if not images:
                logger.warning(f"   No images found")
                continue

            loc_reviews = []
            for img_data in images:
                # Support both S3 URLs and local paths
                img_path = img_data.get('s3_url') or img_data.get('url') or img_data.get('local_path')
                img_index = img_data.get('index', 1)

                if not img_path:
                    logger.warning(f"   Image path/URL not found in data")
                    continue

                # Skip file existence check for URLs
                if not (img_path.startswith('http://') or img_path.startswith('https://')) and not os.path.exists(img_path):
                    logger.warning(f"   Image not found: {img_path}")
                    continue

                review = self.review_image(
                    img_path, loc_name, 'location', prompt_data, img_index
                )

                if review:
                    loc_reviews.append(review)
                    decision = review.get('decision')
                    self.statistics[decision] = self.statistics.get(decision, 0) + 1
                    self.statistics['total_reviewed'] += 1

            # Update existing entry if present (selective review), otherwise append new
            existing_entry = None
            for i, entry in enumerate(self.review_results['locations']):
                if entry.get('id') == loc_id:
                    existing_entry = i
                    break

            new_entry = {
                'id': loc_id,
                'name': loc_name,
                'reviews': loc_reviews
            }

            if existing_entry is not None:
                self.review_results['locations'][existing_entry] = new_entry
            else:
                self.review_results['locations'].append(new_entry)

        # Review prop images
        logger.info("\nReviewing prop images...")
        for prop_data in self.generated_images.get('props', []):
            prop_name = prop_data.get('name', 'Unknown')
            prop_id = prop_data.get('id')
            is_product_prop = prop_data.get('is_product', False)

            # Skip if selective review and this asset is not in the list
            if assets_to_review and prop_id not in assets_to_review:
                logger.info(f"\n⏭Skipping {prop_name} (not in review list)")
                continue

            logger.info(f"\n📸 Prop: {prop_name}" + (" [UPLOADED PRODUCT — restricted review]" if is_product_prop else ""))

            images = prop_data.get('images', [])
            # Find prompt data by UUID or name
            prompt_data = None
            for p_data in self.final_prompts.get('props', []):
                if p_data.get('id') == prop_id or p_data.get('name') == prop_name:
                    prompt_data = p_data
                    break

            if not prompt_data:
                prompt_data = {}

            if not images:
                logger.warning(f"   No images found")
                continue

            prop_reviews = []
            for img_data in images:
                # Support both S3 URLs and local paths
                img_path = img_data.get('s3_url') or img_data.get('url') or img_data.get('local_path')
                img_index = img_data.get('index', 1)

                if not img_path:
                    logger.warning(f"   Image path/URL not found in data")
                    continue

                # Skip file existence check for URLs
                if not (img_path.startswith('http://') or img_path.startswith('https://')) and not os.path.exists(img_path):
                    logger.warning(f"   Image not found: {img_path}")
                    continue

                review = self.review_image(
                    img_path, prop_name, 'prop', prompt_data, img_index
                )

                if review:
                    # PRODUCT FIDELITY PROTECTION:
                    # The uploaded product image must never be regenerated.
                    # Only visibility/presentation issues allow 'needs_edit'.
                    # Shape, size, text, logo, and color are fixed — do not evaluate them.
                    if is_product_prop and review.get('decision') == 'regenerate':
                        review['decision'] = 'approved'
                        review['notes'] = (
                            "Product image auto-approved: regeneration is not permitted for "
                            "uploaded product images. Shape, size, text, and branding are fixed."
                        )
                        logger.info("   ⚠️ Product image 'regenerate' overridden → 'approved' (fidelity lock)")

                    prop_reviews.append(review)
                    decision = review.get('decision')
                    self.statistics[decision] = self.statistics.get(decision, 0) + 1
                    self.statistics['total_reviewed'] += 1

            # Update existing entry if present (selective review), otherwise append new
            existing_entry = None
            for i, entry in enumerate(self.review_results['props']):
                if entry.get('id') == prop_id:
                    existing_entry = i
                    break

            new_entry = {
                'id': prop_id,
                'name': prop_name,
                'reviews': prop_reviews
            }

            if existing_entry is not None:
                self.review_results['props'][existing_entry] = new_entry
            else:
                self.review_results['props'].append(new_entry)

        logger.info("\n✓ Image review completed!")
        self._print_review_summary()

        return self.review_results

    def _print_review_summary(self) -> None:
        """Print summary of review results"""

        logger.info("\n" + "─"*60)
        logger.info("REVIEW SUMMARY")
        logger.info("─"*60)

        logger.info(f"\nTotal Images Reviewed: {self.statistics['total_reviewed']}")
        logger.info(f"Approved: {self.statistics.get('approved', 0)}")
        logger.info(f"✏Needs Edit: {self.statistics.get('needs_edit', 0)}")
        logger.info(f"Regenerate: {self.statistics.get('regenerate', 0)}")

        # Detailed breakdown
        logger.info("\n" + "─"*60)
        logger.info("DETAILED RESULTS")
        logger.info("─"*60)

        for asset_type in ['characters', 'locations', 'props']:
            asset_list = self.review_results.get(asset_type, [])
            if asset_list:
                logger.info(f"\n{asset_type.upper()}:")
                for asset_data in asset_list:
                    asset_name = asset_data.get('name', 'Unknown')
                    for review in asset_data.get('reviews', []):
                        decision = review.get('decision', 'unknown')
                        score = review.get('overall_score', 0)
                        emoji = "✅" if decision == "approved" else "✏️" if decision == "needs_edit" else "🔄"
                        logger.info(f"  {emoji} {asset_name} (#{review.get('image_index', 1)}): {decision.upper()} - {score}/100")

    def get_assets_by_decision(self, decision: str) -> Dict[str, List[Dict]]:
        """
        Get all assets with a specific decision

        Args:
            decision: 'approved', 'needs_edit', or 'regenerate'

        Returns:
            Dictionary of assets filtered by decision
        """
        filtered = {
            "characters": [],
            "locations": [],
            "props": []
        }

        for asset_type in ['characters', 'locations', 'props']:
            for asset_data in self.review_results.get(asset_type, []):
                matching_reviews = [r for r in asset_data.get('reviews', []) if r.get('decision') == decision]
                if matching_reviews:
                    filtered[asset_type].append({
                        'id': asset_data.get('id'),
                        'name': asset_data.get('name'),
                        'reviews': matching_reviews
                    })

        return filtered

    def generate_edit_instructions(self) -> Dict[str, Any]:
        """
        Generate edit instructions for assets that need editing

        Returns:
            Dictionary with edit instructions for each asset
        """
        needs_edit = self.get_assets_by_decision('needs_edit')
        edit_instructions = {}

        for asset_type in ['characters', 'locations', 'props']:
            for asset_data in needs_edit.get(asset_type, []):
                asset_name = asset_data.get('name', 'Unknown')
                asset_id = asset_data.get('id')
                for review in asset_data.get('reviews', []):
                    key = f"{asset_name}_#{review.get('image_index', 1)}"
                    edit_instructions[key] = {
                        'asset_id': asset_id,
                        'asset_name': asset_name,
                        'asset_type': asset_type,
                        'image_path': review.get('image_path'),
                        'edit_instructions': review.get('feedback', {}).get('for_edit'),
                        'issues': review.get('assessment', {}).get('issues', [])
                    }

        return edit_instructions

    def generate_regeneration_prompts(self) -> Dict[str, Any]:
        """
        Generate modified prompts for assets that need regeneration

        Returns:
            Dictionary with regeneration guidance for each asset
        """
        regenerate = self.get_assets_by_decision('regenerate')
        regen_prompts = {}

        for asset_type in ['characters', 'locations', 'props']:
            for asset_data in regenerate.get(asset_type, []):
                asset_name = asset_data.get('name', 'Unknown')
                asset_id = asset_data.get('id')
                for review in asset_data.get('reviews', []):
                    key = f"{asset_name}_#{review.get('image_index', 1)}"
                    regen_prompts[key] = {
                        'asset_id': asset_id,
                        'asset_name': asset_name,
                        'asset_type': asset_type,
                        'original_image_path': review.get('image_path'),
                        'regeneration_guidance': review.get('feedback', {}).get('for_regeneration'),
                        'missing_elements': review.get('assessment', {}).get('missing_elements', []),
                        'issues': review.get('assessment', {}).get('issues', []),
                        'score': review.get('overall_score', 0)
                    }

        return regen_prompts

    def rewrite_prompts_for_regeneration(self, optimized_prompts: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rewrite prompts using Gemini based on regeneration feedback

        Args:
            optimized_prompts: Current optimized prompts from Agent 4

        Returns:
            Modified prompts dictionary with rewritten prompts for regenerate assets
        """
        import copy

        logger.info("\n" + "="*60)
        logger.info("REWRITING PROMPTS FOR REGENERATION")
        logger.info("="*60)

        # Get regeneration data
        regeneration_data = self.generate_regeneration_prompts()

        if not regeneration_data:
            logger.info("   ℹNo assets need regeneration")
            return optimized_prompts

        modified_prompts = copy.deepcopy(optimized_prompts)

        for asset_key, guidance in regeneration_data.items():
            asset_id = guidance.get('asset_id')
            asset_name = guidance.get('asset_name')
            asset_type = guidance.get('asset_type')

            logger.info(f"\n🔧 Rewriting prompt for: {asset_name} ({asset_type})")

            # Find the asset in prompts (asset_type is already plural: "characters", "locations", "props")
            if asset_type not in modified_prompts:
                logger.warning(f"   Asset type {asset_type} not found in prompts")
                continue

            for asset in modified_prompts[asset_type]:
                if asset.get('id') == asset_id:
                    # Get original prompt data
                    final_prompt = asset.get('final_prompt', {})
                    original_prompt = final_prompt.get('prompt', '')

                    # Extract feedback
                    issues = guidance.get('issues', [])
                    missing_elements = guidance.get('missing_elements', [])
                    regeneration_guidance_text = guidance.get('regeneration_guidance', '')

                    # Create rewrite prompt for Gemini
                    rewrite_prompt = f"""You are an expert prompt engineer for AI image generation.

Your task is to rewrite an image generation prompt to fix the issues identified by an AI image reviewer.

ORIGINAL PROMPT:
{original_prompt}

ISSUES FOUND:
{chr(10).join(f"- {issue}" for issue in issues) if issues else "None"}

MISSING ELEMENTS:
{chr(10).join(f"- {elem}" for elem in missing_elements) if missing_elements else "None"}

REGENERATION GUIDANCE:
{regeneration_guidance_text}

INSTRUCTIONS:
1. Keep the core concept and style of the original prompt
2. Address ALL issues and missing elements
3. Make the prompt more specific and detailed where issues were found
4. Maintain the same format and structure
5. Do NOT add extra commentary or explanations

Return ONLY the rewritten prompt text, nothing else."""

                    try:
                        response = self.client.models.generate_content(
                            model="gemini-3.1-pro-preview",
                            contents=rewrite_prompt
                        )

                        new_prompt = response.text.strip()

                        # Update the prompt
                        asset['final_prompt']['prompt'] = new_prompt

                        logger.info(f"   Prompt rewritten successfully")
                        logger.info(f"   Original: {original_prompt[:80]}...")
                        logger.info(f"   New: {new_prompt[:80]}...")

                    except Exception as e:
                        logger.error(f"   Failed to rewrite prompt: {e}")
                        # Keep original prompt if rewrite fails

                    break

        return modified_prompts

    def run_full_pipeline(
        self,
        agent5_metadata_path: str,
        agent4_output_path: str
    ) -> Dict[str, Any]:
        """
        Run the complete Agent 6 pipeline

        Args:
            agent5_metadata_path: Path to Agent 5 metadata JSON
            agent4_output_path: Path to Agent 4 output JSON

        Returns:
            Dictionary with review results
        """
        # Step 1: Load Agent 5 metadata
        self.load_agent5_output(agent5_metadata_path)

        # Step 2: Load Agent 4 prompts
        self.load_agent4_output(agent4_output_path)

        # Step 3: Review all images
        results = self.review_all_images()

        return {
            "status": "completed",
            "review_results": results,
            "statistics": self.statistics,
            "next_steps": {
                "approved_count": self.statistics.get('approved', 0),
                "needs_edit_count": self.statistics.get('needs_edit', 0),
                "regenerate_count": self.statistics.get('regenerate', 0)
            }
        }


def main():
    """Example usage of Agent 6"""
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)

    # Initialize agent
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("ERROR: GEMINI_API_KEY not found in environment")
        return

    agent = ImageReviewerAgent(api_key=api_key)

    logger.info("Agent 6: Image Reviewer initialized")
    logger.info("Use run_full_pipeline() with Agent 5 and Agent 4 output paths")


if __name__ == "__main__":
    main()
