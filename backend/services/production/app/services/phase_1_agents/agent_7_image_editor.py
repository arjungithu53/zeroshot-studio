#!/usr/bin/env python3
"""
Agent 7: Image Edit Agent
==========================
Updates prompts for images that need editing and applies edits using Freepik SeeDream 4 Edit API.
Uses Gemini 2.5 Pro to generate targeted edit prompts based on Agent 6 feedback.

Flow:
1. Load review results from Agent 6
2. For each image marked as "needs_edit"
3. Generate targeted edit prompt using Gemini
4. Apply edits using SeeDream 4 Edit API
5. Save edited images locally
6. Track edit metadata
"""

import google.generativeai as genai
from google import genai as genai_new
from google.genai import types as genai_types
import requests
import json
import os
import time
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from io import BytesIO
import sys
from PIL import Image as PILImage

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)



class ImageEditAgent:
    """
    Agent 7: Edits images using AI based on Agent 6 feedback

    This agent takes images that need minor corrections and applies
    targeted edits using SeeDream 4 Edit API with AI-generated prompts.
    """

    def __init__(self, gemini_api_key: str, model_name: str = "gemini-3.1-pro-preview"):
        """
        Initialize Image Edit Agent using Nano Banana (Gemini Flash Image).

        Args:
            gemini_api_key: Google AI API key for prompt generation and image editing
        
            model_name: Gemini model to use for prompt generation
        """
        # Prompt generation
        genai.configure(api_key=gemini_api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)

        # Image editing via Nano Banana
        self.gemini_client = genai_new.Client(api_key=gemini_api_key)
        self.gemini_image_model = "gemini-3.1-flash-image-preview"
        logger.info("✓ Agent 7 initialized with Nano Banana (Gemini Flash Image) for image editing")

        self.review_results = {}
        self.edit_instructions = {}
        self.edit_prompts = {}
        self.edited_images = {}
        self.failed_edits = []

    def load_agent6_output(self, agent6_output_path: str) -> None:
        """
        Load review results from Agent 6

        Args:
            agent6_output_path: Path to Agent 6 review results JSON
        """
        with open(agent6_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.review_results = data.get('review_results', {})
        self.edit_instructions = data.get('edit_instructions', {})

        needs_edit_count = len(self.edit_instructions)

        logger.info(f"✓ Loaded Agent 6 review results from: {agent6_output_path}")
        logger.info(f"   Images needing edit: {needs_edit_count}")

    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Encode image to base64 string

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded string or None if failed
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"   Error encoding image: {e}")
            return None

    def _create_edit_prompt_generation_request(
        self,
        asset_name: str,
        asset_type: str,
        original_prompt: str,
        edit_instructions: str,
        issues: List[str]
    ) -> str:
        """
        Create prompt for Gemini to generate an edit prompt

        Args:
            asset_name: Name of the asset
            asset_type: Type of asset
            original_prompt: Original generation prompt
            edit_instructions: Specific edit instructions from Agent 6
            issues: List of issues identified

        Returns:
            Prompt for Gemini to generate edit instructions
        """
        prompt = f"""
You are an expert AI image editing prompt engineer specializing in Nano Banana (Gemini Flash Image /
gemini-3.1-flash-image-preview) edit prompts. Your task is to create a precise, targeted edit prompt
that fixes specific issues in an AI-generated image while preserving everything that is correct.

**ASSET INFORMATION:**
- Asset Name: {asset_name}
- Asset Type: {asset_type}

**ORIGINAL GENERATION PROMPT:**
{original_prompt}

**ISSUES IDENTIFIED:**
{json.dumps(issues, indent=2)}

**EDIT INSTRUCTIONS FROM REVIEWER:**
{edit_instructions}

**YOUR TASK:**
Generate a clear, targeted edit prompt for Nano Banana that will fix the specific issues identified.

**NANO BANANA EDIT PROMPT GUIDELINES:**

1. **Semantic Masking — Target Only What Needs Changing**
   - Nano Banana uses text-driven semantic masking: it identifies the region from your description and edits only that area
   - Describe the target region precisely: "the background behind the subject", "the artifact in the upper-right corner", "the character's left hand"
   - Everything you do NOT mention will be preserved as-is — use this to your advantage

2. **Always Include an Explicit Preserve Clause**
   - Every edit prompt MUST end with what to keep unchanged
   - Formula: `[Action verb] + [Target region] + [Desired result] + [Preserve clause]`
   - Example: "Replace the background behind the character with a smooth solid neutral grey; preserve the character's exact pose, costume, facial features, skin tone, and lighting completely unchanged"
   - This is the most important guideline — Nano Banana responds well to explicit preservation instructions

3. **Use Positive Framing — Describe What You WANT**
   - Say "solid empty neutral grey background" NOT "no people in background, remove environment"
   - Say "correct the left hand anatomy to a natural relaxed position" NOT "fix the weird hand"
   - Positive descriptions produce better semantic targeting than negations

4. **Edit Types Supported (in order of safety)**
   - **SAFE**: Background replacement/neutralization (describe the desired background positively)
   - **SAFE**: Artifact removal (describe the clean result you want, plus preserve clause)
   - **SAFE**: Color correction (describe the desired color state specifically)
   - **MODERATE**: Anatomy correction (describe the correct form; limit to one body part per edit)
   - **RISKY**: Lighting changes (can alter composition — use "maintain current framing and composition" explicitly)
   - **RISKY**: Adding new elements (prefer regeneration over adding)

5. **Prompt Structure**
   - Start with an action verb (replace, remove, correct, adjust, neutralize)
   - Specify the exact semantic region (behind the subject, in the top-right corner, on the character's hand)
   - Describe the desired result in positive terms
   - End with an explicit preserve clause
   - Keep concise but complete (20-50 words ideal for Nano Banana)

6. **Examples of Good Edit Prompts:**
   - "Replace the background behind the character with a smooth, solid neutral grey studio backdrop; preserve the character's exact pose, costume, face, and lighting completely unchanged"
   - "Remove the blurred artifact in the upper-right corner of the background; keep the subject, framing, and all other elements completely unchanged"
   - "Neutralize the background to a plain off-white studio color; keep subject position, lighting on the subject, and all costume details exactly as they are"
   - "Correct the distorted anatomy of the left hand to a natural relaxed position; preserve all other body parts, clothing, and background unchanged"
   - "Adjust the overall color temperature to be slightly warmer and more natural; maintain current exposure, composition, and subject details"

7. **Examples of BAD Edit Prompts (DO NOT USE):**
   - ❌ "Add rim lighting" (adds new element, can alter composition)
   - ❌ "Enhance sharpness and detail" (vague, can over-process or introduce artifacts)
   - ❌ "Make it better" (not actionable)
   - ❌ "Remove the bad background" (negative framing — describe what you want instead)
   - ❌ "Add depth of field" (changes composition irreversibly)
   - ❌ Any edit without a preserve clause

8. **What to Avoid:**
   - No vague enhancement terms ("enhance", "improve", "better", "fix it")
   - No multiple complex changes in one prompt — one targeted change at a time
   - No edits that would crop or reframe the subject
   - No adding significant new elements (use regeneration instead)

**CRITICAL FOR DIFFERENT ASSET TYPES:**

- **Characters**: ONLY edit if background is NOT neutral, or if there are clear artifacts/anatomy distortions. Preserve neutral background, pose, and costume at all costs
- **Props**: ONLY edit if there are people/hands/animals present, or background isn't neutral. Preserve object isolation
- **Locations**: Maintain full environmental context while fixing only the specific issue identified

**WHEN TO RECOMMEND "NO EDIT NEEDED" INSTEAD:**
If the issues are:
- Minor subjective improvements (slightly better lighting, slightly sharper, etc.)
- Likely to make the image worse (composition changes, cropping, reframing)
- Not critical for production use (the image already works)

In these cases, set edit_prompt to "NO EDIT RECOMMENDED - Image is production-ready as-is" and explain in rationale.

**OUTPUT FORMAT (JSON):**
{{
    "asset_name": "{asset_name}",
    "asset_type": "{asset_type}",
    "edit_prompt": "Concise, targeted edit instruction (10-50 words)...",
    "edit_rationale": "Brief explanation of why this edit will fix the issues",
    "expected_changes": [
        "Specific change 1",
        "Specific change 2"
    ],
    "guidance_scale": 2.5,
    "edit_strength": "subtle|moderate|strong"
}}

**EDIT STRENGTH GUIDELINES:**
- subtle (1.5-2.0): Minor color/lighting adjustments
- moderate (2.0-3.0): Standard corrections, object fixes
- strong (3.0-4.0): Major changes, background replacement

Generate the edit prompt now.
"""
        return prompt

    def generate_edit_prompt(
        self,
        asset_name: str,
        asset_type: str,
        edit_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate edit prompt using Gemini

        Args:
            asset_name: Name of the asset
            asset_type: Type of asset
            edit_data: Edit instruction data from Agent 6

        Returns:
            Edit prompt data or None if failed
        """
        logger.info(f"\n   🔧 Generating edit prompt for: {asset_name}")

        is_product = edit_data.get('is_product', False)
        original_prompt = edit_data.get('original_prompt', '')
        edit_instructions = edit_data.get('edit_instructions', '')
        issues = edit_data.get('issues', [])

        request_prompt = self._create_edit_prompt_generation_request(
            asset_name, asset_type, original_prompt, edit_instructions, issues
        )

        try:
            response = self.model.generate_content(request_prompt)
            response_text = response.text

            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                json_text = json_text.replace('```json', '').replace('```', '').strip()

                edit_prompt_data = json.loads(json_text)

                # PRODUCT FIDELITY LOCK: prepend a hard constraint for uploaded product images.
                # Shape, size, proportions, text, logo, label, and color must never be altered.
                if is_product:
                    raw_prompt = edit_prompt_data.get('edit_prompt', '')
                    if raw_prompt and 'NO EDIT RECOMMENDED' not in raw_prompt.upper():
                        product_lock = (
                            "CRITICAL — PRODUCT FIDELITY: Under NO circumstances alter the product's "
                            "shape, size, proportions, text, logo, label, color, or branding. "
                            "Permitted adjustments ONLY: background, shadows, reflections, or lighting. "
                            "The product itself must remain pixel-perfect. "
                        )
                        edit_prompt_data['edit_prompt'] = product_lock + raw_prompt
                        logger.info("   🔒 Product fidelity lock applied to edit prompt")

                logger.info(f"   ✓ Edit prompt: {edit_prompt_data.get('edit_prompt', '')[:80]}...")

                return edit_prompt_data

        except Exception as e:
            logger.error(f"   Failed to generate edit prompt: {e}")
            return None

    def _edit_image_nanobanana(self, image_path: str, edit_prompt: str) -> Optional[str]:
        """
        Edit image using Nano Banana (Gemini Flash Image).

        Args:
            image_path: Local file path or URL to the image
            edit_prompt: Edit instruction prompt

        Returns:
            Local path of the edited image, or None if failed
        """
        try:
            logger.info("   📤 Submitting edit request to Nano Banana (Gemini Flash Image)...")

            # Load image from local path or URL
            if image_path.startswith("http://") or image_path.startswith("https://"):
                img_response = requests.get(image_path, timeout=60)
                if img_response.status_code != 200:
                    logger.error(f"   Failed to download image: {img_response.status_code}")
                    return None
                reference_image = PILImage.open(BytesIO(img_response.content))
            else:
                reference_image = PILImage.open(image_path)

            response = self.gemini_client.models.generate_content(
                model=self.gemini_image_model,
                contents=[edit_prompt, reference_image],
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )

            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img = PILImage.open(BytesIO(part.inline_data.data))
                    # Save to a temp file alongside the original
                    base = os.path.splitext(image_path)[0] if not image_path.startswith("http") else "/tmp/edited"
                    out_path = f"{base}_nanobanana_edited.png"
                    img.save(out_path)
                    logger.info(f"   ✓ Nano Banana edit saved: {out_path}")
                    return out_path

            logger.error("   No image in Nano Banana response")
            return None

        except Exception as e:
            logger.error(f"   Nano Banana edit request failed: {e}")
            return None

    def _download_image(self, image_url: str, save_path: str) -> bool:
        """Download edited image from URL"""
        try:
            response = requests.get(image_url, timeout=60)

            if response.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                with open(save_path, 'wb') as f:
                    f.write(response.content)

                return True
            else:
                logger.error(f"   Download failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"   Download error: {e}")
            return False

    def edit_images(self, output_dir: str = "output/edited_images") -> Dict[str, Any]:
        """
        Edit all images that need editing

        Args:
            output_dir: Directory to save edited images

        Returns:
            Dictionary containing edit results
        """
        logger.info("\n" + "="*60)
        logger.info("AGENT 7: IMAGE EDITING STARTING")
        logger.info("="*60)

        if not self.edit_instructions:
            logger.warning("\nNo images need editing")
            return {"status": "no_edits_needed"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(output_dir, timestamp)

        self.edited_images = {}

        for key, edit_data in self.edit_instructions.items():
            asset_name = edit_data.get('asset_name')
            asset_type = edit_data.get('asset_type')
            image_path = edit_data.get('image_path')

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {asset_name} ({asset_type})")
            logger.info(f"{'='*60}")

            is_url = image_path and (image_path.startswith("http://") or image_path.startswith("https://"))
            if not image_path or (not is_url and not os.path.exists(image_path)):
                logger.warning(f"   Image not found: {image_path}")
                self.failed_edits.append({
                    'asset_id': edit_data.get('asset_id'),
                    'asset_name': asset_name,
                    'reason': 'Image file not found'
                })
                continue

            # Step 1: Generate edit prompt using Gemini
            edit_prompt_data = self.generate_edit_prompt(asset_name, asset_type, edit_data)

            if not edit_prompt_data:
                self.failed_edits.append({
                    'asset_id': edit_data.get('asset_id'),
                    'asset_name': asset_name,
                    'reason': 'Failed to generate edit prompt'
                })
                continue

            self.edit_prompts[key] = edit_prompt_data

            # Check if edit is recommended
            edit_prompt = edit_prompt_data.get('edit_prompt', '')

            if "NO EDIT RECOMMENDED" in edit_prompt.upper():
                logger.info(f"   ℹNo edit recommended - image is already production-ready")
                logger.info(f"   Rationale: {edit_prompt_data.get('edit_rationale', '')}")
                self.edited_images[key] = {
                    'asset_id': edit_data.get('asset_id'),
                    'asset_name': asset_name,
                    'asset_type': asset_type,
                    'original_image': image_path,
                    'edit_skipped': True,
                    'skip_reason': edit_prompt_data.get('edit_rationale', ''),
                    'edit_prompt_data': edit_prompt_data,
                    'edit_timestamp': datetime.now().isoformat()
                }
                continue

            # Step 2: Submit edit request to Nano Banana
            safe_name = asset_name.replace(' ', '_').replace('/', '_')
            filename = f"{safe_name}_edited.png"
            save_path = os.path.join(base_output_dir, asset_type, filename)

            logger.info(f"   📤 Submitting edit request to Nano Banana...")
            edited_local_path = self._edit_image_nanobanana(image_path, edit_prompt)
            if not edited_local_path:
                self.failed_edits.append({
                    'asset_id': edit_data.get('asset_id'),
                    'asset_name': asset_name,
                    'reason': 'Nano Banana edit request failed'
                })
                continue
            import shutil
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.move(edited_local_path, save_path)
            edited_imgs = [{'index': 1, 'local_path': save_path, 'filename': filename}]

            self.edited_images[key] = {
                'asset_id': edit_data.get('asset_id'),
                'asset_name': asset_name,
                'asset_type': asset_type,
                'original_image': image_path,
                'edit_prompt': edit_prompt,
                'edit_prompt_data': edit_prompt_data,
                'edited_images': edited_imgs,
                'edit_timestamp': datetime.now().isoformat()
            }

        logger.info("\n✓ Image editing completed!")
        self._print_edit_summary()

        return self.edited_images

    def _print_edit_summary(self) -> None:
        """Print summary of edits"""
        logger.info("\n" + "─"*60)
        logger.info("EDIT SUMMARY")
        logger.info("─"*60)

        total_edited = len(self.edited_images)
        total_failed = len(self.failed_edits)

        logger.info(f"\nSuccessfully edited: {total_edited}")
        logger.error(f"Failed edits: {total_failed}")

        if self.edited_images:
            logger.info("\nEdited Assets:")
            for key, data in self.edited_images.items():
                num_images = len(data.get('edited_images', []))
                logger.info(f"  • {data['asset_name']} ({data['asset_type']}): {num_images} image(s)")

        if self.failed_edits:
            logger.error("\nFailed Edits:")
            for failure in self.failed_edits:
                logger.info(f"  • {failure['asset_name']}: {failure['reason']}")

    def update_agent6_results_with_edits(self, agent6_output_path: str) -> str:
        """
        Create updated Agent 6 results where successfully edited images are marked as approved

        Args:
            agent6_output_path: Path to original Agent 6 review results

        Returns:
            Path to updated Agent 6 results file
        """
        # Load original Agent 6 results
        with open(agent6_output_path, 'r', encoding='utf-8') as f:
            agent6_data = json.load(f)

        # Get the current categories
        approved = agent6_data.get('assets_approved', {})
        needs_edit = agent6_data.get('assets_needs_edit', {})
        regenerate = agent6_data.get('assets_regenerate', {})

        # Move successfully edited images from needs_edit to approved
        for key, edit_data in self.edited_images.items():
            asset_name = edit_data.get('asset_name')
            asset_id = edit_data.get('asset_id')
            asset_type = edit_data.get('asset_type')

            # Check if this was a successful edit (not skipped)
            if not edit_data.get('edit_skipped'):
                edited_imgs = edit_data.get('edited_images', [])
                if edited_imgs and len(edited_imgs) > 0:
                    # Find and remove from needs_edit (now a list)
                    asset_found = None
                    for idx, asset_data in enumerate(needs_edit.get(asset_type, [])):
                        if asset_data.get('id') == asset_id or asset_data.get('name') == asset_name:
                            asset_found = (idx, asset_data)
                            break

                    if asset_found:
                        idx, asset_data = asset_found
                        original_reviews = asset_data.get('reviews', [])

                        # Update the image path in the review to point to edited image
                        for review in original_reviews:
                            review['image_path'] = edited_imgs[0].get('local_path')
                            review['edited_by_agent7'] = True
                            review['original_image_path'] = edit_data.get('original_image')

                        # Move to approved (maintaining list structure)
                        approved[asset_type].append({
                            'id': asset_id,
                            'name': asset_name,
                            'reviews': original_reviews
                        })

                        # Remove from needs_edit
                        needs_edit[asset_type].pop(idx)

                        logger.info(f"   ✓ Moved {asset_name} to approved (edited successfully)")

        # Create updated Agent 6 output
        updated_data = agent6_data.copy()
        updated_data['assets_approved'] = approved
        updated_data['assets_needs_edit'] = needs_edit
        updated_data['updated_by_agent7'] = True
        updated_data['agent7_timestamp'] = datetime.now().isoformat()

        # Save updated file
        output_dir = os.path.dirname(agent6_output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent6_review_results_updated_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=2)

        logger.info(f"\n✓ Updated Agent 6 results saved to: {filepath}")
        return filepath


    def run_full_pipeline(self, agent6_output_path: str) -> Dict[str, Any]:
        """
        Run the complete Agent 7 pipeline

        Args:
            agent6_output_path: Path to Agent 6 review results JSON

        Returns:
            Dictionary with edit results
        """
        # Step 1: Load Agent 6 review results
        self.load_agent6_output(agent6_output_path)

        if not self.edit_instructions:
            logger.info("\n✓ No images need editing - all approved or need regeneration")
            return {
                "status": "no_edits_needed",
                "message": "All images were either approved or need regeneration"
            }

        # Step 2: Edit images
        results = self.edit_images()

        # Step 4: Update Agent 6 results with edited images marked as approved
        updated_agent6_path = self.update_agent6_results_with_edits(agent6_output_path)

        return {
            "status": "completed",
            "edited_images": results,
            "failed_edits": self.failed_edits,
            "updated_agent6_path": updated_agent6_path
        }


def main():
    """Example usage of Agent 7"""
    from pathlib import Path
    from dotenv import load_dotenv

    # Load environment variables
    env_path = Path(__file__).parent / '.env'
    load_dotenv(env_path)

    # Initialize agent
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not gemini_key:
        logger.error("ERROR: GEMINI_API_KEY not found in environment")
        return

    agent = ImageEditAgent(gemini_api_key=gemini_key)

    logger.info("Agent 7: Image Edit Agent initialized (Nano Banana)")
    logger.info("Use run_full_pipeline() with Agent 6 output path")


if __name__ == "__main__":
    main()
