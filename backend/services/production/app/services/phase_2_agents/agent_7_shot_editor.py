#!/usr/bin/env python3
"""
Agent 7: Shot Image Editor (Phase 2)
=====================================
Edits shot images that need improvements based on Agent 15 feedback.
Uses Gemini 2.5 Pro to generate targeted edit prompts and Freepik SeeDream 4 Edit API for editing.

Flow:
1. Receive shots marked as "need edit" from Agent 15
2. For each shot:
   - Generate targeted edit prompt using Gemini based on Agent 15 feedback
   - Download current image from S3
   - Apply edits using SeeDream 4 Edit API
   - Upload edited image to S3
   - Track edit version (v1, v2, v3)
3. Return edited shots for Agent 15 re-review
"""

import os
import logging
import requests
import tempfile
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google import genai as genai_new
from google.genai import types as genai_types
from io import BytesIO
from PIL import Image as PILImage


# Import S3 client
from infrastructure.s3.upload import S3ImageUploader

logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"

# COMMENTED OUT: Old Freepik configuration
# FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")


@dataclass
class EditResult:
    """Result of editing a single shot"""
    shot_id: str
    original_s3_url: str
    edited_s3_url: str
    edit_version: str  # v1, v2, or v3
    edit_prompt: str
    edit_instructions: str  # From Agent 15
    edit_timestamp: str
    success: bool
    error_message: Optional[str] = None


class ShotEditorAgent:
    """
    Agent 7: Edits shot images based on Agent 15 feedback
    
    This agent takes shots that need minor corrections and applies
    targeted edits using SeeDream 4 Edit API with AI-generated prompts.
    """

    def __init__(
        self,
        gemini_api_key: str = GEMINI_API_KEY,
        model_name: str = MODEL_NAME
    ):
        """
        Initialize Shot Editor Agent

        Args:
            gemini_api_key: Google AI API key for prompt generation

            model_name: Gemini model to use
        """
        # Configure Gemini
        genai.configure(api_key=gemini_api_key, transport="rest")
        self.model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # NanoBanana: Google Gemini Flash Image for editing
        self.gemini_edit_client = genai_new.Client(api_key=gemini_api_key)
        self.gemini_image_model = "gemini-3.1-flash-image-preview"

        # S3 uploader - use bucket from env var or fallback to zeroshot-v1
        s3_bucket = os.getenv("production_S3_BUCKET_NAME", "zeroshot-v1")
        self.s3_uploader = S3ImageUploader(bucket_name=s3_bucket)

        # Results tracking
        self.edit_results = []
        self.failed_edits = []

        logger.info("="*60)
        logger.info("AGENT 7: SHOT EDITOR AGENT INITIALIZED (NANO BANANA)")
        logger.info("="*60)
        logger.info(f"Gemini Model: {model_name}")
        logger.info(f"Image Edit API: Nano Banana (Gemini Flash Image)")
        logger.info(f"S3 Upload: Configured")
        logger.info("="*60)

    def download_image_from_s3(self, s3_url: str) -> Optional[str]:
        """
        Download image from S3 URL and save to temporary file
        
        Args:
            s3_url: S3 URL of the image
            
        Returns:
            Path to temporary file or None if failed
        """
        try:
            response = requests.get(s3_url, timeout=30)
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_file.write(response.content)
            temp_file.close()
            
            logger.info(f"✓ Downloaded image from S3 to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to download image from S3: {e}")
            return None

    def generate_edit_prompt(
        self,
        shot_id: str,
        edit_instructions: str,
        original_prompt: str,
        issues_found: List[Dict[str, str]],
        is_product_shot: bool = False
    ) -> Dict[str, Any]:
        """
        Generate targeted edit prompt using Gemini (using Phase 1 Agent 7 prompt structure)
        
        Args:
            shot_id: Shot identifier
            edit_instructions: Edit instructions from Agent 15
            original_prompt: Original image generation prompt
            issues_found: List of issues identified by Agent 15
            
        Returns:
            Dict containing edit prompt and metadata
        """
        logger.info(f"Generating edit prompt for {shot_id}...")
        
        prompt = f"""
You are an expert AI image editing prompt engineer specializing in Nano Banana (Gemini Flash Image /
gemini-3.1-flash-image-preview) edit prompts for cinematic shot images.
Your task is to create a precise, targeted edit prompt that fixes specific issues while preserving
the shot's composition, character placement, and framing completely intact.

**SHOT INFORMATION:**
- Shot ID: {shot_id}

**ORIGINAL GENERATION PROMPT:**
{original_prompt}

**ISSUES IDENTIFIED:**
{json.dumps(issues_found, indent=2)}

**EDIT INSTRUCTIONS FROM REVIEWER:**
{edit_instructions}

**YOUR TASK:**
Generate a clear, targeted edit prompt for Nano Banana that fixes the specific issues without
touching anything that is already correct in this shot.

**NANO BANANA EDIT PROMPT GUIDELINES:**

1. **Semantic Masking — Target Only the Problem Region**
   - Nano Banana uses text-driven semantic masking: describe the specific region to change, and it edits ONLY that area
   - Name the region precisely: "the sky in the upper third of the frame", "the background behind the characters", "the artifact in the lower-left corner", "the character's right hand"
   - Everything you do NOT mention stays untouched — use this to protect composition and character placement

2. **Always Include an Explicit Preserve Clause — MANDATORY**
   - Every edit prompt MUST state what to keep unchanged
   - Formula: `[Action verb] + [Target region] + [Desired result] + [Preserve clause]`
   - The preserve clause must protect: character positions, shot framing, camera angle, composition, and scene lighting direction
   - Example: "Adjust the color temperature of the scene to warmer golden tones; preserve all character positions, facial expressions, shot composition, and framing completely unchanged"
   - **PRODUCT SHOT RULE** (applies when is_product_shot=True): The preserve clause MUST also include "preserve the PRODUCT's position, visibility, and prominence in the frame unchanged — it must remain clearly visible and in sharp focus"

3. **Use Positive Framing — Describe What You WANT**
   - Say "warm golden-hour lighting across the scene" NOT "remove the harsh cold light"
   - Say "clear empty sky above the rooftop" NOT "no clouds, no overcast"
   - Say "correct the left hand to a natural relaxed grip" NOT "fix the weird hand"
   - Positive descriptions produce better semantic targeting than negations

4. **Edit Types Supported (in order of safety)**
   - **SAFE**: Color correction / color temperature shift (describe the desired color state)
   - **SAFE**: Artifact removal (describe the clean result + preserve clause)
   - **SAFE**: Background element correction (describe what the area should look like)
   - **MODERATE**: Lighting atmosphere adjustment (must include "maintain current framing and all character positions")
   - **RISKY**: Any anatomy correction on a character (limit to one body part; be very specific about the region)
   - **AVOID**: Adding new characters, objects, or major scene elements → use REGENERATE instead
   - **AVOID**: Changing camera angle, reframing, or cropping → these require REGENERATE

5. **Prompt Structure**
   - Start with an action verb (adjust, correct, neutralize, remove, replace, fix)
   - Name the exact semantic region in the shot
   - Describe the desired result in positive terms
   - End with a preserve clause that protects characters, framing, and composition
   - Keep concise but complete (20-50 words ideal for Nano Banana)

6. **Examples of Good Edit Prompts (shot-scene specific):**
   - "Adjust the color temperature of the entire scene to warm golden-hour tones; preserve all character positions, expressions, shot framing, and composition exactly as they are"
   - "Remove the visual artifact in the upper-right corner of the background; keep all characters, their positions, lighting, and scene composition completely unchanged"
   - "Correct the sky in the upper portion of the frame to a clear blue with natural cloud coverage consistent with the existing lighting; preserve all foreground characters, their placement, and scene composition"
   - "Fix the distorted left hand of the foreground character to a natural relaxed position; preserve the character's pose, position in frame, costume, and all other elements unchanged"
   - "Neutralize the overly saturated greens in the background foliage to a more natural muted tone; maintain all character placements, shot framing, and scene lighting direction"

7. **Examples of BAD Edit Prompts (DO NOT USE):**
   - ❌ "Add rim lighting to the characters" (adds new element, alters composition)
   - ❌ "Enhance sharpness and detail" (vague, can over-process or introduce artifacts)
   - ❌ "Make the scene more cinematic" (not actionable)
   - ❌ "Remove the bad lighting" (negative framing — describe what you want instead)
   - ❌ "Add depth of field" (changes composition irreversibly)
   - ❌ Any edit without a preserve clause

8. **What to Avoid:**
   - No vague enhancement terms ("enhance", "improve", "better", "fix it")
   - No edits that would shift character positions or change shot framing
   - No adding significant new scene elements (use REGENERATE instead)
   - Never combine multiple complex changes in one prompt — one targeted fix at a time

**PRODUCT SHOT CONTEXT:**
{'⚠️  THIS IS A PRODUCT SHOT. The PRODUCT is the hero element of this scene. Your edit prompt MUST preserve the product\'s visibility, position, and prominence. Never edit, move, obscure, or diminish the product. The preserve clause MUST explicitly say: "preserve the PRODUCT position, visibility, and prominence — it must remain clearly in frame and in sharp focus."' if is_product_shot else 'No product requirement for this shot.'}

**WHEN TO RECOMMEND "NO EDIT NEEDED" INSTEAD:**
If the issues are:
- Minor subjective improvements (slightly better lighting, slightly sharper, etc.)
- Likely to make the image worse (composition changes, cropping, reframing)
- Not critical for production use (the shot already works)

In these cases, set edit_prompt to "NO EDIT RECOMMENDED - Shot is production-ready as-is" and explain in rationale.

**OUTPUT FORMAT (JSON):**
{{
    "shot_id": "{shot_id}",
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

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Extract JSON
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                edit_prompt_data = json.loads(json_text)
                
                logger.info(f"✓ Generated edit prompt: {edit_prompt_data.get('edit_prompt', '')[:80]}...")
                return edit_prompt_data
            else:
                raise ValueError("No JSON found in response")
            
        except Exception as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            
            # Fallback: return a simple dict with edit instructions
            return {
                "shot_id": shot_id,
                "edit_prompt": edit_instructions[:100] if edit_instructions else "Fix identified issues",
                "edit_rationale": "Fallback edit prompt due to parsing error",
                "expected_changes": ["Address issues from review"],
                "guidance_scale": 2.5,
                "edit_strength": "moderate"
            }

    # ============================================================
    # COMMENTED OUT: OLD FREEPIK API METHOD
    # ============================================================
    # def edit_image_with_seedream(
    #     self,
    #     image_path: str,
    #     edit_prompt: str,
    #     shot_id: str,
    #     guidance_scale: float = 2.5
    # ) -> Optional[str]:
    #     """
    #     Edit image using Freepik SeeDream 4 Edit API (OLD FREEPIK VERSION)
    #
    #     Args:
    #         image_path: Path to the image file
    #         edit_prompt: Edit prompt
    #         shot_id: Shot identifier for naming
    #         guidance_scale: Guidance scale for editing (1.5-4.0, default 2.5)
    #
    #     Returns:
    #         Path to edited image file or None if failed
    #     """
    #     logger.info(f"Editing image with SeeDream Edit API...")
    #     logger.info(f"Edit prompt: {edit_prompt}")
    #     logger.info(f"Guidance scale: {guidance_scale}")
    #
    #     try:
    #         # Read image file and encode to base64
    #         with open(image_path, 'rb') as f:
    #             import base64
    #             image_data = base64.b64encode(f.read()).decode('utf-8')
    #
    #         # Prepare request payload for SeeDream Edit (using SeeDream v4 Edit endpoint)
    #         endpoint = f"{self.base_url}/seedream-v4-edit"
    #
    #         payload = {
    #             "prompt": edit_prompt,
    #             "reference_images": [f"data:image/png;base64,{image_data}"],
    #             "guidance_scale": guidance_scale,
    #             "aspect_ratio": "square_1_1"
    #         }
    #
    #         # Submit edit request
    #         logger.info(f"📤 Submitting edit request to {endpoint}...")
    #         response = requests.post(
    #             endpoint,
    #             headers=self.headers,
    #             json=payload,
    #             timeout=120
    #         )
    #
    #         if response.status_code != 200:
    #             logger.error(f"❌ API Error: {response.status_code} - {response.text}")
    #             return None
    #
    #         result = response.json()
    #         logger.info(f"✓ API Response received: {result}")
    #
    #         # SeeDream v4 Edit returns a task_id nested in 'data' field
    #         data = result.get('data', {})
    #         task_id = data.get('task_id')
    #         if not task_id:
    #             logger.error(f"No task_id in response: {result}")
    #             return None
    #
    #         # Poll for result using the correct endpoint structure
    #         logger.info(f"⏳ Polling for task {task_id}...")
    #         max_attempts = 30
    #         for attempt in range(max_attempts):
    #             time.sleep(5)  # Wait 5 seconds between polls (same as Phase 1)
    #
    #             status_response = requests.get(
    #                 f"{self.base_url}/seedream-v4-edit/{task_id}",
    #                 headers=self.headers,
    #                 timeout=30
    #             )
    #
    #             if status_response.status_code == 200:
    #                 status_result = status_response.json()
    #
    #                 # Extract data from nested structure (same as Phase 1)
    #                 data = status_result.get('data', status_result)
    #                 status = data.get('status', 'UNKNOWN')
    #
    #                 logger.info(f"   Status: {status} (attempt {attempt + 1}/{max_attempts})")
    #
    #                 if status == 'COMPLETED':
    #                     # Get the generated image URLs
    #                     generated_urls = data.get('generated', [])
    #                     if generated_urls and len(generated_urls) > 0:
    #                         edited_image_url = generated_urls[0]
    #
    #                         if edited_image_url:
    #                             # Download the edited image
    #                             edited_response = requests.get(edited_image_url, timeout=30)
    #                             edited_response.raise_for_status()
    #
    #                             # Save to temporary file
    #                             temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    #                             temp_file.write(edited_response.content)
    #                             temp_file.close()
    #
    #                             logger.info(f"✓ Image edited successfully and saved to: {temp_file.name}")
    #                             return temp_file.name
    #
    #                     logger.error(f"No images in completed response: {data}")
    #                     return None
    #
    #                 elif status == 'FAILED':
    #                     logger.error(f"Task failed: {data}")
    #                     return None
    #                 elif status in ['CREATED', 'IN_PROGRESS']:
    #                     # Continue polling
    #                     continue
    #
    #         logger.error(f"Task timed out after {max_attempts} attempts")
    #         return None
    #
    #     except Exception as e:
    #         logger.error(f"Failed to edit image with SeeDream: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    # ============================================================
    # END OF COMMENTED OUT FREEPIK METHOD
    # ============================================================

    def edit_image_with_nanobanana(
        self,
        image_source: str,
        edit_prompt: str,
        shot_id: str
    ) -> Optional[str]:
        """
        Edit image using Nano Banana (Gemini Flash Image).

        Args:
            image_source: Publicly accessible URL or local path for the source image
            edit_prompt: Edit prompt
            shot_id: Shot identifier for naming

        Returns:
            Path to edited image file or None if failed
        """
        logger.info(f"Editing image with Nano Banana (Gemini Flash Image)...")
        logger.info(f"Source image: {image_source}")
        logger.info(f"Edit prompt: {edit_prompt}")

        try:
            # Load image from URL or local path
            if image_source.startswith("http://") or image_source.startswith("https://"):
                img_response = requests.get(image_source, timeout=60)
                img_response.raise_for_status()
                reference_image = PILImage.open(BytesIO(img_response.content))
            else:
                reference_image = PILImage.open(image_source)

            response = self.gemini_edit_client.models.generate_content(
                model=self.gemini_image_model,
                contents=[edit_prompt, reference_image],
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE"]
                )
            )

            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    img = PILImage.open(BytesIO(part.inline_data.data))
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(temp_file.name)
                    temp_file.close()
                    logger.info(f"✓ Nano Banana edited image saved to: {temp_file.name}")
                    return temp_file.name

            logger.error("❌ No image in Nano Banana response")
            return None

        except Exception as e:
            logger.error(f"❌ Nano Banana edit request failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def edit_shot(
        self,
        shot_id: str,
        current_s3_url: str,
        edit_instructions: str,
        issues_found: List[Dict[str, str]],
        original_prompt: str,
        edit_version: str,
        show_id: str,
        is_product_shot: bool = False
    ) -> EditResult:
        """
        Edit a single shot image
        
        Args:
            shot_id: Shot identifier
            current_s3_url: Current image S3 URL
            edit_instructions: Edit instructions from Agent 15
            issues_found: List of issues from Agent 15
            original_prompt: Original generation prompt
            edit_version: Version of this edit (v1, v2, v3)
            show_id: Show identifier for S3 path
            
        Returns:
            EditResult with edit outcome
        """
        logger.info(f"─"*60)
        logger.info(f"EDITING SHOT: {shot_id} ({edit_version})")
        logger.info(f"─"*60)
        
        try:
            # Step 1: (Optional) Download current image from S3 for local reference
            image_path = self.download_image_from_s3(current_s3_url)
            
            # Step 2: Generate edit prompt using Gemini
            edit_prompt_data = self.generate_edit_prompt(
                shot_id,
                edit_instructions,
                original_prompt,
                issues_found,
                is_product_shot=is_product_shot
            )
            
            # Extract edit prompt
            edit_prompt = edit_prompt_data.get('edit_prompt', '')

            # Check if edit is recommended
            if "NO EDIT RECOMMENDED" in edit_prompt.upper():
                logger.info(f"⚠️  No edit recommended for {shot_id}: {edit_prompt_data.get('edit_rationale', '')}")
                # Clean up
                try:
                    os.unlink(image_path)
                except:
                    pass

                return EditResult(
                    shot_id=shot_id,
                    original_s3_url=current_s3_url,
                    edited_s3_url=current_s3_url,  # Use original image
                    edit_version=edit_version,
                    edit_prompt=edit_prompt,
                    edit_instructions=edit_instructions,
                    edit_timestamp=datetime.now().isoformat(),
                    success=True,  # Success, but no changes made
                    error_message=None
                )

            # Step 3: Edit image with Nano Banana
            edited_image_path = self.edit_image_with_nanobanana(
                current_s3_url, edit_prompt, shot_id
            )
            
            if not edited_image_path:
                # Clean up
                try:
                    if image_path:
                        os.unlink(image_path)
                except:
                    pass
                
                return EditResult(
                    shot_id=shot_id,
                    original_s3_url=current_s3_url,
                    edited_s3_url="",
                    edit_version=edit_version,
                    edit_prompt=edit_prompt,
                    edit_instructions=edit_instructions,
                    edit_timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message="Failed to edit image with SeeDream"
                )
            
            # Step 4: Upload edited image to S3 (productionvideos bucket, edited_images folder)
            s3_key = f"edited_images/{show_id}/shots/{shot_id}/{edit_version}.png"
            edited_s3_url = self.s3_uploader.upload_image(edited_image_path, s3_key)
            
            # Clean up temporary files
            try:
                if image_path:
                    os.unlink(image_path)
                os.unlink(edited_image_path)
            except:
                pass
            
            if not edited_s3_url:
                return EditResult(
                    shot_id=shot_id,
                    original_s3_url=current_s3_url,
                    edited_s3_url="",
                    edit_version=edit_version,
                    edit_prompt=edit_prompt,
                    edit_instructions=edit_instructions,
                    edit_timestamp=datetime.now().isoformat(),
                    success=False,
                    error_message="Failed to upload edited image to S3"
                )
            
            logger.info(f"✓ Shot {shot_id} edited successfully ({edit_version})")
            logger.info(f"  Edited image S3 URL: {edited_s3_url}")
            
            return EditResult(
                shot_id=shot_id,
                original_s3_url=current_s3_url,
                edited_s3_url=edited_s3_url,
                edit_version=edit_version,
                edit_prompt=edit_prompt,
                edit_instructions=edit_instructions,
                edit_timestamp=datetime.now().isoformat(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to edit shot {shot_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return EditResult(
                shot_id=shot_id,
                original_s3_url=current_s3_url,
                edited_s3_url="",
                edit_version=edit_version,
                edit_prompt="",
                edit_instructions=edit_instructions,
                edit_timestamp=datetime.now().isoformat(),
                success=False,
                error_message=str(e)
            )

    def edit_shots_batch(
        self,
        shots_to_edit: List[Dict[str, Any]],
        edit_loop_iterations: Dict[str, int],
        show_id: str,
        product_shot_ids: Optional[set] = None
    ) -> Dict[str, Any]:
        """
        Edit a batch of shots that need editing
        
        Args:
            shots_to_edit: List of shot data with edit requirements
            edit_loop_iterations: Current iteration count per shot
            show_id: Show identifier for S3 paths
            
        Returns:
            Dictionary with edit results
        """
        logger.info("="*60)
        logger.info("AGENT 7: SHOT EDITING STARTING")
        logger.info("="*60)
        logger.info(f"Shots to edit: {len(shots_to_edit)}")
        
        self.edit_results = []
        self.failed_edits = []
        
        for shot_data in shots_to_edit:
            shot_id = shot_data['shot_id']
            current_iteration = edit_loop_iterations.get(shot_id, 0)
            edit_version = f"v{current_iteration + 1}"  # v1, v2, or v3
            
            # Get current S3 URL (either from previous edit or original generation)
            current_s3_url = shot_data.get('current_s3_url', '')
            if not current_s3_url:
                logger.warning(f"No S3 URL found for {shot_id}, skipping")
                self.failed_edits.append({
                    'shot_id': shot_id,
                    'reason': 'No S3 URL found for current image'
                })
                continue
            
            # Edit the shot
            edit_result = self.edit_shot(
                shot_id=shot_id,
                current_s3_url=current_s3_url,
                edit_instructions=shot_data.get('edit_instructions', ''),
                issues_found=shot_data.get('issues_found', []),
                original_prompt=shot_data.get('original_prompt', ''),
                edit_version=edit_version,
                show_id=show_id,
                is_product_shot=bool(product_shot_ids and shot_id in product_shot_ids)
            )
            
            if edit_result.success:
                self.edit_results.append(edit_result)
            else:
                self.failed_edits.append({
                    'shot_id': shot_id,
                    'reason': edit_result.error_message,
                    'edit_version': edit_version
                })
        
        logger.info("="*60)
        logger.info("AGENT 7: SHOT EDITING COMPLETED")
        logger.info("="*60)
        logger.info(f"Successful edits: {len(self.edit_results)}")
        logger.info(f"Failed edits: {len(self.failed_edits)}")
        
        return {
            "edit_results": [asdict(r) for r in self.edit_results],
            "failed_edits": self.failed_edits,
            "total_edited": len(self.edit_results),
            "total_failed": len(self.failed_edits)
        }

    def save_edit_report(
        self,
        output_dir: str = None
    ) -> str:
        """
        Save edit report to JSON file

        Args:
            output_dir: Directory to save report. Defaults to an `outputs/`
                folder next to this file, regardless of the process's cwd.

        Returns:
            Path to saved report
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "agent_7_shot_edits")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent7_shot_edit_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        report = {
            "agent": "Agent 7: Shot Editor Agent",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_edits": len(self.edit_results),
                "successful_edits": len(self.edit_results),
                "failed_edits": len(self.failed_edits)
            },
            "edit_results": [asdict(r) for r in self.edit_results],
            "failed_edits": self.failed_edits
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Edit report saved to: {filepath}")
        return filepath


def main():
    """Example usage of Agent 7 Shot Editor"""
    logger.info("Agent 7: Shot Editor Agent")
    logger.info("Usage: Initialize agent and call edit_shots_batch(shots_to_edit, edit_loop_iterations, show_id)")


if __name__ == "__main__":
    main()

