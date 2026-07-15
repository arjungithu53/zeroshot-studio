"""
Agent 15: Image Review & Validation Agent
==========================================
Reviews generated images from Agent 14 using Gemini 2.5 Pro vision API to validate:
- Camera angles and POV accuracy
- Character placement and composition
- Scene consistency with shot requirements
- Technical quality issues

Flow:
1. Load Agent 14 generated images metadata from MongoDB
2. For each image:
   - Download image from S3 URL
   - Analyze using Gemini 2.5 Pro vision
   - Compare against shot requirements (angle, POV, composition)
   - Check for technical issues (lighting, focus, artifacts)
   - Validate character consistency
3. Make decision: APPROVE, REGENERATE, or EDIT
4. For EDIT decisions, generate minimal edit instructions (1-2 lines)
5. Output review report and save to MongoDB
"""

import os
import logging
import json
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import PIL.Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"


@dataclass
class ImageReview:
    """Review result for a single image"""
    shot_id: str
    image_s3_url: str
    decision: str  # "APPROVE", "REGENERATE", or "EDIT"
    confidence: float  # 0.0 to 1.0
    issues_found: List[Dict[str, str]]  # [{"category": "angle", "severity": "critical", "description": "..."}]
    edit_instructions: Optional[str]  # 1-2 line edit instruction if decision is EDIT
    analysis: Dict[str, Any]  # Full analysis from Gemini
    review_timestamp: str


@dataclass
class ReviewSummary:
    """Summary of all reviews"""
    total_images: int
    approved: int
    regenerate: int
    edit: int
    critical_issues: int
    reviews: List[ImageReview]


class ImageReviewAgent:
    """
    Agent 15: Reviews generated images using Gemini 2.5 Pro vision API
    
    Validates images against shot requirements and provides actionable feedback.
    """

    def __init__(self, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        """
        Initialize Image Review Agent with Gemini vision API

        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model name
        """
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=self.api_key, transport="rest")
        
        # Initialize vision model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        self.reviews = []
        
        logger.info("="*60)
        logger.info("AGENT 15: IMAGE REVIEW AGENT INITIALIZED")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Vision API: Enabled")
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
            # Download image
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

    def _prepare_image_for_gemini(self, image_path: str):
        """
        Load and prepare image for Gemini vision API (inline, no Files API upload).

        Args:
            image_path: Path to image file

        Returns:
            PIL.Image for inline passing to generate_content()
        """
        try:
            image = PIL.Image.open(image_path)
            logger.info(f"✓ Image loaded for Gemini (inline)")
            return image
        except Exception as e:
            logger.error(f"Failed to load image for Gemini: {e}")
            raise

    def _build_review_prompt(self, shot_metadata: Dict[str, Any], shot_id: str, is_product_shot: bool = False) -> str:
        """
        Build detailed review prompt for Gemini vision API
        
        Args:
            shot_metadata: Combined metadata for the shot
            shot_id: Shot identifier
            
        Returns:
            Review prompt string
        """
        # Extract metadata from different sources
        shot_design = shot_metadata.get('shot_design', {})
        prompt_modifications = shot_metadata.get('prompt_modifications', {})
        
        # Extract key requirements
        required_angle = shot_design.get('metadata', {}).get('required_angle', 'unknown')
        original_description = shot_design.get('metadata', {}).get('original_description', '')
        corrected_prompt = prompt_modifications.get('corrected_prompt', '')
        characters = shot_design.get('metadata', {}).get('characters_found', [])
        spatial_placement = shot_design.get('metadata', {}).get('spatial_placement', {})
        
        prompt = f"""You are an expert cinematographer and image quality analyst reviewing a generated image for shot {shot_id}.

**SHOT REQUIREMENTS:**

**Required Camera Angle:** {required_angle}

**Original Shot Description:**
{original_description}

**Detailed Generation Prompt:**
{corrected_prompt}

**Characters Expected:** {', '.join(characters)}

**Spatial Placement Requirements:**
{json.dumps(spatial_placement, indent=2)}

---

{"**⚠️  PRODUCT SHOT — ADDITIONAL MANDATORY CHECK:**" + chr(10) + "This shot requires a PRODUCT to be prominently visible in the image. Before all other checks:" + chr(10) + "- Is the PRODUCT clearly visible, in sharp focus, and prominently placed in the frame?" + chr(10) + "- If the PRODUCT is absent, obscured, blurry, or cropped out → decision MUST be REGENERATE." + chr(10) + "- If the PRODUCT is present but poorly positioned or partially hidden → decision MUST be EDIT." + chr(10) if is_product_shot else ""}
**YOUR TASK:**

Analyze the provided image and evaluate it against the shot requirements above. Focus on:

1. **Camera Angle & POV Accuracy** (CRITICAL)
   - Is the camera angle correct? (e.g., if "close_up" is required, is it actually a close-up?)
   - For POV shots, is the perspective accurate?
   - Does the composition match the described framing?

2. **Character Placement & Composition**
   - Are all expected characters present?Are they the correct "characters" as described (e.g., if the description requires a "white tiger", does the image show a "white tiger")?
   - Are they positioned correctly according to spatial placement requirements?
   - Does the composition follow the rule of thirds or other specified guidelines?
   - Does the background/environment match the description?

3. **Scene Consistency**
   - Does the lighting match the description?
   - Is the background/environment correct?
   - Are there any inconsistencies with the scene requirements?

4. **Technical Quality**
   - Are there visible artifacts, distortions, or generation errors?
   - Is the focus correct?
   - Are there any anatomical issues with characters?
   (Note: Do not analyze aspect ratio here, per the critical rule.)

{"5. **Product Visibility** (CRITICAL — product shots only)" + chr(10) + "   - Is the PRODUCT clearly visible and in sharp focus?" + chr(10) + "   - Is it prominently placed so viewers immediately notice it?" + chr(10) + "   - If absent or obscured → REGENERATE; if partially hidden → EDIT" + chr(10) if is_product_shot else ""}
**DECISION CRITERIA:**

- **APPROVE**: Image meets all requirements with only minor, acceptable variations
- **EDIT**: Image is mostly correct but has 1-2 specific issues that can be fixed with minimal edits (e.g., adjust lighting, slight reframe)
- **REGENERATE**: Image has critical issues with angle/POV, character placement, or multiple problems requiring full regeneration

**OUTPUT FORMAT (JSON):**

```json
{{
  "decision": "APPROVE|EDIT|REGENERATE",
  "confidence": 0.0-1.0,
  "issues_found": [
    {{
      "category": "angle|pov|composition|lighting|technical|character",
      "severity": "critical|major|minor",
      "description": "Detailed description of the issue"
    }}
  ],
  "edit_instructions": "1-2 line edit instruction if decision is EDIT, otherwise null",
  "analysis": {{
    "angle_accuracy": "Assessment of camera angle correctness",
    "pov_accuracy": "Assessment of POV accuracy (if applicable)",
    "composition_match": "How well composition matches requirements",
    "character_placement": "Assessment of character positioning",
    "lighting_match": "How well lighting matches description",
    "technical_quality": "Overall technical quality assessment (excluding aspect ratio)",
    "overall_assessment": "Brief overall summary"

  }}
}}
```

Analyze the image carefully and provide your review in the JSON format above."""

        return prompt

    def review_image(
        self,
        s3_url: str,
        shot_id: str,
        shot_metadata: Dict[str, Any],
        is_product_shot: bool = False
    ) -> ImageReview:
        """
        Review a single image using Gemini vision API
        
        Args:
            s3_url: S3 URL of the image
            shot_id: Shot identifier
            shot_metadata: Combined metadata for the shot
            
        Returns:
            ImageReview object with review results
        """
        logger.info(f"─"*60)
        logger.info(f"REVIEWING: {shot_id}")
        logger.info(f"Image S3 URL: {s3_url}")
        logger.info(f"─"*60)
        
        try:
            # Download image from S3
            temp_image_path = self.download_image_from_s3(s3_url)
            
            if not temp_image_path:
                raise Exception("Failed to download image from S3")
            
            # Upload to Gemini
            uploaded_file = self._prepare_image_for_gemini(temp_image_path)
            
            # Build review prompt
            review_prompt = self._build_review_prompt(shot_metadata, shot_id, is_product_shot=is_product_shot)
            
            # Call Gemini vision API
            logger.info(f"📤 Sending to Gemini vision API...")
            response = self.model.generate_content([review_prompt, uploaded_file])
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            review_data = json.loads(response_text)
            
            # Create ImageReview object
            review = ImageReview(
                shot_id=shot_id,
                image_s3_url=s3_url,
                decision=review_data['decision'],
                confidence=review_data['confidence'],
                issues_found=review_data['issues_found'],
                edit_instructions=review_data.get('edit_instructions'),
                analysis=review_data['analysis'],
                review_timestamp=datetime.now().isoformat()
            )
            
            # Print review summary
            logger.info(f"✓ Review completed")
            logger.info(f"Decision: {review.decision} (confidence: {review.confidence:.2f})")
            logger.info(f"Issues found: {len(review.issues_found)}")
            
            if review.issues_found:
                for issue in review.issues_found:
                    severity_icon = "🔴" if issue['severity'] == 'critical' else "🟡" if issue['severity'] == 'major' else "🔵"
                    logger.info(f"   {severity_icon} [{issue['category']}] {issue['description']}")
            
            if review.edit_instructions:
                logger.info(f"📝 Edit instructions: {review.edit_instructions}")
            
            # Clean up temporary file
            try:
                os.unlink(temp_image_path)
            except:
                pass
            
            return review
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Raw response: {response_text}")
            
            # Create a fallback review
            return ImageReview(
                shot_id=shot_id,
                image_s3_url=s3_url,
                decision="REGENERATE",
                confidence=0.0,
                issues_found=[{
                    "category": "technical",
                    "severity": "critical",
                    "description": f"Review failed - could not parse API response: {e}"
                }],
                edit_instructions=None,
                analysis={"error": str(e), "raw_response": response_text},
                review_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Review failed for {shot_id}: {e}")
            
            # Create a fallback review
            return ImageReview(
                shot_id=shot_id,
                image_s3_url=s3_url,
                decision="REGENERATE",
                confidence=0.0,
                issues_found=[{
                    "category": "technical",
                    "severity": "critical",
                    "description": f"Review failed with error: {e}"
                }],
                edit_instructions=None,
                analysis={"error": str(e)},
                review_timestamp=datetime.now().isoformat()
            )

    def review_all_images(
        self,
        generated_images: List[Dict],
        shot_designs: List[Dict],
        modified_prompts: List[Dict],
        product_shot_ids: Optional[set] = None
    ) -> ReviewSummary:
        """
        Review all generated images from Agent 14
        
        Args:
            generated_images: List of generated image data from Agent 14
            shot_designs: List of shot designs from Agent 12
            modified_prompts: List of modified prompts from Agent 13
            
        Returns:
            ReviewSummary with all review results
        """
        logger.info("="*60)
        logger.info("STARTING IMAGE REVIEW")
        logger.info("="*60)
        
        if not generated_images:
            raise ValueError("No images provided for review")
        
        for image_data in generated_images:
            shot_id = image_data['shot_id']
            s3_url = image_data.get('s3_url', '')
            
            if not s3_url:
                logger.warning(f"No S3 URL found for {shot_id}, skipping review")
                continue
            
            # Get shot metadata from agents 12 and 13
            shot_design = next((s for s in shot_designs if s['shot_id'] == shot_id), {})
            prompt_modifications = next((p for p in modified_prompts if p['shot_id'] == shot_id), {})
            
            shot_metadata = {
                'shot_id': shot_id,
                'shot_design': shot_design,
                'prompt_modifications': prompt_modifications
            }
            
            # Review image
            review = self.review_image(
                s3_url, shot_id, shot_metadata,
                is_product_shot=bool(product_shot_ids and shot_id in product_shot_ids)
            )
            self.reviews.append(review)
        
        # Create summary
        summary = ReviewSummary(
            total_images=len(self.reviews),
            approved=sum(1 for r in self.reviews if r.decision == "APPROVE"),
            regenerate=sum(1 for r in self.reviews if r.decision == "REGENERATE"),
            edit=sum(1 for r in self.reviews if r.decision == "EDIT"),
            critical_issues=sum(
                1 for r in self.reviews
                for issue in r.issues_found
                if issue['severity'] == 'critical'
            ),
            reviews=self.reviews
        )
        
        logger.info("="*60)
        logger.info("REVIEW COMPLETED")
        logger.info("="*60)
        self._print_summary(summary)
        
        return summary

    def _print_summary(self, summary: ReviewSummary) -> None:
        """Print review summary"""
        logger.info("─"*60)
        logger.info("📊 REVIEW SUMMARY")
        logger.info("─"*60)
        logger.info(f"Total images reviewed: {summary.total_images}")
        logger.info(f"✅ Approved: {summary.approved}")
        logger.info(f"✏️  Edit required: {summary.edit}")
        logger.info(f"🔄 Regenerate required: {summary.regenerate}")
        logger.info(f"🔴 Critical issues: {summary.critical_issues}")
        
        if summary.edit > 0:
            logger.info(f"\n📝 EDIT INSTRUCTIONS:")
            for review in summary.reviews:
                if review.decision == "EDIT" and review.edit_instructions:
                    logger.info(f"   {review.shot_id}: {review.edit_instructions}")

    def save_review_report(
        self,
        summary: ReviewSummary,
        output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_15_image_reviews"
    ) -> str:
        """
        Save review report to JSON file
        
        Args:
            summary: ReviewSummary object
            output_dir: Directory to save report
            
        Returns:
            Path to saved report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent15_review_report_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        report = {
            "agent": "Agent 15: Image Review Agent",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_images": summary.total_images,
                "approved": summary.approved,
                "edit": summary.edit,
                "regenerate": summary.regenerate,
                "critical_issues": summary.critical_issues
            },
            "reviews": [asdict(review) for review in summary.reviews]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Review report saved to: {filepath}")
        return filepath


def save_results(
    summary: ReviewSummary,
    output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_15_image_reviews"
) -> str:
    """
    Save agent 15 review results to JSON file
    
    Args:
        summary: ReviewSummary object
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agent15_review_report_{timestamp}.json"
    
    report = {
        "agent": "Agent 15: Image Review Agent",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": summary.total_images,
            "approved": summary.approved,
            "edit": summary.edit,
            "regenerate": summary.regenerate,
            "critical_issues": summary.critical_issues
        },
        "reviews": [asdict(review) for review in summary.reviews]
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Results saved to: {filename}")
    return filename


def main():
    """Example usage of Agent 15"""
    logger.info("Agent 15: Image Review Agent")
    logger.info("Usage: Initialize agent and call review_all_images(generated_images, shot_designs, modified_prompts)")


if __name__ == "__main__":
    main()

