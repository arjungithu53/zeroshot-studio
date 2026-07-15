"""
Agent 15A: Prompt Regeneration Agent
====================================
Analyzes current image and edit instructions to generate an improved prompt for regeneration.

Flow:
1. Receive edit_instructions (from agent_15), image_s3_link (current/latest), and older_prompt
2. Download the image from S3
3. Analyze the image using Gemini vision API
4. Generate an updated prompt that addresses the regeneration requirements
5. Output: { updated_prompt, updated_edit_instructions, older_prompt }
"""

import os
import logging
import json
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import PIL.Image
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)

# Configuration — GOOGLE_API_KEY is the working Gemini key in this env
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-pro-preview"


@dataclass
class RegenerationOutput:
    """Output from agent_15A"""
    updated_prompt: str
    updated_edit_instructions: str
    older_prompt: str
    shot_id: str
    image_s3_url: str
    regeneration_timestamp: str
    reasoning: str
    analysis: Dict[str, Any]


class PromptRegenerationAgent:
    """
    Agent 15A: Regenerates image prompts based on review feedback
    
    Takes the current image, edit instructions, and original prompt,
    analyzes them together, and generates an improved prompt for regeneration.
    """
    
    def __init__(self, api_key: str = API_KEY, model_name: str = MODEL_NAME):
        """
        Initialize Prompt Regeneration Agent
        
        Args:
            api_key: Google API key for Gemini
            model_name: Model name (gemini-3.1-pro-preview)
        """
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
        
        logger.info("="*60)
        logger.info("AGENT 15A: PROMPT REGENERATION AGENT INITIALIZED")
        logger.info("="*60)
        logger.info(f"Model: {self.model_name}")
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
    
    def _build_regeneration_prompt(
        self,
        older_prompt: str,
        edit_instructions: str,
        shot_id: str,
        shot_metadata: Optional[Dict[str, Any]] = None,
        is_product_shot: bool = False
    ) -> str:
        """
        Build prompt for Gemini to regenerate the image generation prompt
        
        Args:
            older_prompt: Original prompt that generated the current image
            edit_instructions: Edit instructions from Agent 15 (why regeneration is needed)
            shot_id: Shot identifier
            shot_metadata: Optional additional shot metadata
            
        Returns:
            Regeneration prompt string
        """
        # Extract metadata if available
        shot_design = shot_metadata.get('shot_design', {}) if shot_metadata else {}
        prompt_modifications = shot_metadata.get('prompt_modifications', {}) if shot_metadata else {}
        
        required_angle = shot_design.get('metadata', {}).get('required_angle', 'unknown') if shot_design else 'unknown'
        original_description = shot_design.get('metadata', {}).get('original_description', '') if shot_design else ''
        
        prompt = f"""You are an expert AI image generation prompt engineer specializing in creating detailed, precise prompts for cinematic image generation.

**CONTEXT:**

A generated image needs to be **regenerated** (not edited) because it has issues that require a complete re-generation with an improved prompt.

**SHOT INFORMATION:**
- Shot ID: {shot_id}
- Required Camera Angle: {required_angle}
- Original Shot Description: {original_description}

**ORIGINAL PROMPT (that generated the current image):**
{older_prompt}

**ISSUES IDENTIFIED (requiring regeneration):**
{edit_instructions}

**YOUR TASK:**

Analyze the provided image (which was generated using the "Original Prompt" above) and the "Issues Identified" section. Your goal is to generate a **significantly improved prompt** that will address all the regeneration requirements when used to generate a new image.

**PROMPT GENERATION GUIDELINES:**

1. **Address All Issues**
   - Carefully review the "Issues Identified" section
   - Ensure your updated prompt explicitly addresses each issue mentioned
   - If camera angle is wrong, correct it in the prompt
   - If character placement is incorrect, specify correct placement
   - If composition doesn't match requirements, describe the correct composition

2. **Preserve What Works**
   - Keep elements from the original prompt that are correct
   - Maintain visual style and mood if appropriate
   - Preserve character descriptions and scene context that are accurate

3. **Improve Precision with Cinematic Language**
   - **Camera & Framing:** Specify camera angle (eye-level, low-angle, high-angle), movement (static, handheld), and framing (close-up, medium shot, wide shot)
   - **Lens:** Name the lens — "24mm wide-angle for expansive environmental context", "85mm telephoto for portrait compression with background bokeh", "35mm balanced standard view"
   - **Lighting — direction and quality:** "soft golden-hour sidelight from camera-left casting long shadows across the ground", "hard overhead practical fluorescent with sharp downward shadows", "overcast diffuse daylight with flat even fill and no directional shadows"
   - **Materiality (for realistic style):** Replace vague references with specific physical descriptions — NOT "stone wall" but "rough-hewn limestone blocks with weathered mortar joints and surface moss"; NOT "wooden floor" but "wide-plank oak floorboards with a faded polyurethane finish and visible grain"
   - **Film Stock / Color Grade (for realistic style):** "Kodak Vision3 500T cinema color grade with warm shadows and natural grain", "ARRI Log-C with muted highlights and accurate skin tones", or "clean digital neutral color balance"
   - **Physical Grounding:** Describe how characters physically connect to the environment — "feet firmly planted on the wet cobblestones", "back pressed against the rough brick surface", "hand gripping the rusted metal railing" — this prevents the composited/pasted-in look
   - Be explicit about character placement, spatial relationships, and spatial anchoring

4. **Prompt Structure** — follow this order:
   - `[Camera framing & angle]` → `[Subject position & action in the scene]` → `[Physical grounding with environment]` → `[Shared lighting — direction, quality, effect on all elements]` → `[Atmosphere & mood]` → `[Technical specs — lens, film stock/color grade, style prefix]`
   - Be concise but comprehensive (100-300 words ideal)
   - Use positive framing — describe what you WANT, not what to avoid

5. **Updated Edit Instructions**
   - Generate a concise summary (1-2 lines) of what improvements the new prompt addresses
   - This will help track what changed and why

**OUTPUT FORMAT (JSON):**

```json
{{
  "updated_prompt": "Complete, detailed prompt for regenerating the image with all improvements...",
  "updated_edit_instructions": "Brief summary (1-2 lines) of key improvements made to address regeneration requirements",
  "reasoning": "Brief explanation of why these changes will address the regeneration requirements",
  "analysis": {{
    "issues_addressed": [
      "Issue 1 description",
      "Issue 2 description"
    ],
    "improvements_made": [
      "Improvement 1 description",
      "Improvement 2 description"
    ],
    "elements_preserved": [
      "Element 1 that was correct",
      "Element 2 that was correct"
    ]
  }}
}}
```

{"**PRODUCT SHOT — CRITICAL REQUIREMENT:**" + chr(10) + "This is a product shot. The regenerated prompt MUST ensure the PRODUCT is prominently visible, clearly in focus, and the central hero element of the scene. Explicitly describe the product's placement in the scene — position, surface contact, and how the scene lighting falls on it. Do NOT describe the product's visual appearance (the generator has the reference image). If the previous image failed because the product was missing or obscured, this is the primary issue to fix." + chr(10) if is_product_shot else ""}
Generate the updated prompt now. Focus on creating a prompt that will produce a significantly better image that addresses all the regeneration requirements."""

        return prompt
    
    def regenerate_prompt(
        self,
        shot_id: str,
        image_s3_url: str,
        older_prompt: str,
        edit_instructions: str,
        shot_metadata: Optional[Dict[str, Any]] = None,
        is_product_shot: bool = False
    ) -> RegenerationOutput:
        """
        Regenerate prompt for a shot based on current image and issues
        
        Args:
            shot_id: Shot identifier
            image_s3_url: S3 URL of the current image
            older_prompt: Original prompt that generated the current image
            edit_instructions: Edit/regeneration instructions from Agent 15
            shot_metadata: Optional additional shot metadata
            
        Returns:
            RegenerationOutput with updated prompt and instructions
        """
        logger.info(f"─"*60)
        logger.info(f"REGENERATING PROMPT: {shot_id}")
        logger.info(f"Image S3 URL: {image_s3_url}")
        logger.info(f"─"*60)
        
        try:
            # Download image from S3
            temp_image_path = self.download_image_from_s3(image_s3_url)
            
            if not temp_image_path:
                raise Exception("Failed to download image from S3")
            
            # Upload to Gemini
            uploaded_file = self._prepare_image_for_gemini(temp_image_path)
            
            # Build regeneration prompt
            regeneration_prompt = self._build_regeneration_prompt(
                older_prompt,
                edit_instructions,
                shot_id,
                shot_metadata,
                is_product_shot=is_product_shot
            )
            
            # Call Gemini vision API
            logger.info(f"📤 Analyzing image and generating updated prompt...")
            response = self.model.generate_content([regeneration_prompt, uploaded_file])
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean markdown formatting
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            regeneration_data = json.loads(response_text)
            
            # Create RegenerationOutput object
            output = RegenerationOutput(
                updated_prompt=regeneration_data['updated_prompt'],
                updated_edit_instructions=regeneration_data.get('updated_edit_instructions', edit_instructions),
                older_prompt=older_prompt,
                shot_id=shot_id,
                image_s3_url=image_s3_url,
                regeneration_timestamp=datetime.now().isoformat(),
                reasoning=regeneration_data.get('reasoning', ''),
                analysis=regeneration_data.get('analysis', {})
            )
            
            # Print summary
            logger.info(f"✓ Prompt regeneration completed")
            logger.info(f"  Updated prompt length: {len(output.updated_prompt)} chars")
            logger.info(f"  Reasoning: {output.reasoning[:100]}..." if output.reasoning else "  No reasoning provided")
            
            # Clean up temporary file
            try:
                os.unlink(temp_image_path)
            except:
                pass
            
            return output
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'N/A'}")
            
            # Create a fallback output
            return RegenerationOutput(
                updated_prompt=older_prompt,
                updated_edit_instructions=edit_instructions,
                older_prompt=older_prompt,
                shot_id=shot_id,
                image_s3_url=image_s3_url,
                regeneration_timestamp=datetime.now().isoformat(),
                reasoning="Fallback: Could not parse API response, using original prompt",
                analysis={"error": str(e)}
            )
            
        except Exception as e:
            logger.error(f"Prompt regeneration failed for {shot_id}: {e}")
            
            # Create a fallback output
            return RegenerationOutput(
                updated_prompt=older_prompt,
                updated_edit_instructions=edit_instructions,
                older_prompt=older_prompt,
                shot_id=shot_id,
                image_s3_url=image_s3_url,
                regeneration_timestamp=datetime.now().isoformat(),
                reasoning=f"Fallback: Regeneration failed with error: {e}",
                analysis={"error": str(e)}
            )
    
    def regenerate_prompts_batch(
        self,
        shots_to_regenerate: List[Dict[str, Any]],
        shot_designs: Optional[List[Dict]] = None,
        modified_prompts: Optional[List[Dict]] = None,
        product_shot_ids: Optional[set] = None
    ) -> List[RegenerationOutput]:
        """
        Regenerate prompts for multiple shots
        
        Args:
            shots_to_regenerate: List of shot data with regeneration requirements
                Each dict should have: shot_id, image_s3_url, older_prompt, edit_instructions
            shot_designs: Optional list of shot designs from Agent 12
            modified_prompts: Optional list of modified prompts from Agent 13
            
        Returns:
            List of RegenerationOutput objects
        """
        logger.info("="*60)
        logger.info("AGENT 15A: PROMPT REGENERATION STARTING")
        logger.info("="*60)
        logger.info(f"Shots to regenerate: {len(shots_to_regenerate)}")
        
        results = []
        
        for shot_data in shots_to_regenerate:
            shot_id = shot_data.get('shot_id')
            image_s3_url = shot_data.get('image_s3_url', '')
            older_prompt = shot_data.get('older_prompt', '')
            edit_instructions = shot_data.get('edit_instructions', '')
            
            if not shot_id or not image_s3_url or not older_prompt:
                logger.warning(f"Missing required data for {shot_id}, skipping")
                continue
            
            # Build shot metadata from optional sources
            shot_metadata = {}
            if shot_designs:
                shot_design = next((s for s in shot_designs if s.get('shot_id') == shot_id), {})
                if shot_design:
                    shot_metadata['shot_design'] = shot_design
            
            if modified_prompts:
                prompt_modifications = next((p for p in modified_prompts if p.get('shot_id') == shot_id), {})
                if prompt_modifications:
                    shot_metadata['prompt_modifications'] = prompt_modifications
            
            # Regenerate prompt
            result = self.regenerate_prompt(
                shot_id=shot_id,
                image_s3_url=image_s3_url,
                older_prompt=older_prompt,
                edit_instructions=edit_instructions,
                shot_metadata=shot_metadata if shot_metadata else None,
                is_product_shot=bool(product_shot_ids and shot_id in product_shot_ids)
            )
            
            results.append(result)
        
        logger.info("="*60)
        logger.info("AGENT 15A: PROMPT REGENERATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Successful regenerations: {len(results)}")
        
        return results


def save_results(results: List[RegenerationOutput], output_dir: str = "backend/services/production/app/services/phase_2_agents/outputs/agent_15A") -> str:
    """Save Agent 15A results to JSON file"""
    from pathlib import Path
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agent15A_regenerated_prompts_{timestamp}.json"

    # Calculate statistics
    successful_regenerations = len([r for r in results if r.updated_prompt != r.older_prompt])
    failed_regenerations = len([r for r in results if r.updated_prompt == r.older_prompt])

    output_data = {
        'agent': 'Agent 15A: Prompt Regeneration Agent',
        'timestamp': datetime.now().isoformat(),
        'regenerated_prompts': [asdict(r) for r in results],
        'statistics': {
            'total_shots_processed': len(results),
            'successful_regenerations': successful_regenerations,
            'failed_regenerations': failed_regenerations,
            'success_rate': f"{(successful_regenerations / len(results) * 100):.1f}%" if results else "0%"
        }
    }

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {filename}")
    print(f"  Total shots processed: {len(results)}")
    print(f"  Successful regenerations: {successful_regenerations}")
    print(f"  Failed regenerations: {failed_regenerations}")

    return filename


def main():
    """Example usage of Agent 15A"""
    logger.info("Agent 15A: Prompt Regeneration Agent")
    logger.info("Usage: Initialize agent and call regenerate_prompts_batch(shots_to_regenerate)")

if __name__ == "__main__":
    main()

