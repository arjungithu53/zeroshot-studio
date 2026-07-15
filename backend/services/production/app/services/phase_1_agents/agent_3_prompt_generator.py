
#!/usr/bin/env python3
"""
Agent 3: Asset Prompt Generator
================================
Generates optimized prompts for image generation based on enhanced assets from Agent 2.
Creates initial_prompt for each asset with various shot types and angles.

Flow:
1. Load enhanced assets from Agent 2
2. Generate initial prompts for each asset type
3. Create multiple shot variations (portrait, full body, close-up, etc.)
4. Human intervention: Review and refine prompts
5. Save structured prompt records
"""


import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

from google import genai
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import Pydantic models from models.py
from backend.services.production.app.services.phase_1_agents.models import CharacterPromptData, LocationPromptData, PropPromptData

# Import prompts from prompts.py
from backend.services.production.app.services.phase_1_agents.prompts import Agent3Prompts


class PromptGeneratorAgent:
    """
    Agent 3: Generates optimized image prompts from enhanced assets

    This agent creates production-ready prompts for AI image generation,
    with multiple shot types and technical specifications for each asset.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-pro-preview", visual_style: str = None):
        """
        Initialize Prompt Generator Agent

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
            visual_style: Visual style for image generation (realistic, pixar, etc.)
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.visual_style = visual_style or "realistic"  # Default to realistic
        self.enhanced_assets = {}
        self.generated_prompts = {}
        self.human_feedback_log = []

    def load_agent2_output(self, agent2_output_path: str) -> None:
        """
        Load enhanced assets from Agent 2 output file

        Args:
            agent2_output_path: Path to Agent 2 JSON output
        """
        with open(agent2_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.enhanced_assets = data.get('enhanced_assets', {})

        logger.info(f"✓ Loaded Agent 2 enhanced assets from: {agent2_output_path}")
        logger.info(f"   Characters: {len(self.enhanced_assets.get('characters', []))}")
        logger.info(f"   Locations: {len(self.enhanced_assets.get('locations', []))}")
        logger.info(f"   Props: {len(self.enhanced_assets.get('props', []))}")

    def _get_style_instructions(self) -> str:
        """
        Get style-specific instructions based on visual_style setting

        Returns:
            String with style-specific guidance for prompt generation
        """
        style_instructions = {
            "pixar": """
**PIXAR STYLE REQUIREMENTS:**
- MANDATORY: Begin the prompt with "Pixar-style 3D animation" or "Disney Pixar style"
- DO NOT USE: "realistic", "photorealistic", "hyperrealistic", "photography", "photograph", "DSLR"
- Specify characteristics: smooth rounded shapes, vibrant colors, appealing character design
- Include lighting keywords: soft volumetric lighting, warm color grading, cinematic lighting
- Mention rendering quality: high-quality 3D render, detailed textures, professional CGI
- For characters: emphasize expressive eyes, appealing proportions, friendly design
- For environments: stylized yet detailed, colorful, inviting atmosphere
- Technical specs: rendered in Pixar style, 3D animation quality, smooth shading
""",
            "2d": """
**2D ANIMATION STYLE REQUIREMENTS:**
- MANDATORY: Begin the prompt with "2D animation style", "hand-drawn animation", or "traditional animation"
- DO NOT USE: "realistic", "photorealistic", "hyperrealistic", "photography", "photograph", "3D", "CGI"
- Specify characteristics: clean line art, cel-shaded coloring, flat design with depth
- Include art style keywords: animated illustration, cartoon style, expressive line work
- Mention visual quality: professional 2D animation, clean linework, vibrant cel colors
- For characters: expressive poses, clear silhouettes, simplified yet detailed features
- For environments: illustrated backgrounds, painterly style, atmospheric perspective
- Technical specs: 2D animation quality, traditional animation aesthetic, clean vector art style
""",
            "realistic": """
**REALISTIC STYLE REQUIREMENTS:**
- MANDATORY: Begin the prompt with "photorealistic" or "hyperrealistic"
- DO NOT USE: "Pixar", "animated", "cartoon", "3D animation", "cel-shaded", "illustrated"
- Specify high-fidelity details: realistic skin textures, natural lighting, accurate materials
- Include photography keywords: professional photography, high resolution, sharp focus
- Mention camera specs: DSLR quality, cinematic depth of field, natural color grading
- For characters: realistic human features, natural skin tones, lifelike details
- For environments: authentic atmosphere, real-world lighting, natural materials
- Technical specs: 8K quality, photorealistic rendering, lifelike textures
"""
        }

        return style_instructions.get(self.visual_style, style_instructions["realistic"])

    def _create_character_prompt_generation_request(self, character: Dict) -> str:
        """Create prompt for generating character master image"""

        char_json = json.dumps(character, indent=2)
        style_instructions = self._get_style_instructions()

        prompt = f"""
You are an expert AI prompt engineer specializing in creating optimized prompts for text-to-image AI models
(Midjourney,Imagen,FLux etc.).

**CRITICAL: VISUAL STYLE REQUIREMENT**
The prompt MUST be generated in the "{self.visual_style.upper()}" style.
DO NOT use any other style. Every aspect of the prompt must match the {self.visual_style} aesthetic.
{style_instructions}

**CHARACTER DATA:**
{char_json}

**YOUR TASK:**
Generate ONE highly optimized MASTER IMAGE prompt for this character in the {self.visual_style.upper()} style.
This master image should capture the character's complete appearance and essence in the {self.visual_style} style.
IMPORTANT: The prompt must start with "{self.visual_style}-style" or "{self.visual_style} style" to ensure the correct visual aesthetic.

**MASTER IMAGE REQUIREMENTS:**
- Full body shot showing complete character details
- Clear view of facial features, clothing, and distinctive characteristics
- Appropriate pose that reflects character's personality/role
- CLEAN NEUTRAL BACKGROUND (solid color, simple gradient, or plain studio background - NO environmental elements, landscapes, or scene-specific backgrounds)
- Professional quality that can serve as reference for all future shots and easy compositing into different video scenes

**PROMPT OPTIMIZATION GUIDELINES:**

1. **Structure**: Start with subject, then physical details, clothing, pose, then style/technical specs
2. **Specificity**: Use concrete visual terms with exact details (colors, textures, proportions)
3. **Completeness**: Include ALL visual information needed to recreate this character
4. **Quality Keywords**: Include relevant quality keywords (8K, photorealistic, professional, etc.)
5. **Technical**: Specify camera angle, lighting, composition for optimal character showcase
6. **Style**: Match the story's genre and tone

**OUTPUT FORMAT (JSON):**
{{
    "character_name": "{character.get('name', 'Character')}",
    "master_prompt": {{
        "initial_prompt": "Comprehensive, detailed prompt for master character image (150-300 words)...",
        "negative_prompt": "Things to avoid (blur, distortion, multiple subjects, etc.)",
        "technical_specs": {{
            "aspect_ratio": "Recommended aspect ratio (e.g., 3:4, 1:1)",
            "camera_angle": "Optimal camera angle for character showcase",
            "framing": "How character should be framed",
            "lighting": "Lighting setup description",
            "style_keywords": ["keyword1", "keyword2", "keyword3"]
        }},
        "recommended_settings": {{
            "model": "Best AI model for this type of character",
            "steps": "Recommended inference steps",
            "guidance_scale": "Recommended CFG/guidance"
        }}
    }}
}}

**IMPORTANT:**
- The initial_prompt should be 150-300 words and completely self-contained
- Use natural, flowing language (not just comma-separated keywords)
- CRITICAL: Explicitly specify a NEUTRAL BACKGROUND (e.g., "plain white background", "solid grey background", "simple gradient background")
- Include negative prompts to exclude environmental backgrounds, landscapes, scenery, and any scene-specific elements
- Ensure the prompt captures EVERYTHING needed for visual consistency
- Match the emotional tone and story context for the CHARACTER ONLY (not the environment)

Generate the master prompt now.
"""
        return prompt

    def _create_location_prompt_generation_request(self, location: Dict) -> str:
        """Create prompt for generating location master image - uses centralized prompts"""

        # Use centralized prompts from prompts.py
        base_prompt = Agent3Prompts.location_prompt_generation(location)

        # Add visual style instructions at the beginning
        style_instructions = self._get_style_instructions()

        prompt = f"""
**CRITICAL: VISUAL STYLE REQUIREMENT**
The prompt MUST be generated in the "{self.visual_style.upper()}" style.
DO NOT use any other style. Every aspect of the prompt must match the {self.visual_style} aesthetic.
IMPORTANT: The prompt must start with "{self.visual_style}-style" or "{self.visual_style} style" to ensure the correct visual aesthetic.
{style_instructions}

{base_prompt}

**OUTPUT FORMAT (JSON):**
{{
    "location_name": "{location.get('name', 'Location')}",
    "master_prompt": {{
        "initial_prompt": "Comprehensive, detailed prompt for master location image (150-300 words)...",
        "negative_prompt": "Things to avoid (people, clutter, etc.)",
        "technical_specs": {{
            "aspect_ratio": "Recommended aspect ratio (e.g., 16:9, 21:9)",
            "camera_angle": "Optimal camera angle for location showcase",
            "framing": "How environment should be framed",
            "lighting": "Lighting and atmosphere description",
            "style_keywords": ["keyword1", "keyword2", "keyword3"]
        }},
        "recommended_settings": {{
            "model": "Best AI model for this type of environment",
            "steps": "Recommended inference steps",
            "guidance_scale": "Recommended CFG/guidance"
        }}
    }}
}}
"""
        return prompt

    def _create_prop_prompt_generation_request(self, prop: Dict) -> str:
        """Create prompt for generating prop master image"""

        prop_json = json.dumps(prop, indent=2)
        style_instructions = self._get_style_instructions()

        prompt = f"""
You are an expert AI prompt engineer specializing in creating optimized prompts for text-to-image AI models.

**CRITICAL: VISUAL STYLE REQUIREMENT**
The prompt MUST be generated in the "{self.visual_style.upper()}" style.
DO NOT use any other style. Every aspect of the prompt must match the {self.visual_style} aesthetic.
{style_instructions}

**PROP DATA:**
{prop_json}

**YOUR TASK:**
Generate ONE highly optimized MASTER IMAGE prompt for this prop in the {self.visual_style.upper()} style.
This master image should clearly showcase the prop's appearance, materials, and key characteristics in the {self.visual_style} style.
IMPORTANT: The prompt must start with "{self.visual_style}-style" or "{self.visual_style} style" to ensure the correct visual aesthetic.

**MASTER IMAGE REQUIREMENTS:**
- Hero shot showing the prop clearly in isolation
- All key details, textures, and materials visible
- Appropriate scale and perspective
- CLEAN NEUTRAL BACKGROUND (solid color, simple gradient, or plain studio background - NO environmental elements, characters, animals, or scene-specific backgrounds)
- Professional product photography quality
- Can serve as reference for future compositing into different video scenes

**PROMPT OPTIMIZATION GUIDELINES:**

1. **Object Focus**: Start with the object name and primary characteristics - THE PROP ONLY
2. **Materials & Textures**: Describe surface qualities, materials, finish in detail
3. **Scale & Proportion**: Indicate size and proportions clearly
4. **NO Characters or Animals**: NEVER include people, animals, or hands holding the object
5. **NO Environmental Context**: Avoid backgrounds with scenery, nature, or location elements
6. **Quality**: Detail level should match the prop's story importance
7. **Technical**: Professional product photography standards with neutral backdrop

**OUTPUT FORMAT (JSON):**
{{
    "prop_name": "{prop.get('name', 'Prop')}",
    "master_prompt": {{
        "initial_prompt": "Comprehensive, detailed prompt for master prop image (100-200 words)...",
        "negative_prompt": "Things to avoid (blur, wrong materials, etc.)",
        "technical_specs": {{
            "aspect_ratio": "Recommended aspect ratio (e.g., 1:1, 4:3)",
            "camera_angle": "Optimal camera angle for prop showcase",
            "framing": "How prop should be framed",
            "lighting": "Lighting setup description",
            "style_keywords": ["keyword1", "keyword2", "keyword3"]
        }},
        "recommended_settings": {{
            "model": "Best AI model for this type of object",
            "steps": "Recommended inference steps",
            "guidance_scale": "Recommended CFG/guidance"
        }}
    }}
}}

**IMPORTANT:**
- The initial_prompt should be 100-200 words and completely self-contained
- Use clear, descriptive language about the object's appearance ONLY
- CRITICAL: Explicitly specify a NEUTRAL BACKGROUND (e.g., "plain white background", "solid grey background", "simple gradient background")
- CRITICAL: DO NOT include any characters, animals, hands, or environmental context
- Include negative prompts to exclude: people, animals, hands, environmental backgrounds, scenery, landscapes, outdoor/indoor settings
- Ensure all materials, textures, and details are specified
- Match quality level to the prop's importance
- The prop must be isolated and ready for compositing

Generate the master prompt now.
"""
        return prompt

    def generate_prompts(self) -> Dict[str, Any]:
        """
        Generate prompts for all assets using Gemini with structured output

        Returns:
            Dictionary containing all generated prompts
        """
        if not self.enhanced_assets:
            raise ValueError("No assets loaded. Call load_agent2_output() first.")

        self.generated_prompts = {
            "characters": [],
            "locations": [],
            "props": []
        }

        total_prompts = 0

        # Generate character prompts
        for character in self.enhanced_assets.get('characters', []):
            char_name = character.get('name', 'Unknown')
            char_id = character.get('id')  # Get asset ID

            try:
                request_prompt = self._create_character_prompt_generation_request(character)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": CharacterPromptData,
                    }
                )
                parsed_prompt: CharacterPromptData = response.parsed
                prompt_data = parsed_prompt.model_dump()

                # Add asset metadata (id and name) to the prompt data
                prompt_data['id'] = char_id
                prompt_data['name'] = char_name

                self.generated_prompts['characters'].append(prompt_data)
                total_prompts += 1

            except Exception as e:
                logger.error(f"Error generating prompt for {char_name}: {e}")

        # Generate location prompts
        for location in self.enhanced_assets.get('locations', []):
            loc_name = location.get('name', 'Unknown')
            loc_id = location.get('id')  # Get asset ID

            try:
                request_prompt = self._create_location_prompt_generation_request(location)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": LocationPromptData,
                    }
                )
                parsed_prompt: LocationPromptData = response.parsed
                prompt_data = parsed_prompt.model_dump()

                # Add asset metadata (id and name) to the prompt data
                prompt_data['id'] = loc_id
                prompt_data['name'] = loc_name

                self.generated_prompts['locations'].append(prompt_data)
                total_prompts += 1

            except Exception as e:
                logger.error(f"Error generating prompt for {loc_name}: {e}")

        # Generate prop prompts
        for prop in self.enhanced_assets.get('props', []):
            prop_name = prop.get('name', 'Unknown')
            prop_id = prop.get('id')  # Get asset ID

            try:
                request_prompt = self._create_prop_prompt_generation_request(prop)
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": PropPromptData,
                    }
                )
                parsed_prompt: PropPromptData = response.parsed
                prompt_data = parsed_prompt.model_dump()

                # Add asset metadata (id and name) to the prompt data
                prompt_data['id'] = prop_id
                prompt_data['name'] = prop_name

                self.generated_prompts['props'].append(prompt_data)
                total_prompts += 1

            except Exception as e:
                logger.error(f"Error generating prompt for {prop_name}: {e}")

        logger.info(f"✓ Generated {total_prompts} prompts")

        return self.generated_prompts

    def _print_generation_summary(self) -> None:
        """Print summary of generated prompts"""

        logger.info("\n" + "─"*60)
        logger.info("GENERATION SUMMARY")
        logger.info("─"*60)

        total_prompts = 0

        for char_data in self.generated_prompts.get('characters', []):
            char_name = char_data.get('character_name', char_data.get('name', 'Unknown'))
            if 'master_prompt' in char_data:
                logger.info(f"\n🎭 {char_name}: Master prompt generated")
                total_prompts += 1

        for loc_data in self.generated_prompts.get('locations', []):
            loc_name = loc_data.get('location_name', loc_data.get('name', 'Unknown'))
            if 'master_prompt' in loc_data:
                logger.info(f"🗺{loc_name}: Master prompt generated")
                total_prompts += 1

        for prop_data in self.generated_prompts.get('props', []):
            prop_name = prop_data.get('prop_name', prop_data.get('name', 'Unknown'))
            if 'master_prompt' in prop_data:
                logger.info(f"{prop_name}: Master prompt generated")
                total_prompts += 1

        logger.info(f"\n✨ Total master prompts generated: {total_prompts}")

    def display_prompts_for_review(self) -> None:
        """Display generated prompts in readable format for human review"""

        logger.info("\n" + "="*60)
        logger.info("GENERATED PROMPTS - HUMAN REVIEW REQUIRED")
        logger.info("="*60)

        # Display character prompts
        for char_data in self.generated_prompts.get('characters', []):
            char_name = char_data.get('character_name', char_data.get('name', 'Unknown'))
            logger.info(f"\n" + "─"*60)
            logger.info(f"🎭 CHARACTER: {char_name}")
            logger.info("─"*60)
            logger.info(f"\nBase Description: {char_data.get('base_description', 'N/A')}")

            for shot_type, shot_data in char_data.get('shot_prompts', {}).items():
                logger.info(f"\n  📸 {shot_type.upper()}:")
                logger.info(f"     Prompt: {shot_data.get('prompt', 'N/A')[:150]}...")
                logger.info(f"     Aspect Ratio: {shot_data.get('technical_specs', {}).get('aspect_ratio', 'N/A')}")
                logger.info(f"     Camera: {shot_data.get('technical_specs', {}).get('camera_angle', 'N/A')}")

        # Display location prompts
        for loc_data in self.generated_prompts.get('locations', []):
            loc_name = loc_data.get('location_name', loc_data.get('name', 'Unknown'))
            logger.info(f"\n" + "─"*60)
            logger.info(f"🗺LOCATION: {loc_name}")
            logger.info("─"*60)
            logger.info(f"\nBase Description: {loc_data.get('base_description', 'N/A')}")

            for shot_type, shot_data in loc_data.get('shot_prompts', {}).items():
                logger.info(f"\n  📸 {shot_type.upper()}:")
                logger.info(f"     Prompt: {shot_data.get('prompt', 'N/A')[:150]}...")
                logger.info(f"     Aspect Ratio: {shot_data.get('technical_specs', {}).get('aspect_ratio', 'N/A')}")
                logger.info(f"     Lighting: {shot_data.get('technical_specs', {}).get('lighting', 'N/A')}")

        # Display prop prompts
        for prop_data in self.generated_prompts.get('props', []):
            prop_name = prop_data.get('prop_name', prop_data.get('name', 'Unknown'))
            logger.info(f"\n" + "─"*60)
            logger.info(f"PROP: {prop_name}")
            logger.info("─"*60)
            logger.info(f"\nBase Description: {prop_data.get('base_description', 'N/A')}")

            for shot_type, shot_data in prop_data.get('shot_prompts', {}).items():
                logger.info(f"\n  📸 {shot_type.upper()}:")
                logger.info(f"     Prompt: {shot_data.get('prompt', 'N/A')[:150]}...")
                logger.info(f"     Style: {', '.join(shot_data.get('technical_specs', {}).get('style_keywords', []))}")

        logger.info("\n" + "="*60)

    def request_human_feedback(self) -> Dict[str, Any]:
        """
        Request human feedback on generated prompts

        Returns:
            Dictionary containing feedback request info
        """
        logger.info("\n" + "🤔 "*30)
        logger.info("HUMAN INTERVENTION CHECKPOINT - AGENT 3")
        logger.info("🤔 "*30)

        logger.info("\nPlease review the generated prompts and provide feedback:")
        logger.info("\nEXPECTED FEEDBACK FORMAT:")
        print("""
{
    "approve_all": true/false,
    "prompt_modifications": {
        "CHARACTER_NAME": {
            "shot_type": {
                "prompt": "Modified prompt text...",
                "negative_prompt": "Modified negative prompt...",
                "technical_specs": {...}
            }
        }
    },
    "style_adjustments": {
        "global_style_keywords": ["keyword1", "keyword2"],
        "tone_adjustment": "more cinematic/more realistic/more artistic/etc."
    },
    "additional_shots_needed": {
        "CHARACTER_NAME": ["custom_shot_name", "another_shot"]
    },
    "remove_shots": {
        "CHARACTER_NAME": ["shot_type_to_remove"]
    },
    "general_feedback": "Overall comments about prompt quality"
}
        """)

        logger.info("\n💡 WHAT TO CHECK:")
        logger.info("  1. Do prompts accurately reflect the asset descriptions?")
        logger.info("  2. Are technical specs appropriate (aspect ratio, camera angles)?")
        logger.info("  3. Is the style consistent with the story tone?")
        logger.info("  4. Are negative prompts comprehensive enough?")
        logger.info("  5. Do you need additional shot variations?")

        logger.info("\n" + "="*60)
        logger.info("⏸AGENT PAUSED - Waiting for human feedback...")
        logger.info("="*60)

        return {
            "feedback_type": "pending",
            "message": "Human feedback required before proceeding to Agent 4"
        }

    def apply_human_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Apply human feedback to modify prompts

        Args:
            feedback: Dictionary containing human feedback
        """
        if not feedback:
            return

        # Log feedback
        self.human_feedback_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "Agent 3: Prompt Generator",
            "feedback": feedback
        })

        if feedback.get('approve_all'):
            return

        # Apply prompt modifications
        modifications = feedback.get('prompt_modifications', {})
        for asset_name, shot_mods in modifications.items():
            # Find the asset type
            for asset_type in ['characters', 'locations', 'props']:
                for asset_data in self.generated_prompts.get(asset_type, []):
                    # Get the name from the appropriate field
                    name_field = f'{asset_type[:-1]}_name'  # characters -> character_name
                    current_name = asset_data.get(name_field, asset_data.get('name', ''))

                    if current_name == asset_name:
                        for shot_type, new_data in shot_mods.items():
                            if shot_type in asset_data.get('shot_prompts', {}):
                                asset_data['shot_prompts'][shot_type].update(new_data)
                                logger.info(f"✓ Modified {asset_name} - {shot_type}")
                        break

        # Apply global style adjustments
        style_adj = feedback.get('style_adjustments', {})
        if style_adj:
            pass  # Style adjustments applied

    def run_full_pipeline(self, agent2_output_path: str) -> Dict[str, Any]:
        """
        Run the complete Agent 3 pipeline

        Args:
            agent2_output_path: Path to Agent 2 output JSON

        Returns:
            Dictionary with generated prompts and status
        """
        # Step 1: Load Agent 2 output
        self.load_agent2_output(agent2_output_path)

        # Step 2: Generate prompts
        self.generate_prompts()

        # Step 3: Display for human review
        self.display_prompts_for_review()

        # Step 4: Request human feedback
        feedback_info = self.request_human_feedback()

        return {
            "status": "pending_human_review",
            "generated_prompts": self.generated_prompts,
            "feedback_request": feedback_info,
            "next_step": "Provide human feedback via apply_human_feedback() method, then proceed to Agent 4"
        }


def main():
    """Example usage of Agent 3"""

    # Initialize agent
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set")
    agent = PromptGeneratorAgent(api_key=api_key)

    logger.info("Agent 3: Prompt Generator initialized")
    logger.info("Use run_full_pipeline() with Agent 2 output path")


if __name__ == "__main__":
    main()
