#!/usr/bin/env python3
"""
Agent 4: Prompt Optimizer
==========================
Reviews and optimizes the initial prompts from Agent 3 to create final prompts.
Refines wording, adds missing details, and ensures maximum quality.

Flow:
1. Load initial prompts from Agent 3
2. Review and optimize each prompt
3. Create final_prompt with improvements
4. Human intervention: Review optimizations
5. Save final optimized prompts
"""

from google import genai
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Import Pydantic models from models.py
from backend.services.production.app.services.phase_1_agents.models import OptimizedPromptData

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)



class PromptOptimizerAgent:
    """
    Agent 4: Optimizes and refines prompts from Agent 3

    This agent reviews initial prompts and creates optimized final prompts
    with improved wording, additional details, and better structure.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-pro-preview", visual_style: str = None):
        """
        Initialize Prompt Optimizer Agent

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
            visual_style: Visual style for image generation (realistic, pixar, etc.)
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.visual_style = visual_style or "realistic"  # Default to realistic
        self.initial_prompts = {}
        self.optimized_prompts = {}
        self.human_feedback_log = []

    def load_agent3_output(self, agent3_output_path: str) -> None:
        """
        Load initial prompts from Agent 3 output file

        Args:
            agent3_output_path: Path to Agent 3 JSON output
        """
        with open(agent3_output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.initial_prompts = data.get('generated_prompts', {})

        logger.info(f"✓ Loaded Agent 3 prompts from: {agent3_output_path}")
        logger.info(f"   Characters: {len(self.initial_prompts.get('characters', []))}")
        logger.info(f"   Locations: {len(self.initial_prompts.get('locations', []))}")
        logger.info(f"   Props: {len(self.initial_prompts.get('props', []))}")

    def _get_style_instructions(self) -> str:
        """
        Get style-specific instructions based on visual_style setting

        Returns:
            String with style-specific guidance for prompt optimization
        """
        style_instructions = {
            "pixar": """
**PIXAR STYLE OPTIMIZATION REQUIREMENTS:**
- MANDATORY: Ensure prompt begins with "Pixar-style 3D animation" or "Disney Pixar style"
- REMOVE ALL: "realistic", "photorealistic", "hyperrealistic", "photography", "photograph", "DSLR" keywords
- Reinforce and enhance: smooth rounded shapes, vibrant colors, appealing character design
- Strengthen lighting keywords: soft volumetric lighting, warm color grading, cinematic lighting
- Improve rendering quality descriptions: high-quality 3D render, detailed textures, professional CGI
- For characters: enhance expressive eyes, appealing proportions, friendly design elements
- For environments: strengthen stylized yet detailed atmosphere, colorful and inviting mood
- Technical refinements: Pixar-quality rendering, 3D animation excellence, smooth professional shading
""",
            "2d": """
**2D ANIMATION STYLE OPTIMIZATION REQUIREMENTS:**
- MANDATORY: Ensure prompt begins with "2D animation style", "hand-drawn animation", or "traditional animation"
- REMOVE ALL: "realistic", "photorealistic", "hyperrealistic", "photography", "3D", "CGI" keywords
- Reinforce and enhance: clean line art, cel-shaded coloring, flat design with depth
- Strengthen art style keywords: animated illustration quality, cartoon style, expressive line work
- Improve visual quality descriptions: professional 2D animation, clean linework, vibrant cel colors
- For characters: enhance expressive poses, clear silhouettes, simplified yet detailed features
- For environments: strengthen illustrated backgrounds, painterly style, atmospheric perspective
- Technical refinements: 2D animation quality, traditional animation aesthetic, clean vector art style
""",
            "realistic": """
**REALISTIC STYLE OPTIMIZATION REQUIREMENTS (RAW PHOTOGRAPHY):**
- MANDATORY: Ensure prompt begins with "Raw, unretouched photograph" or "Candid documentary photo"
- REMOVE ALL: "photorealistic", "hyperrealistic", "8K", "masterpiece", "professional photography", "perfect", "cinematic", "3D", "render"
- Reinforce Biological Reality (Characters): explicitly mention "visible skin pores", "subtle peach fuzz", "natural skin oil/texture", "slight under-eye bags", "stray flyaway hairs", "asymmetrical features". DO NOT use "flawless" or "smooth skin".
- Reinforce Material Entropy (Props/Clothes): explicitly mention "micro-wrinkles", "stray lint", "subtle fabric wear", "dust specks", "fingerprints on smooth surfaces".
- Strengthen Optical Physics: Instead of "good lighting", use "ISO 800 sensor noise", "slight halation around highlights", "subtle chromatic aberration at the edges", "Kodak Portra 400 film grain".
- Camera Specs: "Shot on Sony FX3, 50mm f/1.8 lens", "ARRI Alexa Mini RAW capture".
- Technical refinements: The goal is gritty, unretouched reality, not a glossy magazine cover.
"""
        }

        return style_instructions.get(self.visual_style, style_instructions["realistic"])

    def _create_optimization_request(self, asset_name: str, asset_type: str, initial_prompt_data: Dict) -> str:
        """Create prompt for optimizing an initial prompt - uses centralized prompts"""

        # Import centralized prompts
        from backend.services.production.app.services.phase_1_agents.prompts import Agent4Prompts

        # Use centralized prompts from prompts.py
        base_prompt = Agent4Prompts.prompt_optimization(asset_name, asset_type, initial_prompt_data)

        # Add visual style instructions at the beginning
        style_instructions = self._get_style_instructions()

        prompt = f"""
**CRITICAL: VISUAL STYLE REQUIREMENT**
The optimized prompt MUST maintain the "{self.visual_style.upper()}" style from the initial prompt.
DO NOT change the style to realistic, photorealistic, or any other style unless that is the specified visual_style.
If the initial prompt incorrectly uses a different style, FIX IT to match the {self.visual_style} style.
{style_instructions}

{base_prompt}

**OUTPUT FORMAT (JSON):**
{{
    "asset_name": "{asset_name}",
    "asset_type": "{asset_type}",
    "optimization_analysis": {{
        "strengths": ["What works well in the initial prompt"],
        "improvements_needed": ["What could be enhanced"],
        "added_elements": ["New details or refinements added"]
    }},
    "final_prompt": {{
        "prompt": "Optimized final prompt (200-350 words, natural flowing language)...",
        "negative_prompt": "Enhanced negative prompt with comprehensive exclusions...",
        "technical_specs": {{
            "aspect_ratio": "Recommended aspect ratio",
            "camera_angle": "Optimal camera angle",
            "framing": "Framing description",
            "lighting": "Lighting details",
            "style_keywords": ["optimized", "keywords"]
        }},
        "recommended_settings": {{
            "model": "Best AI model recommendation",
            "steps": "Recommended steps",
            "guidance_scale": "Recommended guidance"
        }}
    }},
    "comparison": {{
        "initial_word_count": "count",
        "final_word_count": "count",
        "detail_level_improvement": "percentage or description",
        "key_changes": ["Major changes made"]
    }}
}}

**IMPORTANT:**
- CRITICAL: Maintain consistent {self.visual_style} style throughout the entire prompt

Generate the optimized prompt now.
"""
        return prompt

    def optimize_prompts(self) -> Dict[str, Any]:
        """
        Optimize all prompts using Gemini with structured output

        Returns:
            Dictionary containing all optimized prompts
        """
        if not self.initial_prompts:
            raise ValueError("No prompts loaded. Call load_agent3_output() first.")

        self.optimized_prompts = {
            "characters": [],
            "locations": [],
            "props": []
        }

        total_optimized = 0

        # Optimize character prompts
        for char_data in self.initial_prompts.get('characters', []):
            char_name = char_data.get('name') or char_data.get('character_name', 'Unknown')
            char_id = char_data.get('id')
            try:
                request_prompt = self._create_optimization_request(
                    char_name,
                    "character",
                    char_data.get('master_prompt', {})
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": OptimizedPromptData,
                    }
                )
                parsed_optimization: OptimizedPromptData = response.parsed
                optimized_data = parsed_optimization.model_dump()
                # Preserve UUID and name
                optimized_data['id'] = char_id
                optimized_data['name'] = char_name
                self.optimized_prompts['characters'].append(optimized_data)
                total_optimized += 1

            except Exception as e:
                logger.error(f"Error optimizing {char_name}: {e}")

        # Optimize location prompts
        for loc_data in self.initial_prompts.get('locations', []):
            loc_name = loc_data.get('name') or loc_data.get('location_name', 'Unknown')
            loc_id = loc_data.get('id')
            try:
                request_prompt = self._create_optimization_request(
                    loc_name,
                    "location",
                    loc_data.get('master_prompt', {})
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": OptimizedPromptData,
                    }
                )
                parsed_optimization: OptimizedPromptData = response.parsed
                optimized_data = parsed_optimization.model_dump()
                # Preserve UUID and name
                optimized_data['id'] = loc_id
                optimized_data['name'] = loc_name
                self.optimized_prompts['locations'].append(optimized_data)
                total_optimized += 1

            except Exception as e:
                logger.error(f"Error optimizing {loc_name}: {e}")

        # Optimize prop prompts
        for prop_data in self.initial_prompts.get('props', []):
            prop_name = prop_data.get('name') or prop_data.get('prop_name', 'Unknown')
            prop_id = prop_data.get('id')
            try:
                request_prompt = self._create_optimization_request(
                    prop_name,
                    "prop",
                    prop_data.get('master_prompt', {})
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=request_prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": OptimizedPromptData,
                    }
                )
                parsed_optimization: OptimizedPromptData = response.parsed
                optimized_data = parsed_optimization.model_dump()
                # Preserve UUID and name
                optimized_data['id'] = prop_id
                optimized_data['name'] = prop_name
                self.optimized_prompts['props'].append(optimized_data)
                total_optimized += 1

            except Exception as e:
                logger.error(f"Error optimizing {prop_name}: {e}")

        logger.info(f"✓ Optimized {total_optimized} prompts")

        return self.optimized_prompts

    def _print_optimization_summary(self) -> None:
        """Print summary of optimizations"""

        logger.info("\n" + "─"*60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("─"*60)

        total_optimized = 0

        for char_data in self.optimized_prompts.get('characters', []):
            char_name = char_data.get('name') or char_data.get('asset_name', 'Unknown')
            if 'final_prompt' in char_data:
                logger.info(f"\n🎭 {char_name}: Optimized")
                total_optimized += 1

        for loc_data in self.optimized_prompts.get('locations', []):
            loc_name = loc_data.get('name') or loc_data.get('asset_name', 'Unknown')
            if 'final_prompt' in loc_data:
                logger.info(f"🗺{loc_name}: Optimized")
                total_optimized += 1

        for prop_data in self.optimized_prompts.get('props', []):
            prop_name = prop_data.get('name') or prop_data.get('asset_name', 'Unknown')
            if 'final_prompt' in prop_data:
                logger.info(f"{prop_name}: Optimized")
                total_optimized += 1

        logger.info(f"\n✨ Total prompts optimized: {total_optimized}")

    def display_optimizations_for_review(self) -> None:
        """Display optimizations in readable format for human review"""

        logger.info("\n" + "="*60)
        logger.info("OPTIMIZED PROMPTS - HUMAN REVIEW REQUIRED")
        logger.info("="*60)

        # Display character optimizations
        for char_data in self.optimized_prompts.get('characters', []):
            char_name = char_data.get('name') or char_data.get('asset_name', 'Unknown')
            logger.info(f"\n" + "─"*60)
            logger.info(f"🎭 CHARACTER: {char_name}")
            logger.info("─"*60)

            analysis = char_data.get('optimization_analysis', {})
            logger.info(f"\n💪 STRENGTHS:")
            for strength in analysis.get('strengths', []):
                logger.info(f"   • {strength}")

            logger.info(f"\n🔧 IMPROVEMENTS MADE:")
            for improvement in analysis.get('improvements_needed', []):
                logger.info(f"   • {improvement}")

            logger.info(f"\n✨ ADDED ELEMENTS:")
            for element in analysis.get('added_elements', []):
                logger.info(f"   • {element}")

            final = char_data.get('final_prompt', {})
            logger.info(f"\nFINAL OPTIMIZED PROMPT:")
            logger.info(f"   {final.get('prompt', 'N/A')[:250]}...")

        # Display location optimizations
        for loc_data in self.optimized_prompts.get('locations', []):
            loc_name = loc_data.get('name') or loc_data.get('asset_name', 'Unknown')
            logger.info(f"\n" + "─"*60)
            logger.info(f"🗺LOCATION: {loc_name}")
            logger.info("─"*60)

            analysis = loc_data.get('optimization_analysis', {})
            logger.info(f"\n🔧 IMPROVEMENTS MADE:")
            for improvement in analysis.get('improvements_needed', []):
                logger.info(f"   • {improvement}")

            final = loc_data.get('final_prompt', {})
            logger.info(f"\nFINAL OPTIMIZED PROMPT:")
            logger.info(f"   {final.get('prompt', 'N/A')[:250]}...")

        # Display prop optimizations
        for prop_data in self.optimized_prompts.get('props', []):
            prop_name = prop_data.get('name') or prop_data.get('asset_name', 'Unknown')
            logger.info(f"\n" + "─"*60)
            logger.info(f"PROP: {prop_name}")
            logger.info("─"*60)

            analysis = prop_data.get('optimization_analysis', {})
            logger.info(f"\n🔧 IMPROVEMENTS MADE:")
            for improvement in analysis.get('improvements_needed', []):
                logger.info(f"   • {improvement}")

            final = prop_data.get('final_prompt', {})
            logger.info(f"\nFINAL OPTIMIZED PROMPT:")
            logger.info(f"   {final.get('prompt', 'N/A')[:250]}...")

        logger.info("\n" + "="*60)

    def request_human_feedback(self) -> Dict[str, Any]:
        """
        Request human feedback on optimized prompts

        Returns:
            Dictionary containing feedback request info
        """
        logger.info("\n" + "🤔 "*30)
        logger.info("HUMAN INTERVENTION CHECKPOINT - AGENT 4")
        logger.info("🤔 "*30)

        logger.info("\nPlease review the optimized prompts and provide feedback:")
        logger.info("\nEXPECTED FEEDBACK FORMAT:")
        print("""
{
    "approve_all": true/false,
    "final_modifications": {
        "ASSET_NAME": {
            "prompt": "Final custom prompt if needed...",
            "negative_prompt": "Final custom negative prompt...",
            "technical_specs": {...}
        }
    },
    "revert_to_initial": ["ASSET_NAME"],  // Assets where initial was better
    "general_feedback": "Overall assessment of optimizations"
}
        """)

        logger.info("\n💡 WHAT TO CHECK:")
        logger.info("  1. Are optimizations actually improvements?")
        logger.info("  2. Did any important details get lost?")
        logger.info("  3. Is the language natural and flowing?")
        logger.info("  4. Are negative prompts comprehensive?")
        logger.info("  5. Ready for production use?")

        logger.info("\n" + "="*60)
        logger.info("⏸AGENT PAUSED - Waiting for human feedback...")
        logger.info("="*60)

        return {
            "feedback_type": "pending",
            "message": "Human feedback required before finalizing prompts"
        }

    def apply_human_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Apply human feedback to finalize prompts

        Args:
            feedback: Dictionary containing human feedback
        """
        if not feedback:
            return

        # Log feedback
        self.human_feedback_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "Agent 4: Prompt Optimizer",
            "feedback": feedback
        })

        if feedback.get('approve_all'):
            return

        # Apply final modifications (now using UUID or name to find assets)
        modifications = feedback.get('final_modifications', {})
        for asset_identifier, new_data in modifications.items():
            # Find and update the asset (search by UUID or name)
            for asset_type in ['characters', 'locations', 'props']:
                for asset_data in self.optimized_prompts.get(asset_type, []):
                    if asset_data.get('id') == asset_identifier or asset_data.get('name') == asset_identifier:
                        asset_data['final_prompt'].update(new_data)

        # Revert any optimizations if needed
        revert_list = feedback.get('revert_to_initial', [])
        for asset_identifier in revert_list:
            for asset_type in ['characters', 'locations', 'props']:
                # Find in initial prompts
                initial_asset = None
                for init_asset in self.initial_prompts.get(asset_type, []):
                    if init_asset.get('id') == asset_identifier or init_asset.get('name') == asset_identifier:
                        initial_asset = init_asset
                        break

                if initial_asset:
                    # Find and update in optimized prompts
                    for opt_asset in self.optimized_prompts.get(asset_type, []):
                        if opt_asset.get('id') == asset_identifier or opt_asset.get('name') == asset_identifier:
                            opt_asset['final_prompt'] = initial_asset.get('master_prompt', {})

    def run_full_pipeline(self, agent3_output_path: str) -> Dict[str, Any]:
        """
        Run the complete Agent 4 pipeline

        Args:
            agent3_output_path: Path to Agent 3 output JSON

        Returns:
            Dictionary with optimized prompts and status
        """
        # Step 1: Load Agent 3 output
        self.load_agent3_output(agent3_output_path)

        # Step 2: Optimize prompts
        self.optimize_prompts()

        # Step 3: Display for human review
        self.display_optimizations_for_review()

        # Step 4: Request human feedback
        feedback_info = self.request_human_feedback()

        return {
            "status": "pending_human_review",
            "optimized_prompts": self.optimized_prompts,
            "feedback_request": feedback_info,
            "next_step": "Provide human feedback via apply_human_feedback() method, then save final prompts"
        }


def main():
    """Example usage of Agent 4"""

    # Initialize agent
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set")
    agent = PromptOptimizerAgent(api_key=api_key)

    logger.info("Agent 4: Prompt Optimizer initialized")
    logger.info("Use run_full_pipeline() with Agent 3 output path")


if __name__ == "__main__":
    main()
