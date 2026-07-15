"""
Prompt Feasibility Checker
Uses Gemini to validate if a prompt is technically feasible to generate
with available image generation models and assets
"""

import logging
from typing import Dict, List, Optional
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class PromptFeasibilityChecker:
    """Checks if image prompts are technically feasible"""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.client = gemini_client or GeminiClient(model_name="gemini-3.1-pro-preview")

    def check_feasibility(
        self,
        prompt: str,
        shot_type: str,
        available_assets: List[Dict],
        model_type: str = "flux_with_ip_adapter"
    ) -> Dict:
        """
        Check if a prompt is feasible to generate

        Args:
            prompt: The image generation prompt
            shot_type: Type of shot (Wide Shot, CU, MCU, etc.)
            available_assets: List of available reference assets
            model_type: Target image generation model

        Returns:
            {
                "is_feasible": bool,
                "confidence": float (0-1),
                "issues": List[str],
                "suggestions": List[str],
                "modified_prompt": Optional[str]
            }
        """

        system_prompt = """You are an expert AI image generation advisor specializing in spatial composition and character placement.
Your job is to analyze image generation prompts and determine if they are technically feasible to create with modern
text-to-image models (like Imagen4, FLUX).

Consider these factors:
1. **Asset Availability**: Can the required angles/poses be achieved with available reference images?
2. **Model Capabilities**: Can the model handle the requested composition?
3. **Technical Constraints**: Are there impossible camera angles, physics violations, or composition issues?
4. **Consistency**: Will the output maintain character consistency with reference images?
5. **Spatial Placement**: Are character positions clearly defined?

Respond in JSON format with:
{
    "is_feasible": true/false,
    "confidence": 0.0-1.0,
    "issues": ["list of identified problems"],
    "suggestions": ["list of recommendations"],
    "modified_prompt": "improved prompt (null if original is perfect)"
}"""

        # Build asset description
        asset_descriptions = []
        for asset in available_assets:
            asset_type = asset.get('type', 'unknown')
            if asset_type == 'character':
                asset_descriptions.append(
                    f"- Character {asset['character']}: {asset['angle']} view (confidence: {asset['confidence']})"
                )
            elif asset_type == 'location':
                asset_descriptions.append(
                    f"- Location {asset['location']}: {asset['angle']} view (confidence: {asset['confidence']})"
                )
            elif asset_type == 'prop':
                asset_descriptions.append(
                    f"- Prop {asset['prop']}: {asset['angle']} view (confidence: {asset['confidence']})"
                )
            else:
                # Fallback for old format
                if 'character' in asset:
                    asset_descriptions.append(
                        f"- {asset['character']}: {asset['angle']} view (confidence: {asset['confidence']})"
                    )
        assets_text = "\n".join(asset_descriptions) if asset_descriptions else "No reference assets available"

        user_prompt = f"""Analyze this image generation request:

**Shot Type**: {shot_type}
**Prompt**: {prompt}
**Model Type**: {model_type}
**Available Reference Assets**:
{assets_text}

Is this prompt feasible to generate? Provide detailed analysis."""

        try:
            response = self.client.call_sync(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="prompt_feasibility_check",
                expect_json=True
            )
            return response

        except Exception as e:
            logger.error(f"Feasibility check failed: {e}")
            # Fallback response if API fails
            return {
                "is_feasible": True,  # Assume feasible if check fails
                "confidence": 0.5,
                "issues": [f"Feasibility check failed: {str(e)}"],
                "suggestions": [],
                "modified_prompt": None
            }

    def enhance_spatial_placement(
        self,
        prompt: str,
        shot_type: str,
        scene_context: Optional[str] = None
    ) -> Dict:
        """
        Enhance prompt with explicit spatial placement instructions

        Args:
            prompt: Original prompt
            shot_type: Camera shot type
            scene_context: Additional scene context

        Returns:
            {
                "enhanced_prompt": str,
                "placement_details": {
                    "character_positions": [...],
                    "environmental_anchors": [...],
                    "composition_notes": str
                }
            }
        """

        system_prompt = """You are a cinematography and composition expert working with Nano Banana (Gemini Flash Image). Add EXPLICIT spatial placement and physical grounding instructions to image generation prompts.

CRITICAL RULES:
1. **ONLY use environmental elements EXPLICITLY mentioned in the original prompt** — do not hallucinate new surfaces or locations
2. Characters MUST be physically grounded on the surface mentioned in the prompt — specify the contact point
3. Specify foreground/midground/background depth plane for each element
4. Describe how the existing light source in the scene falls on the character from their position
5. Preserve ALL existing cinematic language in the prompt (lens specs, film stock, lighting setups, style prefix) — only ADD, never remove

PHYSICAL GROUNDING LANGUAGE (use these patterns):
- Feet contact: "feet firmly planted on the [surface]", "boots pressing into the [material]"
- Body contact: "shoulder pressed against the [surface]", "back leaning against the [material]", "hand gripping the [object]"
- Seated: "seated on the [surface], weight resting on [material]"
- Crouched/action: "crouched low on the [surface], [body part] making contact with [material]"

LIGHTING INTERACTION (add based on character's position in the scene):
- "lit by the [existing light source described in prompt] from [direction], casting a [shadow description] on the [nearby surface]"
- "the same [light quality] that illuminates the [environment element] falls across the character from [direction]"
- This prevents the character from looking like they were composited onto the scene separately

Output JSON format:
{
    "enhanced_prompt": "detailed prompt with spatial placement, physical grounding, and lighting interaction added",
    "placement_details": {
        "character_positions": ["character: position description with surface contact"],
        "environmental_anchors": ["element: depth plane position"],
        "composition_notes": "overall composition strategy including foreground/midground/background breakdown"
    }
}"""

        context_text = f"\n**Scene Context**: {scene_context}" if scene_context else ""

        user_prompt = f"""Enhance this prompt with explicit spatial placement and physical grounding:

**Shot Type**: {shot_type}
**Original Prompt**: {prompt}{context_text}

Add physical grounding details (how characters make contact with surfaces), lighting interaction (how the scene's light falls on the character from their position), and foreground/midground/background depth placement. Preserve all existing cinematic language exactly."""

        try:
            response = self.client.call_sync(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="spatial_placement_enhancement",
                expect_json=True
            )
            return response

        except Exception as e:
            logger.error(f"Spatial enhancement failed: {e}")
            # Fallback: return original prompt
            return {
                "enhanced_prompt": prompt,
                "placement_details": {
                    "character_positions": [],
                    "environmental_anchors": [],
                    "composition_notes": f"Enhancement failed: {str(e)}"
                }
            }

    def get_metrics_summary(self) -> Dict:
        """Get API call metrics"""
        return self.client.get_metrics_summary()

