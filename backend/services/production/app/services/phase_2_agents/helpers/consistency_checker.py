"""
Scene Consistency Checker
Ensures visual and narrative consistency across shot sequences
"""

import logging
from typing import Dict, List, Optional
from .gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Checks scene consistency across multiple shots"""

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        self.client = gemini_client or GeminiClient(model_name="gemini-3.1-pro-preview")
        self.scene_context = {}  # Store established scene elements

    def check_shot_consistency(
        self,
        current_shot: Dict,
        previous_shots: List[Dict],
        scene_context: Optional[Dict] = None
    ) -> Dict:
        """
        Check if current shot is consistent with previous shots and scene context

        Args:
            current_shot: {
                "shot_id": str,
                "description": str,
                "prompt": str (if available)
            }
            previous_shots: List of previous shot dictionaries
            scene_context: Optional scene establishment info

        Returns:
            {
                "is_consistent": bool,
                "confidence": float,
                "inconsistencies": [{type, description, severity}, ...],
                "corrected_prompt": str (if needed),
                "consistency_notes": str
            }
        """

        system_prompt = """You are a film continuity supervisor specializing in visual consistency.
Your job is to identify inconsistencies between shots in a scene and ensure continuity.

Check for:

1. **Environmental Consistency**:
   - If scene is "empty park" it should stay empty
   - Weather/lighting should remain consistent
   - Background elements shouldn't appear/disappear randomly

2. **Character Consistency**:
   - Character appearance/clothing should match
   - Positions should make logical sense
   - DO NOT add characters for "continuity" if they are not in the shot description

3. **Temporal Consistency**:
   - Time of day should match
   - Actions should follow logical sequence

4. **Spatial Consistency**:
   - Objects/characters should maintain relative positions

Output JSON:
{
    "is_consistent": true/false,
    "confidence": 0.0-1.0,
    "inconsistencies": [
        {
            "type": "environmental|character|temporal|spatial",
            "description": "detailed description",
            "severity": "critical|warning|minor",
            "affected_element": "what element is inconsistent"
        }
    ],
    "corrected_prompt": "prompt with consistency fixes (null if no issues)",
    "consistency_notes": "overall assessment"
}"""

        # Build context from previous shots
        previous_context = self._build_shot_context(previous_shots)
        scene_info = self._format_scene_context(scene_context) if scene_context else "Not specified"

        user_prompt = f"""Analyze this shot for consistency issues:

**Scene Context**: {scene_info}

**Previous Shots**:
{previous_context}

**Current Shot**:
- Shot ID: {current_shot.get('shot_id')}
- Description: {current_shot.get('description')}
- Prompt: {current_shot.get('prompt', current_shot.get('description'))}

Identify any inconsistencies with the established scene."""

        try:
            response = self.client.call_sync(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="consistency_check",
                expect_json=True
            )
            return response

        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
            return {
                "is_consistent": True,  # Assume consistent if check fails
                "confidence": 0.5,
                "inconsistencies": [],
                "corrected_prompt": None,
                "consistency_notes": f"Check failed: {str(e)}"
            }

    def establish_scene_baseline(
        self,
        establishing_shot: Dict
    ) -> Dict:
        """
        Extract and store baseline scene elements from establishing shot

        Args:
            establishing_shot: First shot that establishes the scene

        Returns:
            {
                "environment": {...},
                "lighting": str,
                "characters_present": [...],
                "key_elements": [...]
            }
        """

        system_prompt = """Extract the key scene elements that establish the baseline for consistency.

These elements MUST remain consistent throughout the scene:
- Environmental state (empty/crowded, indoor/outdoor)
- Time of day / lighting conditions
- Weather conditions
- Key background elements
- Initially present characters

Output JSON:
{
    "environment": {
        "location": str,
        "state": "empty|crowded|moderate",
        "key_features": [...]
    },
    "lighting": {
        "time_of_day": str,
        "conditions": str
    },
    "characters_present": [...],
    "key_background_elements": [...],
    "mood": str
}"""

        user_prompt = f"""Extract baseline scene elements from this establishing shot:

**Shot Description**: {establishing_shot.get('description')}
**Shot Prompt**: {establishing_shot.get('prompt', establishing_shot.get('description'))}

What are the key elements that must stay consistent?"""

        try:
            response = self.client.call_sync(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stage="scene_baseline_extraction",
                expect_json=True
            )

            # Store scene context for future checks
            self.scene_context = response
            return response

        except Exception as e:
            logger.error(f"Scene baseline extraction failed: {e}")
            return {
                "environment": {},
                "lighting": {},
                "characters_present": [],
                "key_background_elements": [],
                "mood": "unknown"
            }

    def _build_shot_context(self, shots: List[Dict]) -> str:
        """Format previous shots for prompt"""
        if not shots:
            return "This is the first shot"

        context_lines = []
        for shot in shots[-3:]:  # Last 3 shots for context
            context_lines.append(
                f"Shot {shot.get('shot_id')}: {shot.get('description')}"
            )

        return "\n".join(context_lines)

    def _format_scene_context(self, context: Dict) -> str:
        """Format scene context dictionary to readable string"""
        lines = []

        if 'environment' in context:
            env = context['environment']
            lines.append(f"Environment: {env.get('location', 'unknown')}")
            lines.append(f"State: {env.get('state', 'unknown')}")

        if 'lighting' in context:
            light = context['lighting']
            lines.append(f"Time: {light.get('time_of_day', 'unknown')}")

        if 'characters_present' in context:
            chars = ', '.join(context['characters_present'])
            lines.append(f"Characters: {chars}")

        return "; ".join(lines) if lines else "Unknown scene context"

    def get_metrics_summary(self) -> Dict:
        """Get API call metrics"""
        return self.client.get_metrics_summary()

