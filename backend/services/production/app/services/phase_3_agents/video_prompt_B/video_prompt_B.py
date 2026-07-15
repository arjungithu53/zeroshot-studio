#!/usr/bin/env python3
"""
Agent 16: Multi-Shot Video Prompt Generator
============================================
Generates concise video prompts (<30 words) for multi-shot strategy shots,
ensuring visual consistency with previously established character/object keyframes.

Strategy: multi_shot
- Used when a character/object from a previous generate_new shot needs to be
  shown from a new angle or perspective
- Maintains visual consistency by referencing the established keyframe
- Focuses on the new perspective/action while keeping subject consistent

Flow:
1. Receive multi-shot description and reference shot information
2. Receive reference image path
3. Generate concise video prompt focusing on new perspective
4. Return prompt with metadata (<30 words)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import google.generativeai as genai

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)



@dataclass
class MultiShotPrompt:
    """Generated video prompt for a multi-shot"""
    shot_id: str
    reference_shot_id: str
    reference_image_path: str
    video_prompt: str
    word_count: int
    original_description: str
    character_or_object: str
    new_perspective: str
    timestamp: str


class MultiShotVideoGenerator:
    """
    Agent 16: Generates video prompts for multi-shot strategy

    Takes multi-shot description and reference image to create concise
    video prompts that maintain character/object consistency with their
    established keyframe images.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-pro-preview") -> None:
        """
        Initialize Multi-Shot Video Generator

        Args:
            api_key: Google API key for Gemini
            model_name: Gemini model to use
        """
        self.multi_shot_prompts: List[MultiShotPrompt] = []
        genai.configure(api_key=api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)

    def extract_key_elements(self, description: str) -> Tuple[str, str, str]:
        """
        Extract key elements from shot description

        Args:
            description: Shot description text

        Returns:
            Tuple of (character/object, action/perspective, setting)
        """
        desc_lower = description.lower()

        # Extract character/object from description
        character_keywords = {
            'puppy': 'puppy',
            'dog': 'dog',
            'squirrel': 'squirrel',
            'car': 'car',
            'wheel': 'wheel',
            'wizard': 'wizard',
            'staff': 'staff',
            'eagle': 'eagle'
        }

        character = 'subject'
        for keyword, name in character_keywords.items():
            if keyword in desc_lower:
                character = name
                break

        # Common perspective indicators
        perspective_keywords = {
            'close-up': 'close up view',
            'cu on': 'close up on',
            'medium shot': 'medium shot',
            'wide shot': 'wide shot',
            'pov': 'point of view',
            'looking up': 'looking upward',
            'looking down': 'looking down',
            'from above': 'overhead view',
            'from below': 'low angle view',
            'side view': 'profile view',
            'back': 'rear view'
        }

        perspective = 'shot'
        for keyword, desc in perspective_keywords.items():
            if keyword in desc_lower:
                perspective = desc
                break

        # Extract action verbs
        action_verbs = ['sitting', 'standing', 'running', 'watching', 'perched',
                       'looking', 'sniffing', 'wagging', 'tilting', 'spinning',
                       'expression changes', 'changes']
        action = ''
        for verb in action_verbs:
            if verb in desc_lower:
                action = verb
                break

        # Extract setting
        setting_keywords = ['park', 'lake', 'tree', 'branch', 'grass', 'ground']
        setting = 'scene'
        for keyword in setting_keywords:
            if keyword in desc_lower:
                setting = keyword
                break

        return character, f"{action} {perspective}".strip(), setting

    def generate_video_prompt(
        self,
        shot_description: str,
        reference_context: Optional[str] = None
    ) -> str:
        """
        Generate concise video prompt for multi-shot using Gemini API

        Args:
            shot_description: Description of the multi-shot
            reference_context: Optional context from reference shot

        Returns:
            Video prompt string (<30 words)
        """
        prompt_for_gemini = f"""You are a video prompt generator specializing in creating sequenced video prompts for a video generation AI. Your task is to combine multiple related shot descriptions into a single, cohesive prompt (under 30 words) that describes a short visual sequence, thoughtfully incorporating the most appropriate camera movements to enhance the visual narrative.

This multi_shot strategy is used when the input describes a sequence of views or actions involving the same character, object, or environment, typically transitioning between different angles, focal points, or using dynamic camera work. The generated prompt must describe the visual scene objectively, focusing on camera angles, subject actions, and camera movements, without explicitly stating a character's Point of View (POV).

Your generated prompt must describe the entire sequence, making explicit transitions using [CUT TO] where logical. Ensure absolute visual consistency of the main subject(s) across the described sequence.

**Instructions:**
1.  Given a set of related shot descriptions, combine them. The first description typically establishes the primary subject.
2.  Clearly delineate different views or actions within the sequence using [CUT TO].
3.  **Thoughtfully integrate specific camera movements when they best serve the visual storytelling and enhance flow. If a camera movement is already present in the input `shot_description`, *include it*. If no movement is specified but one would significantly improve the scene's impact, clarity, or transition, choose the most appropriate one from the following, considering standard filmmaking conventions:**
    *   **`Camera dollies into` / `Camera pushes in` / `Camera zooms into`:** Use for transitions from a wider shot to a closer view, especially for intricate details, a reveal, or increasing intensity/focus on a subject. "Pushes in" implies a more aggressive or direct forward movement.
    *   **`Camera dollies out` / `Camera pulls out` / `Camera zooms out`:** Use for transitions from a closer shot to a wider view, often to reveal context, establish scale, or convey a sense of departure/isolation. "Pulls out" implies a retreat or broadening perspective.
    *   **`Camera tracking` / `Camera follows`:** Use when the camera moves alongside a subject, maintaining a consistent distance, typically for action sequences or moving through an environment with the subject.
    *   **`Camera pans` / `Camera tilts`:** Use *within* a continuous shot when the camera rotates horizontally (`pans`) or vertically (`tilts`) from a fixed point, often to reveal an environment, follow a slow movement, or connect two elements. These are less common for direct [CUT TO] transitions unless setting up a new view.
    *   **`Camera cranes up/down` / `Jib movement`:** Consider for dramatic reveals, establishing grand scale, or complex vertical transitions.
    *   **Only include movements when they genuinely improve the shot's readability, impact, or narrative contribution within the word limit.**
4.  Maintain character, object, and environment consistency implicitly; do not explicitly state 'same appearance' or character POVs (e.g., 'Character A's POV'). Instead, describe the scene as a camera would observe it (e.g., 'Low angle shot of the subject').
5.  The total prompt length must be under 30 words.

**Dialogue Handling**: If the shot description includes dialogue, omit the quoted text. Instead, describe the visual action of the character speaking (e.g., 'conversing intensely', 'whispering', 'shouting').Dialogue Handling: Include the specific dialogue in quotation marks. Precede the dialogue with the character speaking and a verb describing their delivery (e.g., the rugged soldier yells, "Get down!"). Keep the visual description around the dialogue extremely brief to stay under the word limit.

**Character Identification**: When multiple characters are present, identify the speaker using a unique physical trait in 4-5 words (e.g., "the woman in the blue scarf"). if a character is non-distinct or impossible to define, refer to them as "the other character."

**Generic Examples:**

**Example 1 (Input with movement - preserve it):**
Input:
Previous Master Shot Context: "A large, ancient tree stands in a clearing."
Current Shot Description: "Camera pushes in to a close-up of the tree's gnarled roots, pulsating faintly."
Strategy: multi_shot
Output: "A large, ancient tree stands in a clearing. [CUT TO] Camera pushes in to gnarled roots, pulsating faintly."

**Example 2 (Input without movement - agent adds one):**
Input:
Previous Master Shot Context: "A hero leaps across a chasm."
Current Shot Description: "Wide shot of the hero landing gracefully on the far side."
Strategy: multi_shot
Output: "Hero leaps across chasm. [CUT TO] Wide shot, camera tracking hero landing gracefully on far side."

**Example 3 (Input without movement - agent adds one):**
Input:
Previous Master Shot Context: "A shimmering crystal orb rests on a velvet cushion."
Current Shot Description: "An extreme close-up of the orb's internal swirling lights."
Strategy: multi_shot
Output: "Shimmering crystal orb on velvet cushion. [CUT TO] Camera zooms into extreme close-up of its internal swirling lights."

**Example 4 (Input with movement - preserve it):**
Input:
Previous Master Shot Context: "A mysterious figure stands behind a wall."
Current Shot Description: "Camera cranes up, revealing a sprawling city beyond them."
Strategy: multi_shot
Output: "Mysterious figure behind wall. [CUT TO] Camera cranes up, revealing sprawling city beyond."

**Example 5 (Dialogue Handling):**
Input:
Previous Master Shot Context: "A detective approaches a suspect in a dimly lit room."
Current Shot Description: "A detective stands in the rain and says, 'I will find you, no matter what.'"
Strategy: multi_shot
Output: "Detective approaches suspect in dimly lit room. [CUT TO] Medium shot of a determined detective standing in the rain, speaking ominous threats into the void."
**YOUR TASK:**

Previous Master Shot Context (from generate_new shot): "{reference_context if reference_context else 'N/A'}"
Current Shot Description (multi_shot): "{shot_description}"
Strategy: multi_shot

**Output only the video prompt (under 50 words), nothing else:**"""
        try:
            response = self.model.generate_content(prompt_for_gemini)
            video_prompt = response.text.strip()

            # Ensure it ends with a period
            if not video_prompt.endswith('.'):
                video_prompt += '.'

            # Verify word count
            word_count = len(video_prompt.split())
            if word_count > 30:
                # Truncate to 30 words if needed
                words = video_prompt.split()
                video_prompt = ' '.join(words[:30]) + '.'
                logger.warning(f"   Gemini output was {word_count} words, truncated to 30")

            return video_prompt

        except Exception as e:
            logger.error(f"   Gemini API error: {e}")
            # Fallback to simple extraction
            return self._fallback_prompt_generation(shot_description)

    def _fallback_prompt_generation(self, shot_description: str) -> str:
        """
        Fallback prompt generation if Gemini API fails

        Args:
            shot_description: Description of the shot

        Returns:
            Simple video prompt
        """
        character, perspective, setting = self.extract_key_elements(shot_description)
        desc_lower = shot_description.lower()

        # Simple prompt generation
        prompt_parts = []

        if 'cu' in desc_lower or 'close-up' in desc_lower:
            prompt_parts.append(f"Close up on the {character}")
        elif 'medium' in desc_lower:
            prompt_parts.append(f"Medium shot of the {character}")
        elif 'wide' in desc_lower:
            prompt_parts.append(f"Wide shot of the {character}")
        else:
            prompt_parts.append(f"The {character}")

        # Add key action
        if 'pov' in desc_lower or 'looking' in desc_lower:
            prompt_parts.append("from point of view")
        elif 'watching' in desc_lower or 'alert' in desc_lower:
            prompt_parts.append("watching")
        elif 'perched' in desc_lower:
            prompt_parts.append("perched on branch")

        prompt = ", ".join(prompt_parts) + "."
        return prompt

    def generate_multi_shot_prompt(
        self,
        shot_id: str,
        shot_description: str,
        reference_shot_id: str,
        reference_image_path: str,
        reference_context: Optional[str] = None
    ) -> MultiShotPrompt:
        """
        Generate video prompt for a specific multi-shot

        Args:
            shot_id: ID of the multi-shot
            shot_description: Description of the multi-shot
            reference_shot_id: ID of the reference generate_new shot
            reference_image_path: Path to reference keyframe image
            reference_context: Optional context from reference shot

        Returns:
            MultiShotPrompt object with generated prompt
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Multi-Shot {shot_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Reference: {reference_shot_id}")
        logger.info(f"Image: {reference_image_path}")

        # Generate video prompt
        video_prompt = self.generate_video_prompt(shot_description, reference_context)
        word_count = len(video_prompt.split())

        logger.info(f"\nGenerated prompt ({word_count} words):")
        logger.info(f"   {video_prompt}")

        # Extract character and perspective
        character, perspective, _ = self.extract_key_elements(shot_description)

        # Create result object
        result = MultiShotPrompt(
            shot_id=shot_id,
            reference_shot_id=reference_shot_id,
            reference_image_path=reference_image_path,
            video_prompt=video_prompt,
            word_count=word_count,
            original_description=shot_description,
            character_or_object=character,
            new_perspective=perspective,
            timestamp=datetime.now().isoformat()
        )

        self.multi_shot_prompts.append(result)
        return result

    def _print_summary(self, results: List[MultiShotPrompt]) -> None:
        """Print summary of generated prompts"""
        logger.info("\n" + "─"*60)
        logger.info("GENERATION SUMMARY")
        logger.info("─"*60)

        logger.info(f"✨ Total video prompts generated: {len(results)}")

        if results:
            avg_words = sum(r.word_count for r in results) / len(results)
            logger.info(f"📏 Average prompt length: {avg_words:.1f} words")
            logger.info(f"✓ All prompts under 30 words: {all(r.word_count <= 30 for r in results)}")

    def save_results(self, output_dir: str = "phase_3_agents/output") -> str:
        """
        Save generated video prompts to JSON

        Args:
            output_dir: Directory to save output file

        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent17_multi_shot_prompts_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        output_data = {
            "agent": "Agent 17: Multi-Shot Video Prompt Generator (Prompt B)",
            "timestamp": datetime.now().isoformat(),
            "multi_shot_prompts": [asdict(p) for p in self.multi_shot_prompts],
            "statistics": {
                "total_prompts": len(self.multi_shot_prompts),
                "avg_word_count": sum(p.word_count for p in self.multi_shot_prompts) / len(self.multi_shot_prompts) if self.multi_shot_prompts else 0,
                "max_word_count": max((p.word_count for p in self.multi_shot_prompts), default=0),
                "min_word_count": min((p.word_count for p in self.multi_shot_prompts), default=0),
                "all_under_30_words": all(p.word_count <= 30 for p in self.multi_shot_prompts)
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Results saved to: {filepath}")
        return filepath


def main() -> None:
    """Example usage of Agent 16"""
    logger.info("Agent 16: Multi-Shot Video Prompt Generator")
    logger.info("\nUsage:")
    logger.info("  api_key = os.getenv('GOOGLE_API_KEY')")
    logger.info("  agent = MultiShotVideoGenerator(api_key=api_key)")
    logger.info("  result = agent.generate_multi_shot_prompt(")
    logger.info("      shot_id='1.5A',")
    logger.info("      shot_description='PUPPY POV of the squirrel',")
    logger.info("      reference_shot_id='1.4',")
    logger.info("      reference_image_path='path/to/squirrel_keyframe.png',")
    logger.info("      reference_context='Squirrel perched on branch watching'")
    logger.info("  )")
    logger.info("\n  Generated prompt will be under 30 words, using Gemini API")


if __name__ == "__main__":
    main()
