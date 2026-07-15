"""
Agent 13: Prompt Modifier Agent
Analyzes warnings from Agent 12, uses Gemini API to correct prompts for consistency/feasibility issues,
and re-selects the best available assets from the existing library.
"""


import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

import os
import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from bson import ObjectId

import google.generativeai as genai

from .helpers.asset_library import AssetLibrary, ANGLE_FALLBACKS


@dataclass
class WarningAnalysis:
    """Analysis of warnings from Agent 12"""
    warning_type: str  # consistency, ai_check, asset
    severity: str  # critical, warning, info
    description: str
    affected_elements: List[str]
    suggested_fix: str


@dataclass
class ModifiedShotOutput:
    """Output from Agent 13 for a modified shot"""
    shot_id: str
    original_prompt: Optional[str]
    corrected_prompt: str
    original_assets: List[Dict]
    corrected_assets: List[Dict]
    warnings_analyzed: List[WarningAnalysis]
    warnings_resolved: List[str]
    warnings_remaining: List[str]
    feasibility_score: float
    feasibility_change: float
    modifications_made: List[str]
    metadata: Dict


class PromptModifierAgent:
    """Agent 13: Analyzes warnings and corrects prompts using Gemini API"""

    def __init__(self, asset_library: AssetLibrary, api_key: str, model_name: str = "gemini-3.1-pro-preview", visual_style: str = None):
        self.asset_library = asset_library
        genai.configure(api_key=api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)
        self.scene_baseline = None
        self.shot_history: List[ModifiedShotOutput] = []
        self.visual_style = visual_style
        if not self.visual_style:
            raise ValueError("visual_style is required for PromptModifierAgent")
        logger.info(f"Agent 13 initialized with visual_style: {self.visual_style}")

    def _fetch_visual_style_from_movies(self, movie_id: str) -> str:
        """
        Fetch visual_style from movies collection using movie_id
        
        Args:
            movie_id: Movie ID (ObjectId string) to query
            
        Returns:
            Visual style string (must exist in the document)
        """
        try:
            from backend.services.production.app.config import get_mongo_factory
            from backend.shared.utils.mongodb_validators import validate_object_id
            from fastapi import HTTPException
            
            mongo_factory = get_mongo_factory()
            client, movies_collection = mongo_factory.get_collection("movies")
            
            try:
                movie_obj_id = validate_object_id(movie_id)
            except (ValueError, HTTPException) as e:
                logger.error(f"Invalid movie_id format: {e}")
                raise ValueError(f"Invalid movie_id format: {str(e)}") from e
                
            # Query movies collection by _id
            movie = movies_collection.find_one(
                {"_id": movie_obj_id},
                {"global_settings.visual_style": 1}
            )
            
            if movie:
                visual_style = movie.get("global_settings", {}).get("visual_style")
                logger.info(f"✓ Fetched visual_style from movies collection: {visual_style}")
                if not visual_style:
                    raise ValueError(f"Visual style missing in movies document for movie_id: {movie_id}")
                return visual_style
            raise ValueError(f"Movie not found with _id: {movie_id}")
                
        except Exception as e:
            logger.error(f"Error fetching visual_style from movies collection: {e}")
            raise

    def _get_style_instructions(self, visual_style: str) -> str:
        """
        Get style-specific instructions based on visual_style setting
        
        Args:
            visual_style: The visual style (realistic, pixar, anime, etc.)
            
        Returns:
            Style-specific instructions string
        """
        style_instructions = {
            "realistic": """
RAW PHOTOREALISTIC STYLE REQUIREMENTS:
- Ban "AI" words: Do NOT use "8K", "hyperrealistic", "masterpiece", "ethereal", or "perfect".
- Photographic Realism: Demand optical flaws. "ISO 800 sensor noise", "subtle film grain", "lens halation", "chromatic aberration".
- Biological/Material Entropy: Describe exact imperfect material properties: "worn full-grain leather with scuffs", "skin with visible pores and natural texture", "stray flyaway hairs", "lint and micro-wrinkles on fabric".
- Natural color grading: "Kodak Vision3 500T cinema color grade", "raw un-graded ARRI Log-C capture", "natural, slightly desaturated tones".
- Camera body context: "shot on ARRI Alexa Mini LF", "Sony FX3", "RED V-RAPTOR" for photographic credibility.
- Lighting specificity: Anchor lighting to physical reality — "practical fluorescent overhead casting unflattering downward shadows", "bounced window light with uneven falloff", "harsh practical streetlamp". Do NOT use "luminous golden glow".
- The corrected prompt MUST start with "Raw unretouched photograph" or "Candid cinematic still".
""",
            
            "pixar": """
PIXAR 3D ANIMATION STYLE REQUIREMENTS:
- Stylized 3D character designs with exaggerated features
- Smooth, polished surfaces with subtle subsurface scattering
- Vibrant, saturated color palette with warm tones
- Soft, indirect lighting with minimal harsh shadows
- Rounded, appealing shapes and proportions
- Expressive character poses and emotions
- Clean, detailed textures without photorealism
- The corrected prompt MUST start with "pixar-style" or "3D animated Pixar style"
""",
            
            "2d": """
2D ANIMATION STYLE REQUIREMENTS:
- Hand-drawn or digitally painted aesthetic
- Flat or limited shading with cel-shading techniques
- Bold outlines and clear silhouettes
- Simplified but expressive character designs
- Stylized backgrounds with artistic interpretation
- Limited color palette with intentional color choices
- Traditional animation composition and staging
- The corrected prompt MUST start with "2d-style" or "2D animated style"
""",
            
            "anime": """
ANIME STYLE REQUIREMENTS:
- Japanese animation aesthetic with distinctive character designs
- Large expressive eyes and stylized facial features
- Dynamic poses and action compositions
- Vibrant colors with cel-shading
- Detailed backgrounds with atmospheric depth
- Dramatic lighting and shadow effects
- Sharp lines and clean linework
- The corrected prompt MUST start with "anime-style" or "Japanese anime style"
"""
        }
        
        style_key = visual_style.lower()
        if style_key not in style_instructions:
            logger.warning(
                f"Unsupported visual_style '{visual_style}'. "
                f"Supported: {list(style_instructions.keys())}. Falling back to 'realistic'."
            )
            style_key = "realistic"
        return style_instructions[style_key]

    def analyze_warnings(self, shot_design_output: Dict) -> List[WarningAnalysis]:
        """Analyze warnings from Agent 12 output"""
        warnings = shot_design_output.get('warnings', [])
        analyzed = []

        for warning in warnings:
            analysis = self._parse_warning(warning, shot_design_output)
            if analysis:
                analyzed.append(analysis)

        return analyzed

    def _parse_warning(self, warning: str, shot_data: Dict) -> Optional[WarningAnalysis]:
        """Parse a warning string into structured analysis"""

        # Consistency warnings
        if '[Consistency-CRITICAL]' in warning:
            return self._parse_consistency_critical(warning, shot_data)
        elif '[Consistency-WARNING]' in warning:
            return self._parse_consistency_warning(warning, shot_data)
        elif '[Consistency]' in warning:
            return WarningAnalysis(
                warning_type='consistency',
                severity='info',
                description=warning.replace('[Consistency]', '').strip(),
                affected_elements=[],
                suggested_fix='No action needed'
            )

        # AI Check warnings
        elif '[AI Check]' in warning:
            return self._parse_ai_check(warning, shot_data)

        # Asset warnings
        elif 'asset' in warning.lower() or 'angle' in warning.lower():
            return self._parse_asset_warning(warning, shot_data)

        # Generic warning
        else:
            return WarningAnalysis(
                warning_type='generic',
                severity='warning',
                description=warning,
                affected_elements=[],
                suggested_fix='Review manually'
            )

    def _parse_consistency_critical(self, warning: str, shot_data: Dict) -> WarningAnalysis:
        """Parse critical consistency warnings"""
        clean_warning = warning.replace('[Consistency-CRITICAL]', '').strip()
        affected = []
        suggested_fix = ""

        # Detect specific issues
        if 'indoor' in clean_warning.lower() and 'outdoor' in clean_warning.lower():
            affected = ['environment', 'setting']
            suggested_fix = "Correct environment to match scene baseline"
        elif 'introduces a new character' in clean_warning.lower():
            affected = self._extract_characters_from_warning(clean_warning)
            suggested_fix = "Remove unexpected character or explain their presence"

        return WarningAnalysis(
            warning_type='consistency',
            severity='critical',
            description=clean_warning,
            affected_elements=affected,
            suggested_fix=suggested_fix
        )

    def _parse_consistency_warning(self, warning: str, shot_data: Dict) -> WarningAnalysis:
        """Parse non-critical consistency warnings"""
        clean_warning = warning.replace('[Consistency-WARNING]', '').strip()

        return WarningAnalysis(
            warning_type='consistency',
            severity='warning',
            description=clean_warning,
            affected_elements=['spatial_continuity'],
            suggested_fix="Establish spatial relationship with previous shots"
        )

    def _parse_ai_check(self, warning: str, shot_data: Dict) -> WarningAnalysis:
        """Parse AI check warnings"""
        clean_warning = warning.replace('[AI Check]', '').strip()
        affected = []
        suggested_fix = ""

        if 'Asset Mismatch' in clean_warning:
            affected = self._extract_asset_names(shot_data)
            suggested_fix = "Re-select correct character asset from library"
        elif 'Shot Type Contradiction' in clean_warning:
            affected = ['shot_type', 'framing']
            suggested_fix = "Adjust shot description to match shot type"

        return WarningAnalysis(
            warning_type='ai_check',
            severity='warning',
            description=clean_warning,
            affected_elements=affected,
            suggested_fix=suggested_fix
        )

    def _parse_asset_warning(self, warning: str, shot_data: Dict) -> WarningAnalysis:
        """Parse asset-related warnings"""
        severity = 'critical' if 'No suitable asset' in warning else 'warning'
        affected = self._extract_asset_names(shot_data)

        if 'fallback angle' in warning.lower():
            suggested_fix = "Accept fallback angle or adjust prompt"
        elif 'Angle changed' in warning:
            suggested_fix = "Find asset with matching angle if available"
        else:
            suggested_fix = "Review asset library for alternatives"

        return WarningAnalysis(
            warning_type='asset',
            severity=severity,
            description=warning,
            affected_elements=affected,
            suggested_fix=suggested_fix
        )

    def _extract_characters_from_warning(self, warning: str) -> List[str]:
        """
        Extract character names from warning text by matching against the asset library.
        Falls back to extracting any ALL_CAPS_WITH_UNDERSCORES tokens from the warning.
        """
        import re
        characters = []
        warning_lower = warning.lower()

        # First: match against known characters in the asset library
        if self.asset_library:
            for char_name in self.asset_library.get_available_characters():
                if char_name.lower().replace('_', ' ') in warning_lower:
                    characters.append(char_name)

        # Fallback: extract UPPER_CASE_TOKENS that look like asset names
        if not characters:
            tokens = re.findall(r'\b[A-Z][A-Z0-9_]{2,}\b', warning)
            characters = list(dict.fromkeys(tokens))  # deduplicate, preserve order

        return characters

    def _extract_asset_names(self, shot_data: Dict) -> List[str]:
        """Extract asset names from shot data"""
        assets = shot_data.get('selected_assets', [])
        names = []
        for asset in assets:
            name = asset.get('character') or asset.get('location') or asset.get('prop') or asset.get('name')
            if name:
                names.append(name)
        return names

    def modify_shot(self, shot_design_output: Dict, scene_context: Optional[Dict] = None, is_product_shot: bool = False) -> ModifiedShotOutput:
        """
        Main method to modify a shot based on warnings using Gemini API
        
        Args:
            shot_design_output: Output from Agent 12
            scene_context: Scene baseline context
            
        Returns:
            ModifiedShotOutput with corrected prompts and assets
        """

        # Establish baseline from first shot if not set
        if self.scene_baseline is None and scene_context:
            self.scene_baseline = scene_context

        # Analyze warnings
        warning_analyses = self.analyze_warnings(shot_design_output)

        # Separate by severity
        critical_warnings = [w for w in warning_analyses if w.severity == 'critical']
        other_warnings = [w for w in warning_analyses if w.severity != 'critical']

        # Get original prompt and data
        original_prompt = shot_design_output.get('prompt', '')
        original_description = shot_design_output.get('metadata', {}).get('original_description', '')
        
        if not original_prompt:
            original_prompt = original_description or "No prompt available"

        original_assets = shot_design_output.get('selected_assets', [])

        # Only process if there are actionable warnings
        if not critical_warnings and not other_warnings:
            # No warnings to fix, but still need to add locations/props if missing
            corrected_assets = self._reselect_assets(
                original_prompt,
                original_assets,
                shot_design_output,
                [],
                {},
                {},
                {}
            )

            return ModifiedShotOutput(
                shot_id=shot_design_output['shot_id'],
                original_prompt=original_prompt,
                corrected_prompt=original_prompt,
                original_assets=original_assets,
                corrected_assets=corrected_assets,
                warnings_analyzed=[],
                warnings_resolved=[],
                warnings_remaining=[],
                feasibility_score=shot_design_output.get('feasibility_score', 1.0),
                feasibility_change=0.0,
                modifications_made=[],
                metadata={
                    'generation_strategy': shot_design_output.get('generation_strategy'),
                    'original_feasibility': shot_design_output.get('feasibility_score', 1.0),
                    'critical_issues_count': 0,
                    'total_warnings': 0
                }
            )

        # Use Gemini to fix the prompt
        gemini_result = self._use_gemini_to_fix_prompt(
            original_prompt,
            warning_analyses,
            shot_design_output,
            self.scene_baseline,
            is_product_shot=is_product_shot
        )

        corrected_prompt = gemini_result['corrected_prompt']

        if is_product_shot and "product" not in corrected_prompt.lower():
            logger.warning(
                f"[Agent 13] Product shot {shot_design_output['shot_id']}: corrected_prompt "
                f"does not mention the product. Appending product placement reminder."
            )
            corrected_prompt = (
                corrected_prompt.rstrip()
                + " The PRODUCT must be prominently placed in the foreground, "
                "clearly visible, fully lit, and in sharp focus."
            )

        modifications_made = gemini_result['modifications']
        warnings_resolved = gemini_result['resolved_warnings']
        required_assets = gemini_result.get('required_assets', {})
        required_locations = gemini_result.get('required_locations', {})
        required_props = gemini_result.get('required_props', {})

        # Re-select assets based on Gemini's fixes
        corrected_assets = self._reselect_assets(
            corrected_prompt,
            original_assets,
            shot_design_output,
            warning_analyses,
            required_assets,
            required_locations,
            required_props
        )

        # Determine remaining warnings
        warnings_remaining = []
        for warning in warning_analyses:
            if warning.description not in warnings_resolved:
                warnings_remaining.append(warning.description)

        # Calculate feasibility improvement
        original_feasibility = shot_design_output.get('feasibility_score', 0.5)
        new_feasibility = self._calculate_new_feasibility(
            corrected_assets,
            warnings_remaining,
            len(warnings_resolved)
        )
        feasibility_change = new_feasibility - original_feasibility

        return ModifiedShotOutput(
            shot_id=shot_design_output['shot_id'],
            original_prompt=original_prompt,
            corrected_prompt=corrected_prompt,
            original_assets=original_assets,
            corrected_assets=corrected_assets,
            warnings_analyzed=warning_analyses,
            warnings_resolved=warnings_resolved,
            warnings_remaining=warnings_remaining,
            feasibility_score=new_feasibility,
            feasibility_change=feasibility_change,
            modifications_made=modifications_made,
            metadata={
                'generation_strategy': shot_design_output.get('generation_strategy'),
                'original_feasibility': original_feasibility,
                'critical_issues_count': len(critical_warnings),
                'total_warnings': len(warning_analyses)
            }
        )

    def _use_gemini_to_fix_prompt(self, original_prompt: str, warnings: List[WarningAnalysis],
                                   shot_data: Dict, scene_baseline: Optional[Dict],
                                   is_product_shot: bool = False) -> Dict:
        """Use Gemini API to intelligently fix the prompt based on warnings"""

        # Build context for Gemini
        warnings_text = "\n".join([
            f"- [{w.severity.upper()}] {w.warning_type}: {w.description}\n  Suggested fix: {w.suggested_fix}"
            for w in warnings
        ])

        scene_context = ""
        if scene_baseline:
            scene_context = f"""
**SCENE BASELINE (from first shot):**
- Environment: {scene_baseline.get('environment', 'N/A')}
- Characters: {', '.join(scene_baseline.get('characters', []))}
- Description: {scene_baseline.get('description', 'N/A')}
"""

        available_assets_info = self._get_available_assets_info(shot_data)

        # Get original description to understand intent
        original_description = shot_data['metadata'].get('original_description', '')

        # Determine if this is introducing new characters
        scene_chars = scene_baseline.get('characters', []) if scene_baseline else []
        current_chars = shot_data['metadata'].get('characters_found', [])
        new_characters = [c for c in current_chars if c not in scene_chars]

        character_context = ""
        if new_characters:
            character_context = f"""
**CHARACTER INTRODUCTION:**
This shot introduces NEW characters: {', '.join(new_characters)}
This is acceptable for story progression. The "empty park" baseline refers to the absence of crowds/people, not animals.
"""

        # Get visual style instructions
        style_instructions = self._get_style_instructions(self.visual_style)

        visual_style_context = f"""
**VISUAL STYLE REQUIREMENT:**
Project Visual Style: {self.visual_style.upper()}

{style_instructions}

CRITICAL: Your corrected_prompt MUST start with "{self.visual_style}-style" to ensure the image generator uses the correct aesthetic.
Example start: "{self.visual_style}-style [rest of your prompt here]"
"""

        product_context = ""
        if is_product_shot:
            product_context = """
**PRODUCT SHOT — CRITICAL RULES:**
This shot requires a PRODUCT to appear in the final image. The product is the hero element.
- The corrected_prompt MUST explicitly describe the product's placement in the scene (e.g., "the PRODUCT rests on the wooden table in the foreground, fully lit by the scene light, clearly visible and in sharp focus").
- DO NOT describe the product's visual appearance — the image generator receives a reference image of the product directly. Just describe WHERE it is placed, HOW it interacts with the scene lighting, and that it must be PROMINENTLY VISIBLE.
- Never write a corrected_prompt for this shot that omits the product or places it out-of-frame.
- In required_assets, include an entry for PRODUCT with role "foreground hero element".
"""

        prompt_for_gemini = f"""
You are an expert AI Cinematographer, Director of Photography, and Prompt Engineer. Your primary directive is to ensure every shot in a sequence is visually coherent, emotionally resonant, and technically feasible for generation. You will analyze prompts, but your ultimate loyalty is to the **Original Shot Description (The Intent)**, not the text of the current prompt you are given.

{visual_style_context}
{product_context}
#### Guiding Principles: The Director's Logic

1. **The Shot List is Truth:** The `original_description` is the director's vision. If the `CURRENT PROMPT` contradicts it in framing, subject, or action, the `CURRENT PROMPT` is wrong and must be either heavily corrected or discarded entirely.
2. **Specificity Over Ambiguity:** Your goal is to leave no room for AI misinterpretation. You will lock down variables like camera angle, lens type, and lighting to guarantee the desired outcome.
3. **Narrative and Emotional Continuity is Paramount:** A shot is not just an image; it's a beat in a story. Your prompts must connect to the previous shot and advance the emotional arc. You will describe actions and expressions, not just static scenes.
4. The Principle of Holistic Scene Description: While specificity is key, your prompt must describe a single, unified scene, not a collection of separate elements. To avoid the "composite assembly" issue where characters look pasted onto a background, you must explicitly describe the interactions between subjects and their environment.
Describe Shared Lighting: Explicitly state how the same light source affects all elements in the frame, and detail the characteristics of that light. For example: "The same harsh Mediterranean sun that bakes the terracotta tiles also casts a sharp highlight on the puppy's fur and the squirrel's back. Lighting: strong, direct sunlight creates dramatic shadows and highlights, ensuring natural and consistent illumination."
Describe Physical Interaction: Detail the physical connection between characters and the environment. For example: "The puppy and squirrel are firmly planted on the roof, their weight subtly pressing on the tiles, casting short, dark shadows directly beneath them consistent with the high-angle sun."Detail the physical connection between characters and the environment, especially when an action originates from or directly involves an object. For example: "The puppy and squirrel are firmly planted on the roof, their weight subtly pressing on the tiles, casting short, dark shadows directly beneath them consistent with the high-angle sun." Or, for dynamic interaction: "The puppy launches itself from the surface of the slightly dented satellite dish, its paws pushing off with visible force, sending grit and debris scattering from the dish's surface."
Describe Spatial Relationships: Use language that binds the characters together within the scene. For example: "The puppy and squirrel sit side-by-side, sharing the same focal plane..."
Foreground Subject Integration (CRITICAL FOCUS) : Explicitly ensure foreground subjects are sharp, detailed, and visually integrated into the environment's lighting and perspective, matching the surrounding elements' quality. For example: "The black lab puppy and feisty squirrel, rendered with exceptional detail, are bathed in the same harsh sunlight, their fur and features sharply defined against the sun-baked tiles."
5. **Prioritize Action and Specific Focus (CRITICAL PRINCIPLE):** When an original_description details a specific action (e.g., 'skidding', 'climbing', 'leaping') or demands an extreme close-up on a particular element with no background, these directives are paramount. The prompt must vividly describe the peak moment of that action and rigorously exclude any elements not explicitly requested for extreme close-ups, ensuring the AI cannot misinterpret the subject or the required level of detail. Focus on the consequences and visual cues of the action (e.g., kicked-up dust, strained muscles, determined gaze).When an original_description details a specific action (e.g., 'skidding', 'climbing', 'leaping') or demands an extreme close-up on a particular element with no background, these directives are paramount. The prompt MUST vividly describe the peak moment of that action, including the precise physical mechanics (e.g., 'paws splayed wide, claws extended, desperately scraping'), its immediate visual consequences (e.g., 'grit and dust explode outwards', 'backlit spray'), and rigorously exclude any elements not explicitly requested for extreme close-ups. Ensure the AI cannot misinterpret the subject or the required level of detail. Focus on the consequences and visual cues of the action (e.g., kicked-up dust, strained muscles, determined gaze).

**FRAMING CONFLICT RESOLUTION (CRITICAL — READ BEFORE WRITING ANY PROMPT):**
Principles 4 and 5 are MUTUALLY EXCLUSIVE based on the shot's framing type. You MUST apply only one:

- If the `original_description` requires an **extreme close-up, macro shot, detail shot, or tight single-element focus**:
  → **Principle 5 WINS. Principle 4 does NOT apply.**
  → Do NOT add full-body spatial context, standing posture, feet-on-floor grounding, or spatial relationships to the environment.
  → Do NOT describe where the subject is standing, their full body position, or how they connect to the wider environment.
  → Describe ONLY the single element being filmed in close-up (e.g., "the lip," "the finger," "the eye").
  → A close-up prompt that also describes full-body standing = a contradictory prompt that will ALWAYS fail generation.

- If the `original_description` requires a **wide shot, full-body shot, establishing shot, or environmental shot**:
  → **Principle 4 WINS.** Apply all spatial grounding, physical interaction, and scene integration as normal.

**NEVER write a prompt that simultaneously describes extreme close-up framing AND full-body spatial positioning. These are physically incompatible and will cause image generation failure.**


---

**YOUR TASK: A Step-by-Step Directorial Review**

You will follow this precise workflow for every shot:

**Step 1: Establish Intent.**
Read the `ORIGINAL SHOT DESCRIPTION` first. This is your mission. Understand the required framing (CU, WS, etc.), the characters involved, the key action, and the emotional purpose of the shot.

**Step 2: Triage the `CURRENT PROMPT`.**
Compare the `CURRENT PROMPT` against the Intent from Step 1.
* **Is it salvageable?** (e.g., correct framing and subject, but wrong environment).
* **Is it fundamentally flawed?** (e.g., describes a close-up when the intent is a wide shot).
* **Is it missing or nonsensical?**

**Step 3: Execute Your Path.**

* **PATH A: The Prompt is Salvageable (Correction & Enhancement)**
    1. Address all `IDENTIFIED ISSUES` (warnings) by making targeted edits.
    2. **Enhance with Cinematic Language:** Inject specific, professional terms to add precision.
        * **Camera:** Is it `static`, `handheld`, `panning`? What is the `camera level` (eye-level, low-angle, high-angle)?
        * **Lens:** What lens achieves the desired look? `Wide-angle lens (24mm)` for expansive shots, `Telephoto lens (85mm)` for portraits with compressed, blurry backgrounds (bokeh). `Macro lens (100mm)` for extreme close-up detail.
        * **Lighting:** Be specific — name the setup and direction. `Soft golden-hour sidelight from camera-left casting long shadows`, `hard overhead fluorescent with downward shadows and cool color temperature`, `overcast daylight with flat diffuse fill and no directional shadows`, `dramatic practical backlight creating a rim highlight on the subject's shoulder`.
        * **Materiality (for REALISTIC style):** Replace generic material references with specific physical descriptions — NOT "wooden table" but "aged pine table with deep grain lines and ring stains from years of use"; NOT "metal door" but "galvanized steel door with surface rust blooms at the lower corners and scuffed paint at handle height".
        * **Film Stock / Color Grade (for REALISTIC style):** Ground the image in photographic reality — "Kodak Vision3 500T color grade with warm shadows and natural grain", "ARRI Log-C color science with muted highlights", or "clean digital neutral with accurate color rendering".
        * **Physical Grounding:** Describe how subjects physically connect to the environment to prevent the composited look — "feet planted firmly on the wet cobblestones", "shoulder pressed against the rough brick surface", "both hands gripping the rusted railing".
    3. Follow the "MINIMAL CHANGES" principle *only* for the parts of the prompt that are already correct.

* **PATH B: The Prompt is Flawed or Missing (Full Generation)**
    1. **Discard the `CURRENT PROMPT` entirely.** Do not try to fix it.
    2. **Generate a new prompt from scratch** based *only* on the `ORIGINAL SHOT DESCRIPTION` and the established `SCENE CONTEXT`.
    3. Build this new prompt using all the best practices from "Enhance with Cinematic Language" above — camera, lens, lighting direction/quality, materiality (for realistic style), film stock/color grade (for realistic style), and physical grounding.
    4. Structure the new prompt as: `[Camera framing & angle] → [Subject position & action] → [Physical grounding in environment] → [Shared lighting description] → [Atmosphere & mood] → [Technical/style specs]`
    5. Explicitly describe character actions, expressions, and emotional transitions (e.g., "Its sad expression softens, its ears twitch and lift with nascent curiosity").
    6. Weave in continuity links (e.g., "the squirrel from Shot 1.4," "its gaze directed towards the puppy's location").

**Step 4: Handle Complex Sequences.**
If the `ORIGINAL SHOT DESCRIPTION` describes a multi-part sequence (like a "Shot/Reverse-Shot" or "Generate from Last Frame"), you MUST break it down into multiple, clearly labeled prompts within the `corrected_prompt` field (e.g., "PROMPT A: [First part of action]. PROMPT B: [Second part of action].").

**Step 5: Define Asset Requirements.**
Based on your final corrected/generated prompt, specify the `needed_angle` for each character asset required to film the shot correctly. **Choose from the existing asset angles only:**
- **close_up**: frontal/face view, emotional close-ups, facial expressions
- **wide_shot**: full body view, front-facing, establishing shots, showing full character
- **profile_left**: left side view of character
- **profile_right**: right side view of character
- **back_shot**: rear view, from behind, over-the-shoulder perspectives

**CRITICAL MATCHING RULES:**
- If the shot description says "back shot", "from behind", "rear view", "over-the-shoulder" → MUST use **back_shot**
- If the shot description says "close up", "face", "facial expression", "eyes" → MUST use **close_up**
- If the shot description says "wide", "full body", "establishing", "entire character" → MUST use **wide_shot**
- If the shot description says "side view", "profile", "left side" → MUST use **profile_left**
- If the shot description says "side view", "profile", "right side" → MUST use **profile_right**

Justify your choice with a clear reason based on the shot composition and camera angle.

**Step 6: Identify Required Locations and Props.**
Analyze the corrected prompt and identify ALL locations and props that should be present in the shot. For each one:
- Extract the exact name or create a descriptive name (e.g., "TERRACOTTA_ROOFTOPS", "WEATHERED_DRAINPIPE", "SATELLITE_DISH")
- Specify the role it plays in the shot (background, foreground, interaction point)
- This is CRITICAL for proper scene composition

---

**SCENE CONTEXT:**
{scene_context}
{character_context}

**SHOT INFORMATION:**
- Shot ID: {shot_data['shot_id']}
- Generation Strategy: {shot_data.get('generation_strategy', 'N/A')}
- Required Camera Angle: {shot_data['metadata'].get('required_angle', 'N/A')}

**ORIGINAL SHOT DESCRIPTION (The Intent):**
{original_description}

**CURRENT PROMPT (May have issues):**
{original_prompt}

**AVAILABLE ASSETS IN LIBRARY:**
{available_assets_info}

**Asset Angle Definitions:**
- **close_up**: frontal/face view - use for emotional moments, facial expressions, reactions
- **wide_shot**: full body, front-facing - use for establishing shots, showing full character and environment
- **back_shot**: rear view, from behind - use for over-the-shoulder shots, shots from behind character
- **profile_left**: left side view - use for profile shots facing left
- **profile_right**: right side view - use for profile shots facing right
- **master**: neutral reference (fallback only)

**IDENTIFIED ISSUES:**
{warnings_text}

---

**OUTPUT FORMAT (JSON):**
{{
    "corrected_prompt": "{self.visual_style}-style [The fully corrected prompt with cinematic language starting with the style prefix]...",
    "modifications_made": [
        "Specific change 1 with reasoning",
        "Specific change 2 with reasoning"
    ],
    "resolved_warnings": [
        "Warning text that was resolved"
    ],
    "required_assets": {{
        "CHARACTER_NAME": {{
            "needed_angle": "one of: close_up, wide_shot, profile_left, profile_right, back_shot",
            "reason": "Detailed explanation of why this specific angle is needed for the shot composition"
        }}
    }},
    "required_locations": {{
        "LOCATION_NAME": {{
            "role": "background/foreground/setting",
            "description": "How this location appears in the shot"
        }}
    }},
    "required_props": {{
        "PROP_NAME": {{
            "role": "interaction/background/foreground",
            "description": "How this prop is used in the shot"
        }}
    }},
    "explanation": "Your complete reasoning process: which path you chose (A or B), why, and how you enhanced the prompt"
}}

---

**CRITICAL GUIDELINES:**

* **PRIORITIZE THE ORIGINAL DESCRIPTION:** This is the most important rule. It overrides everything else.
* **LOCK DOWN VARIABLES:** Never assume the generator knows the best camera height, lens, or lighting. Specify it.
* **DESCRIBE ACTION & EMOTION:** Turn static descriptions into dynamic, emotional beats. Instead of "sad face," write "Its large, glistening eyes convey deep sadness, its mouth trembles in a soft whimper."
* **BUILD EXPLICIT CONTINUITY:** Use phrases that link this shot to others in the scene.
* **DISCARD AND REBUILD WHEN NECESSARY:** Do not waste time trying to fix a prompt that is fundamentally wrong. Your job is to produce the *correct* prompt, not just a *corrected* one.
* **DO NOT ADD CHARACTERS FOR CONTINUITY:** NEVER add characters to a shot that are not explicitly mentioned in the ORIGINAL SHOT DESCRIPTION. If the original shot description only mentions a squirrel, DO NOT add the puppy to the frame for "continuity" or "spatial relationship." Each shot should only include the characters specified in its original description. Continuity is maintained through editing and sequencing, not by forcing characters into every frame.
* **REMOVE CAMERA MOVEMENT INSTRUCTIONS:** This is for STATIC IMAGE generation, NOT video. REMOVE all camera movement terms like "pushing in," "pulling back," "panning," "tracking," "dolly," "zoom," etc. Instead, describe the FINAL FRAME COMPOSITION. Camera position should be "static" with specified angle (eye-level, low-angle, high-angle). Focus on what the final image looks like, not how the camera gets there.
* **CHOOSE ASSETS FROM EXISTING ANGLES ONLY:** You MUST select from: close_up, wide_shot, profile_left, profile_right, back_shot. Match the angle to the shot composition:
  - "back shot", "from behind", "over-the-shoulder", "rear view", "following from behind" → **back_shot**
  - "close up", "face", "frontal", "looking at camera", "emotional expression", "eyes", "facial detail" → **close_up**
  - "wide shot", "full body", "establishing", "wide view", "entire character" → **wide_shot**
  - "side view", "profile", "left side" → **profile_left**
  - "side view", "profile", "right side" → **profile_right**
* **ALWAYS IDENTIFY LOCATIONS AND PROPS:** Every shot occurs somewhere and may involve objects. Always include `required_locations` and `required_props` in your output, even if they're empty objects. Extract these from the scene description dynamically.

---

**EXAMPLE OF GOOD FIXES:**

**Example: Handling a Missing/Flawed Prompt (Path B)**

* **Shot ID:** 1.3
* **ORIGINAL SHOT DESCRIPTION:** "Generate from Last Frame (MCU Puppy Keyframe 1.2). Digitally pull the camera back to a Medium Shot. Prompt the sniffing action."
* **CURRENT PROMPT:** `""` (Empty)
* **IDENTIFIED ISSUES:** Prompt is missing.

**✓ GOOD OUTPUT (JSON):**
```json
{{
    "corrected_prompt": "Image-to-Image Generation. Using the final frame of Shot 1.2 as the starting point, digitally pull the camera back to a Medium Shot. The camera remains static at eye-level. Shot with a 50mm lens maintaining shallow depth of field. The black lab puppy is now fully in frame, standing on all fours on lush green grass. Its head is lowered, nose actively sniffing the ground with visible nostril flare. Soft golden hour light illuminates the puppy's glossy black fur. The camera remains focused on the puppy with the park's bokeh background of soft greens and blues. The puppy's body language shows a subtle shift—its tense, sad posture begins to relax slightly, ears perking forward with nascent curiosity as it discovers an interesting scent.",
    "modifications_made": [
        "Generated new prompt from scratch as original was empty (Path B)",
        "Translated 'pull back' instruction into clear Image-to-Image generation directive",
        "Added specific camera details: static, eye-level, 50mm lens",
        "Added lighting specification: soft golden hour light",
        "Described sniffing action with physical detail: lowered head, nostril flare",
        "Enhanced emotional beat: transition from sadness to curiosity through body language"
    ],
    "resolved_warnings": [
        "Prompt is missing."
    ],
    "required_assets": {{
        "BLACK_LAB_PUPPY": {{
            "needed_angle": "wide_shot",
            "reason": "The shot pulls back to Medium Shot showing the full puppy body on all fours. Wide_shot asset provides full body, front-facing view needed for this framing. The previous close_up is the starting keyframe, but the final composed shot requires wide_shot for full body visibility."
        }}
    }},
    "explanation": "The original prompt was empty. Following Step 2 triage, this is Path B—fundamentally flawed/missing. I discarded the empty prompt and generated from scratch based solely on the ORIGINAL SHOT DESCRIPTION. I locked down camera variables (static, eye-level, 50mm lens), specified lighting (golden hour), described the physical action (sniffing with nostril flare), and built emotional continuity (sad posture transitioning to curiosity). For assets, the shot requires wide_shot to show the full puppy body after pulling back from the close_up keyframe."
}}
```

**Example 2: Salvageable Prompt (Path A)**
* **ORIGINAL SHOT DESCRIPTION:** "Close-up of puppy's face showing sadness"
* **CURRENT PROMPT:** "Puppy sitting on wooden floor indoors looking sad"
* **IDENTIFIED ISSUES:** Indoor environment contradicts outdoor park baseline

**✓ GOOD OUTPUT (JSON):**
```json
{{
    "corrected_prompt": "Close-up of the black lab puppy's face, shot with an 85mm telephoto lens creating shallow depth of field. Static camera at low angle, slightly below eye-level to emphasize vulnerability. The puppy sits on lush green grass in the outdoor park. Its large, glistening dark eyes convey deep sadness, brow slightly furrowed. Its mouth trembles in a soft, barely audible whimper. Soft, diffused natural light illuminates the puppy's face, catching the moisture in its eyes. The park background melts into soft bokeh of muted greens.",
    "modifications_made": [
        "Changed 'wooden floor indoors' to 'lush green grass in the outdoor park' to match scene baseline (Path A - environment fix)",
        "Added camera specification: 85mm telephoto lens, static, low angle",
        "Added lighting detail: soft, diffused natural light",
        "Enhanced emotional description: glistening eyes, furrowed brow, trembling mouth, whimper",
        "Specified bokeh background for continuity"
    ],
    "resolved_warnings": [
        "[Consistency-CRITICAL] Indoor setting contradicts outdoor park baseline"
    ],
    "required_assets": {{
        "BLACK_LAB_PUPPY": {{
            "needed_angle": "close_up",
            "reason": "Shot explicitly calls for 'Close-up of puppy's face'. Close_up asset provides the frontal/face view needed to capture facial expressions and emotional details like eyes, brow, and mouth."
        }}
    }},
    "explanation": "Path A—prompt is salvageable. The framing (close-up) and subject (puppy face) are correct, only environment is wrong. I made targeted fix: changed indoor/wooden floor to outdoor park/grass. Then enhanced with cinematic language: specified 85mm telephoto for portrait, low angle for vulnerability, described emotional beat in detail (glistening eyes, furrowed brow, whimper), and locked down lighting (soft, diffused natural light)."
}}
```
"""

        try:
            response = self.model.generate_content(prompt_for_gemini)
            
            # Check for blocked content before accessing response.text
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason
                if block_reason:
                    logger.warning(
                        f"Gemini API blocked content for shot {shot_data['shot_id']}: "
                        f"block_reason={block_reason}, "
                        f"block_reason_message={getattr(response.prompt_feedback, 'block_reason_message', 'N/A')}"
                    )
                    # Return original prompt as fallback when content is blocked
                    return {
                        'corrected_prompt': original_prompt,
                        'modifications': [],
                        'resolved_warnings': [],
                        'explanation': f'Content blocked by Gemini API: {block_reason}. Using original prompt as fallback.',
                        'required_assets': {},
                        'required_locations': {},
                        'required_props': {}
                    }
            
            # Check if candidates exist before accessing response.text
            if not response.candidates or len(response.candidates) == 0:
                logger.warning(
                    f"Gemini API returned empty candidates for shot {shot_data['shot_id']}. "
                    f"Prompt feedback: {getattr(response, 'prompt_feedback', 'N/A')}"
                )
                return {
                    'corrected_prompt': original_prompt,
                    'modifications': [],
                    'resolved_warnings': [],
                    'explanation': 'No candidates in response. Using original prompt as fallback.',
                    'required_assets': {},
                    'required_locations': {},
                    'required_props': {}
                }
            
            response_text = response.text

            # Extract JSON from response
            import json
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                json_text = json_text.replace('```json', '').replace('```', '').strip()
                result = json.loads(json_text)

                return {
                    'corrected_prompt': result.get('corrected_prompt', original_prompt),
                    'modifications': result.get('modifications_made', []),
                    'resolved_warnings': result.get('resolved_warnings', []),
                    'explanation': result.get('explanation', ''),
                    'required_assets': result.get('required_assets', {}),
                    'required_locations': result.get('required_locations', {}),
                    'required_props': result.get('required_props', {})
                }
            else:
                logger.warning(f"Could not parse Gemini JSON response for shot {shot_data['shot_id']}")
                return {
                    'corrected_prompt': original_prompt,
                    'modifications': [],
                    'resolved_warnings': [],
                    'explanation': 'Failed to parse response',
                    'required_assets': {},
                    'required_locations': {},
                    'required_props': {}
                }

        except Exception as e:
            logger.error(f"Gemini API error for shot {shot_data['shot_id']}: {e}")
            return {
                'corrected_prompt': original_prompt,
                'modifications': [],
                'resolved_warnings': [],
                'explanation': f'API error: {str(e)}',
                'required_assets': {},
                'required_locations': {},
                'required_props': {}
            }

    def _get_available_assets_info(self, shot_data: Dict) -> str:
        """Get formatted info about available assets"""
        lines = []

        # Get characters from current shot
        chars_in_shot = shot_data['metadata'].get('characters_found', [])

        lines.append("**CHARACTERS:**")
        for char_name in chars_in_shot:
            # assets is a list of AssetInfo objects
            assets_list = self.asset_library.assets.get(char_name, [])
            if assets_list:
                angles = [asset.angle for asset in assets_list]
                lines.append(f"- {char_name}: Available angles = {', '.join(angles)}")

        # List all available locations
        lines.append("\n**LOCATIONS:**")
        for asset_name, assets_list in self.asset_library.assets.items():
            if assets_list and assets_list[0].type == 'location':
                lines.append(f"- {asset_name}: Available")

        # List all available props
        lines.append("\n**PROPS:**")
        for asset_name, assets_list in self.asset_library.assets.items():
            if assets_list and assets_list[0].type == 'prop':
                lines.append(f"- {asset_name}: Available")

        return "\n".join(lines) if lines else "No assets available"

    def _reselect_assets(self, corrected_prompt: str, original_assets: List[Dict],
                        shot_data: Dict, warnings: List[WarningAnalysis],
                        gemini_required_assets: Dict = None,
                        gemini_required_locations: Dict = None,
                        gemini_required_props: Dict = None) -> List[Dict]:
        """
        Re-select assets based on corrected prompt, warnings, and Gemini's recommendations.

        IMPORTANT: Characters and locations are LOCKED from shot design agent (CSV-based).
        Only props can be requeried from the asset library.
        """

        new_assets = []

        # ===== CHARACTERS: LOCKED FROM SHOT DESIGN AGENT (CSV-based) =====
        # Characters come from CSV in shot design agent, so we MUST use them as-is
        # We can only adjust the angle based on Gemini's recommendation
        character_assets_from_shot_design = [a for a in original_assets if a.get('type') == 'character']

        for char_asset in character_assets_from_shot_design:
            char_name = char_asset.get('character')
            if not char_name:
                continue

            # Start with the angle from shot design agent
            best_angle = char_asset.get('angle', shot_data['metadata'].get('required_angle', 'close_up'))

            # Use Gemini's recommendation if available
            if gemini_required_assets and char_name in gemini_required_assets:
                recommended = gemini_required_assets[char_name]
                gemini_angle = recommended.get('needed_angle')
                if gemini_angle:
                    best_angle = gemini_angle
                    logger.info(f"[Agent 13] Using Gemini's recommended angle '{gemini_angle}' for character '{char_name}'")

            # Find asset with the best angle
            asset = self.asset_library.find_asset(
                char_name,
                best_angle,
                fallback_angles=ANGLE_FALLBACKS.get(best_angle, [])
            )

            if asset:
                new_assets.append({
                    'type': 'character',
                    'character': char_name,
                    'angle': asset.angle,
                    'local_path': asset.local_path,
                    'url': asset.url,
                    'confidence': asset.confidence
                })
                logger.info(f"[Agent 13] Locked character '{char_name}' from shot design with angle '{asset.angle}'")
            else:
                # If we can't find the asset, keep the original
                new_assets.append(char_asset)
                logger.warning(f"[Agent 13] Could not find asset for locked character '{char_name}', keeping original")

        # ===== LOCATIONS: LOCKED FROM SHOT DESIGN AGENT (CSV-based) =====
        # Locations come from CSV in shot design agent, so we MUST use them as-is
        location_assets_from_shot_design = [a for a in original_assets if a.get('type') == 'location']

        for loc_asset in location_assets_from_shot_design:
            # Keep location assets exactly as they were from shot design agent
            new_assets.append(loc_asset)
            loc_name = loc_asset.get('location') or loc_asset.get('name')
            logger.info(f"[Agent 13] Locked location '{loc_name}' from shot design agent")

        # ===== PROPS: CAN BE REQUERIED (description-based in shot design agent) =====
        # Props are not from CSV, so we can requery based on Gemini's recommendations or prompt analysis
        props_needed = []
        if gemini_required_props:
            props_needed = list(gemini_required_props.keys())
            logger.info(f"[Agent 13] Using Gemini's prop recommendations: {props_needed}")
        else:
            props_needed = self._extract_props_from_prompt(corrected_prompt, shot_data)
            logger.info(f"[Agent 13] Extracted props from prompt: {props_needed}")

        for prop_name in props_needed:
            # Try to find exact match first
            asset = self.asset_library.find_asset(prop_name, 'master')
            if not asset:
                # Try to find partial match
                asset = self._find_prop_by_keyword(prop_name)

            if asset:
                new_assets.append({
                    'type': 'prop',
                    'name': prop_name,
                    'angle': asset.angle,
                    'local_path': asset.local_path,
                    'url': asset.url,
                    'confidence': asset.confidence
                })
                logger.info(f"[Agent 13] Requeried and added prop '{prop_name}' from asset library")

        # If we couldn't find any assets, return original
        return new_assets if new_assets else original_assets

    def _extract_characters_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract character names from prompt text.

        NOTE: This method is NO LONGER used for character asset selection in _reselect_assets.
        Characters are now LOCKED from shot design agent (CSV-based).
        This method may be used for analysis or reference purposes only.
        """
        characters = []
        prompt_lower = prompt.lower()

        # Character mappings
        if 'puppy' in prompt_lower or 'labrador' in prompt_lower or 'dog' in prompt_lower:
            if 'BLACK_LAB_PUPPY' in self.asset_library.assets:
                characters.append('BLACK_LAB_PUPPY')

        if 'squirrel' in prompt_lower:
            if 'FEISTY_SQUIRREL' in self.asset_library.assets:
                characters.append('FEISTY_SQUIRREL')

        return characters

    def _extract_locations_from_prompt(self, prompt: str, shot_data: Dict) -> List[str]:
        """
        Extract location names from prompt text and shot metadata.

        NOTE: This method is NO LONGER used for location asset selection in _reselect_assets.
        Locations are now LOCKED from shot design agent (CSV-based).
        This method may be used for analysis or reference purposes only.
        """
        locations = []
        prompt_lower = prompt.lower()

        # Location mappings based on keywords
        if 'park' in prompt_lower or 'lakeside' in prompt_lower:
            if 'LAKESIDE_PARK' in self.asset_library.assets:
                locations.append('LAKESIDE_PARK')

        if 'lake' in prompt_lower and 'THE_LAKE_(BACKGROUND_ELEMENT)' in self.asset_library.assets:
            locations.append('THE_LAKE_(BACKGROUND_ELEMENT)')

        # Also check scene baseline for locations
        if self.scene_baseline:
            scene_desc = self.scene_baseline.get('description', '').lower()
            if 'park' in scene_desc and 'LAKESIDE_PARK' in self.asset_library.assets:
                if 'LAKESIDE_PARK' not in locations:
                    locations.append('LAKESIDE_PARK')

        return locations

    def _extract_props_from_prompt(self, prompt: str, shot_data: Dict) -> List[str]:
        """
        Extract prop names from prompt text.

        NOTE: Props CAN be requeried from asset library (not CSV-based in shot design agent).
        This method is actively used for prop asset selection in _reselect_assets.
        """
        props = []
        prompt_lower = prompt.lower()

        # Prop mappings
        if 'nut' in prompt_lower or 'acorn' in prompt_lower:
            if 'NUT' in self.asset_library.assets:
                props.append('NUT')

        return props

    def _find_location_by_keyword(self, location_name: str) -> Optional[Any]:
        """Find location asset by keyword matching"""
        location_lower = location_name.lower()

        # Check all assets for location type
        for asset_name, assets_list in self.asset_library.assets.items():
            if assets_list and assets_list[0].type == 'location':
                # Check for keyword matches
                asset_name_lower = asset_name.lower()
                # Extract keywords from both names
                keywords = location_lower.replace('_', ' ').split()
                for keyword in keywords:
                    if keyword in asset_name_lower:
                        return assets_list[0]  # Return first asset (usually master)

        return None

    def _find_prop_by_keyword(self, prop_name: str) -> Optional[Any]:
        """Find prop asset by keyword matching"""
        prop_lower = prop_name.lower()

        # Check all assets for prop type
        for asset_name, assets_list in self.asset_library.assets.items():
            if assets_list and assets_list[0].type == 'prop':
                # Check for keyword matches
                asset_name_lower = asset_name.lower()
                # Extract keywords from both names
                keywords = prop_lower.replace('_', ' ').split()
                for keyword in keywords:
                    if keyword in asset_name_lower:
                        return assets_list[0]  # Return first asset (usually master)

        return None

    def _calculate_new_feasibility(self, assets: List[Dict],
                                   warnings_remaining: List[str],
                                   warnings_resolved_count: int) -> float:
        """Calculate new feasibility score after modifications"""

        if not assets:
            return 0.1  # Very low if no assets

        # Base score from asset quality
        avg_confidence = sum(a.get('confidence', 0) for a in assets) / len(assets)

        # Bonus for resolved warnings
        resolution_bonus = min(0.3, warnings_resolved_count * 0.05)

        # Penalty for remaining critical warnings
        critical_remaining = sum(1 for w in warnings_remaining if 'CRITICAL' in w)
        critical_penalty = critical_remaining * 0.15

        # Penalty for other remaining warnings
        other_penalty = (len(warnings_remaining) - critical_remaining) * 0.05

        final_score = avg_confidence + resolution_bonus - critical_penalty - other_penalty
        return max(0.0, min(1.0, final_score))

    def process_shot_list(self, agent12_output: Dict) -> List[ModifiedShotOutput]:
        """Process entire shot list from Agent 12"""
        shot_designs = agent12_output.get('shot_designs', [])
        results = []

        # Establish scene baseline from first shot
        if shot_designs:
            first_shot = shot_designs[0]
            self.scene_baseline = {
                'shot_id': first_shot['shot_id'],
                'description': first_shot['metadata'].get('original_description', ''),
                'environment': 'outdoor park by lake',
                'characters': first_shot['metadata'].get('characters_found', [])
            }

        logger.info("\n" + "="*60)
        logger.info("AGENT 13: PROMPT MODIFIER STARTING")
        logger.info("="*60)

        for i, shot in enumerate(shot_designs, 1):
            logger.info(f"\nProcessing Shot {shot['shot_id']} ({i}/{len(shot_designs)})...")
            modified = self.modify_shot(shot, self.scene_baseline)
            results.append(modified)
            self.shot_history.append(modified)

            if modified.modifications_made:
                logger.info(f"   ✓ {len(modified.modifications_made)} modifications made")
                logger.info(f"   ✓ Resolved {len(modified.warnings_resolved)} warnings")
            else:
                logger.info(f"   • No modifications needed")

        logger.info("\n" + "="*60)
        logger.info("✓ AGENT 13 COMPLETED")
        logger.info("="*60)

        return results


def load_agent12_output(filepath: str) -> Dict:
    """Load Agent 12 output JSON"""
    import json
    with open(filepath, 'r') as f:
        return json.load(f)


def save_results(results: List[ModifiedShotOutput], output_dir: str = "phase_2_agents/outputs/agent_prompt_modifier"):
    """Save Agent 13 results to JSON"""
    from pathlib import Path
    from datetime import datetime
    import json
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agent13_modified_prompts_{timestamp}.json"

    # Calculate statistics
    total_warnings_resolved = sum(len(r.warnings_resolved) for r in results)
    total_warnings_remaining = sum(len(r.warnings_remaining) for r in results)
    avg_feasibility_improvement = sum(r.feasibility_change for r in results) / len(results) if results else 0

    output_data = {
        'agent': 'Agent 13: Prompt Modifier Agent',
        'timestamp': datetime.now().isoformat(),
        'modified_shots': [asdict(r) for r in results],
        'statistics': {
            'total_shots': len(results),
            'total_warnings_resolved': total_warnings_resolved,
            'total_warnings_remaining': total_warnings_remaining,
            'avg_feasibility_improvement': avg_feasibility_improvement,
            'shots_improved': sum(1 for r in results if r.feasibility_change > 0),
            'shots_degraded': sum(1 for r in results if r.feasibility_change < 0),
            'shots_unchanged': sum(1 for r in results if r.feasibility_change == 0)
        }
    }

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n✓ Results saved to: {filename}")
    logger.info(f"  Total shots processed: {len(results)}")
    logger.info(f"  Warnings resolved: {total_warnings_resolved}")
    logger.info(f"  Warnings remaining: {total_warnings_remaining}")
    logger.info(f"  Avg feasibility improvement: {avg_feasibility_improvement:+.3f}")
    logger.info(f"  Shots improved: {sum(1 for r in results if r.feasibility_change > 0)}")

    return filename

