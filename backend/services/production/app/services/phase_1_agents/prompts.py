#!/usr/bin/env python3
"""
Centralized Prompts for Phase 1 Agents
=======================================
All agent prompts in one place for easy maintenance and consistency.
"""

from typing import Dict, Any, List, Optional, Tuple
import json


class Agent1Prompts:
    """Prompts for Agent 1: Asset Generator"""

    @staticmethod
    def asset_extraction(script_content: str, csv_entities: Optional[Dict[str, Any]] = None, product_image_available: bool = False) -> str:
        """
        Prompt for extracting assets from script with optional CSV entity mapping.

        Args:
            script_content: The script text to analyze
            csv_entities: Optional CSV entity mapping with character/location names from shotlist

        Returns:
            Formatted prompt string
        """
        base_prompt = f"""
You are an expert script analyst specializing in asset extraction for visual production.

Analyze the following script and extract ALL assets in three categories:

1. CHARACTERS: Every named character, character group, or entity with visual presence
2. LOCATIONS: Every distinct location, setting, or environment
3. PROPS: Every significant MOVABLE object, tool, vehicle, or item that characters interact with

For EACH asset, provide:
- Name (exact as it appears in script)
- Description (detailed visual description)
- Key attributes (specific to asset type)
- Scenes where it appears (scene numbers or descriptions)
- Importance level (critical, important, background)

**IMPORTANT GUIDELINES:**
- Include ALL assets, even if mentioned briefly
- Look for implied assets (e.g., if someone "drives away", there's a vehicle)
- Don't create duplicates - same asset with different names should be merged
- Be thorough - missing assets will cause production issues later

**CRITICAL - WHAT IS NOT A PROP:**
- Environmental elements (grass, trees, rocks, water, sky, clouds, etc.) → These are part of LOCATIONS
- Background scenery or landscape features → Part of LOCATIONS
- Natural terrain or ground cover → Part of LOCATIONS
- ONLY include objects that are distinct, movable items that characters interact with

**CHARACTER ATTRIBUTES:**
- Physical appearance (age, gender, build, distinctive features)
- Clothing/costume style
- Character role (protagonist, antagonist, supporting)
- Key visual traits that make them recognizable

**LOCATION ATTRIBUTES:**
- Setting type (interior/exterior, urban/rural, etc.)
- Time of day (if specified)
- Atmosphere/mood
- Key visual elements
- Lighting conditions

**PROP ATTRIBUTES:**
- Material and appearance
- Size and scale
- Condition (new, worn, futuristic, etc.)
- How it's used in the story

**SCRIPT TO ANALYZE:**
{script_content}

Extract all assets following the guidelines above. Be thorough and detailed.
"""

        product_image_section = ""
        if product_image_available:
            product_image_section = """

**PRODUCT IMAGE RULE — READ CAREFULLY:**
A real product image has been uploaded for this project. The script features an advertised product.
When extracting props, identify which single prop represents the main advertised product being featured in this video.
Mark that prop with `"is_product": true` in your JSON output.
Only ONE prop should have `"is_product": true`.
Do NOT generate or imagine a replacement image for this prop — the real image is already available.
"""

        # Add CSV entity mapping section if provided
        if csv_entities and csv_entities.get('has_entity_data'):
            csv_characters = csv_entities.get('unique_characters', [])
            csv_locations = csv_entities.get('unique_locations', [])

            csv_section = f"""

**CRITICAL - CSV ENTITY MAPPING (SOURCE OF TRUTH):**

The following entities are pre-defined in the shotlist CSV. These names are the GROUND TRUTH and MUST be used exactly as listed.

**CSV CHARACTERS ({len(csv_characters)} total):**
{', '.join(csv_characters) if csv_characters else 'None specified'}

**CSV LOCATIONS ({len(csv_locations)} total):**
{', '.join(csv_locations) if csv_locations else 'None specified'}

**MANDATORY MAPPING RULES:**
1. When you find a character or location in the script, check if it semantically matches ANY entity from the CSV lists above
2. If it matches, you MUST use the CSV name EXACTLY (already normalized to UPPERCASE_WITH_UNDERSCORES)
3. Use the 'name' field for the CSV name and store the CSV name in 'csv_name' field as well
4. Include script details in the 'description' field, but the name must match CSV
5. ONLY extract characters and locations that appear in the CSV lists — the list is CLOSED AND FINAL
6. Your output MUST contain EXACTLY {len(csv_characters)} character(s) — no more, no fewer
7. Props are NOT in the CSV — extract them normally from the script

**MAPPING EXAMPLES:**
- Script mentions "Dense Forest" or "jungle area" → CSV has "JUNGLE" → Use name="JUNGLE", csv_name="JUNGLE", description="Dense tropical forest with..."
- Script mentions "Alex" or "Alexander" → CSV has "ALEX" → Use name="ALEX", csv_name="ALEX", description="[Character details from script]"
- Script mentions "Sarah Thompson" or "Dr. Thompson" → CSV has "SARAH" → Use name="SARAH", csv_name="SARAH", description="[Character details]"
- Script mentions "old cabin" → CSV has "CABIN" → Use name="CABIN", csv_name="CABIN"

**NON-NEGOTIABLE HARD RULES — THESE OVERRIDE ALL OTHER INSTRUCTIONS:**
- Your output characters list MUST contain EXACTLY these {len(csv_characters)} name(s): {', '.join(csv_characters) if csv_characters else 'None'}
- Your output locations list MUST contain ONLY names from: {', '.join(csv_locations) if csv_locations else 'None'}
- ANY character or location name NOT in the CSV lists above is STRICTLY FORBIDDEN in your output
- DO NOT add implied bystanders, background crowd, unnamed extras, or ANY person not explicitly listed in the CSV above
- If the script mentions a character who is not in the CSV, SILENTLY DISCARD them — do not include them under any name or alias
- DO NOT "log it as a note" — non-CSV entities must not appear anywhere in your JSON output
- Props are the ONLY category where you may freely extract entities not listed in the CSV

If you cannot find script details for a CSV character or location, still include it with a placeholder description derived from the name — a sparse entry is better than a missing one.
"""
            return base_prompt + csv_section + product_image_section

        return base_prompt + product_image_section


class Agent2Prompts:
    """Prompts for Agent 2: Asset Reviewer"""

    @staticmethod
    def asset_review(script_content: str, assets: Dict[str, Any]) -> str:
        """Prompt for reviewing and enhancing assets"""
        assets_json = json.dumps(assets, indent=2)

        return f"""
You are an expert asset quality reviewer for visual production. Your job is to review extracted assets and ensure they are:
1. COMPLETE - No missing assets from the script
2. ACCURATE - Descriptions match the script exactly
3. DETAILED - Descriptions are rich enough for image generation
4. CONSISTENT - No duplicates or conflicting information
5. PRODUCTION-READY - All necessary visual details are present

**ORIGINAL SCRIPT:**
{script_content}

**EXTRACTED ASSETS TO REVIEW:**
{assets_json}

**YOUR REVIEW TASKS:**

1. **COMPLETENESS CHECK:**
   - Are there any IMPLIED assets not explicitly mentioned but necessary? (e.g., "drives away" implies a vehicle)
   - Are there background elements that should be assets? (e.g., furniture, weather elements, ambient objects)
   - Check for any characters, locations, or props mentioned even briefly

2. **DUPLICATE DETECTION:**
   - Are there duplicate assets with different names?
   - Are there redundant entries that should be merged?
   - List any duplicates found with suggested merge strategy

3. **DESCRIPTION ENHANCEMENT:**
   - For each asset, evaluate if the description is detailed enough for AI image generation
   - Suggest improvements: add colors, textures, scale, mood, artistic style references
   - Enhance physical descriptions with specific visual details
   - Add missing attributes (lighting, materials, proportions, etc.)

4. **ACCURACY VERIFICATION:**
   - Do descriptions match what's in the script?
   - Are there any contradictions or inconsistencies?
   - Are importance levels correctly assigned?

5. **EDGE CASES:**
   - Assets that appear in multiple forms (e.g., character with/without costume)
   - Time-sensitive assets (e.g., location at different times of day)
   - Assets with state changes (e.g., damaged prop vs intact prop)

6. **OVERALL ASSESSMENT:**
   - Provide quality scores for completeness, accuracy, detail level, and production readiness (0-100)
   - List overall recommendations

Provide a thorough, detailed review. Be critical but constructive.
"""


class Agent3Prompts:
    """Prompts for Agent 3: Prompt Generator"""

    @staticmethod
    def character_prompt_generation(character: Dict[str, Any]) -> str:
        """Generate optimized prompt for character master image"""
        char_json = json.dumps(character, indent=2)

        return f"""
You are an expert AI prompt engineer specializing in creating optimized prompts for Google Imagen 4.0,
a state-of-the-art photorealistic text-to-image model. Your prompts must produce realistic, high-detail
images with accurate materials, textures, and lighting.

**CHARACTER DATA:**
{char_json}

**YOUR TASK:**
Generate ONE highly optimized MASTER IMAGE prompt for this character that serves as the definitive visual reference.
This master image should capture the character's complete appearance and essence with photographic realism.

**MASTER IMAGE REQUIREMENTS:**
- Full body shot showing complete character details
- Clear view of facial features, clothing, and distinctive characteristics
- Appropriate pose that reflects character's personality/role
- CLEAN NEUTRAL BACKGROUND (solid color, simple gradient, or plain studio background - NO environmental elements, landscapes, or scene-specific backgrounds)
- Professional photographic quality that can serve as reference for all future shots and easy compositing into different video scenes

**PROMPT STRUCTURE (follow this order):**
[Subject + physical detail] → [Clothing material specifics] → [Pose/action] → [Lighting setup] → [Camera/lens] → [Film stock/style]

**PROMPT OPTIMIZATION GUIDELINES:**

1. **Structure**: Follow the formula above — subject first, then physical details, clothing, pose, lighting, camera, style
2. **Materiality & Texture**: Specify exact fabric types and surface finishes — NOT "jacket" but "worn navy blue tweed jacket with frayed lapels and visible weave texture"; NOT "boots" but "scuffed dark leather ankle boots with metal buckle detail". Describe skin tone photographically (e.g., "warm medium-brown skin with subtle natural highlights")
3. **Lighting Design**: Use named photographic lighting setups — "three-point softbox studio setup", "soft Rembrandt lighting from camera-left", "diffuse overcast daylight through a large north-facing window". Describe shadow quality and direction
4. **Camera & Lens**: Specify equipment for photographic realism — "shot on Canon EOS R5, 85mm f/1.4 prime lens, shallow depth of field, subject in sharp focus, neutral background in soft bokeh"
5. **Film Stock / Color Grade**: Ground the image in photographic reality — "shot on Kodak Portra 400, warm natural skin tones, slight organic grain" or "digital capture, clean neutral color balance, true-to-life color rendering"
6. **Completeness**: Include ALL visual information needed to recreate this character consistently
7. **Positive Framing**: Describe what you WANT, not what to avoid — say "empty neutral grey studio background" not "no environment"

**IMPORTANT:**
- The initial_prompt should be 150-300 words and completely self-contained
- Use natural, flowing narrative language (not comma-separated keyword lists)
- CRITICAL: Explicitly specify a NEUTRAL BACKGROUND (e.g., "plain white background", "solid grey background", "simple gradient background")
- Include negative prompts to exclude environmental backgrounds, landscapes, scenery, and any scene-specific elements
- Ensure the prompt captures EVERYTHING needed for visual consistency
- Match the emotional tone and story context for the CHARACTER ONLY (not the environment)

Generate the master prompt now.
"""

    @staticmethod
    def location_prompt_generation(location: Dict[str, Any]) -> str:
        """Generate optimized prompt for location master image"""
        loc_json = json.dumps(location, indent=2)

        return f"""
You are an expert AI prompt engineer specializing in creating optimized prompts for Google Imagen 4.0,
a state-of-the-art photorealistic text-to-image model. Your prompts must produce realistic aerial images
with rich environmental texture, accurate atmospheric lighting, and cinematic detail.

**LOCATION DATA:**
{loc_json}

**YOUR TASK:**
Generate ONE highly optimized MASTER IMAGE prompt for this location that serves as the definitive visual reference.
This master image should capture the location's complete atmosphere, layout, and visual characteristics from the air.

**MASTER IMAGE REQUIREMENTS:**
- MANDATORY: Start your prompt with "aerial shot" or "bird's-eye view" - NEVER use "establishing shot"
- A breathtaking, high-altitude cinematic bird's-eye view looking down from above to capture the sprawling layout, vast scale, and entire environmental topography from an expansive aerial perspective
- Rich surface textures and materials visible from altitude
- Clear representation of lighting, atmosphere, and mood
- Include all distinctive visual elements mentioned
- Cinematic composition that showcases the location's essence
- Professional quality suitable as primary reference

**PROMPT STRUCTURE (follow this order):**
[Aerial framing] → [Surface materiality & texture from above] → [Atmospheric lighting & time of day] → [Aerial camera specs] → [Color palette/grading]

**PROMPT OPTIMIZATION GUIDELINES:**

1. **Environment First**: MANDATORY AERIAL START. Your prompt MUST begin with either "aerial shot" or "bird's-eye view" - NEVER use "establishing shot" or any ground-level terminology. Begin with a high-altitude vantage point describing the location as seen from the sky.
2. **Aerial Materiality & Texture**: Describe surface textures visible from altitude — canopy foliage density and color variation, roof tile patterns and materials (terracotta, slate, corrugated metal), terrain soil color and texture, water surface reflectivity and movement patterns, road/path materials (asphalt crack patterns, dirt track ruts). Be specific: NOT "forest" but "dense rainforest canopy with uneven emerald-green treetop texture and scattered light gaps"
3. **Atmospheric Lighting**: Specify time of day and sun angle for realistic shadow casting — "golden hour light at 15-degree angle casting long warm shadows across the terrain", "overcast midday with soft diffuse light and minimal shadows", "blue hour twilight with deep indigo sky and warm artificial lights below". Include atmospheric haze or clarity
4. **Aerial Camera Specs**: Use realistic drone/aerial equipment — "shot from DJI Mavic 3 Pro at 120m altitude, DJI Zenmuse X7 lens, 16mm wide-angle, top-down orthographic perspective" or "high-altitude satellite-style view, ultra-wide 10mm lens, tack-sharp environmental detail"
5. **Color Palette**: Define the environmental color grading — warm ochre and terracotta earth tones, cool steel-blue urban palette, lush saturated greens, muted winter browns. Reference a cinematic look if appropriate (e.g., "Fujifilm Velvia 50 saturation, rich environmental color")
6. **Mood & Scale**: Convey the emotional tone of looking down upon a vast world — the sense of scale, isolation, density, or grandeur

**IMPORTANT:**
- The initial_prompt MUST start with "aerial shot" or "bird's-eye view" - this is NON-NEGOTIABLE
- NEVER use "establishing shot" - this is FORBIDDEN for locations
- The initial_prompt should be 150-300 words and completely self-contained
- Use natural, flowing narrative language describing the environment
- Include negative prompts to avoid: ground-level view, eye-level perspective, walking view, standing perspective, horizon view
- Capture the full atmosphere, lighting, and spatial layout from an AERIAL perspective only
- Match the emotional tone from the story

Generate the master prompt now.
"""

    @staticmethod
    def prop_prompt_generation(prop: Dict[str, Any]) -> str:
        """Generate optimized prompt for prop master image"""
        prop_json = json.dumps(prop, indent=2)

        return f"""
You are an expert AI prompt engineer specializing in creating optimized prompts for Google Imagen 4.0,
a state-of-the-art photorealistic text-to-image model. Your prompts must produce realistic product-quality
images with precise material description, surface texture, and professional studio lighting.

**PROP DATA:**
{prop_json}

**YOUR TASK:**
Generate ONE highly optimized MASTER IMAGE prompt for this prop that serves as the definitive visual reference.
This master image should clearly showcase the prop's appearance, materials, and key characteristics with
photographic precision.

**MASTER IMAGE REQUIREMENTS:**
- Hero shot showing the prop clearly in isolation
- All key details, textures, and materials visible with maximum clarity
- Appropriate scale and perspective
- CLEAN NEUTRAL BACKGROUND (solid color, simple gradient, or plain studio background - NO environmental elements, characters, animals, or scene-specific backgrounds)
- Professional product photography quality
- Can serve as reference for future compositing into different video scenes

**PROMPT STRUCTURE (follow this order):**
[Object name + primary material] → [Surface texture & finish detail] → [Scale & proportion] → [Product lighting setup] → [Camera/lens for detail] → [Digital capture style]

**PROMPT OPTIMIZATION GUIDELINES:**

1. **Object Focus**: Start with the object name and primary material — THE PROP ONLY, isolated
2. **Materiality & Texture — BE SPECIFIC**: Describe the exact physical makeup — NOT "metal sword" but "hand-forged carbon steel longsword with a pattern-welded blade showing visible Damascus swirl pattern, polished to a mirror finish on the flat, rough satin on the fuller"; NOT "wooden box" but "aged mahogany jewelry box with dovetail corner joints, deep walnut grain visible through matte lacquer finish, brass corner fittings with green patina". Include surface finish (matte, gloss, satin, brushed, hammered, weathered, aged)
3. **Scale & Proportion**: Indicate size relative to context — "life-size", "approximately 30cm tall", "hand-sized"
4. **Product Lighting**: Specify a product photography lighting setup — "three-point softbox studio setup with white fill card", "rim lighting to define material edges against the background", "single large octabox from above-left casting clean directional shadow". The lighting should reveal texture and material accurately
5. **Camera & Lens for Detail**: "shot on Canon EOS R5 with 100mm f/2.8L macro lens, pin-sharp product focus, shallow depth of field on background"
6. **Digital Capture**: "clean digital product photography, ultra-high resolution, no grain, clinical material accuracy, true-to-life color rendering"
7. **NO Characters or Animals**: NEVER include people, animals, or hands holding the object
8. **Positive Framing**: Say "isolated on plain white studio background" not "no people, no environment"

**IMPORTANT:**
- The initial_prompt should be 100-200 words and completely self-contained
- Use clear, precise language about the object's physical appearance ONLY
- CRITICAL: Explicitly specify a NEUTRAL BACKGROUND (e.g., "plain white background", "solid grey background", "simple gradient background")
- CRITICAL: DO NOT include any characters, animals, hands, or environmental context
- Include negative prompts to exclude: people, animals, hands, environmental backgrounds, scenery, landscapes, outdoor/indoor settings
- Ensure all materials, textures, finishes, and surface details are specified precisely
- Match detail level to the prop's story importance
- The prop must be isolated and ready for compositing

Generate the master prompt now.
"""


class Agent4Prompts:
    """Prompts for Agent 4: Prompt Optimizer"""

    @staticmethod
    def prompt_optimization(asset_name: str, asset_type: str, initial_prompt_data: Dict[str, Any]) -> str:
        """Optimize an initial prompt"""
        prompt_json = json.dumps(initial_prompt_data, indent=2)

        return f"""
You are an expert AI prompt optimization specialist for Google Imagen 4.0 and Gemini Nano Banana
(gemini-3.1-flash-image-preview). Your job is to take an existing image generation prompt and refine it
to produce more photorealistic, detail-rich, and cinematically grounded results.

**ASSET TYPE:** {asset_type}
**ASSET NAME:** {asset_name}

**INITIAL PROMPT DATA:**
{prompt_json}

**YOUR TASK:**
Review the initial_prompt and create an optimized final_prompt that improves upon it.

**OPTIMIZATION STRATEGIES:**

1. **Enhanced Clarity**: Make descriptions more precise and unambiguous
2. **Better Flow**: Improve sentence structure for natural language flow — use narrative prose, not comma-separated keyword lists
3. **Visual Richness**: Add sensory details (textures, lighting nuances, atmosphere)
4. **Technical Precision**: Ensure all technical terms are accurate and specific
5. **Consistency**: Maintain all key information while enhancing quality
6. **Positive Framing**: Describe what you WANT, not what to avoid — "empty neutral studio background" not "no environment, no people"
7. **Keyword Optimization**: Add relevant quality and style keywords strategically
8. **Materiality & Texture Upgrade**: Replace vague material references with specific physical descriptions — NOT "leather jacket" but "worn dark brown full-grain leather jacket with visible crease lines at the elbows and faded brass zipper hardware"; NOT "stone wall" but "rough-hewn limestone blocks with weathered mortar joints and moss patches in the lower courses". Apply to ALL asset types
9. **Photographic Lighting Specification**: Add a named lighting setup appropriate to the asset type — for characters/props: "three-point softbox studio setup", "Rembrandt lighting from camera-left", "large north-facing window diffusion"; for locations: "golden hour at 20-degree sun angle casting long directional shadows", "overcast sky producing flat diffuse light"
10. **Camera & Lens Specification**: Insert photographic equipment details for realistic rendering — characters: "Canon EOS R5, 85mm f/1.4 prime, shallow depth of field, sharp subject, soft background bokeh"; props: "100mm f/2.8 macro lens, clinical product sharpness"; locations: "DJI Mavic 3 Pro drone at 80-120m altitude, 16mm wide-angle"
11. **Film Stock / Color Grade**: Add a realistic photographic grounding — "shot on Kodak Portra 400, warm natural skin tones, subtle organic grain" for characters; "Phase One IQ4 digital back, zero grain, ultra-high-resolution clinical detail" for props; "Fujifilm Velvia 50, rich saturated environmental color" for locations
12. **STRICT AERIAL FOR LOCATIONS**: If the {asset_type} is a LOCATION, the final_prompt MUST start with the exact phrase "aerial shot" or "bird's-eye view" - NEVER use "establishing shot". You must force the perspective to be a high-altitude shot, even if the initial_prompt describes a ground-level scene. Start the final prompt with: "Aerial shot of..." or "Bird's-eye view of..." followed by "a breathtaking, high-altitude cinematic view looking down to capture the sprawling layout, vast scale, and entire environmental topography from an expansive aerial perspective." Every subsequent detail must describe the scene as viewed from the sky (e.g., canopy tops, roof patterns, river veins, and geographic footprints).
13. **CRITICAL FOR CHARACTERS**: ALWAYS enforce NEUTRAL/PLAIN backgrounds (solid colors, gradients, studio backgrounds). NEVER add environmental elements, landscapes, or scene-specific backgrounds for character assets

**WHAT TO PRESERVE:**
- All core visual information
- Character/location/prop identity
- Story context and tone
- Technical specifications

**WHAT TO IMPROVE:**
- Materiality and texture specificity (this is the highest-impact change)
- Photographic lighting, camera, and film stock grounding
- Clarity and precision of descriptions
- Flow and readability
- Negative prompt comprehensiveness
- FOR CHARACTERS ONLY: Ensure neutral background is explicitly specified and environmental elements are excluded in negative prompts

**IMPORTANT:**
- The final prompt should be MORE detailed than the initial prompt; if it is already detailed enough, don't hallucinate
- Maintain natural, flowing narrative language (not just keyword lists)
- Add material, texture, lighting, and camera details to ALL asset types
- Enhance negative prompts to prevent common AI errors
- FOR CHARACTER ASSETS: MANDATORY neutral background specification (e.g., "plain white background", "solid grey studio background"). This is CRITICAL for I2V compositing
- FOR CHARACTER ASSETS: Negative prompt MUST exclude: landscapes, environmental backgrounds, scenery, outdoor/indoor settings, lakes, parks, etc.
- FOR PROP ASSETS: MANDATORY neutral background specification. NEVER include characters, animals, hands, or environmental context
- FOR PROP ASSETS: Negative prompt MUST exclude: people, animals, hands, characters, environmental backgrounds, scenery, landscapes, outdoor/indoor settings
- ONLY LOCATIONS should have environmental context and backgrounds
- Ensure the prompt is production-ready
- FOR LOCATION ASSETS: The final_prompt MUST start with "aerial shot" or "bird's-eye view" (NEVER "establishing shot"). You are strictly forbidden from generating ground-level or eye-level descriptions. The final output must be an unambiguous aerial shot viewed from above.
- LOCATION NEGATIVE PROMPT: You must explicitly include: establishing shot, eye-level view, ground-level perspective, low-angle shot, walking-path view, horizon-only shot, interior perspective, ground camera, standing view.

Generate the optimized prompt now.
"""


class Agent6Prompts:
    """Prompts for Agent 6: Image Reviewer"""

    @staticmethod
    def image_review(asset_name: str, asset_type: str, asset_description: str, image_analysis: str) -> str:
        """Review generated image quality"""
        return f"""
You are an expert visual quality assurance reviewer for AI-generated images in video production.

**ASSET INFORMATION:**
- Name: {asset_name}
- Type: {asset_type}
- Expected Description: {asset_description}

**GENERATED IMAGE ANALYSIS:**
{image_analysis}

**YOUR TASK:**
Review this generated image and provide comprehensive quality assessment.

**EVALUATION CRITERIA:**

1. **Technical Quality (1-10)**
   - Resolution and clarity
   - Proper rendering without artifacts
   - Correct proportions and anatomy (if applicable)
   - No distortions or abnormalities

2. **Visual Accuracy (1-10)**
   - Matches the description
   - All key features present
   - Correct colors and materials
   - Appropriate lighting and atmosphere

3. **Artistic Quality (1-10)**
   - Aesthetic appeal
   - Professional production value
   - Appropriate style and tone
   - Cohesive visual design

4. **Consistency (1-10)**
   - Matches the story's visual style
   - Compatible with other assets
   - Consistent lighting and perspective
   - Ready for video compositing

5. **Production Readiness**
   - Can this be used in final production?
   - What issues need to be fixed?
   - Regeneration needed? Editing needed?

Provide detailed, actionable feedback.
"""


# Utility function to get all prompts
def get_prompt(agent_number: int, prompt_type: str, **kwargs) -> str:
    """
    Get a specific prompt by agent number and type

    Args:
        agent_number: Agent number (1-8)
        prompt_type: Type of prompt to retrieve
        **kwargs: Additional parameters needed for the prompt

    Returns:
        Formatted prompt string
    """
    prompt_map = {
        1: Agent1Prompts,
        2: Agent2Prompts,
        3: Agent3Prompts,
        4: Agent4Prompts,
        6: Agent6Prompts,
    }

    agent_class = prompt_map.get(agent_number)
    if not agent_class:
        raise ValueError(f"No prompts defined for Agent {agent_number}")

    prompt_method = getattr(agent_class, prompt_type, None)
    if not prompt_method:
        raise ValueError(f"Prompt type '{prompt_type}' not found for Agent {agent_number}")

    return prompt_method(**kwargs)
