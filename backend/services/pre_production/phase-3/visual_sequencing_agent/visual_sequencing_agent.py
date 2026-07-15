import json
import logging
import os
import time
from typing import List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase3.visual_sequencing_agent")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
SCRIPT_COLLECTION = os.getenv("COLLECTION_SCRIPT", "script")


class TextSuper(BaseModel):
    content: str = Field(description="Text content of the super overlay.")
    placement: str = Field(description="Screen position: top, center, or bottom.")
    duration_s: float = Field(description="Duration in seconds the text super is visible.")


class ShotEntry(BaseModel):
    window_id: str = Field(description="Timeline window ID from master_timeline, e.g. W01, W02.")
    framing: str = Field(description="Cinematic camera framing for 9:16 vertical, e.g. 'ECU lips 85mm lens', 'MS creator at chaotic desk 24mm wide'.")
    subject: str = Field(description="Highly specific description of who/what is in frame, including wardrobe, posture, and micro-expressions.")
    action: str = Field(description="Choreography-level detail of what happens on screen. Break down exact blocking, prop handling, and exact moves.")
    lighting: str = Field(description="Detailed lighting direction, e.g. 'harsh fluorescent with green cast', 'soft warm golden backlight with negative fill'.")
    text_supers: List[TextSuper] = Field(default_factory=list, description="Text overlays for this shot. May be empty.")
    notes: Optional[str] = Field(default=None, description="Detailed production notes: specific camera moves (whip pan, snap zoom), depth of field, set dressing, and production design details.")
    location_heading: str = Field(description="Fountain scene heading for this shot. Format strictly as INT./EXT. LOCATION - DAY/NIGHT. Derived from the environmental context: lighting quality, set dressing, and implied space. E.g. 'INT. CORPORATE BREAKROOM - DAY', 'INT. BATHROOM - MORNING', 'EXT. ROOFTOP - NIGHT'.")
    character_intro: str = Field(default="", description="On the FIRST window where this character appears in the shot list, write their screenplay introduction: 'CHARACTER NAME (age range, defining physical trait, wardrobe)'. E.g. 'CREATOR (late 20s, heavy-lidded eyes, oversized beige blazer)'. Empty string if the character has already appeared in a prior window, or if no character is present.")


class VisualSequencingResult(BaseModel):
    shot_list: List[ShotEntry] = Field(description="One shot entry per master_timeline window.")
    reasoning: str = Field(description="Step-by-step reasoning for shot design decisions.")
    key_frame_rationale: str = Field(description="Rationale for key frame selection across the sequence.")
    mobile_compliance_check: str = Field(description="Verification that all shots comply with 9:16 mobile-first constraints.")
    status: str = Field(default="pending", description="Execution status.")


async def run_visual_sequencing_agent(project_id: str, db, revision_prefix: str | None = None) -> VisualSequencingResult:
    agent_key = "visual_sequencing_agent"
    run_start = time.time()
    logger.info(f"Initializing Agent [3]... | project_id={project_id}")

    if not GEMINI_API_KEY:
        logger.error("Agent [3]: Missing GEMINI_API_KEY environment variable.")
        raise ValueError("GEMINI_API_KEY is not configured")

    try:
        logger.info(f"[{agent_key}] Fetching data for project_id={project_id}")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project '{project_id}' not found")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document for '{project_id}' not found")

        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        if not strategy_doc:
            raise ValueError(f"Strategy document for '{project_id}' not found")

        pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)})
        if not pipeline_doc:
            logger.info(f"[{agent_key}] Pipeline document not found. Continuing with empty pipeline context.")
            pipeline_doc = {}

        script_doc = await db[SCRIPT_COLLECTION].find_one({"project_id": str(project_id)})
        if not script_doc:
            logger.info(f"[{agent_key}] Script document not found. Continuing with empty script context.")
            script_doc = {}

        # --- Extract from ideation_doc ---
        phase_2_output = ideation_doc.get("phase_2_output", ideation_doc)

        approved_concepts = phase_2_output.get("approved_concepts", ideation_doc.get("approved_concepts", []))
        if not approved_concepts:
            raise ValueError("No approved concepts found in ideation document")

        concept_index = 0
        concept = approved_concepts[concept_index]
        concept_id = str(concept.get("concept_id") or f"concept_{concept_index + 1}")

        format_group = phase_2_output.get("format_group", ideation_doc.get("format_group", "N"))
        is_visual = format_group == "V"
        beats = concept.get("composition_beats", []) if is_visual else concept.get("story_beats", [])
        beat_label = "COMPOSITION BEATS" if is_visual else "STORY BEATS"
        if not beats:
            raise ValueError(f"{'composition_beats' if is_visual else 'story_beats'} are missing for the selected approved concept")

        # scene_intelligence
        scene_intelligence = phase_2_output.get("scene_intelligence", ideation_doc.get("scene_intelligence", {}))
        selected_role = scene_intelligence.get("selected_role")
        placement_timing = scene_intelligence.get("placement_timing")
        atomic_elements = scene_intelligence.get("atomic_elements", {})

        # scene_integration_plan
        scene_integration_plan = phase_2_output.get("scene_integration_plan", ideation_doc.get("scene_integration_plan", {}))
        integration_strategy = scene_integration_plan.get("integration_strategy")
        scene_brief_for_generator = scene_integration_plan.get("scene_brief_for_generator")

        # platform_rules
        platform_rules = phase_2_output.get("platform_rules", ideation_doc.get("platform_rules", {}))
        authenticity_signal = platform_rules.get("authenticity_signal")
        soft_rules = platform_rules.get("soft_rules", [])

        # product_image_s3_url
        product_image_s3_url = phase_2_output.get("product_image_s3_url", "")

        # video type conditioning
        video_type_final = ideation_doc.get("video_type_final", project_doc.get("video_type", ""))
        video_type_conditioning_notes = ideation_doc.get("video_type_conditioning_notes", "")

        # --- Extract from project_doc ---
        product_details = project_doc.get("product_details", "")
        number_of_shots = project_doc.get("number_of_shots")
        if number_of_shots is None:
            logger.warning(f"[{agent_key}] number_of_shots not found in project doc | project_id={project_id}")

        # --- Extract from strategy_doc ---
        visual_context_summary = strategy_doc.get("visual_context_summary")

        # --- Extract from script_doc (Agent 1 + Agent 2 outputs) ---
        master_timeline = script_doc.get("master_timeline", {})
        master_timeline_windows = master_timeline.get("windows", []) if isinstance(master_timeline, dict) else []

        offer_constraints = script_doc.get("offer_constraints", {})
        visual_directive = offer_constraints.get("visual_directive", "")
        text_super_max_words = offer_constraints.get("text_super_max_words")
        cta_channel_rules = offer_constraints.get("cta_channel_rules", [])

        # --- Validations ---
        if not master_timeline_windows:
            raise ValueError("master_timeline.windows is missing in script document (Agent 1 must run first)")

        logger.info(
            f"[{agent_key}] Extracted key inputs: concept_id={concept_id}, "
            f"beats_count={len(beats)} (format_group={format_group}), "
            f"master_timeline_windows_count={len(master_timeline_windows)}, "
            f"selected_role_present={bool(selected_role)}, "
            f"placement_timing_present={bool(placement_timing)}, "
            f"atomic_elements_keys={list(atomic_elements.keys()) if isinstance(atomic_elements, dict) else 'N/A'}, "
            f"integration_strategy_present={bool(integration_strategy)}, "
            f"scene_brief_present={bool(scene_brief_for_generator)}, "
            f"visual_directive_present={bool(visual_directive)}, "
            f"text_super_max_words={text_super_max_words}, "
            f"cta_channel_rules_count={len(cta_channel_rules)}, "
            f"authenticity_signal={authenticity_signal}, "
            f"soft_rules_count={len(soft_rules)}, "
            f"product_details_present={bool(product_details)}, "
            f"visual_context_summary_present={bool(visual_context_summary)}, "
            f"product_image_s3_url_present={bool(product_image_s3_url)}, "
            f"pipeline_agent_logs_count={len(pipeline_doc.get('agent_logs', []))}"
        )

        # --- Build prompt ---
        prompt = f"""You are a master-level cinematic and mobile-first visual director for short-form video. You build exhaustively detailed, production-ready shot lists. Every shot description you write must be densely packed with visual information: exact camera movements, specific lighting setups, character micro-expressions, wardrobe details, production design, set dressing, and precise framing. Do not provide basic or generic descriptions. You think in vertical frames (9:16), in key frames that carry the whole idea, and in the principle that the simplest video needs only one great image built into or out of.

You are grounded strictly in the provided timeline, scene intelligence, and product visual context. Do not invent out-of-bounds settings, but deeply enrich the implied environments and actions with hyper-specific visual details.

You are building the shot list for concept: {concept_id}

{beat_label}:
{json.dumps(beats, ensure_ascii=True)}

MASTER TIMELINE (windows to fill):
{json.dumps(master_timeline_windows, ensure_ascii=True)}

SHOT STRUCTURE (non-negotiable):
Each window in the master timeline is exactly 8 seconds long. You must produce exactly {number_of_shots} ShotEntry objects — one per window, in order W01 through W{str(number_of_shots).zfill(2) if number_of_shots else "NN"}. Each ShotEntry represents one complete, uninterrupted camera setup that occupies the full 8-second window. Do NOT merge windows. Do NOT split a single window into multiple ShotEntry objects.

MANDATORY SCENE BRIEF:
Role: {selected_role}
Placement: {placement_timing}
Atomic elements — character action: {atomic_elements.get('character_action', 'N/A')}
Atomic elements — emotional tone: {atomic_elements.get('emotional_tone', 'N/A')}
Atomic elements — product interaction: {atomic_elements.get('product_interaction_type', 'N/A')}
Atomic elements — environmental context: {atomic_elements.get('environmental_context', 'N/A')}
Atomic elements — symbolic meaning: {atomic_elements.get('symbolic_meaning', 'N/A')}
Scene brief for generator: {scene_brief_for_generator}
Integration strategy: {integration_strategy}

OFFER WINDOW VISUAL CONSTRAINTS:
Visual directive: {visual_directive}
Text super max words: {text_super_max_words}
CTA channel rules: {json.dumps(cta_channel_rules, ensure_ascii=True)}

PRODUCT VISUAL CONTEXT (from product image and brand analysis):
{visual_context_summary}

PRODUCT:
{product_details}

PLATFORM REQUIREMENTS:
{json.dumps(soft_rules, ensure_ascii=True)}
Authenticity signal required: {authenticity_signal}

FORMAT TYPE: {video_type_final}
FORMAT CONDITIONING DIRECTIVES:
{video_type_conditioning_notes}

These directives override default cinematic conventions wherever they conflict. Apply them to every shot decision below:
- UGC / Organic-style: Describe shots as a handheld phone camera would capture them — tight, slightly imperfect framing, available light only. No professional lighting setups. No formal cinematography language (no lens mm values, no "dolly", no "negative fill"). Action must feel spontaneous and reactive, not choreographed. Environments should be lived-in and incidental, not dressed. The creator is speaking directly to camera, not performing for a crew.
- Testimonial / Real person: Emphasize physical product interaction and visible results. Lighting feels functional, not artistic. Framing favors the face — reactions and sincerity above composition.
- Animation / Illustrated: Visual descriptions define symbolic transformations and abstract sequences impossible in live action. Environments can be stylized or non-literal.
- Narrative / Mini-film: Shots may be more deliberately composed. Pacing can be slower. Multi-character framing is permitted.
- Satire / Comedy: Exaggerate the enemy visually. Shots can lean into absurdity. The brand moment is straight-faced contrast.
- Product Beauty: NO human talent in any shot. The product is the only subject. Describe macro-level texture shots (cream on skin-tone surface, liquid pour, powder swirl, ingredient reveal), extreme close-ups of packaging details, and dramatic lighting setups that reveal material quality (dewy sheen, translucency, crystalline particulates). Every shot is a still-life composition. Lighting is the primary expressive tool — describe it in full: direction, color temperature, specular highlights, shadow fall.
- Flatlay: NO human talent. Camera angle is strictly top-down (90° overhead). Product sits on a flat, styled surface. Describe exact prop placement, surface material (white marble, linen, stone, raw wood), color palette, and any botanical or ingredient elements. Movement is a slow, barely perceptible drift or a single prop sliding into frame. Negative space is compositional — describe what is NOT in the frame as deliberately as what is.
- CGI/3D Product: NO human talent. Describe the product in a fully rendered, non-photorealistic environment. Physics are non-literal — the product can float, rotate, be surrounded by particles, have its materials morph or shimmer. Every shot describes a specific material quality (frosted glass, brushed metal, glossy lacquer, soft silicone), a lighting rig (HDRI dome, rim lights, volumetric god rays), and an environmental concept (void space, branded color-field, abstract landscape). Think Dyson, Apple, and Dior product films.

TASK:
Step 1 — Identify the single key frame: the one image that, if shown as a still poster, would telegraph the entire concept's emotional idea. This is your visual anchor. Build all other shots into or out of this image. State what the key frame is before proceeding.

Step 2 — For each window in the master_timeline, provide EXTREME DETAIL for the following:
- framing: Exact camera angle, lens implication (e.g., wide 14mm, tight 85mm), and distance (e.g., Extreme Close Up, Medium Wide).
- subject: Highly specific description of the subject, including wardrobe textures, screen direction, posture, and micro-expressions.
- action: Choreography-level detail of what is physically happening. Break down the micro-actions, prop handling, and exact blocking in the space. Make it undeniable.
- lighting: Cinematic lighting directions (e.g., 'harsh top-down fluorescent with a green cast', 'soft warm golden hour backlight with negative fill on the right side').
- text_supers: Text content, screen placement (top/center/bottom), and duration in seconds.
- notes (use this field heavily): Include specific camera movement (e.g., 'whip pan left', 'slow push in on dolly', 'kinetic handheld snap zoom'), depth of field cues, and meticulous production design/set dressing details.
- location_heading: Write the Fountain scene heading for this shot. Format strictly as INT./EXT. LOCATION - DAY/NIGHT. Derive from the environmental context in the shot (lighting character, set dressing, implied space). Examples: 'INT. CORPORATE BREAKROOM - DAY', 'INT. BATHROOM - MORNING', 'EXT. CITY ROOFTOP - NIGHT'. Every window must have a location_heading.
- character_intro: If this is the FIRST window in the shot list where this character appears, write their screenplay introduction: 'CHARACTER NAME (age range, gender , ethnicity, defining physical trait, wardrobe detail)'. Example: 'CREATOR (late 20s, female , south indian , heavy-lidded eyes, oversized beige office blazer)'. Write an empty string if (a) this character has already been introduced in a prior window, or (b) no character is present in this shot.

Step 3 — Place the mandatory scene exactly within {placement_timing}. Preserve all atomic elements. The scene brief overrides any generic shot logic.

Step 4 — Apply the offer window constraints strictly. Text supers in the offer window must not exceed {text_super_max_words} words.

Step 5 — Apply 9:16 vertical framing to every shot. Flag any shot where the subject placement would be destructively cropped in vertical format.

CONSTRAINTS (apply last):
- NO GENERIC DESCRIPTIONS. "Something beautiful and mythic" or "person looking happy at desk" is forbidden. Every visual field must paint a vivid, meticulous, actionable picture.
- No shot may be empty — every window must have a distinctly detailed visual event.
- AI VIDEO MODEL LIMITS (CRITICAL): The destination is a generative AI video model (like Veo/Sora). YOU MUST AVOID complex physics, instantaneous environmental morphing (e.g., 'background instantly fractures', 'turning into a wind tunnel'), high-speed particle interactions (e.g., 'flying papers', 'extreme wind on hair'), or chaotic hyper-kinetic environments. 
- CAMERA MOVEMENT: Keep camera moves simple, stable, and grounded in physical reality (e.g., slow pan, subtle push-in, static composition). Do not use 'snap zoom', 'whip pan', or extreme speed. Keep environments stable.
- Each window is exactly 8 seconds.
- Shot count is non-negotiable: the output array must have exactly {number_of_shots} entries.
- One-to-one mapping between windows and shots is mandatory. Do not produce more or fewer entries than the number of windows.
- Output as a JSON array, one entry per window_id.

Return only valid JSON matching the schema exactly.
Do not include markdown, prose preamble, or code fences.
Do not add extra fields or omit required fields.
"""

        invoke_start = time.time()
        logger.info(f"Agent [3]: Preparing to call Gemini model={GEMINI_MODEL}...")
        try:
            client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
            )
            logger.info(f"Agent [3]: Gemini Client instantiated. Sending prompt...")

            if revision_prefix:
                existing_script_doc = await db[SCRIPT_COLLECTION].find_one(
                    {"project_id": str(project_id)}, {"shot_list": 1}
                )
                previous_shots = (existing_script_doc or {}).get("shot_list") or []
                if previous_shots:
                    revision_block = (
                        revision_prefix
                        + "\n\nYOUR PREVIOUS SHOT LIST — copy every shot VERBATIM except the windows listed above:\n"
                        + json.dumps(previous_shots, indent=2)
                    )
                else:
                    revision_block = revision_prefix
                prompt = revision_block + "\n\n" + prompt

            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": VisualSequencingResult.model_json_schema(),
                    "automatic_function_calling": {"disable": True},
                }
            )

            api_duration = time.time() - invoke_start
            logger.info(f"Agent [3]: Gemini API call completed in {api_duration:.2f}s")
            logger.debug(f"Agent [3]: Raw response length={len(response.text)} chars")

            cleaned_json = _clean_json_string(response.text)
            parsed_data = json.loads(cleaned_json)
            logger.info(f"Agent [3]: Successfully parsed JSON response.")

            result = VisualSequencingResult(**parsed_data)
            result.status = "completed"
            logger.info(f"Agent [3]: Successfully validated structured output with Pydantic.")
        except Exception as llm_error:
            if "503" in str(llm_error) or "429" in str(llm_error):
                logger.error(f"Gemini API overloaded: {llm_error}")
            else:
                logger.error(f"Agent [3]: Gemini invocation failed: {llm_error}", exc_info=True)
            raise

        logger.info("Agent [3]: Updating SCRIPT and PIPELINE collections...")
        await db[SCRIPT_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "project_id": str(project_id),
                    "shot_list": [shot.model_dump() for shot in result.shot_list],
                }
            },
            upsert=True,
        )

        pipeline_log = {
            "agent_id": 3,
            "agent_name": "visual_sequencing_agent",
            "execution_time_utc": time.time(),
            "api_duration_s": round(api_duration, 2),
            "duration_s": round(time.time() - run_start, 2),
            "reasoning": result.reasoning,
            "key_frame_rationale": result.key_frame_rationale,
            "mobile_compliance_check": result.mobile_compliance_check,
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )

        total_duration = time.time() - run_start
        logger.info("Agent [3]: Successfully updated SCRIPT and PIPELINE collections.")
        logger.info(f"Agent [3]: Complete! Total duration {total_duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed with error: {str(e)}", exc_info=True)
        raise
