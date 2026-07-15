import json
import logging
import os
import time
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

logger = logging.getLogger("zeroshot.phase2.visual_structure_agent")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")


class CompositionBeat(BaseModel):
    beat_label: str = Field(description="Label for this visual beat (e.g. 'Texture Reveal', 'Scale Surprise', 'Ingredient Cascade')")
    visual_function: str = Field(description="What this beat accomplishes visually and emotionally for the viewer")
    seconds_budget: float = Field(description="Duration of this beat in seconds")


class VisualStructureOutput(BaseModel):
    composition_flow: List[CompositionBeat] = Field(description="Ordered sequence of visual beats. Total seconds_budget must equal video_length_seconds.")
    texture_inventory: List[str] = Field(description="Specific textures, materials, and surfaces that must appear. Min 3 items.")
    lighting_sequence: List[str] = Field(description="Lighting states in order: ambient setup, key reveals, accent moments. Min 3 entries.")
    motion_choreography: List[str] = Field(description="Camera and product movement types in sequence (e.g. 'slow push into macro', 'orbital rotation', 'locked overhead drift'). One per beat.")
    visual_hook_mechanism: str = Field(description="What makes the first frame impossible to scroll past — described as a single, specific visual event or property.")
    reasoning: str = Field(description="Reasoning for structural choices tied to the video type and brand.")
    status: Optional[str] = Field(default="completed")


def _clean_json_string(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


async def run_visual_structure_agent(project_id: str, db) -> VisualStructureOutput:
    """
    Phase 2-V Agent: Visual Structure Agent.
    Generates the visual composition architecture for Group V formats (Product Beauty,
    Flatlay, CGI/3D Product, Animated). Replaces narrative_skeleton_generator and
    narrative_skeleton_planner for visual-first formats.
    RUN CONDITION: format_group == "V" only.
    """
    agent_key = "visual_structure_agent"
    logger.info("[%s] Starting | project_id=%s", agent_key, project_id)
    start_time = time.time()

    try:
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project not found: {project_id}")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    except Exception as e:
        logger.error("[%s] DB fetch failed | error=%s", agent_key, e)
        raise

    video_type_final = ideation_doc.get("video_type_final", "Product Beauty")
    video_length_seconds = project_doc.get("video_length_seconds", 30)
    video_type_conditioning_notes = ideation_doc.get("video_type_conditioning_notes", "")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    narrative_budget = ideation_doc.get("narrative_budget", {})

    # Scene intelligence is available if scene_deconstruction ran (preferred_scene was set)
    scene_intelligence = ideation_doc.get("scene_intelligence", {})
    atomic_elements = scene_intelligence.get("atomic_elements", {})

    product_details = project_doc.get("product_details", "")
    strategy_agents = strategy_doc.get("agents", {})
    human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth", "")
    brand_adjective = strategy_agents.get("brand_adjective", "")

    format_rules = {
        "product beauty": (
            "No human talent. The product is the sole subject. "
            "Every beat explores the object itself — texture, liquid movement, ingredient close-ups, material surfaces. "
            "Lighting and composition carry the emotional arc. Hook is a visually arresting product macro shot. "
            "VO is minimal — silence is an asset. The product must be visible and central in every beat."
        ),
        "flatlay": (
            "No human talent. Product shot from directly above on a clean, styled surface. "
            "Props (ingredients, botanicals, fabric swatches, tools) arranged to reinforce brand codes. "
            "Color palette and negative space are primary expressive tools. "
            "Movement is subtle — a slow drift or a single prop entering frame. "
            "Text supers carry the informational load."
        ),
        "cgi/3d product": (
            "No human talent. Product exists in a fully stylized, rendered environment. "
            "Physics can be non-literal (floating, slow-motion liquid, particle effects, material morphs). "
            "Brand world-building is the primary task — every surface, light source, and environmental element reinforces brand identity. "
            "This is a brand film for the object, not a demo."
        ),
        "animated": (
            "No live-action talent required. Visual metaphor and symbolic transformation are permitted and encouraged. "
            "Physics and scale can be non-literal. Tone can heighten to epic or surreal. "
            "Movement should feel purposefully designed — not generic motion graphics. "
            "Every animated element should reinforce the brand's emotional positioning."
        ),
    }
    format_key = video_type_final.lower().strip()
    format_rule = format_rules.get(format_key, format_rules["product beauty"])

    prompt = f"""You are the Visual Structure Agent for a {video_type_final} advertisement.

Your purpose is to design the visual composition architecture that will govern all downstream concept generation for this format. This is NOT a narrative skeleton. There are no characters, no story arcs, no dialogue. This is a visual choreography plan.

FORMAT RULES FOR {video_type_final.upper()}:
{format_rule}

INPUTS:
- Product: {product_details}
- Brand adjective: {brand_adjective}
- Human truth (emotional foundation): {human_truth}
- Video length: {video_length_seconds} seconds
- Video type conditioning notes: {video_type_conditioning_notes}
- Brand guardrails: {json.dumps(brand_guardrails, indent=2)}
- Narrative budget (timing guidance): {json.dumps(narrative_budget, indent=2)}
- Scene atomic elements (from preferred scene if set): {json.dumps(atomic_elements, indent=2)}

INGREDIENT-FIRST RULE (non-negotiable):
Your primary visual raw material is the product's actual physical ingredients — not the brand's narrative enemy, competitive conflict, or strategic pain point. Read the product description above and extract every hero ingredient. Name them. Describe their specific macro-scale visual properties: color, translucency, crystalline structure, liquid viscosity, powder dispersion, surface texture. Every composition beat, every texture inventory item, every lighting decision must be anchored to these real materials.

DO NOT translate the brand's narrative enemy into a visual hazard (e.g. if the brand fights "AC-dried skin", do not invent frost crystals as your opening visual). The enemy belongs to narrative formats. In {video_type_final}, the product's own ingredients are the conflict, the resolution, and the story — all three.

A visual built on real ingredients is always more memorable than a visual built on a metaphor for a pain point, because the ingredient IS the product. A citrus cross-section backlit to translucency IS Vitamin C. A turmeric cloud dissolving into cream IS the product's formula. These images sell the product directly. Frost metaphors sell a concept about the product.

The only permitted exception: if the product's physical use-case naturally produces a genuine visual contrast (e.g. sunscreen meeting UV light), that contrast must emerge from the product's material reality — not from an abstract lifestyle metaphor.

TASK:
Design the visual structure for this {video_type_final} ad. Start by listing the hero ingredients and their visual properties, then build every beat from those materials.

Step 1 — Composition flow:
Generate a sequence of visual beats. Each beat is a distinct visual event (not a narrative event). Label each beat by its visual function (e.g. "Macro texture reveal", "Scale shift to product silhouette", "Ingredient cascade", "Color payoff", "Brand mark arrival"). Assign seconds_budget to each beat. Total seconds_budget MUST exactly equal {video_length_seconds}.

Step 2 — Texture inventory:
List the specific textures, materials, and surfaces that must appear in this clip. Be precise and evocative (e.g. "matte ceramic glaze with hairline crackle", "raw crystalline mineral surface", "liquid suspension with slow settling particle"). Minimum 3 items.

Step 3 — Lighting sequence:
Describe the lighting states in order across the clip. Include: ambient setup light, key reveal light, accent moments. Be specific (e.g. "warm side-fill from 45° with soft shadow", "hard top-light to emphasize surface texture", "backlit product rim to separate from background").

Step 4 — Motion choreography:
Assign one camera or product movement type per composition beat. Use precise movement language (e.g. "locked macro — no movement", "slow push from 30cm to 5cm", "orbital rotation at 45° tilt", "overhead drift left to right at 2cm/sec").

Step 5 — Visual hook mechanism:
Describe in one specific sentence what makes the first frame impossible to scroll past. This must be a concrete visual property or event, not a vague description (e.g. "The first frame is an extreme macro of the product's crystalline surface catching a single ray of backlight — a detail the viewer has never seen at this scale").

Return strictly valid JSON matching the output schema.
"""

    invoke_start = time.time()
    logger.info("[%s] Calling Gemini model=%s", agent_key, GEMINI_MODEL)
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": VisualStructureOutput.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info("[%s] Gemini call completed | duration=%.2fs", agent_key, api_duration)

        cleaned = _clean_json_string(response.text)
        parsed = json.loads(cleaned)
        result = VisualStructureOutput(**parsed)
        result.status = "completed"

    except Exception as e:
        logger.error("[%s] Gemini call failed | error=%s", agent_key, e)
        raise

    total_duration = time.time() - start_time

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "visual_structure": {
                    "composition_flow": [b.model_dump() for b in result.composition_flow],
                    "texture_inventory": result.texture_inventory,
                    "lighting_sequence": result.lighting_sequence,
                    "motion_choreography": result.motion_choreography,
                    "visual_hook_mechanism": result.visual_hook_mechanism,
                },
                "status.visual_structure_agent": "completed",
                "updated_at": time.time(),
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_key": agent_key,
            "status": "completed",
            "reasoning": result.reasoning,
            "duration_secs": total_duration,
            "api_duration_secs": api_duration,
            "timestamp": time.time(),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
    except Exception as e:
        logger.error("[%s] DB save failed | error=%s", agent_key, e)
        raise

    logger.info("[%s] Completed | duration=%.2fs | project_id=%s", agent_key, total_duration, project_id)
    return result
