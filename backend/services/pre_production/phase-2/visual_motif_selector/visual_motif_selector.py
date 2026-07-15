import json
import logging
import os
import time
from typing import List, Literal, Optional

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

logger = logging.getLogger("zeroshot.phase2.visual_motif_selector")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")


VISUAL_MOTIFS = Literal[
    "texture-reveal",
    "scale-surprise",
    "transformation-reveal",
    "material-showcase",
    "spatial-composition",
    "color-story",
    "light-play",
]


class MotifScore(BaseModel):
    motif: str
    brand_fit: int = Field(description="1-10: How well this motif serves the brand adjective and guardrails")
    product_fit: int = Field(description="1-10: How well this motif showcases the product's strongest visual properties")
    format_fit: int = Field(description="1-10: How native this motif is to the stated video format")
    total: int


class VisualMotifResult(BaseModel):
    selected_motif: VISUAL_MOTIFS = Field(description="The primary organizing visual motif for this concept portfolio")
    motif_rationale: str = Field(description="One paragraph explaining why this motif was selected over alternatives")
    visual_micro_policy: str = Field(description="Concise rules governing how the motif must be applied across all concepts — hook requirements, composition constraints, texture emphasis, lighting rules, what is forbidden")
    failure_modes: List[str] = Field(description="3-5 specific ways this motif can fail for this product/format combination")
    motif_scoring_table: List[MotifScore] = Field(description="Scores for all 7 motifs")
    reasoning: str = Field(description="Full reasoning log for pipeline")
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


async def run_visual_motif_selector(project_id: str, db) -> VisualMotifResult:
    """
    Phase 2-V Agent: Visual Motif Selector.
    Selects the organizing visual motif that governs the portfolio's visual logic.
    Replaces narrative_archetype_selector for Group V formats.
    RUN CONDITION: format_group == "V" only.
    """
    agent_key = "visual_motif_selector"
    logger.info("[%s] Starting | project_id=%s", agent_key, project_id)
    start_time = time.time()

    try:
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project not found: {project_id}")

        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
        if not ideation_doc:
            raise ValueError(f"Ideation document not found: {project_id}")

        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    except Exception as e:
        logger.error("[%s] DB fetch failed | error=%s", agent_key, e)
        raise

    video_type_final = ideation_doc.get("video_type_final", "Product Beauty")
    video_type_conditioning_notes = ideation_doc.get("video_type_conditioning_notes", "")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    visual_structure = ideation_doc.get("visual_structure", {})

    product_details = project_doc.get("product_details", "")
    strategy_agents = strategy_doc.get("agents", {})
    human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth", "")
    brand_adjective = strategy_agents.get("brand_adjective", "")

    prompt = f"""You are the Visual Motif Selector for a {video_type_final} advertisement portfolio.

Your purpose is to select the single organizing visual motif that governs the psychological and aesthetic logic of all concepts in this portfolio. This is the visual-format equivalent of a narrative archetype — the deep structural principle that makes the portfolio feel coherent and intentional.

THE 7 VISUAL MOTIFS:
1. texture-reveal: The organizing principle is the progressive revelation of surface texture. Best when the product has physically compelling material properties (grain, weave, crystalline structure, liquid viscosity). Each concept is organized around a different texture encounter.

2. scale-surprise: The organizing principle is unexpected scale contrast — macro shots that reveal details invisible to the naked eye, or scale shifts that recontextualize the product's size and presence. Best when the product has interesting micro-scale properties or when oversized/undersized presentation creates surprise.

3. transformation-reveal: The organizing principle is a physical or material state change. The product or its ingredients undergo visible transformation (dissolving, blooming, crystallizing, melting, expanding). Best when the product has active ingredient stories or before/after visual properties.

4. material-showcase: The organizing principle is the pure celebration of material quality — the richness, precision, and craftsmanship of what the product is made of. Best for premium/luxury products where material provenance is a brand signal.

5. spatial-composition: The organizing principle is the arrangement of objects in space. Negative space, balance, proportion, and the relationship between the product and its surrounding elements is the primary expressive tool. Best for Flatlay formats and minimalist brand codes.

6. color-story: The organizing principle is color — hue relationships, color temperature shifts, saturation progressions, and the product's chromatic identity. Each concept is organized around a different color narrative. Best when the product has distinctive color or packaging.

7. light-play: The organizing principle is the behavior of light on the product's surface — refraction, reflection, translucency, shadow patterns. Best when the product has interesting optical properties (glass, liquids, metallic surfaces, transparent packaging).

INPUTS:
- Video type: {video_type_final}
- Product: {product_details}
- Brand adjective: {brand_adjective}
- Human truth: {human_truth}
- Visual structure (generated): {json.dumps(visual_structure, indent=2)}
- Video type conditioning notes: {video_type_conditioning_notes}
- Brand guardrails: {json.dumps(brand_guardrails, indent=2)}

TASK:
Step 1 — Score each of the 7 motifs on 3 dimensions (1-10 each):
- brand_fit: Does this motif serve the brand adjective and honor the brand guardrails?
- product_fit: Does this motif showcase the product's strongest visual properties?
- format_fit: How native is this motif to the stated video format?

Step 2 — Select the highest-scoring motif. If scores tie, prefer the motif with the highest product_fit.

Step 3 — Write the visual_micro_policy: the specific rules governing how this motif must be applied across all concepts. Include: hook requirements, composition constraints, what ingredient textures/elements must appear, lighting rules, what is explicitly forbidden.

CRITICAL RULE FOR visual_micro_policy: Every rule you write must be grounded in the product's actual ingredient properties — their specific colors, material states, surface textures, and physical behaviors at macro scale. Do NOT write rules that import the brand's narrative enemy or lifestyle conflict as a visual element (e.g. "every concept must open with frost crystals representing AC damage" is WRONG — frost has nothing to do with the product's ingredients).

The test for every rule you write: does this rule reference a real material property of the product or its ingredients? If yes, keep it. If the rule references an abstract metaphor for a brand pain point (frost, dust, heat waves, corrosion, etc.), DELETE it and replace it with an ingredient-specific equivalent.

WRONG: "Every concept must open with jagged frost crystals being shattered by the product."
RIGHT: "Every concept must open with an extreme macro of a hero ingredient — e.g. turmeric powder dispersing in warm golden light, or a citrus cross-section revealing backlit translucency — that is visually arresting at 0 seconds."

WRONG: "Lighting must begin cold and harsh to simulate environmental damage."
RIGHT: "Lighting must begin neutral to reveal the ingredient's true color, then shift to warm directional light to activate the ingredient's luminous properties."

Step 4 — List 3-5 specific failure modes: ways this motif can fail for this particular product/format combination.

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
                "response_json_schema": VisualMotifResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info("[%s] Gemini call completed | duration=%.2fs", agent_key, api_duration)

        cleaned = _clean_json_string(response.text)
        parsed = json.loads(cleaned)
        result = VisualMotifResult(**parsed)
        result.status = "completed"

    except Exception as e:
        logger.error("[%s] Gemini call failed | error=%s", agent_key, e)
        raise

    total_duration = time.time() - start_time

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "selected_visual_motif": {
                    "selected_motif": result.selected_motif,
                    "motif_rationale": result.motif_rationale,
                    "visual_micro_policy": result.visual_micro_policy,
                    "failure_modes": result.failure_modes,
                },
                "status.visual_motif_selector": "completed",
                "updated_at": time.time(),
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_key": agent_key,
            "status": "completed",
            "selected_motif": result.selected_motif,
            "reasoning": result.reasoning,
            "motif_scoring_table": [s.model_dump() for s in result.motif_scoring_table],
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
