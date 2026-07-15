import json
import logging
import os
import time
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

logger = logging.getLogger("zeroshot.phase2.diversity_manifest_generator")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")

HOOK_MECHANISM_TYPES = [
    "pattern-interrupt",
    "direct-address",
    "contrast-reveal",
    "question-open",
    "demonstration",
    "transformation-reveal",
    "social-proof",
    "authority",
    "curiosity",
    "comedic-inversion",
]

EMOTIONAL_REGISTERS = [
    "aspirational",
    "anxious",
    "curious",
    "satisfied",
    "defiant",
    "amused",
    "moved",
    "surprised",
    "proud",
    "relieved",
]

ARGUMENT_ANGLES = [
    "benefit",
    "objection-kill",
    "social-proof",
    "authority",
    "transformation",
    "urgency",
    "comparison",
    "identity",
    "curiosity",
    "humor",
]


class ConceptSlot(BaseModel):
    slot_id: str = Field(description="Slot identifier: S01 through S10")
    hook_mechanism_type: str = Field(description="One of the 10 hook mechanism types — must be unique across all slots")
    emotional_register: str = Field(description="The primary emotional state this concept must evoke in the viewer")
    argument_angle: str = Field(description="The persuasion strategy this concept employs")
    buyer_problem: str = Field(description="The specific audience pain point or desire this slot addresses — must be distinct from other slots")
    format_constraint: str = Field(description="Format-specific execution constraint for this slot, derived from the video type and format_group")


class DiversityManifestResult(BaseModel):
    concept_slots: List[ConceptSlot] = Field(description="Exactly 10 ConceptSlots, each with a distinct hook_mechanism_type")
    reasoning: str = Field(description="Reasoning for the slot design decisions — how diversity was enforced")
    diversity_guarantee: str = Field(description="One paragraph confirming that no two slots share the same hook_mechanism_type, and that emotional_register and argument_angle are distributed across the portfolio")
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


async def run_diversity_manifest_generator(project_id: str, db) -> DiversityManifestResult:
    """
    Pre-generation agent that creates a DiversityManifest: 10 typed ConceptSlots with
    distinct hook mechanisms, emotional registers, and argument angles.
    The concept_generator fills exactly one concept per slot, guaranteeing structural
    differentiation before generation begins.
    RUN CONDITION: ALWAYS — runs before concept_generator for both Group N and Group V.
    """
    agent_key = "diversity_manifest_generator"
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

    format_group = ideation_doc.get("format_group", "N")
    video_type_final = ideation_doc.get("video_type_final", "UGC")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    platform_rules = ideation_doc.get("platform_rules", {})
    priority_directives = ideation_doc.get("priority_directives", {})

    strategy_agents = strategy_doc.get("agents", {})
    human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth", "")
    enemy = strategy_agents.get("conflict_identification", {}).get("enemy", "")
    offer_hook = strategy_agents.get("value_prop_and_offer", {}).get("offer_hook", "")
    persona = strategy_agents.get("audience_persona", {})

    product_details = project_doc.get("product_details", "")

    # Format-specific constraint template for slots
    if format_group == "V":
        format_slot_note = (
            f"This is a {video_type_final} format (Group V — visual-first). "
            "format_constraint must specify a visual execution constraint (e.g. 'hook must be an extreme macro shot', "
            "'no human elements — product only', 'motion must be product-driven not camera-driven'). "
            "Do NOT specify narrative or character constraints."
        )
    else:
        format_slot_note = (
            f"This is a {video_type_final} format (Group N — narrative-first). "
            "format_constraint must specify a narrative execution constraint appropriate for this format "
            "(e.g. 'hook must break the fourth wall', 'must contain a specific result claim', "
            "'tone must be absurdist — enemy is exaggerated'). "
            "Visual constraints are secondary."
        )

    prompt = f"""You are the Diversity Manifest Generator for an advertising concept portfolio.

Your purpose is to design 10 structural slots that concept_generator must fill — one concept per slot. These slots enforce diversity BEFORE generation begins, preventing all concepts from sharing the same hook mechanism, emotional register, or argument angle.

This is the most important diversity enforcement in the pipeline. If two slots share the same hook_mechanism_type, the portfolio will produce redundant concepts that reduce A/B learning value and converge on the same audience segment.

AVAILABLE HOOK MECHANISM TYPES (10 total — each slot must use a different one):
{json.dumps(HOOK_MECHANISM_TYPES, indent=2)}

AVAILABLE EMOTIONAL REGISTERS:
{json.dumps(EMOTIONAL_REGISTERS, indent=2)}

AVAILABLE ARGUMENT ANGLES:
{json.dumps(ARGUMENT_ANGLES, indent=2)}

CAMPAIGN CONTEXT:
- Product: {product_details}
- Human truth: {human_truth}
- Enemy (the problem the product solves): {enemy}
- Offer hook: {offer_hook}
- Video type: {video_type_final} (format_group: {format_group})
- Platform rules: {json.dumps(platform_rules, indent=2)}
- Brand guardrails: {json.dumps(brand_guardrails, indent=2)}
- Priority directives: {json.dumps(priority_directives, indent=2)}
- Audience persona: {json.dumps(persona, indent=2)}

FORMAT SLOT GUIDANCE:
{format_slot_note}

RULES:
1. Generate exactly 10 ConceptSlots (S01 through S10).
2. Each slot must use a DIFFERENT hook_mechanism_type — no repeats across the 10 slots.
3. No two slots may share the same emotional_register (variety is required).
4. No two slots may share the same argument_angle (variety is required).
5. Each slot's buyer_problem must address a DIFFERENT specific audience concern or desire.
6. The format_constraint must be specific and executable — not vague ("make it interesting") but concrete ("hook must show the product from below at table level — a perspective the viewer has never seen").

SLOT DESIGN LOGIC:
- S01-S03: High-interrupt, immediate-hook slots. Hook physics are disruptive (pattern-interrupt, direct-address, comedic-inversion). Designed for cold audiences.
- S04-S06: Proof and credibility slots. Hook builds trust before asking for belief (social-proof, authority, demonstration). Designed for warm audiences who need evidence.
- S07-S09: Desire and aspiration slots. Hook creates want before revealing the product (curiosity, transformation-reveal, contrast-reveal). Designed for audiences who are open but unconvinced.
- S10: Experimental slot. Must score ≥4 on both hook intensity (pattern-interrupt or comedic-inversion) and ironic/unexpected framing. This is the high-risk/high-reward concept.

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
                "response_json_schema": DiversityManifestResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info("[%s] Gemini call completed | duration=%.2fs", agent_key, api_duration)

        cleaned = _clean_json_string(response.text)
        parsed = json.loads(cleaned)
        result = DiversityManifestResult(**parsed)
        result.status = "completed"

    except Exception as e:
        logger.error("[%s] Gemini call failed | error=%s", agent_key, e)
        raise

    # Validate: exactly 10 slots with unique hook_mechanism_types
    hooks_used = [slot.hook_mechanism_type for slot in result.concept_slots]
    unique_hooks = set(hooks_used)
    if len(result.concept_slots) != 10:
        raise ValueError(f"[{agent_key}] Expected 10 concept slots, got {len(result.concept_slots)}")
    if len(unique_hooks) < 8:
        logger.warning("[%s] Only %d unique hook types across 10 slots — diversity is below target", agent_key, len(unique_hooks))

    total_duration = time.time() - start_time

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "diversity_manifest": {
                    "concept_slots": [s.model_dump() for s in result.concept_slots],
                    "diversity_guarantee": result.diversity_guarantee,
                },
                "status.diversity_manifest_generator": "completed",
                "updated_at": time.time(),
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_key": agent_key,
            "agent_name": agent_key,
            "status": "completed",
            "reasoning": result.reasoning,
            "unique_hook_types": len(unique_hooks),
            "slot_count": len(result.concept_slots),
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

    logger.info("[%s] Completed | %d slots | %d unique hooks | duration=%.2fs", agent_key, len(result.concept_slots), len(unique_hooks), total_duration)
    return result
