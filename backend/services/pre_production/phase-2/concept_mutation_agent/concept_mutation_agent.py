import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

logger = logging.getLogger("zeroshot.phase2.concept_mutation_agent")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "90000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")


class MutatedConcept(BaseModel):
    concept_id: str = Field(description="Must match the original concept_id exactly.")
    mutation_applied: str = Field(description="One sentence describing exactly what was changed and why.")
    mutated_fields: Dict[str, Any] = Field(
        description=(
            "Only the fields that were changed. All other fields are inherited verbatim from the original. "
            "Group V mutable fields: concept_hook, composition_beats, texture_moments, motion_choreography, "
            "lighting_rationale, virality_lever, cta_framing, risk_tags. "
            "Group N mutable fields: concept_hook, story_beats, character_description, virality_lever, "
            "cta_framing, risk_tags. "
            "NEVER include: concept_id, offer_placement, category, kill_score, format_adherence_score, provenance."
        )
    )


class ConceptMutationResult(BaseModel):
    regen_round: int = Field(description="Which regeneration round this is (1, 2, or 3).")
    mutated_concepts: List[MutatedConcept]
    reasoning: str = Field(description="Overall reasoning for the mutation decisions.")
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


def _get_mutation_instruction(concept_id: str, guidance: dict) -> Optional[str]:
    """Find the mutation instruction for a concept from kill_switch_guidance."""
    for entry in guidance.get("regeneration_guidance", []):
        if entry.get("concept_id") == concept_id:
            return entry.get("specific_mutation")
    for entry in guidance.get("eliminated_guidance", []):
        if entry.get("concept_id") == concept_id:
            return entry.get("fix_required")
    return None


def _get_lowest_dimension(concept_id: str, guidance: dict) -> str:
    """Identify the lowest-scoring dimension from the kill_score_breakdown."""
    weights = {
        "virality_score": 0.20,
        "constraint_score": 0.20,
        "platform_score": 0.15,
        "novelty_score": 0.20,
        "offer_clarity_score": 0.10,
        "format_adherence_score": 0.15,
    }
    for entry in guidance.get("kill_score_breakdown", []):
        if entry.get("concept_id") == concept_id:
            scores = {dim: entry.get(dim, 10) for dim in weights}
            return min(scores, key=lambda d: scores[d])
    return "novelty_score"


async def run_concept_mutation_agent(project_id: str, db) -> ConceptMutationResult:
    """
    Concept Mutation Agent — runs in the regen loop after concept_kill_switch.
    Takes all non-approved concepts, applies targeted mutations based on kill switch guidance,
    and writes the improved concepts back to ideation.concept_portfolio.
    """
    agent_key = "concept_mutation_agent"
    start_time = time.time()
    logger.info("[%s] Starting | project_id=%s", agent_key, project_id)

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

    # ---------------------------------------------------------------------------
    # Identify non-approved concepts and their mutation instructions
    # ---------------------------------------------------------------------------
    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    if not concept_portfolio:
        raise ValueError("concept_portfolio is empty — nothing to mutate")

    guidance = ideation_doc.get("kill_switch_guidance", {})
    if not guidance:
        raise ValueError("kill_switch_guidance is missing — kill_switch must run before mutation agent")

    phase_2_output = ideation_doc.get("phase_2_output", {})
    approved_ids = {c.get("concept_id") for c in phase_2_output.get("approved_concepts", [])}

    # Determine which regen round we're on (count pipeline logs from this agent)
    pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    prior_mutation_logs = [
        log for log in pipeline_doc.get("agent_logs", [])
        if log.get("agent_name") == agent_key
    ]
    regen_round = len(prior_mutation_logs) + 1
    logger.info("[%s] Regen round %d | project_id=%s", agent_key, regen_round, project_id)

    # Build list of concepts to mutate
    concepts_to_mutate = [c for c in concept_portfolio if c.get("concept_id") not in approved_ids]
    if not concepts_to_mutate:
        logger.info("[%s] All concepts already approved — nothing to mutate", agent_key)
        return ConceptMutationResult(
            regen_round=regen_round,
            mutated_concepts=[],
            reasoning="All concepts already approved — no mutation needed.",
            status="skipped",
        )

    logger.info("[%s] Mutating %d concepts | round=%d", agent_key, len(concepts_to_mutate), regen_round)

    # ---------------------------------------------------------------------------
    # Format context
    # ---------------------------------------------------------------------------
    format_group = ideation_doc.get("format_group", "N")
    is_visual = format_group == "V"
    video_type_final = ideation_doc.get("video_type_final", "UGC")
    product_details = project_doc.get("product_details", "")

    if is_visual:
        motif_data = ideation_doc.get("selected_visual_motif", {})
        style_policy = motif_data.get("visual_micro_policy", "")
        mutable_fields = ["concept_hook", "composition_beats", "texture_moments", "motion_choreography", "lighting_rationale", "virality_lever", "cta_framing", "risk_tags"]
        format_rule = (
            "This is a VISUAL-FIRST format. Mutations must remain ingredient-anchored. "
            "composition_beats must describe visual frame evolution — texture change, angle shift, lighting reveal. "
            "NEVER add story_beats, character arcs, or narrative structure. "
            "NEVER add human talent for Product Beauty/Flatlay/CGI-3D formats. "
            "Every composition beat must reference a real physical ingredient or material property."
        )
    else:
        archetype_data = ideation_doc.get("selected_archetype", {})
        style_policy = archetype_data.get("micro_policy", "")
        mutable_fields = ["concept_hook", "story_beats", "character_description", "virality_lever", "cta_framing", "risk_tags"]
        format_rule = (
            "This is a NARRATIVE-FIRST format. story_beats must describe the narrative arc. "
            "character_description must be specific enough for Veo character consistency. "
            "The archetype micro_policy governs tone throughout."
        )

    # Build per-concept mutation briefs
    mutation_briefs = []
    for concept in concepts_to_mutate:
        cid = concept.get("concept_id")
        instruction = _get_mutation_instruction(cid, guidance)
        lowest_dim = _get_lowest_dimension(cid, guidance)
        kill_score = concept.get("kill_score", 0)
        format_adherence = concept.get("format_adherence_score", 0)

        mutation_briefs.append({
            "concept_id": cid,
            "current_kill_score": kill_score,
            "current_format_adherence": format_adherence,
            "lowest_scoring_dimension": lowest_dim,
            "specific_mutation_instruction": instruction or f"Improve the {lowest_dim} dimension. Make the concept more specific to the product and audience.",
            "original_concept": concept,
        })

    strat_agents = strategy_doc.get("agents", {})
    human_truth = strat_agents.get("central_human_truth", {}).get("human_truth", "")
    enemy = strat_agents.get("conflict_identification", {}).get("enemy", "")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})

    prompt = f"""You are the Concept Mutation Agent. You perform targeted, surgical improvements to advertising concepts that scored below the approval threshold.

MUTATION PRINCIPLES:
1. Apply ONLY the specific mutation instruction for each concept. Do not rewrite the whole concept.
2. The mutation must directly address the lowest-scoring dimension.
3. Preserve: concept_id, offer_placement, category, provenance, and all other non-mutated fields exactly.
4. After mutation, the concept must still be traceable to this specific product and audience.

FORMAT: {video_type_final} (Group {'V — Visual-First' if is_visual else 'N — Narrative-First'})
{format_rule}

PRODUCT: {product_details}
HUMAN TRUTH: {human_truth}
ENEMY: {enemy}
STYLE POLICY: {style_policy}
BRAND GUARDRAILS: {json.dumps(brand_guardrails, indent=2)}

MUTABLE FIELDS (only these may change): {json.dumps(mutable_fields)}

CONCEPTS TO MUTATE (regen round {regen_round}):
{json.dumps(mutation_briefs, indent=2)}

SCORING CONTEXT:
- Concepts need ≥7.5 composite to be approved
- virality_score (20%): specificity of virality mechanic integration
- constraint_score (20%): hard constraint compliance
- platform_score (15%): platform survivability, no unsubstantiated claims
- novelty_score (20%): pattern interrupt integration with product-grounded reason
- offer_clarity_score (10%): offer communicable within offer window
- format_adherence_score (15%): does this concept actually look/feel like {video_type_final}?

TASK:
For each concept in the list:
1. Read the specific_mutation_instruction carefully
2. Identify which field(s) in the mutable_fields list need to change to address the instruction
3. Write the improved version of ONLY those fields
4. Return the concept_id, a one-sentence mutation_applied description, and ONLY the changed fields in mutated_fields

Return strictly valid JSON matching the output schema. Include every concept from the input list.
"""

    invoke_start = time.time()
    logger.info("[%s] Calling Gemini model=%s | mutating %d concepts", agent_key, GEMINI_MODEL, len(concepts_to_mutate))
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
                "response_json_schema": ConceptMutationResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info("[%s] Gemini call completed | duration=%.2fs", agent_key, api_duration)

        cleaned = _clean_json_string(response.text)
        parsed = json.loads(cleaned)
        result = ConceptMutationResult(**parsed)
        result.regen_round = regen_round
        result.status = "completed"

    except Exception as e:
        logger.error("[%s] Gemini call failed | error=%s", agent_key, e)
        raise

    # ---------------------------------------------------------------------------
    # Apply mutations to concept_portfolio
    # ---------------------------------------------------------------------------
    mutated_by_id = {m.concept_id: m for m in result.mutated_concepts}
    updated_portfolio = []

    for concept in concept_portfolio:
        cid = concept.get("concept_id")
        mutation = mutated_by_id.get(cid)
        if mutation and mutation.mutated_fields:
            # Merge: start with original, overlay only the mutated fields
            updated = dict(concept)
            for field, value in mutation.mutated_fields.items():
                # Safety: never allow mutation of protected fields
                if field not in ("concept_id", "offer_placement", "category", "kill_score",
                                 "format_adherence_score", "provenance", "_review_format_score",
                                 "_review_brief_score"):
                    updated[field] = value
            # Clear stale scores so kill switch re-evaluates from scratch
            updated.pop("kill_score", None)
            updated.pop("format_adherence_score", None)
            updated.pop("_review_format_score", None)
            updated.pop("_review_brief_score", None)
            updated_portfolio.append(updated)
            logger.info("[%s] Mutated %s: %s", agent_key, cid, mutation.mutation_applied)
        else:
            updated_portfolio.append(concept)

    total_duration = time.time() - start_time

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "concept_portfolio": updated_portfolio,
                "status.concept_mutation_agent": f"round_{regen_round}_completed",
                "updated_at": time.time(),
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_name": agent_key,
            "regen_round": regen_round,
            "concepts_mutated": len(mutated_by_id),
            "reasoning": result.reasoning,
            "mutations": [m.model_dump() for m in result.mutated_concepts],
            "duration_secs": total_duration,
            "api_duration_secs": api_duration,
            "timestamp": time.time(),
            "status": "completed",
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
    except Exception as e:
        logger.error("[%s] DB save failed | error=%s", agent_key, e)
        raise

    logger.info("[%s] Round %d complete | %d concepts mutated | duration=%.2fs",
                agent_key, regen_round, len(mutated_by_id), total_duration)
    return result
