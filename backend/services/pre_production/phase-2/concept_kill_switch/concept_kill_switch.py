import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from bson import ObjectId
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.concept_kill_switch")

# ---------------------------------------------------------------------------
# Scoring weights — must sum to 100%
# ---------------------------------------------------------------------------
WEIGHTS = {
    "virality": 0.20,
    "constraint": 0.20,
    "platform": 0.15,
    "novelty": 0.20,
    "offer_clarity": 0.10,
    "format_adherence": 0.15,
}

COMPOSITE_APPROVAL_THRESHOLD = 7.5
FORMAT_ADHERENCE_MINIMUM = 5.0   # Hard floor — below this eliminates regardless of composite


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class KillScoreBreakdown(BaseModel):
    concept_id: str
    virality_score: float = Field(description="Specificity of virality mechanic integration (weight 20%)")
    constraint_score: float = Field(description="Hard constraint compliance (weight 20%)")
    platform_score: float = Field(description="Platform survivability and unsubstantiated claims check (weight 15%)")
    novelty_score: float = Field(description="Conventions-to-break integration with product-grounded reason (weight 20%)")
    offer_clarity_score: float = Field(description="Offer communicable within offer window (weight 10%)")
    format_adherence_score: float = Field(description="Does the concept actually look/feel/execute like the stated video type? Group V: visual-first, no character arc required. Group N: format-native authenticity register. (weight 15%)")
    idea_integrity_deduction: float = Field(description="Deduction if project idea is non-null and concept fails integrity test")
    interest_filter_deduction: float = Field(description="Deduction if concept was flagged 'boring' by interest_filter_agent")
    composite_score: float = Field(description="Weighted composite of all dimensions minus deductions")
    decision: str = Field(description="'approved', 'flagged_for_regen', or 'eliminated'")


class EliminatedConceptLog(BaseModel):
    concept_id: str
    score: float
    failure_reason: str
    fix_required: str


class RegenerationGuidance(BaseModel):
    concept_id: str
    specific_mutation: str


class ConceptKillSwitchResult(BaseModel):
    reasoning: str
    kill_score_breakdown: List[KillScoreBreakdown]
    eliminated_concepts_log: List[EliminatedConceptLog]
    regeneration_guidance: List[RegenerationGuidance]
    status: Optional[str] = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _clean_json_string(json_string: str) -> str:
    cleaned = json_string.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


async def fetch_agent_output(pipeline_collection, project_id, agent_id):
    """Returns the output of the MOST RECENT log entry for agent_id.
    Uses last-wins scan so regen rounds 2 and 3 read fresh interest_filter data,
    not the stale entry from round 1."""
    doc = await pipeline_collection.find_one(
        {"project_id": str(project_id)},
        {"agent_logs": 1},
    )
    if not doc:
        raise Exception("Pipeline doc not found")
    result = None
    for log in doc.get("agent_logs", []):
        if log.get("agent_id") == agent_id:
            result = log.get("output")
    if result is None:
        raise Exception(f"Agent {agent_id} output not found")
    return result


# ---------------------------------------------------------------------------
# Format adherence rubric (injected into prompt)
# ---------------------------------------------------------------------------
def _build_format_adherence_rubric(format_group: str, video_type_final: str) -> str:
    if format_group == "V":
        return f"""FORMAT ADHERENCE SCORING RUBRIC (weight 15%) — {video_type_final} (Group V — Visual-First):

Score 9-10: Concept is entirely visual-first. composition_beats describe ONLY visual flow (texture change, angle shift, lighting reveal, product interaction). No character arc, no dialogue requirement, no human talent (for Product Beauty/Flatlay). Visual hook is specific and format-native.
Score 7-8: Concept is mostly visual-first with minor narrative contamination (e.g. an implied emotional journey).
Score 5-6: Concept has partial narrative structure but could still be executed as a visual format with adaptation.
Score 3-4: Concept has a clear narrative arc or character requirement that conflicts with this visual format.
Score 1-2: Concept is a narrative marketing concept — has story_beats, character arc, dialogue. This is a format collapse failure.

AUTOMATIC ELIMINATION RULE: Any concept with format_adherence_score < 5.0 is ELIMINATED regardless of composite score."""
    else:
        return f"""FORMAT ADHERENCE SCORING RUBRIC (weight 15%) — {video_type_final} (Group N — Narrative-First):

Score 9-10: Concept feels completely native to {video_type_final}. Hook physics match the format. Character register is correct and specific. story_beats are narrative beats, not visual descriptions.
Score 7-8: Concept is mostly format-native with minor register inconsistencies.
Score 5-6: Concept could work in this format but feels like a generic ad rather than format-specific content.
Score 3-4: Concept feels like a brand film template applied to the wrong format, or lacks character specificity.
Score 1-2: Concept has no format-specific properties — it could be any format.

Format-specific requirements for {video_type_final}:
- UGC: hook must feel unscripted/creator-native, first-person register, specific personal claim
- Testimonial: must contain at least one concrete result claim, emotional sincerity over cleverness
- Satire: enemy must be personified/exaggerated, brand is the straight-faced solution
- Narrative: multi-beat emotional arc with proper setup and payoff
- Realistic: believable, non-exaggerated, naturalistic execution

AUTOMATIC ELIMINATION RULE: Any concept with format_adherence_score < 5.0 is ELIMINATED regardless of composite score."""


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------
async def run_concept_kill_switch_agent(project_id: str, db: Any) -> Dict[str, Any]:
    logger.info(f"Initializing Agent 25 [concept_kill_switch] for project_id={project_id}...")
    start_time = time.time()

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as exc:
        raise ValueError(f"Invalid project_id {project_id}")

    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    if not ideation_doc:
        raise ValueError(f"Ideation not found in {IDEATION_COLLECTION}")

    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
    if not strategy_doc:
        raise ValueError(f"Strategy not found in {STRATEGY_COLLECTION}")

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    constraint_graph = ideation_doc.get("constraint_graph", {})
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    idea_core_rules = ideation_doc.get("idea_core_rules", {})
    platform_rules = ideation_doc.get("platform_rules", {})
    format_group = ideation_doc.get("format_group", "N")
    video_type_final = ideation_doc.get("video_type_final", "UGC")

    strategy_agents = strategy_doc.get("agents", {})
    competitive_landscape = strategy_agents.get("competitive_landscape", {})
    conventions_to_break = competitive_landscape.get("conventions_to_break", [])
    conventions_to_follow = competitive_landscape.get("conventions_to_follow", [])
    venn_model = strategy_agents.get("strategy_models", {}).get("venn_model", {})
    brand_can_say = venn_model.get("brand_can_say", [])
    positioning_statement = strategy_agents.get("positioning_alignment", {}).get("positioning_statement", "")

    # Interest filter flags from pipeline
    try:
        interest_data = await fetch_agent_output(
            pipeline_collection=db[PIPELINE_COLLECTION],
            project_id=project_id,
            agent_id=24,
        )
        interest_filter_flags = {
            "concept_interest_scores": interest_data.get("concept_interest_scores", []),
            "boring_flags": interest_data.get("boring_flags", []),
            "thumb_stopper_flags": interest_data.get("thumb_stopper_flags", []),
        } if interest_data else {}
    except Exception as e:
        logger.warning(f"Agent 25: Could not fetch interest_filter output: {e}")
        interest_filter_flags = {}

    format_adherence_rubric = _build_format_adherence_rubric(format_group, video_type_final)

    input_data = {
        "concept_portfolio": concept_portfolio,
        "constraint_graph": constraint_graph,
        "brand_guardrails": brand_guardrails,
        "idea_core_rules": idea_core_rules,
        "platform_rules": platform_rules,
        "format_group": format_group,
        "video_type_final": video_type_final,
        "conventions_to_break": conventions_to_break,
        "conventions_to_follow": conventions_to_follow,
        "brand_can_say": brand_can_say,
        "positioning_statement": positioning_statement,
        "interest_filter_flags": interest_filter_flags,
    }

    prompt = f"""You are concept_kill_switch — the final hard quality filter for advertising concepts.

Compute a composite kill_score per concept out of 10 using the weighted formula below.
The composite score determines the decision. FORMAT ADHERENCE HAS A HARD FLOOR — see rules.

SCORING FORMULA (weights must be applied exactly):
- virality_score (weight 20%): specificity of virality mechanic integration — is it embedded in a specific beat or just mentioned?
- constraint_score (weight 20%): hard constraint violation = 0. Soft violations (conventions_to_follow) reduce score by 1pt per missing item (max 2pt reduction).
- platform_score (weight 15%): obeys platform_rules. Unsubstantiated claims (not in brand_can_say) reduce score by 1.5pt.
- novelty_score (weight 20%): conventions_to_break integration with product-grounded reason. Generic inversions score 3-4.
- offer_clarity_score (weight 10%): offer communicable within offer window.
- format_adherence_score (weight 15%): see rubric below.

DEDUCTIONS (applied after weighted composite):
- idea_integrity_deduction: -2 if project idea is non-null and concept fails integrity tests.
- interest_filter_deduction: -1.5 if concept flagged 'boring' by interest_filter_agent.

composite_score = (virality*0.20 + constraint*0.20 + platform*0.15 + novelty*0.20 + offer_clarity*0.10 + format_adherence*0.15) - idea_integrity_deduction - interest_filter_deduction

DECISION RULES:
- format_adherence_score < {FORMAT_ADHERENCE_MINIMUM}: decision = "eliminated" (format floor violation — regardless of composite)
- composite_score < 6.5: decision = "eliminated"
- composite_score 6.5-7.4: decision = "flagged_for_regen"
- composite_score >= {COMPOSITE_APPROVAL_THRESHOLD}: decision = "approved"

{format_adherence_rubric}

For each eliminated concept, provide: failure_reason (the specific dimension that failed and why) and fix_required (a concrete, actionable change to the concept that would raise it above threshold).

For each flagged concept, provide: specific_mutation (the exact dimension and direction needed to reach approved status).

Input Data:
{json.dumps(input_data, indent=2)}

Schema:
{json.dumps(ConceptKillSwitchResult.model_json_schema(), indent=2)}
"""

    invoke_start = time.time()
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
                "response_json_schema": ConceptKillSwitchResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        result = ConceptKillSwitchResult(**parsed_data)
        result.status = "completed"

    except Exception as e:
        logger.error(f"Agent 25: Gemini failed: {str(e)}")
        raise e

    # ---------------------------------------------------------------------------
    # Apply decisions — no fallback approval
    # ---------------------------------------------------------------------------
    approved_concepts = []
    score_mapping = {kb.concept_id: kb for kb in result.kill_score_breakdown}
    product_image_obj = project_doc.get("product_image", {}) if isinstance(project_doc, dict) else {}
    product_image_s3_url = (
        project_doc.get("product_image_s3_url")
        or product_image_obj.get("s3_url", "")
    )

    # Also pull phase2_review scores to use as a tiebreaker
    phase2_review = ideation_doc.get("phase2_review", {})
    review_scores = {
        r.get("concept_id"): r
        for r in phase2_review.get("concept_reviews", [])
        if isinstance(r, dict)
    }

    for concept in concept_portfolio:
        cid = concept.get("concept_id")
        breakdown = score_mapping.get(cid)
        if breakdown:
            concept["kill_score"] = breakdown.composite_score
            concept["format_adherence_score"] = breakdown.format_adherence_score
            # Embed phase2_review scores for sort tiebreaking
            review = review_scores.get(cid, {})
            concept["_review_format_score"] = review.get("format_adherence_score", 5)
            concept["_review_brief_score"] = review.get("brief_fidelity_score", 5)
            if breakdown.decision == "approved":
                approved_concepts.append(concept)

    # Sort approved concepts: best composite first, format adherence as tiebreaker,
    # then brief fidelity. Phase 3 reads approved_concepts[0] — this ensures it always
    # gets the highest-quality concept, not simply the first one generated.
    approved_concepts.sort(
        key=lambda c: (
            c.get("kill_score", 0),
            c.get("format_adherence_score", 0),
            c.get("_review_format_score", 0),
            c.get("_review_brief_score", 0),
        ),
        reverse=True,
    )
    logger.info(
        "Agent 25: Approved concept ranking: %s",
        [(c.get("concept_id"), round(c.get("kill_score", 0), 2)) for c in approved_concepts],
    )

    # approved_concepts may be empty — the orchestrator regen loop handles routing.
    # Do NOT promote flagged concepts or raise here; that is now the regen loop's job.
    if approved_concepts:
        logger.info("Agent 25: %d/%d concepts approved", len(approved_concepts), len(concept_portfolio))
    else:
        non_eliminated = sum(
            1 for kb in result.kill_score_breakdown
            if kb.decision != "eliminated"
        )
        logger.warning(
            "Agent 25: No approved concepts (≥7.5). %d flagged_for_regen / %d eliminated — "
            "routing to regen loop | project_id=%s",
            non_eliminated,
            len(concept_portfolio) - non_eliminated,
            project_id,
        )

    phase_2_output = {
        "approved_concepts": approved_concepts,
        "constraint_graph": constraint_graph,
        "narrative_budget": ideation_doc.get("narrative_budget", {}),
        "scene_intelligence": ideation_doc.get("scene_intelligence", {}),
        "scene_integration_plan": ideation_doc.get("scene_integration_plan", {}),
        "visual_structure": ideation_doc.get("visual_structure", {}),
        "selected_archetype": ideation_doc.get("selected_archetype", {}),
        "selected_visual_motif": ideation_doc.get("selected_visual_motif", {}),
        "platform_rules": platform_rules,
        "video_type_final": video_type_final,
        "format_group": format_group,
        "product_image_s3_url": product_image_s3_url,
    }

    total_duration = time.time() - start_time

    # kill_switch_guidance is written to ideation so concept_mutation_agent can read it
    # directly without scanning pipeline logs.
    kill_switch_guidance = {
        "regeneration_guidance": [g.model_dump() for g in result.regeneration_guidance],
        "eliminated_guidance": [
            {"concept_id": e.concept_id, "fix_required": e.fix_required}
            for e in result.eliminated_concepts_log
        ],
        "kill_score_breakdown": [b.model_dump() for b in result.kill_score_breakdown],
    }

    await db[IDEATION_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$set": {
            "concept_portfolio": concept_portfolio,
            "phase_2_output": phase_2_output,
            "kill_switch_guidance": kill_switch_guidance,
            "updated_at": time.time(),
        }},
    )

    pipeline_log = {
        "agent_id": 25,
        "agent_name": "concept_kill_switch",
        "execution_time": total_duration,
        "api_duration": api_duration,
        "status": "completed",
        "approved_count": len(approved_concepts),
        "total_count": len(concept_portfolio),
        "output": result.model_dump(),
        "timestamp": time.time(),
    }
    await db[PIPELINE_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$push": {"agent_logs": pipeline_log}},
        upsert=True,
    )

    logger.info(f"Agent 25: Completed in {total_duration:.2f}s | {len(approved_concepts)} approved")
    return {
        "status": "success",
        "message": "Agent 25 execution complete",
        "data": result.model_dump(),
    }
