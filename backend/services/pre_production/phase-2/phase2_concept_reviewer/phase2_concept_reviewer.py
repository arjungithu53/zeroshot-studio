import json
import logging
import os
import time
from typing import List, Optional

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

logger = logging.getLogger("zeroshot.phase2.phase2_concept_reviewer")

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")


class ConceptReview(BaseModel):
    concept_id: str
    format_adherence_score: int = Field(description="1-10: Does this concept actually look/feel like the stated video type? A Product Beauty concept that has a character arc scores 1. A UGC concept written in brand-voice scores 2.", ge=1, le=10)
    distinctiveness_score: int = Field(description="1-10: Is this concept structurally distinct from the other concepts in this portfolio? Concepts that share hook physics, emotional register, and argument angle with 2+ other concepts score 1-3.", ge=1, le=10)
    brief_fidelity_score: int = Field(description="1-10: Does this concept reflect the brand strategy (human truth, enemy, offer hook, brand adjective) from the creative brief? Generic concepts that could apply to any product score 1-3.", ge=1, le=10)
    overall_pass: bool = Field(description="True if concept passes minimum thresholds: format_adherence >= 6 AND distinctiveness >= 5 AND brief_fidelity >= 5")
    failure_reasons: List[str] = Field(description="Specific, actionable failure reasons. Empty list if overall_pass is True.", default_factory=list)


class Phase2ReviewResult(BaseModel):
    concept_reviews: List[ConceptReview]
    portfolio_summary: str = Field(description="One paragraph summary of portfolio health: what worked, what failed, what the kill switch will need to address")
    format_adherence_portfolio_score: float = Field(description="Mean format_adherence_score across all concepts")
    reasoning: str
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


async def run_phase2_concept_reviewer(project_id: str, db) -> Phase2ReviewResult:
    """
    Phase 2 Concept Reviewer: First reviewer node in Phase 2.
    Reviews each generated concept for format adherence, structural distinctiveness,
    and brief fidelity. Does not block the pipeline — flags are passed to the kill switch.
    RUN CONDITION: ALWAYS — runs after concept_generator for both Group N and Group V.
    """
    agent_key = "phase2_concept_reviewer"
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

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    if not concept_portfolio:
        raise ValueError("concept_portfolio is empty — concept_generator must run before reviewer")

    format_group = ideation_doc.get("format_group", "N")
    video_type_final = ideation_doc.get("video_type_final", "UGC")
    brand_guardrails = ideation_doc.get("brand_guardrails", {})
    diversity_manifest = ideation_doc.get("diversity_manifest", {})

    strategy_agents = strategy_doc.get("agents", {})
    human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth", "")
    enemy = strategy_agents.get("conflict_identification", {}).get("enemy", "")
    offer_hook = strategy_agents.get("value_prop_and_offer", {}).get("offer_hook", "")
    brand_adjective = strategy_agents.get("brand_adjective", "")

    product_details = project_doc.get("product_details", "")

    # Format-specific scoring rubric
    if format_group == "V":
        format_rubric = f"""FORMAT ADHERENCE RUBRIC for {video_type_final} (Group V — Visual-First):
Score 9-10: Concept is entirely visual-first. No character arc. composition_beats describe visual flow, not narrative beats. No human talent requirement (for Product Beauty/Flatlay). Visual hook mechanism is specific and format-native.
Score 7-8: Concept is mostly visual-first but has minor narrative contamination (e.g. a vague emotional journey implied).
Score 5-6: Concept has partial narrative structure but could still be executed as a visual format with adaptation.
Score 3-4: Concept has a clear narrative arc or character requirement that conflicts with the format.
Score 1-2: Concept is completely a narrative format concept — has story beats, character arc, dialogue. This is a format collapse failure."""
    else:
        format_rubric = f"""FORMAT ADHERENCE RUBRIC for {video_type_final} (Group N — Narrative-First):
Score 9-10: Concept feels completely native to {video_type_final}. Hook physics match the format (UGC: unscripted feel; Testimonial: personal experience; Satire: absurdist enemy). Character register is correct.
Score 7-8: Concept is mostly format-native with minor register inconsistencies.
Score 5-6: Concept could work in this format but feels like a generic ad rather than format-specific content.
Score 3-4: Concept feels like a brand film template applied to the wrong format.
Score 1-2: Concept has no format-specific properties — it could be any format."""

    prompt = f"""You are a senior advertising creative reviewer conducting a Phase 2 portfolio review.

Your role is to identify format failures, structural clustering, and brief drift BEFORE the kill switch runs. You are the first reviewer in Phase 2. You do not approve or reject concepts — you score them and flag issues. The kill switch makes the final approval decision.

Be brutally honest. A concept that scores 8 on format_adherence when it clearly has a character arc for a Product Beauty brief is a failure of this review.

CAMPAIGN CONTEXT:
- Product: {product_details}
- Video type: {video_type_final} (format_group: {format_group})
- Human truth: {human_truth}
- Enemy: {enemy}
- Offer hook: {offer_hook}
- Brand adjective: {brand_adjective}
- Brand guardrails: {json.dumps(brand_guardrails, indent=2)}

DIVERSITY MANIFEST (what was requested — used to assess distinctiveness):
{json.dumps(diversity_manifest, indent=2)}

CONCEPT PORTFOLIO TO REVIEW:
{json.dumps(concept_portfolio, indent=2)}

{format_rubric}

DISTINCTIVENESS RUBRIC:
Score 9-10: This concept's hook_mechanism_type, emotional_register, and buyer_problem are all unique in the portfolio.
Score 7-8: 2 of 3 dimensions are unique.
Score 5-6: 1 of 3 dimensions is unique — concept is structurally similar to 1-2 others.
Score 3-4: This concept is essentially the same as another concept with surface theme variation.
Score 1-2: This concept is a near-duplicate of another concept.

BRIEF FIDELITY RUBRIC:
Score 9-10: Human truth, enemy, and brand adjective are all recognizable in the concept. Could not apply to any other product.
Score 7-8: 2 of 3 brief elements are clearly present.
Score 5-6: 1 brief element is present; concept is generic but brand-appropriate.
Score 3-4: Concept is generic marketing copy that could apply to any product in this category.
Score 1-2: Concept has no relationship to the brand strategy inputs.

OVERALL_PASS THRESHOLD:
A concept passes if ALL of these are true:
- format_adherence_score >= 6
- distinctiveness_score >= 5
- brief_fidelity_score >= 5

If any one falls below threshold, overall_pass = false and failure_reasons must be specific and actionable.

Review every concept in concept_portfolio. Return strictly valid JSON matching the output schema.
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
                "response_json_schema": Phase2ReviewResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            },
        )
        api_duration = time.time() - invoke_start
        logger.info("[%s] Gemini call completed | duration=%.2fs", agent_key, api_duration)

        cleaned = _clean_json_string(response.text)
        parsed = json.loads(cleaned)
        result = Phase2ReviewResult(**parsed)
        result.status = "completed"

    except Exception as e:
        logger.error("[%s] Gemini call failed | error=%s", agent_key, e)
        raise

    passing = sum(1 for r in result.concept_reviews if r.overall_pass)
    logger.info("[%s] Review complete | %d/%d concepts passed | mean format adherence=%.1f",
                agent_key, passing, len(result.concept_reviews), result.format_adherence_portfolio_score)

    total_duration = time.time() - start_time

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {
                "phase2_review": {
                    "concept_reviews": [r.model_dump() for r in result.concept_reviews],
                    "portfolio_summary": result.portfolio_summary,
                    "format_adherence_portfolio_score": result.format_adherence_portfolio_score,
                    "passing_count": passing,
                },
                "status.phase2_concept_reviewer": "completed",
                "updated_at": time.time(),
            }},
            upsert=True,
        )

        pipeline_log = {
            "agent_key": agent_key,
            "agent_name": agent_key,
            "status": "completed",
            "passing_concepts": passing,
            "total_concepts": len(result.concept_reviews),
            "format_adherence_portfolio_score": result.format_adherence_portfolio_score,
            "portfolio_summary": result.portfolio_summary,
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
