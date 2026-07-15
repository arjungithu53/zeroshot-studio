import json
import logging
import os
import time
import asyncio
from typing import List, Optional, Literal

from google import genai
from pydantic import BaseModel, Field
from bson import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("zeroshot.narrative_archetype_selector")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))

PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")

# ── Pydantic Models for Structured Output ────────────────────────────────────

class ArchetypeScore(BaseModel):
    archetype: str
    human_truth_fit: int
    enemy_compatibility: int
    offer_integration: int
    total: int

class SelectedArchetype(BaseModel):
    archetype_name: str
    fit_score: int
    micro_policy: str
    rationale: str
    failure_modes: List[str]

class NarrativeArchetypeResult(BaseModel):
    status: Literal["completed", "error", "skipped"]
    reason: Optional[str] = None
    
    # Output Schema
    selected_archetype: Optional[SelectedArchetype] = None
    
    # Pipeline Log Fields
    reasoning: Optional[str] = None
    archetype_scoring_table: Optional[List[ArchetypeScore]] = None
    failure_mode_analysis: Optional[str] = None

# ---------------------------------------------------------------------------
# Core Agent Function
# ---------------------------------------------------------------------------

def _clean_json_string(raw_response: str) -> str:
    """Removes markdown code blocks to safely parse JSON."""
    cleaned = raw_response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

async def run_narrative_archetype_selector(project_id: str, db) -> NarrativeArchetypeResult:
    """
    Agent 14: narrative_archetype_selector.
    RUN CONDITION: ALWAYS.
    Selects the deep emotional archetype that governs the portfolio's psychological logic.
    """
    logger.info("Initializing Agent 14 (Narrative Archetype Selector) | project_id=%s", project_id)
    start_time = time.time()

    # 1. Fetch relevant data
    logger.info(f"Agent 14: Fetching data for project_id={project_id}")
    try:
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})

        if not strategy_doc or not ideation_doc:
            logger.error(f"Agent 14: Missing strategy or ideation documents for project: {project_id}")
            raise ValueError(f"Missing required documents for project: {project_id}")

        logger.info(f"Agent 14: DB fetch successful. Found strategy and ideation docs.")
    except Exception as e:
        logger.error("Agent 14: Failed to fetch documents | error=%s", e)
        return NarrativeArchetypeResult(status="error", reason=str(e))

    # Defensive guard: this agent is only valid for narrative-group formats.
    # The orchestrator routing prevents it from running for Phase 2-V, but guard anyway.
    format_group = ideation_doc.get("format_group", "N")
    if format_group == "V":
        logger.info("Agent 14: Skipping — format_group is V (visual formats use visual_motif_selector) | project_id=%s", project_id)
        return NarrativeArchetypeResult(status="skipped", reason="visual format group — use visual_motif_selector")

    # Retrieve inputs matching schema
    strategy_agents = strategy_doc.get("agents", {})
    human_truth = strategy_agents.get("central_human_truth", {}).get("human_truth")
    
    conflict_ident = strategy_agents.get("conflict_identification", {})
    enemy = conflict_ident.get("enemy")
    enemy_type = conflict_ident.get("enemy_type")
    conflict_statement = conflict_ident.get("conflict_statement")
    
    offer_hook = strategy_agents.get("value_prop_and_offer", {}).get("offer_hook")

    video_type_final = ideation_doc.get("video_type_final")
    narrative_skeleton = ideation_doc.get("narrative_skeleton", {})
    priority_directives = ideation_doc.get("priority_directives", {})

    logger.info(f"Agent 14: Extracted inputs | human_truth={bool(human_truth)}, enemy={bool(enemy)}, video_type_final={video_type_final}")

    # Validation check for critical fields
    if not human_truth or not enemy or not video_type_final or not narrative_skeleton:
        error_msg = "Critical fields missing from DB (human_truth, enemy, video_type_final, or narrative_skeleton)"
        logger.error(f"Agent 14: Aborting - {error_msg} | project_id={project_id}")
        raise ValueError(error_msg)

    # 2. Construct prompt
    prompt = f"""
    You are Agent 14: narrative_archetype_selector.
    
    PURPOSE:
    Select the deep emotional archetype (Ritual, Transformation, Rebellion, Comparison, Curiosity Loop, Social Proof) that governs the portfolio's psychological logic. 
    Score against human_truth fit, enemy compatibility, and offer integration. Write micro-policy rules the Concept Generator follows.

    INPUT DATA:
    "human_truth": "{human_truth}",
    "enemy": "{enemy}",
    "enemy_type": "{enemy_type}",
    "conflict_statement": "{conflict_statement}",
    "offer_hook": "{offer_hook}",
    "video_type_final": "{video_type_final}",
    "narrative_skeleton": {json.dumps(narrative_skeleton)},
    "priority_directives": {json.dumps(priority_directives)}
    
    PROMPT LOGIC:
    This agent chooses the deep structural archetype for the concept portfolio. Unlike the narrative skeleton (which defines the beat sequence), the archetype defines the emotional logic and audience psychology that powers the narrative.

    Evaluate the following archetypes against the human_truth and enemy:
    - Ritual: The product is framed as a sacred, repeated act of self-restoration. Best when the audience craves a sense of belonging to an ancient or larger-than-self practice. Payoff is peace, not triumph.
    - Transformation: A clear before/after arc where the product is the catalyst for visible change. Best when the enemy creates a measurable, visible deficit. Payoff is relief and confidence.
    - Rebellion: The product is an act of defiance against a system, norm, or condition the audience resents. Best when the enemy is a condition or social norm, not a competitor. Payoff is agency and reclamation.
    - Comparison: The product is positioned against what the audience previously used or accepted. Best when the strategic sweet spot allows direct or implicit contrast. Payoff is superiority and vindication.
    - Curiosity Loop: The ad withholds a resolution, creating compulsive engagement. Best in longer formats or when the product has an unexpected benefit. Payoff is revelation.
    - Social Proof: The product's power is demonstrated through witness — others who changed. Best when trust is the primary purchase barrier. Payoff is permission and belonging.

    enemy_type weighting rule:
    Before scoring, apply a prior weight based on enemy_type "{enemy_type}":
    If enemy_type is "condition" or "norm", add +1 to Rebellion and Ritual scores.
    If enemy_type is "competitor", add +1 to Comparison.
    If enemy_type is "behaviour", add +1 to Transformation.
    This prevents archetype selection from being purely abstract when the enemy's nature already signals the correct emotional logic.

    Score each archetype on three dimensions (out of 10):
    (1) Fit with human_truth — does this archetype amplify the emotional core?
    (2) Enemy compatibility — does this archetype give the enemy a clear structural role?
    (3) Offer integration — can the commercial mechanics naturally land in this archetype?

    Select the highest-scoring archetype. Write the full selected_archetype object including micro_policy (rules governing hook types, arc length, interrupt intensity), a one-paragraph rationale, and a list of failure modes.

    OUTPUT FORMAT:
    Output STRICTLY JSON matching this schema. NO markdown formatting.
    """

    # 3. Call Gemini
    invoke_start = time.time()
    logger.info(f"Agent 14: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 14: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": NarrativeArchetypeResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 14: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 14: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 14: Successfully parsed JSON response.")

        result = NarrativeArchetypeResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 14: Successfully validated structured output with Pydantic.")

    except asyncio.TimeoutError:
        logger.error("Agent 14: Call to Gemini API timed out")
        return NarrativeArchetypeResult(status="error", reason="Gemini API Timeout")
    except Exception as exc:
        logger.error("Agent 14: Error generating/parsing | error=%s", exc)
        return NarrativeArchetypeResult(status="error", reason=str(exc))

    total_duration = time.time() - start_time

    # 4. Save results
    logger.info(f"Agent 14: Updating IDEATION and PIPELINE collections for project_id={project_id}...")
    try:
        if result.selected_archetype:
            await db[IDEATION_COLLECTION].update_one(
                {"project_id": str(project_id)},
                {"$set": {
                    "selected_archetype": result.selected_archetype.model_dump(),
                    "status.narrative_archetype_selector": "completed",
                    "updated_at": time.time()
                }},
                upsert=True
            )
            logger.info("Agent 14: IDEATION collection updated with selected_archetype.")

        # Serialize archetype scoring table manually if it exists to ensure purely primitive data in DB
        scoring_table_dump = None
        if result.archetype_scoring_table:
            scoring_table_dump = [score.model_dump() for score in result.archetype_scoring_table]

        pipeline_log = {
            "agent_key": "narrative_archetype_selector",
            "status": "completed",
            "reasoning": result.reasoning,
            "archetype_scoring_table": scoring_table_dump,
            "failure_mode_analysis": result.failure_mode_analysis,
            "duration_secs": total_duration,
            "api_duration_secs": api_duration,
            "timestamp": time.time(),
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info("Agent 14: PIPELINE collection updated with execution log.")

    except Exception as e:
        logger.error(f"Agent 14: Failed to save results to DB | error={e}")
        return NarrativeArchetypeResult(status="error", reason="DB save failed")

    logger.info(f"Agent 14 completed successfully | total_duration={total_duration:.2f}s | project_id={project_id}")
    return result
