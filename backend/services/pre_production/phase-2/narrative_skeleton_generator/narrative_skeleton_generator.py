import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field

from google import genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("zeroshot.phase2.narrative_skeleton_generator")

# ---------------------------------------------------------------------------
# Environment / Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _clean_json_string(raw_text: str) -> str:
    """Strips markdown code fences like ```json ... ```."""
    stripped = raw_text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json"):]
    elif stripped.startswith("```"):
        stripped = stripped[len("```"):]
    
    if stripped.endswith("```"):
        stripped = stripped[:-3]
        
    return stripped.strip()

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class NarrativeBeat(BaseModel):
    beat_label: str
    emotional_function: str
    seconds_budget: float

class SkeletonsScored(BaseModel):
    skeleton: str
    emotional_arc_fit: int
    budget_fit: int
    scene_home_fit: int
    diversity_from_others: int
    total: int

class Agent12Output(BaseModel):
    selected_skeleton: str
    beat_sequence: List[NarrativeBeat]
    # Pipeline Log fields (included to be returned strictly inside the same call)
    reasoning: str
    candidate_skeletons_scored: List[SkeletonsScored]
    selection_rationale: str
    status: Optional[str] = None

# ---------------------------------------------------------------------------
# Runner Function
# ---------------------------------------------------------------------------
async def run_narrative_skeleton_generator(project_id: str, db: Any) -> Agent12Output:
    start_time = time.time()
    logger.info(f"Initializing Agent 12: narrative_skeleton_generator for project_id={project_id}...")

    logger.info(f"Agent 12: Fetching data for project_id={project_id}...")

    # 1. Fetch DB records
    try:
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
        ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    except Exception as e:
        logger.error(f"Agent 12: DB Fetch failed: {e}")
        raise ValueError(f"Agent 12: DB Fetch failed: {e}")

    if not strategy_doc:
        raise ValueError(f"Agent 12: Missing record in STRATEGY_COLLECTION for project {project_id}")
    if not ideation_doc:
        raise ValueError(f"Agent 12: Missing record in IDEATION_COLLECTION for project {project_id}")

    # Defensive guard: narrative skeleton is only valid for narrative-group formats.
    format_group = ideation_doc.get("format_group", "N")
    if format_group == "V":
        logger.info("Agent 12: Skipping — format_group is V (visual formats use visual_structure_agent) | project_id=%s", project_id)
        result = Agent12Output(
            selected_skeleton="skipped",
            beat_sequence=[],
            reasoning="Skipped — visual format group uses visual_structure_agent instead",
            candidate_skeletons_scored=[],
            selection_rationale="N/A — visual format",
            status="skipped",
        )
        return result

    # Extract required fields from strategy
    truth_conflict_platform = strategy_doc.get("truth_conflict_platform", {})
    campaign_platform = truth_conflict_platform.get("selected_platform", "N/A")

    conflict_identification = strategy_doc.get("conflict_identification", {})
    enemy = conflict_identification.get("enemy", "N/A")
    conflict_statement = conflict_identification.get("conflict_statement", "N/A")
    
    central_human_truth_model = strategy_doc.get("central_human_truth", {})
    human_truth = central_human_truth_model.get("human_truth", "N/A")
    
    # Extract required fields from project
    video_length_seconds = project_doc.get("video_length_seconds", 30) if project_doc else 30
    
    # Extract required fields from ideation
    video_type_final = ideation_doc.get("video_type_final", "N/A")
    narrative_budget = ideation_doc.get("narrative_budget", {})
    scene_integration_plan = ideation_doc.get("scene_integration_plan", {})
    priority_directives = ideation_doc.get("priority_directives", {})

    logger.info(
        f"Agent 12: Successfully extracted inputs: "
        f"campaign_platform='{str(campaign_platform)[:30]}...', "
        f"human_truth='{str(human_truth)[:30]}...', "
        f"enemy='{str(enemy)[:30]}...', "
        f"conflict_statement='{str(conflict_statement)[:30]}...', "
        f"video_type_final='{video_type_final}'"
    )

    # 2. Prepare Prompt
    prompt = f"""
You are the narrative_skeleton_generator expert.
Your Purpose is to generate three abstract story flow templates (e.g. 'Shock -> Recognition -> Ritual -> Reclamation -> Offer').
Each beat must be labelled by its emotional function, not content. 
You must select the strongest skeleton based on emotional arc, budget fit, scene home fit, and cross-candidate diversity.

INPUT DATA:
- campaign_platform: {campaign_platform}
- human_truth: {human_truth}
- enemy: {enemy}
- conflict_statement: {conflict_statement}

- video_type_final: {video_type_final}
- video_length_seconds: {video_length_seconds}
- narrative_budget: {json.dumps(narrative_budget, indent=2)}
- scene_integration_plan: {json.dumps(scene_integration_plan, indent=2)}
- priority_directives: {json.dumps(priority_directives, indent=2)}

PROMPT LOGIC & REQUIREMENTS:
This agent creates abstract story flow templates that stabilise concept generation before divergence begins. A narrative skeleton is not a concept — it is the structural backbone that concepts must fill.

Generate three candidate narrative skeletons suited to strategy.agents.truth_conflict_platform.selected_platform and video_type_final. Each skeleton is expressed as a 3–5 beat abstract sequence (e.g. "Shock -> Recognition -> Ritual -> Reclamation -> Offer"). Each beat should be labelled with its emotional function, not its content.

CRITICAL TIMING CONSTRAINT:
The total sum of `seconds_budget` across all beats in the selected `beat_sequence` MUST exactly equal `video_length_seconds` ({video_length_seconds} seconds). Evaluate the narrative_budget allocations provided, but ensure the final assigned seconds exactly add up to {video_length_seconds}.

Skeleton evaluation criteria:
- Does this skeleton create a meaningful emotional arc from strategy.agents.conflict_identification.enemy (opening tension) to the product (resolution)?
- Does it fit within the narrative_budget constraints from Agent 6?
- Does it create the right structural home for the preferred scene's selected_role from Agent 9 (when scene branch ran)?
- Is it distinctly different from the other two candidate skeletons — not a minor variation?

Select the strongest skeleton and write it as the final output. Log all three candidates and scoring rationale to pipeline.

Return the final results matching the requested output schema strictly.
"""

    # 3. Call Gemini
    invoke_start = time.time()
    logger.info(f"Agent 12: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 12: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": Agent12Output.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 12: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 12: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 12: Successfully parsed JSON response.")

        result = Agent12Output(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 12: Successfully validated structured output with Pydantic.")
    except Exception as e:
        logger.error(f"Agent 12: Gemini API Call failed: {e}")
        failure_log = {
            "agent_key": "agent_12_narrative_skeleton_generator",
            "status": "failed",
            "timestamp": time.time(),
            "error_msg": str(e)
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": failure_log}},
            upsert=True
        )
        raise

    # 4. Save to DB
    logger.info(f"Agent 12: Updating IDEATION and PIPELINE collections...")
    
    pipeline_log = {
        "agent_key": "agent_12_narrative_skeleton_generator",
        "duration": time.time() - start_time,
        "api_duration": api_duration,
        "timestamp": time.time(),
        "reasoning": result.reasoning,
        "candidate_skeletons_scored": [sc.model_dump() for sc in result.candidate_skeletons_scored],
        "selection_rationale": result.selection_rationale
    }
    
    # Store the actual ideation fields
    ideation_output = {
        "selected_skeleton": result.selected_skeleton,
        "beat_sequence": [b.model_dump() for b in result.beat_sequence]
    }

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {"narrative_skeleton": ideation_output}}
        )
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info(f"Agent 12: Successfully updated DB.")
    except Exception as e:
        logger.error(f"Agent 12: DB Update failed: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"Agent 12: Completed successfully in {duration:.2f}s")
    return result
