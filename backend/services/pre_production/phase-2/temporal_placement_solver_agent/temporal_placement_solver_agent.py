import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field

from google import genai

logger = logging.getLogger("zeroshot.phase2.temporal_placement_solver_agent")

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

class PlacementAlternativeRejected(BaseModel):
    window: str = Field(..., description="The alternative placement window that was considered.")
    why_rejected: str = Field(..., description="The reasoning for rejecting this alternative window.")

class TemporalPlacementSolverResult(BaseModel):
    placement_timing: str = Field(..., description="Determined precise timing window (e.g., '0-3s', '18-22s', 'final 4 seconds').")
    reasoning: str = Field(..., description="A one-sentence justification referencing which pacing rule governs the decision.")
    pacing_rule_applied: str = Field(..., description="The exact rule applied (e.g., 'Attention decay rule', 'Emotional escalation rule', 'Hook reset rule', 'Offer proximity rule').")
    placement_alternatives_rejected: List[PlacementAlternativeRejected] = Field(..., description="List of alternatives considered and why they were rejected.")
    status: Optional[str] = Field(None, description="Status of the agent execution.")

# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------
async def run_temporal_placement_solver_agent(project_id: str, db: Any) -> TemporalPlacementSolverResult:
    """
    Agent 10: Temporal Placement Solver Agent
    Determines the precise timing window for the scene based on narrative budget, video type, and selected role.
    Applies strict pacing rules for attention decay, emotional escalation, hook reset, and offer proximity.
    """
    logger.info(f"Initializing Agent 10 (Temporal Placement Solver Agent) for project_id={project_id}...")
    start_time = time.time()

    # Load env vars
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))
    
    PROJECTS_COLLECTION = os.getenv("PROJECTS_COLLECTION", "projects")
    IDEATION_COLLECTION = os.getenv("IDEATION_COLLECTION", "ideation")
    PIPELINE_COLLECTION = os.getenv("PIPELINE_COLLECTION", "pipeline")

    if not GEMINI_API_KEY:
        logger.error("Agent 10: GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY enviroment variable not set.")

    # 1. Fetch data
    logger.info(f"Agent 10: Fetching data for project_id={project_id}...")
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
    if not project_doc:
         raise ValueError(f"Project '{project_id}' not found.")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    if not ideation_doc:
         raise ValueError(f"No IDEATION document found for project '{project_id}'.")

    # Extract required inputs
    scene_intelligence = ideation_doc.get("scene_intelligence", {})
    selected_role = scene_intelligence.get("selected_role")
    narrative_budget = ideation_doc.get("narrative_budget", {})
    video_type_final = ideation_doc.get("video_type_final", "Unknown")

    if not selected_role:
         logger.warning(f"Agent 10: Missing 'scene_intelligence.selected_role' for project {project_id}.")

    logger.info(f"Agent 10: Extracted inputs - selected_role='{selected_role}', video_type_final='{video_type_final}'.")

    # 2. Construct Prompt
    prompt = f"""You are the Temporal Placement Solver Agent .
Your task is to determine the precise timing window within the narrative where the preferred scene should appear. 
Timing is not cosmetic — misplaced scenes are the most common reason structurally valid concepts fail in execution.

INPUT DATA:
- Video Type: {video_type_final}
- Selected Scene Role (from Agent 9): {selected_role}
- Narrative Budget Content: {json.dumps(narrative_budget, indent=2)}

INSTRUCTIONS: 
Using the incoming narrative_budget second-by-second allocation and the selected_role above, determine the exact placement window (e.g. "0-3 seconds", "18-22 seconds", "final 4 seconds").

Apply the following attention and pacing rules:
1. Attention decay rule: On feed placements, attention drops 40% after the first 3 seconds. If the selected role is hook disruptor, the scene must begin at 0 seconds with no setup.
2. Emotional escalation rule: In narrative formats, emotional intensity should peak in the final third of the video. If the selected role is emotional climax, the scene must fall in the 65-85% window of total duration.
3. Hook reset rule: In longer formats (30s+), the ad can earn a second attention reset at the midpoint. A scene placed at the 45-55% mark can function as a structural reset that re-engages drifting viewers.
4. Offer proximity rule: If the scene's role is proof demonstration, it must be immediately adjacent to the offer communication beat — separating them by more than 6 seconds breaks commercial intent continuity.

Output a specific placement window with a one-sentence justification referencing which pacing rule governs the decision, plus the list of alternatives rejected. Return JSON adhering exactly to the schema.
"""

    # 3. Call Gemini API
    invoke_start = time.time()
    logger.info(f"Agent 10: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 10: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": TemporalPlacementSolverResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 10: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 10: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 10: Successfully parsed JSON response.")

        result = TemporalPlacementSolverResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 10: Successfully validated structured output with Pydantic.")

    except Exception as e:
        logger.error(f"Agent 10: Gemini API Call failed: {e}")
        # Insert failure log
        failure_log = {
            "agent_key": "agent_10_temporal_placement_solver_agent",
            "status": "failed",
            "timestamp": time.time(),
            "execution_time_sec": time.time() - start_time,
            "error_msg": str(e)
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": failure_log}},
            upsert=True
        )
        raise

    # 4. Save results to DB
    logger.info(f"Agent 10: Updating IDEATION and PIPELINE collections for project_id={project_id}...")
    
    # Partial Write to Ideation
    update_mask = {
        "status.agent_10_temporal_placement_solver": "completed",
        "scene_intelligence.placement_timing": result.placement_timing
    }
    
    await db[IDEATION_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$set": update_mask},
        upsert=True
    )

    # Write to Pipeline Collection
    pipeline_log = {
        "agent_key": "agent_10_temporal_placement_solver_agent",
        "status": "completed",
        "timestamp": time.time(),
        "execution_time_sec": time.time() - start_time,
        "reasoning": result.reasoning,
        "pacing_rule_applied": result.pacing_rule_applied,
        "placement_alternatives_rejected": [alt.model_dump() for alt in result.placement_alternatives_rejected]
    }

    await db[PIPELINE_COLLECTION].update_one(
        {"project_id": str(project_id)},
        {"$push": {"agent_logs": pipeline_log}},
        upsert=True
    )

    total_duration = time.time() - start_time
    logger.info(f"Agent 10: Total duration {total_duration:.2f}s. Completed successfully.")

    return result

