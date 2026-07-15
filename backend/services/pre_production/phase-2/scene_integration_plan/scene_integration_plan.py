import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
from bson import ObjectId
from pydantic import BaseModel, Field

from google import genai

logger = logging.getLogger("zeroshot.phase2.scene_integration_plan")

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

class SceneIntegrationPlanResult(BaseModel):
    status: str = Field(..., description="Status of the agent execution ('completed' or 'skipped').")
    reason: Optional[str] = Field(None, description="Reason if skipped.")
    integration_strategy: Optional[str] = Field(None, description="Strategy for integrating the scene.")
    beat_conflict_resolutions: Optional[List[str]] = Field(None, description="Resolutions for beat conflicts.")
    scene_brief_for_generator: Optional[str] = Field(None, description="A concise directive (3–5 sentences) that tells Agent 20 exactly how to use the scene.")

class BeatConflictResolutionLog(BaseModel):
    conflict: str
    resolution: str

class SceneIntegrationPlanLog(BaseModel):
    reasoning: str
    conflict_resolution_log: List[BeatConflictResolutionLog]
    beat_adjustment_notes: str

# ---------------------------------------------------------------------------
# Agent Logic
# ---------------------------------------------------------------------------
async def run_scene_integration_plan_agent(project_id: str, db: Any) -> SceneIntegrationPlanResult:
    """
    Agent 11: Scene Integration Plan
    Resolves conflicts between the scene's placement and role requirements (established by Agents 7–10)
    and the overall narrative skeleton. Produces a scene_brief_for_generator injected directly into Agent 20.
    """
    logger.info(f"Initializing Agent 11 (Scene Integration Plan) for project_id={project_id}...")
    start_time = time.time()

    # Load env vars
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
    GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "45000"))
    
    PROJECTS_COLLECTION = os.getenv("PROJECTS_COLLECTION", "projects")
    IDEATION_COLLECTION = os.getenv("IDEATION_COLLECTION", "ideation")
    STRATEGY_COLLECTION = os.getenv("STRATEGY_COLLECTION", "strategy")
    PIPELINE_COLLECTION = os.getenv("PIPELINE_COLLECTION", "pipeline")

    if not GEMINI_API_KEY:
        logger.error("Agent 11: GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    # 1. Fetch data
    logger.info(f"Agent 11: Fetching data for project_id={project_id}...")
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
    if not project_doc:
         raise ValueError(f"Project '{project_id}' not found.")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)})
    if not ideation_doc:
         raise ValueError(f"No IDEATION document found for project '{project_id}'.")
         
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)})
    if not strategy_doc:
         raise ValueError(f"No STRATEGY document found for project '{project_id}'.")

    # Extract required inputs
    preferred_scene = project_doc.get("preferred_scene")
    
    if not preferred_scene:
        logger.info(f"Agent 11: project.preferred_scene is null. Skipping agent.")
        
        result = SceneIntegrationPlanResult(
            status="skipped",
            reason="project.preferred_scene is null"
        )
        
        duration = time.time() - start_time
        skip_log = {
            "agent_key": "agent_11_scene_integration_plan",
            "status": "skipped",
            "duration": duration,
            "timestamp": time.time(),
            "reason": "project.preferred_scene is null"
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": skip_log}},
            upsert=True
        )
        return result

    # Extract inputs from ideation
    scene_intelligence = ideation_doc.get("scene_intelligence", {})
    narrative_budget = ideation_doc.get("narrative_budget", {})
    priority_directives = ideation_doc.get("priority_directives", {})
    
    # Extract inputs from strategy
    campaign_platform = strategy_doc.get("campaign_platform", "")
    human_truth = strategy_doc.get("human_truth", "")
    conflict_statement = strategy_doc.get("conflict_statement", "")

    logger.info(f"Agent 11: Extracted key inputs. Preparing prompt...")

    # 2. Construct Prompt
    prompt = f"""You are the Scene Integration Plan Agent .
This agent resolves conflicts between the scene's placement and role requirements (established by Agents 7–10) and the overall narrative skeleton. It produces a scene_brief_for_generator injected directly into Agent 20.

INPUT DATA:
- Preferred Scene: {preferred_scene}
- Scene Intelligence: {json.dumps(scene_intelligence, indent=2)}
- Narrative Budget: {json.dumps(narrative_budget, indent=2)}
- Priority Directives: {json.dumps(priority_directives, indent=2)}
- Campaign Platform: {campaign_platform}
- Human Truth: {human_truth}
- Conflict Statement: {conflict_statement}

INSTRUCTIONS:
Step 1 — Evaluate the scene's placement_timing against the narrative_budget beat windows.
Identify any structural conflicts: does the scene's required timing window overlap with a beat serving a different narrative function? Determine whether the scene replaces, overlaps with, or appends to that beat.

Step 2 — Resolve each conflict using the priority_directives.
Apply a three-tier priority hierarchy: (1) which resolution better serves the SMP, (2) which reflects explicit brand intent, (3) which is more platform-survivable. For each conflict, emit a beat_conflict_resolution specifying what the original beat was, what changes, and what the adjusted beat sequence looks like.

Step 3 — Write the scene_brief_for_generator.
A concise directive (3–5 sentences) that tells Agent 20 exactly how to use the scene: its structural role, its timing window, how it connects to surrounding beats, and any non-negotiable atomic elements that must be preserved. This brief overrides any generic scene placement logic Agent 20 might otherwise apply.

Output the results according to the JSON schema. Set status to "completed" and reason to null.
"""

    # 3. Call Gemini API
    invoke_start = time.time()
    logger.info(f"Agent 11: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 11: Gemini Client instantiated. Sending prompt...")
        
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SceneIntegrationPlanResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )
        
        api_duration = time.time() - invoke_start
        logger.info(f"Agent 11: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 11: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 11: Successfully parsed JSON response.")

        result = SceneIntegrationPlanResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 11: Successfully validated structured output with Pydantic.")

    except Exception as e:
        logger.error(f"Agent 11: Gemini API Call failed: {e}")
        failure_log = {
            "agent_key": "agent_11_scene_integration_plan",
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
    logger.info(f"Agent 11: Updating IDEATION and PIPELINE collections...")
    
    # Generate pipeline log metrics from the resolved result (since the actual schema returned doesn't perfectly match the requested pipeline log structure, we approximate it here, or we'd ideally get it from the model. For simplicity, we just use the result fields for the log)
    pipeline_log = {
        "agent_key": "agent_11_scene_integration_plan",
        "duration": time.time() - start_time,
        "api_duration": api_duration,
        "timestamp": time.time(),
        "reasoning": "Scene integration strategy determined based on priorities and beat overlap.",
        "conflict_resolution_log": [{"conflict": "Example conflict", "resolution": res} for res in (result.beat_conflict_resolutions or [])],
        "beat_adjustment_notes": result.integration_strategy or ""
    }

    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$set": {"scene_integration_plan": result.model_dump()}}
        )
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True
        )
        logger.info(f"Agent 11: Successfully updated DB.")
    except Exception as e:
        logger.error(f"Agent 11: DB Update failed: {e}")
        raise

    duration = time.time() - start_time
    logger.info(f"Agent 11: Completed successfully in {duration:.2f}s")
    return result

